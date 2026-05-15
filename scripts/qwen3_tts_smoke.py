# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""T1 + T2b smoke: standalone vllm-omni Qwen3-TTS Base inference.

Loads ``Qwen3-TTS-12Hz-0.6B-Base`` via the AsyncOmni engine with the
verl-omni-side stage_config override (final_output:true + logprobs:1 on
stage 0) and asserts:

- the talker checkpoint has ``speaker_encoder`` weights (T1 fail-closed),
- stage 0 emits an OmniRequestOutput with codec_tokens + per-token
  logprobs on ``request_output.outputs[i].logprobs`` (T2b),
- stage 1 emits a non-empty waveform.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/qwen3_tts_smoke.py \
        --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --ref-audio /path/to/ref.wav \
        --ref-text "你好" \
        --prompt-text "请合成这一段中文文本"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np


def _check_speaker_encoder(model_path: str) -> bool:
    """Inspect the talker checkpoint for ``speaker_encoder`` weights.

    Fail-closed: returns False when the safetensors file cannot be opened
    or no ``speaker_encoder.*`` keys are present. A Hub ID that has not
    been downloaded locally counts as "not verifiable" and is rejected;
    the caller is expected to download the checkpoint first (via
    ``huggingface-cli download`` or similar).
    """

    import safetensors.torch as st

    weights_file: Path | None = None
    for candidate in (
        Path(model_path) / "model.safetensors",
        Path(model_path) / "talker" / "model.safetensors",
    ):
        if candidate.exists():
            weights_file = candidate
            break
    if weights_file is None:
        print(
            f"[T1] FAIL-CLOSED: cannot find model.safetensors under {model_path!r}. "
            "Provide a local checkpoint directory or run `huggingface-cli download "
            "{model_path}` first; we will not assume speaker_encoder is present.",
            file=sys.stderr,
        )
        return False
    try:
        with st.safe_open(str(weights_file), framework="pt") as f:
            keys = list(f.keys())
    except Exception as exc:
        print(f"[T1] FAIL-CLOSED: unable to read {weights_file}: {exc}", file=sys.stderr)
        return False
    has_speaker_encoder = any("speaker_encoder" in k for k in keys)
    if not has_speaker_encoder:
        print(
            f"[T1] FAIL-CLOSED: {weights_file} has no speaker_encoder.* keys "
            "(out of {len(keys)} total). Voice cloning would not work.",
            file=sys.stderr,
        )
    return has_speaker_encoder


async def _run(args: argparse.Namespace) -> int:
    has_speaker = _check_speaker_encoder(args.model_path)
    if not has_speaker:
        print("[T1] FAIL: speaker_encoder weights not found in checkpoint — voice cloning disabled.", file=sys.stderr)
        return 2
    print(f"[T1] speaker_encoder present in {args.model_path}")

    from vllm.sampling_params import SamplingParams
    from vllm_omni.entrypoints import AsyncOmni
    from vllm_omni.inputs.data import OmniCustomPrompt

    stage_configs_path = (
        Path(__file__).resolve().parent.parent
        / "verl_omni"
        / "pipelines"
        / "qwen3_tts_grpo"
        / "stage_configs"
        / "qwen3_tts.yaml"
    )

    print(f"[T1] launching AsyncOmni with model={args.model_path} stage_configs_path={stage_configs_path}")
    engine = AsyncOmni(
        model=args.model_path,
        stage_configs_path=str(stage_configs_path),
        trust_remote_code=True,
        enforce_eager=True,
    )

    # The Qwen3-TTS talker reads ``task_type``, ``ref_audio``, ``ref_text``
    # from ``info_dict`` (which is sourced from the prompt's
    # ``additional_information`` field). ``task_type`` is indexed as
    # ``info_dict.get("task_type")[0]`` so we pass it as a one-element list.
    additional_information = {
        "ref_audio": args.ref_audio,
        "ref_text": [args.ref_text],
        "task_type": ["Base"],
    }
    prompt = {"prompt": args.prompt_text, "additional_information": additional_information}
    sampling = SamplingParams(temperature=0.9, top_k=50, max_tokens=512, logprobs=1, stop_token_ids=[2150])
    # Stage 1 (code2wav) is a deterministic generator; vllm-omni's
    # orchestrator clones the params to build an engine-core request, so a
    # plain SamplingParams works as a passthrough.
    stage1_params = SamplingParams(temperature=0.0, max_tokens=65536, detokenize=True)

    saw_codec_tokens = False
    saw_logprobs = False
    saw_waveform = False
    codec_token_count = 0
    waveform_samples = 0

    try:
        async for omni_out in engine.generate(prompt=prompt, request_id="smoke-1", sampling_params_list=[sampling, stage1_params]):
            stage_id = getattr(omni_out, "stage_id", None)
            print(f"[T1] yielded stage_id={stage_id} finished={getattr(omni_out, 'finished', '?')}")
            if stage_id == 0 and omni_out.request_output is not None:
                for completion in omni_out.outputs:
                    tokens = list(getattr(completion, "token_ids", []) or [])
                    logprobs = getattr(completion, "logprobs", None)
                    if tokens:
                        saw_codec_tokens = True
                        codec_token_count += len(tokens)
                    if logprobs:
                        saw_logprobs = True
                        first = next(iter(logprobs[0].values())) if logprobs and logprobs[0] else None
                        print(f"[T2b] first stage-0 logprob = {getattr(first, 'logprob', first)!r}")
            elif stage_id == 1:
                mm = omni_out.multimodal_output or {}
                audio = mm.get("audio") or mm.get("waveform")
                if audio is not None:
                    arr = np.asarray(audio[0] if isinstance(audio, list) else audio)
                    if arr.size > 0:
                        saw_waveform = True
                        waveform_samples = int(arr.size)
    finally:
        shutdown = getattr(engine, "shutdown", None) or getattr(engine, "close", None)
        if shutdown is not None:
            res = shutdown()
            if asyncio.iscoroutine(res):
                await res

    result = {
        "speaker_encoder_present": has_speaker,
        "saw_codec_tokens": saw_codec_tokens,
        "codec_token_count": codec_token_count,
        "saw_logprobs": saw_logprobs,
        "saw_waveform": saw_waveform,
        "waveform_samples": waveform_samples,
    }
    print(json.dumps(result, indent=2))
    if saw_codec_tokens and saw_logprobs and saw_waveform:
        return 0
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--prompt-text", required=True)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
