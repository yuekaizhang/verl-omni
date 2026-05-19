# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Torch-only voice-clone smoke for Qwen3-TTS-12Hz-*-Base.

Uses the qwen-tts package's :class:`Qwen3TTSModel.generate_voice_clone`
API directly (no vllm-omni, no Ray, no verl-omni). Goal: prove the base
checkpoint can voice-clone a Chinese AISHELL reference to synthesize a
*different* target sentence so we have a ground-truth reference for the
vllm-omni rollout path.

Usage::

    .venv/bin/python scripts/test_qwen_tts_torch_voice_clone.py \\
        --model-path .hf_cache/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio /tmp/aishell_real/.../BAC009S0739W0269-7441.wav \\
        --ref-text  "索尼昨日发布了一个好消息和一个坏消息。" \\
        --prompt-text "美联航在声明中也为此次事故道歉。" \\
        --language Chinese \\
        --out /tmp/torch_voice_clone.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# Make qwen_tts importable (the repo lives outside the site-packages).
QWEN3_TTS_SOURCE_DIR = os.environ.get(
    "QWEN3_TTS_SOURCE_DIR",
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/Qwen3-TTS",
)
if QWEN3_TTS_SOURCE_DIR and QWEN3_TTS_SOURCE_DIR not in sys.path:
    sys.path.insert(0, QWEN3_TTS_SOURCE_DIR)

from qwen_tts import Qwen3TTSModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True,
                        help="Path or HF id for Qwen3-TTS-12Hz-*-Base.")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--language", default="Chinese",
                        help="One of Chinese / English / Japanese / Korean / German / "
                             "French / Russian / Portuguese / Spanish / Italian / Auto.")
    parser.add_argument("--out", default="/tmp/torch_voice_clone.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl", default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Falls back to sdpa when flash-attn isn't installed.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--x-vector-only", action="store_true",
                        help="Use speaker-embedding-only voice clone "
                             "(no ref_text required, lower fidelity).")
    parser.add_argument(
        "--dump-voice-clone-prompt",
        default=None,
        help="If set, write the dict produced by "
             "``Qwen3TTSModel.create_voice_clone_prompt`` "
             "(ref_code / ref_spk_embedding / x_vector_only_mode / icl_mode) "
             "to this pickle path. The standalone vllm-omni test can then "
             "thread it through ``additional_information.voice_clone_prompt`` "
             "to bypass vllm-omni's internal ``_encode_ref_audio_to_code`` / "
             "``_extract_speaker_embedding`` fallback.",
    )
    args = parser.parse_args()

    print(f"[torch-vc] loading {args.model_path} on {args.device} "
          f"(attn={args.attn_impl})", flush=True)
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    print(f"[torch-vc] loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"[torch-vc] ref_audio={args.ref_audio}", flush=True)
    print(f"[torch-vc] ref_text ={args.ref_text!r}", flush=True)
    print(f"[torch-vc] prompt   ={args.prompt_text!r}", flush=True)
    print(f"[torch-vc] language ={args.language!r}", flush=True)

    if args.dump_voice_clone_prompt:
        import pickle
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only,
        )
        # Mirror Qwen3TTSModel._prompt_items_to_voice_clone_prompt:
        vc_prompt = dict(
            ref_code=[it.ref_code for it in prompt_items],
            ref_spk_embedding=[it.ref_spk_embedding for it in prompt_items],
            x_vector_only_mode=[it.x_vector_only_mode for it in prompt_items],
            icl_mode=[it.icl_mode for it in prompt_items],
        )
        # Convert torch tensors → plain Python lists. vllm-omni's
        # msgspec IPC (engine ↔ worker) doesn't preserve ``torch.Tensor``
        # nor ``numpy.ndarray`` reliably across the boundary; the talker
        # side already special-cases ``list``/``np.ndarray`` for
        # ``ref_spk_embedding`` (see ``qwen3_tts_talker.py:1364-1376``)
        # but ``ref_code`` currently only accepts ``Tensor`` / ``ndarray``
        # (lines 1339-1346). To keep the dump end-to-end serializable,
        # emit *nested Python lists* for both ``ref_code`` (T×Q ints)
        # and ``ref_spk_embedding`` (1024 floats). The vllm-omni
        # standalone test enriches the talker with a list→tensor
        # adapter for ref_code before calling engine.generate.
        def _to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu").contiguous().to(torch.float32).tolist()
            return x
        vc_prompt = {k: [_to_list(v) for v in vs] for k, vs in vc_prompt.items()}
        dump_path = Path(args.dump_voice_clone_prompt)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(vc_prompt, f)
        print(f"[torch-vc] dumped voice_clone_prompt -> {dump_path}", flush=True)
        for k, vs in vc_prompt.items():
            shapes = [tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__ for v in vs]
            print(f"[torch-vc]   {k}: {shapes}", flush=True)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=args.top_k,
        subtalker_top_p=args.top_p,
        subtalker_temperature=args.temperature,
    )

    t1 = time.time()
    torch.cuda.synchronize()
    wavs, sr = tts.generate_voice_clone(
        text=args.prompt_text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
        **gen_kwargs,
    )
    torch.cuda.synchronize()
    print(f"[torch-vc] generated in {time.time() - t1:.1f}s "
          f"n_wavs={len(wavs)} sr={sr}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wavs[0], sr)
    print(f"[torch-vc] wrote {out_path} "
          f"({len(wavs[0]) / sr:.2f}s, dtype={wavs[0].dtype})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
