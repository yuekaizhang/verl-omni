"""Standalone vLLM-Omni Qwen3-TTS rollout test (no Ray, no verl).

Mirrors the official offline-inference demo at
``vllm-omni/examples/offline_inference/text_to_speech/qwen3_tts/end2end.py``
(Base ICL voice-clone path) so the *only* differences vs upstream are:

1. We swap in real AISHELL ref_audio + ref_text + prompt_text.
2. We tee the synthesized audio through Qwen3-ASR to compute CER vs the
   target prompt — that's the gate we need for T14 GRPO reward to be
   meaningful.

The single most important deviation from my prior standalone (which
returned ``'索尼' / '索尼昨日'`` echoes regardless of knob) is the
prompt-token construction: the demo passes
``prompt_token_ids = [0] * estimate_prompt_len(additional_information)``
— a *placeholder of the right length*, not vLLM's tokenization of
``prompt_text``. The Qwen3-TTS talker's ``preprocess`` replaces input
embeddings via ``_build_prompt_embeds`` and uses ``input_ids.shape[0]``
as a span/length budget. Passing a Chinese prompt string instead of the
placeholder produced span_len ≠ embed_len, which silently corrupted the
ICL prefill region.

Usage (inside the Slurm container, USER=root):

    .venv/bin/python scripts/test_vllm_omni_stand_alone.py \\
        --model-path .hf_cache/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio /tmp/aishell_real/.../BAC009S0739W0269-7441.wav \\
        --ref-text "索尼昨日发布了一个好消息和一个坏消息。" \\
        --prompt-text "美联航在声明中也为此次事故道歉。" \\
        --language Chinese \\
        --out-dir /tmp/standalone_tts_out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

# Must be set BEFORE any vllm import — see end2end.py:16.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen3_tts_autoregister  # noqa: E402

qwen3_tts_autoregister.setup()

import torch  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm_omni import AsyncOmni, Omni  # noqa: E402
from vllm_omni.outputs import OmniRequestOutput  # noqa: E402

logger = logging.getLogger(__name__)


def _estimate_prompt_len(
    additional_information: dict,
    model_name: str,
    _cache: dict = {},
) -> int:
    """Port of end2end.py:_estimate_prompt_len.

    Computes the placeholder ``prompt_token_ids`` length the talker
    expects given the ``additional_information`` payload (task_type +
    text + ref_audio + ref_text + language + speaker etc.). The talker's
    ``preprocess`` replaces all input embeddings via ``_build_prompt_embeds``
    but the *length* must match the embeddings produced, otherwise the
    ICL prefill region gets corrupted (which is what produced the
    ``'索尼' / '索尼昨日'`` ref-text echo in my earlier standalone runs).
    """
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    if model_name not in _cache:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)

        speech_tok = None
        try:
            from transformers.utils import cached_file
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import (
                Qwen3TTSTokenizer,
            )

            st_cfg_path = cached_file(model_name, "speech_tokenizer/config.json")
            if st_cfg_path:
                speech_tok = Qwen3TTSTokenizer.from_pretrained(
                    os.path.dirname(st_cfg_path), torch_dtype=torch.bfloat16
                )
        except Exception as e:
            logger.debug("Could not load speech tokenizer: %s", e)

        _cache[model_name] = (tok, getattr(cfg, "talker_config", None), speech_tok)

    tok, tcfg, speech_tok = _cache[model_name]
    task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

    def _estimate_ref_code_len(ref_audio):
        if not isinstance(ref_audio, (str, list)):
            print(f"[standalone-tts] _estimate_ref_code_len: bad type {type(ref_audio)}", flush=True)
            return None
        audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
        if not isinstance(audio_path, str) or not audio_path.strip():
            print(f"[standalone-tts] _estimate_ref_code_len: bad path {audio_path!r}", flush=True)
            return None
        # First fall back to a pure-soundfile read (load_audio is unreliable
        # for local paths on some vllm releases) — this lets us at least
        # get the codec-frame-rate fallback before attempting the heavier
        # speech_tok.encode call.
        try:
            audio, sr = sf.read(audio_path, always_2d=False)
            if audio.ndim > 1:
                audio = audio[:, 0]
            wav_np = np.asarray(audio, dtype=np.float32)
        except Exception as e:
            print(f"[standalone-tts] _estimate_ref_code_len: sf.read failed: {e}", flush=True)
            return None
        # Prefer the codec-tokenizer-exact frame count when available.
        if speech_tok is not None:
            try:
                enc = speech_tok.encode(wav_np, sr=int(sr), return_dict=True)
                ref_code = getattr(enc, "audio_codes", None)
                if isinstance(ref_code, list):
                    ref_code = ref_code[0] if ref_code else None
                if ref_code is not None and hasattr(ref_code, "shape"):
                    shape = ref_code.shape
                    if len(shape) == 2:
                        n = int(shape[0])
                        print(f"[standalone-tts] _estimate_ref_code_len: speech_tok n={n}", flush=True)
                        return n
                    if len(shape) == 3:
                        n = int(shape[1])
                        print(f"[standalone-tts] _estimate_ref_code_len: speech_tok n={n} (shape3)", flush=True)
                        return n
            except Exception as e:
                print(f"[standalone-tts] _estimate_ref_code_len: speech_tok.encode failed: {e}",
                      flush=True)
        codec_hz = getattr(tcfg, "codec_frame_rate", None) or 12
        n = int(len(wav_np) / sr * codec_hz)
        print(f"[standalone-tts] _estimate_ref_code_len: fallback n={n} "
              f"(dur={len(wav_np) / sr:.2f}s @ {codec_hz}Hz)", flush=True)
        return n

    return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
        additional_information=additional_information,
        task_type=task_type,
        tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
        codec_language_id=getattr(tcfg, "codec_language_id", None),
        spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
        estimate_ref_code_len=_estimate_ref_code_len,
    )


def _build_base_input(
    *,
    model_name: str,
    prompt_text: str,
    ref_audio: str,
    ref_text: str,
    language: str,
    x_vector_only_mode: bool = False,
    max_new_tokens: int = 2048,
) -> dict:
    """Mirror end2end.py:get_base_query payload shape exactly."""
    additional_information = {
        "task_type": ["Base"],
        "ref_audio": [ref_audio],
        "ref_text": [ref_text],
        "text": [prompt_text],
        "language": [language],
        "x_vector_only_mode": [x_vector_only_mode],
        "max_new_tokens": [max_new_tokens],
    }
    prompt_len = _estimate_prompt_len(additional_information, model_name)
    return {
        "prompt_token_ids": [0] * prompt_len,
        "additional_information": additional_information,
    }


def _save_wav(out_path: Path, mm: dict) -> tuple[float, int]:
    """Mirror end2end.py:_save_wav. Returns (duration_s, sample_rate)."""
    audio_data = mm["audio"]
    sr_raw = mm["sr"]
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
    if isinstance(audio_data, list):
        # Convert each chunk to torch (handles ndarray/Tensor) then concat.
        audio_tensor = torch.cat(
            [a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a)) for a in audio_data],
            dim=-1,
        )
    else:
        audio_tensor = audio_data if isinstance(audio_data, torch.Tensor) else torch.as_tensor(np.asarray(audio_data))
    wav_np = audio_tensor.float().cpu().numpy().flatten()
    sf.write(out_path, wav_np, samplerate=sr, format="WAV")
    return float(len(wav_np) / sr), sr


async def _run_one(
    engine: AsyncOmni,
    *,
    label: str,
    prompt: dict,
    prompt_text: str,
    ref_audio: str,
    ref_text: str,
    language: str,
    out_dir: Path,
) -> dict:
    """Submit one request, collect the streamed final OmniRequestOutput, save wav."""
    request_id = f"{label}-{uuid4().hex}"
    print(f"[standalone-tts] {label}: request_id={request_id} "
          f"prompt={prompt_text!r} language={language!r} "
          f"prompt_token_ids_len={len(prompt['prompt_token_ids'])}", flush=True)

    final_mm = None
    final_finished = False
    chunk_count = 0
    t0 = time.perf_counter()
    async for stage_output in engine.generate(prompt, request_id=request_id):
        if not isinstance(stage_output, OmniRequestOutput):
            continue
        if stage_output.request_output is None:
            continue
        mm = stage_output.request_output.outputs[0].multimodal_output if (
            stage_output.request_output.outputs
        ) else stage_output.multimodal_output
        if mm and "audio" in mm:
            final_mm = mm
        chunk_count += 1
        if stage_output.finished:
            final_finished = True
            break

    elapsed_s = time.perf_counter() - t0
    dur_s = None
    sr = None
    wav_path = out_dir / f"{label}_sample0.wav"
    if final_mm is not None:
        dur_s, sr = _save_wav(wav_path, final_mm)
        print(f"[standalone-tts] {label}: wrote {wav_path} "
              f"({dur_s:.2f}s @ {sr}Hz, chunks={chunk_count}, "
              f"elapsed={elapsed_s:.1f}s, finished={final_finished})", flush=True)
    else:
        print(f"[standalone-tts] {label}: NO audio produced "
              f"(chunks={chunk_count}, elapsed={elapsed_s:.1f}s)", flush=True)

    return {
        "label": label,
        "request_id": request_id,
        "prompt_text": prompt_text,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "language": language,
        "wav_path": str(wav_path) if final_mm is not None else None,
        "wav_dur_s": dur_s,
        "sample_rate": sr,
        "chunks": chunk_count,
        "elapsed_s": elapsed_s,
        "finished": final_finished,
    }


def _build_sampling_params_list(
    n: int, max_tokens: int, *, seed: int | None = None,
) -> list[SamplingParams] | None:
    """Two-stage SamplingParams mirroring the verl-omni rollout.

    Returns None when ``n == 1`` *and* no seed override is requested, so
    the engine falls back to the stage_config defaults (which is the
    working path end2end.py uses for single-completion inference). When a
    seed is supplied we always build the explicit list so per-sample seed
    variation works even at n=1.
    """
    if (n is None or n <= 1) and seed is None:
        return None
    stage0 = SamplingParams(
        n=max(1, n or 1),
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=max_tokens,
        logprobs=1,
        repetition_penalty=1.0,
        stop_token_ids=[2150],
        seed=seed,
    )
    stage1 = SamplingParams(
        temperature=0.0,
        max_tokens=65536,
        detokenize=True,
    )
    return [stage0, stage1]


def main_sync(args: argparse.Namespace) -> int:
    """Mirror end2end.py:main — sync Omni, one batch, yield final wav per request."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omni_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        dtype="bfloat16",
    )
    if args.stage_config:
        omni_kwargs["stage_configs_path"] = str(Path(args.stage_config).resolve())
    print(f"[standalone-tts] Constructing Omni with engine_args={omni_kwargs}",
          flush=True)
    omni = Omni(**omni_kwargs)

    summaries: list[dict] = []
    # When ``--n-via-loop`` is set we ignore SamplingParams.n (which vllm-omni's
    # Qwen3-TTS pipeline currently collapses to 1) and instead emit ``n`` *separate*
    # requests with different seeds — that's how verl's AgentLoopManager appears
    # to produce GRPO-grouped completions in production (each agent loop iteration
    # → its own request_id, codec sequence, and waveform). Each request becomes a
    # distinct ``sample_index`` in the resulting summary.
    if args.n_via_loop and args.n > 1:
        loop_n = args.n
        sampling_params_list = None
    else:
        loop_n = 1
        sampling_params_list = _build_sampling_params_list(args.n, args.max_new_tokens)

    for lang in args.languages:
        for trial in range(loop_n):
            seed = args.seed_base + trial if args.n_via_loop else None
            if args.n_via_loop:
                sampling_params_list = _build_sampling_params_list(
                    1, args.max_new_tokens, seed=seed,
                )
            prompt = _build_base_input(
                model_name=args.model_path,
                prompt_text=args.prompt_text,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                language=lang,
                max_new_tokens=args.max_new_tokens,
            )
            label = (
                f"Base_lang{lang}_n{args.n}_trial{trial}"
                if args.n_via_loop else f"Base_lang{lang}_n{args.n}"
            )
            print(f"[standalone-tts] {label}: language={lang!r} "
                  f"prompt_token_ids_len={len(prompt['prompt_token_ids'])} "
                  f"n={args.n} loop_trial={trial} seed={seed} "
                  f"prompt_text={args.prompt_text!r}", flush=True)
            t0 = time.perf_counter()
            for stage_outputs in omni.generate(
                [prompt], sampling_params_list=sampling_params_list,
            ):
                request_output = stage_outputs.request_output
                if request_output is None or not request_output.outputs:
                    print(f"[standalone-tts] {label}: empty request_output", flush=True)
                    continue
                print(f"[standalone-tts] {label}: got {len(request_output.outputs)} "
                      f"completion(s) in this stage_output", flush=True)
                for ci, completion in enumerate(request_output.outputs):
                    mm = completion.multimodal_output
                    if not mm or "audio" not in mm:
                        print(f"[standalone-tts] {label}: completion {ci} has no audio",
                              flush=True)
                        continue
                    wav_path = out_dir / f"{label}_sample{ci}.wav"
                    dur_s, sr = _save_wav(wav_path, mm)
                    elapsed_s = time.perf_counter() - t0
                    print(f"[standalone-tts] {label}: wrote {wav_path} "
                          f"({dur_s:.2f}s @ {sr}Hz, elapsed={elapsed_s:.1f}s)",
                          flush=True)
                    summaries.append({
                        "label": label, "language": lang, "n": args.n,
                        "loop_trial": trial, "seed": seed, "completion_index": ci,
                        "prompt_text": args.prompt_text, "ref_audio": args.ref_audio,
                        "ref_text": args.ref_text, "wav_path": str(wav_path),
                        "wav_dur_s": dur_s, "sample_rate": sr, "elapsed_s": elapsed_s,
                    })

    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"[standalone-tts] wrote summary -> {out_dir / 'summary.json'}", flush=True)
    return 0


async def main_async(args: argparse.Namespace) -> int:
    """Async-iterate the stream, logging every event before saving final audio."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omni_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        dtype="bfloat16",
    )
    if args.stage_config:
        omni_kwargs["stage_configs_path"] = str(Path(args.stage_config).resolve())
    print(f"[standalone-tts] Constructing AsyncOmni with engine_args={omni_kwargs}",
          flush=True)
    engine = AsyncOmni(**omni_kwargs)

    sampling_params_list = _build_sampling_params_list(args.n, args.max_new_tokens)
    summaries: list[dict] = []
    for lang in args.languages:
        prompt = _build_base_input(
            model_name=args.model_path,
            prompt_text=args.prompt_text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=lang,
            max_new_tokens=args.max_new_tokens,
        )
        label = f"Base_lang{lang}_n{args.n}_async"
        request_id = f"{label}-{uuid4().hex}"
        print(f"[standalone-tts] {label}: language={lang!r} "
              f"prompt_token_ids_len={len(prompt['prompt_token_ids'])} "
              f"n={args.n} request_id={request_id}", flush=True)
        t0 = time.perf_counter()
        event_idx = 0
        # Track per-completion final audio across the stream.
        per_completion_mm: dict[int, dict] = {}
        async for stage_output in engine.generate(
            prompt, request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            event_idx += 1
            req_out = getattr(stage_output, "request_output", None)
            stage_id = getattr(stage_output, "stage_id", None)
            finished = getattr(stage_output, "finished", False)
            if req_out is None or not req_out.outputs:
                if event_idx <= 5 or finished:
                    print(f"[standalone-tts] {label}: event {event_idx} stage={stage_id} "
                          f"finished={finished} no-request_output", flush=True)
                continue
            n_completions = len(req_out.outputs)
            # Stage 0 emits one event per group sample (n=4 → 4 events at
            # stage 0). Each event's ``outputs[0].index`` carries the
            # sample slot (0..n-1). Stage 1 in turn streams cumulative
            # audio chunks for whichever indexes its decoder actually
            # processes — typically only index 0 for the Qwen3-TTS
            # bundled config. Key completion bookkeeping by the
            # completion-level ``index`` (not the per-event enum index).
            for comp in req_out.outputs:
                comp_index = int(getattr(comp, "index", 0))
                mm = comp.multimodal_output
                if mm and "audio" in mm and mm.get("audio") is not None:
                    per_completion_mm[comp_index] = mm
                # Log token-count info for stage 0 events so we can see
                # whether n=4 actually produced 4 distinct codec sequences.
                if stage_id == 0:
                    n_tok = len(getattr(comp, "token_ids", []) or [])
                    print(f"[standalone-tts] {label}: stage0 event {event_idx} "
                          f"comp.index={comp_index} n_tokens={n_tok} "
                          f"finish={getattr(comp, 'finish_reason', None)!r}",
                          flush=True)
            if event_idx <= 3 or finished:
                first_mm = req_out.outputs[0].multimodal_output or {}
                audio = first_mm.get("audio")
                if isinstance(audio, list):
                    shape_str = f"list(len={len(audio)})"
                elif hasattr(audio, "shape"):
                    shape_str = f"tensor{tuple(audio.shape)}"
                else:
                    shape_str = type(audio).__name__
                print(f"[standalone-tts] {label}: event {event_idx} stage={stage_id} "
                      f"finished={finished} n_completions={n_completions} "
                      f"audio[0]={shape_str}", flush=True)
        elapsed_s = time.perf_counter() - t0
        print(f"[standalone-tts] {label}: stream done in {elapsed_s:.1f}s "
              f"events={event_idx} completions_with_audio={len(per_completion_mm)}",
              flush=True)
        for ci, mm in sorted(per_completion_mm.items()):
            wav_path = out_dir / f"{label}_sample{ci}.wav"
            dur_s, sr = _save_wav(wav_path, mm)
            print(f"[standalone-tts] {label}: wrote {wav_path} "
                  f"({dur_s:.2f}s @ {sr}Hz)", flush=True)
            summaries.append({
                "label": label, "language": lang, "n": args.n,
                "completion_index": ci, "prompt_text": args.prompt_text,
                "ref_audio": args.ref_audio, "ref_text": args.ref_text,
                "wav_path": str(wav_path), "wav_dur_s": dur_s, "sample_rate": sr,
                "elapsed_s": elapsed_s, "event_count": event_idx,
            })

    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"[standalone-tts] wrote summary -> {out_dir / 'summary.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True,
                        help="Path or HF id for Qwen3-TTS-12Hz-*-Base.")
    parser.add_argument("--stage-config", default=None,
                        help="Optional stage_configs override yaml. Omit to use "
                             "the bundled vllm-omni qwen3_tts default.")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument(
        "--languages", nargs="*", default=["Auto", "Chinese"],
        help="Language(s) to sweep. Each runs a separate generate() request.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--out-dir", default="/tmp/standalone_tts_out")
    parser.add_argument(
        "--n", type=int, default=1,
        help="Per-prompt group size for GRPO sampling. n>=2 builds an explicit "
             "stage-0 SamplingParams(n=n,...); n=1 lets the engine use "
             "stage_config defaults (matches end2end demo).",
    )
    parser.add_argument(
        "--mode", choices=("sync", "async"), default="sync",
        help="``sync`` mirrors end2end.py:main with ``Omni.generate(...)``. "
             "``async`` exercises ``AsyncOmni.generate(...)`` and dumps "
             "every streaming event for diagnostic comparison.",
    )
    parser.add_argument(
        "--n-via-loop", action="store_true", default=False,
        help="vllm-omni's Qwen3-TTS pipeline appears to ignore "
             "``SamplingParams.n``; emit ``n`` independent requests with "
             "different seeds instead (matches the apparent fan-out used "
             "by verl's AgentLoopManager in production T14).",
    )
    parser.add_argument("--seed-base", type=int, default=42)
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print("[standalone-tts] WARNING: CUDA_VISIBLE_DEVICES is not set.",
              file=sys.stderr)

    if args.mode == "sync":
        return main_sync(args)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
