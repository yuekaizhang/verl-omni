# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Batch-transcribe a directory of wavs through Qwen3-ASR and print CER.

Companion to :mod:`test_vllm_omni_stand_alone`. Use this to score the
audio produced by the standalone rollout so we can answer:

  * Does Qwen3-TTS-12Hz-0.6B-Base in fact synthesize the requested
    Chinese sentence when called from a clean rollout path?
  * Is there a CER gap between the ``language=None`` (current verl-omni)
    and ``language="Chinese"`` (qwen-tts upstream example) variants?

The CER is computed via the same reward_score utility the trainer uses,
so the numbers here are directly comparable with the ``[T14-reward]``
prints emitted during a verl-omni rollout.

Usage (inside the Slurm container, with the ASR server already running
on :8001):

    .venv/bin/python scripts/asr_transcribe.py \\
        --asr-url http://localhost:8001 \\
        --wav-dir /tmp/standalone_tts_out \\
        --reference "美联航在声明中也为此次事故道歉。"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np


async def main_async(args: argparse.Namespace) -> int:
    import soundfile as sf

    from verl_omni.reward_loop.reward_manager.asr_error_rate import (
        AsrEndpointConfig,
        AsrErrorRateRewardManager,
    )
    from verl_omni.utils.reward_score.asr_error_rate import compute_cer

    wavs = sorted(Path(args.wav_dir).glob(args.glob))
    if not wavs:
        print(f"[asr-transcribe] no wavs match {args.wav_dir}/{args.glob}",
              file=sys.stderr)
        return 1

    rm = AsrErrorRateRewardManager(
        config={},
        endpoint=AsrEndpointConfig(base_url=args.asr_url, model=args.model),
    )

    rows: list[dict] = []
    for wav_path in wavs:
        try:
            data, sr = sf.read(wav_path)
        except Exception as e:
            print(f"[asr-transcribe] {wav_path.name}: read failed — {e}",
                  file=sys.stderr)
            continue
        if data.ndim > 1:
            data = data[:, 0]
        try:
            transcript = await rm.transcribe(np.asarray(data, dtype=np.float32),
                                             int(sr))
        except Exception as e:
            transcript = f"<asr-error: {e}>"
        cer = (
            compute_cer(hypothesis=transcript, reference=args.reference)
            if not transcript.startswith("<asr-error:")
            else float("nan")
        )
        rows.append({
            "wav": wav_path.name,
            "dur_s": round(len(data) / sr, 3),
            "transcript": transcript,
            "cer": cer,
        })
        cer_str = f"{cer:.4f}" if isinstance(cer, float) and cer == cer else str(cer)
        print(
            f"[asr-transcribe] {wav_path.name} ({rows[-1]['dur_s']}s) "
            f"-> {transcript!r} (cer={cer_str})",
            flush=True,
        )

    summary_path = Path(args.wav_dir) / "asr_transcribe_summary.json"
    summary_path.write_text(json.dumps({
        "reference": args.reference,
        "rows": rows,
    }, ensure_ascii=False, indent=2))
    print(f"[asr-transcribe] wrote {summary_path}", flush=True)

    # Per-group means so the A/B comparison is one-glance.
    groups: dict[str, list[float]] = {}
    for r in rows:
        if not isinstance(r["cer"], float) or r["cer"] != r["cer"]:
            continue
        prefix = r["wav"].split("_sample")[0]
        groups.setdefault(prefix, []).append(r["cer"])
    print("\n[asr-transcribe] mean CER per label:")
    for label, cers in sorted(groups.items()):
        print(f"  {label}: mean={sum(cers) / len(cers):.4f} n={len(cers)} "
              f"min={min(cers):.4f} max={max(cers):.4f}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--asr-url", default="http://localhost:8001")
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--wav-dir", required=True,
                        help="Directory containing wavs to transcribe.")
    parser.add_argument("--glob", default="*.wav",
                        help="Glob pattern for wav files inside --wav-dir.")
    parser.add_argument("--reference", required=True,
                        help="Reference text to compute CER against.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
