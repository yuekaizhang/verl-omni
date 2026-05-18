# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""T3 smoke: send one waveform to a live vLLM Qwen3-ASR server and compute CER.

Prereq: launch the ASR server on its own GPU subset, e.g.::

    CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python -m vllm.entrypoints.openai.api_server \\
        --model .hf_cache/Qwen3-ASR-0.6B \\
        --host 0.0.0.0 --port 8001 \\
        --task transcription \\
        --trust-remote-code

Then::

    .venv/bin/python scripts/qwen3_asr_smoke.py \\
        --asr-url http://localhost:8001 \\
        --wav /tmp/aishell_one/.../BAC009S0754W0270-1380.wav \\
        --reference "美联航在声明中也为此次事故道歉"
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

    wav, sr = sf.read(args.wav)
    if wav.ndim > 1:
        wav = wav[:, 0]
    print(f"[T3] wav: shape={wav.shape} sr={sr}")

    rm = AsrErrorRateRewardManager(
        config={},
        endpoint=AsrEndpointConfig(base_url=args.asr_url, model=args.model),
    )

    transcript = await rm.transcribe(np.asarray(wav, dtype=np.float32), int(sr))
    cer = compute_cer(hypothesis=transcript, reference=args.reference)

    result = {
        "wav": args.wav,
        "reference": args.reference,
        "transcript": transcript,
        "cer": cer,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Pass criterion: we got a non-empty transcript and a finite CER. The CER
    # value itself depends on the ASR model and the reference, so we don't
    # gate on a particular threshold — just that the full HTTP path works.
    if transcript and isinstance(cer, float):
        return 0
    print("[T3] FAIL: empty transcript or invalid CER.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asr-url", default="http://localhost:8001")
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
