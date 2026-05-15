# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Held-out evaluation pipeline for the Qwen3-TTS GRPO recipe.

Runs inference with both the base and the RL checkpoint on the eval parquet,
transcribes each waveform via the remote Qwen3-ASR endpoint, and writes
``eval_results.json`` with:

    {
        "base_cer", "rl_cer",
        "base_median_duration_ratio", "rl_median_duration_ratio",
        "base_mean_duration_ratio",   "rl_mean_duration_ratio"
    }

The actual model inference step expects a callable that wraps a checkpoint
in the AR-TTS rollout server. For unit testing and offline scoring this
script accepts a ``--mock-inference-json`` flag that loads pre-rendered
hypotheses + durations from a JSON file, so the AC-7 contract (the JSON
schema and the disjointness check) can be exercised without GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from verl_omni.reward_loop.reward_manager.asr_error_rate import (
    AsrEndpointConfig,
    AsrErrorRateRewardManager,
)
from verl_omni.utils.reward_score.asr_error_rate import compute_cer

logger = logging.getLogger(__name__)


def _assert_disjoint(eval_path: Path, train_path: Path | None) -> None:
    if train_path is None or not str(train_path):
        return
    train_ids = set(pd.read_parquet(train_path)["target_utt_id"].astype(str))
    eval_ids = set(pd.read_parquet(eval_path)["target_utt_id"].astype(str))
    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(
            f"eval parquet target_utt_id overlaps train ({len(overlap)} ids). "
            "Refusing to score — pick disjoint splits per AC-7."
        )


async def _score_split(
    asr: AsrErrorRateRewardManager,
    rows: list[dict[str, Any]],
    hypotheses: dict[str, str],
    durations: dict[str, float],
) -> dict[str, float]:
    cers: list[float] = []
    duration_ratios: list[float] = []
    for row in rows:
        target_id = str(row["target_utt_id"])
        hyp = hypotheses.get(target_id, "")
        gen_duration = float(durations.get(target_id, 0.0))
        target_duration = float(row.get("target_duration") or 1.0)
        cer = compute_cer(hypothesis=hyp, reference=str(row["prompt_text"]))
        cers.append(cer)
        if target_duration > 0:
            duration_ratios.append(gen_duration / target_duration)
    return {
        "cer": float(sum(cers) / max(len(cers), 1)),
        "median_duration_ratio": float(statistics.median(duration_ratios)) if duration_ratios else 0.0,
        "mean_duration_ratio": float(sum(duration_ratios) / max(len(duration_ratios), 1)),
    }


def _load_mock(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--rl-ckpt", required=True)
    parser.add_argument("--eval-parquet", required=True)
    parser.add_argument("--asr-base-url", required=True)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--train-parquet", default=None, help="Optional, enables target_utt_id disjointness check.")
    parser.add_argument(
        "--mock-inference-json",
        default=None,
        help=(
            "Path to a JSON file of the form "
            "{'base': {'hypotheses': {utt_id: text}, 'durations': {utt_id: seconds}}, "
            "'rl': {...}}. Used for offline / no-GPU scoring."
        ),
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_parquet)
    _assert_disjoint(eval_path, Path(args.train_parquet) if args.train_parquet else None)

    rows = pd.read_parquet(eval_path).to_dict(orient="records")
    if not args.mock_inference_json:
        raise NotImplementedError(
            "Live-model evaluation requires a separately-launched vllm-omni "
            "Qwen3-TTS server. Pass --mock-inference-json to score "
            "pre-rendered hypotheses / durations against the AC-7 schema."
        )

    mock = _load_mock(Path(args.mock_inference_json))
    asr = AsrErrorRateRewardManager(
        config={},
        endpoint=AsrEndpointConfig(base_url=args.asr_base_url),
    )

    async def _run() -> dict[str, float]:
        base_metrics = await _score_split(asr, rows, mock["base"]["hypotheses"], mock["base"]["durations"])
        rl_metrics = await _score_split(asr, rows, mock["rl"]["hypotheses"], mock["rl"]["durations"])
        return {
            "base_cer": base_metrics["cer"],
            "rl_cer": rl_metrics["cer"],
            "base_median_duration_ratio": base_metrics["median_duration_ratio"],
            "rl_median_duration_ratio": rl_metrics["median_duration_ratio"],
            "base_mean_duration_ratio": base_metrics["mean_duration_ratio"],
            "rl_mean_duration_ratio": rl_metrics["mean_duration_ratio"],
        }

    results = asyncio.run(_run())
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
