# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for run_eval.py: target_utt_id disjointness + eval_results.json schema."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path)


def _row(utt_id: str) -> dict:
    return {
        "prompt_text": f"text-for-{utt_id}",
        "ref_audio": "/tmp/x.wav",
        "ref_text": "ref",
        "speaker_id": "S0",
        "ref_utt_id": "S0-R0",
        "target_utt_id": utt_id,
        "target_duration": 2.0,
        "data_source": "fixture",
    }


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "verl_omni.trainer.qwen3_tts_grpo.run_eval", *args],
        cwd=Path(__file__).resolve().parent.parent.parent,
        capture_output=True,
        text=True,
    )


def test_disjoint_parquets_produce_eval_results_json(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    eval_ = tmp_path / "eval.parquet"
    _write_parquet(train, [_row("U-train-0"), _row("U-train-1")])
    _write_parquet(eval_, [_row("U-eval-0"), _row("U-eval-1")])

    mock = {
        "base": {
            "hypotheses": {"U-eval-0": "text-for-U-eval-0-WRONG", "U-eval-1": "text-for-U-eval-1"},
            "durations": {"U-eval-0": 2.1, "U-eval-1": 2.0},
        },
        "rl": {
            "hypotheses": {"U-eval-0": "text-for-U-eval-0", "U-eval-1": "text-for-U-eval-1"},
            "durations": {"U-eval-0": 2.0, "U-eval-1": 2.0},
        },
    }
    mock_path = tmp_path / "mock.json"
    mock_path.write_text(json.dumps(mock))
    output = tmp_path / "eval_results.json"

    proc = _run(
        [
            "--base-ckpt",
            "/tmp/base",
            "--rl-ckpt",
            "/tmp/rl",
            "--eval-parquet",
            str(eval_),
            "--asr-base-url",
            "http://asr:8001",
            "--train-parquet",
            str(train),
            "--mock-inference-json",
            str(mock_path),
            "--output",
            str(output),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads(output.read_text())
    for key in (
        "base_cer",
        "rl_cer",
        "base_median_duration_ratio",
        "rl_median_duration_ratio",
        "base_mean_duration_ratio",
        "rl_mean_duration_ratio",
    ):
        assert key in results
        assert isinstance(results[key], float)
    # RL achieved zero CER on both samples while base hit 0.5+ on one — sanity.
    assert results["rl_cer"] < results["base_cer"]


def test_overlapping_target_utt_ids_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    eval_ = tmp_path / "eval.parquet"
    _write_parquet(train, [_row("U-shared-0")])
    _write_parquet(eval_, [_row("U-shared-0")])

    mock_path = tmp_path / "mock.json"
    mock_path.write_text(json.dumps({"base": {"hypotheses": {}, "durations": {}}, "rl": {"hypotheses": {}, "durations": {}}}))
    output = tmp_path / "eval_results.json"

    proc = _run(
        [
            "--base-ckpt",
            "/tmp/base",
            "--rl-ckpt",
            "/tmp/rl",
            "--eval-parquet",
            str(eval_),
            "--asr-base-url",
            "http://asr:8001",
            "--train-parquet",
            str(train),
            "--mock-inference-json",
            str(mock_path),
            "--output",
            str(output),
        ]
    )
    assert proc.returncode != 0
    assert "overlap" in proc.stderr.lower() or "overlap" in proc.stdout.lower()
