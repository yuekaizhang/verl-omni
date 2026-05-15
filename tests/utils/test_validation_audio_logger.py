# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the validation audio artifact logger (AC-8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verl_omni.utils.validation_audio_logger import (
    ArtifactWriteError,
    log_validation_step,
    post_run_check_emitted_artifacts,
)


def _make_sample(i: int, with_target: bool = True) -> dict:
    waveform = np.zeros(24000, dtype=np.float32)
    waveform[1000:5000] = 0.1 * np.sin(np.arange(4000) * 0.1)
    sample = {
        "waveform": waveform,
        "sample_rate": 24000,
        "ref_audio": waveform,
    }
    if with_target:
        sample["target_audio"] = waveform
    return sample


def test_writes_four_generated_and_ref_files(tmp_path: Path) -> None:
    samples = [_make_sample(i) for i in range(5)]
    n = log_validation_step(
        out_dir=tmp_path,
        step=3,
        samples=samples,
        scalar_metrics={"mean_reward": 0.7, "mean_cer": 0.1},
        num_samples=4,
    )
    assert n == 4
    step_dir = tmp_path / "validation_step_000003"
    assert len(list(step_dir.glob("generated_*.wav"))) == 4
    assert len(list(step_dir.glob("ref_audio_*.wav"))) == 4
    assert (step_dir / "metrics.json").exists()
    metrics = json.loads((step_dir / "metrics.json").read_text())
    assert metrics["mean_reward"] == 0.7


def test_target_audio_optional(tmp_path: Path) -> None:
    samples = [_make_sample(i, with_target=False) for i in range(4)]
    n = log_validation_step(
        out_dir=tmp_path,
        step=0,
        samples=samples,
        scalar_metrics={"mean_reward": 0.5},
    )
    assert n == 4
    step_dir = tmp_path / "validation_step_000000"
    # No target_audio files written when the dataset doesn't supply them.
    assert len(list(step_dir.glob("target_audio_*.wav"))) == 0


def test_post_run_check_passes_when_all_steps_have_audio(tmp_path: Path) -> None:
    for step in (0, 1, 2):
        log_validation_step(
            out_dir=tmp_path,
            step=step,
            samples=[_make_sample(0)] * 4,
            scalar_metrics={"mean_reward": 0.0},
        )
    good = post_run_check_emitted_artifacts(tmp_path)
    assert len(good) == 3


def test_post_run_check_flags_empty_step(tmp_path: Path) -> None:
    # Create a step directory but no generated wavs.
    (tmp_path / "validation_step_000007").mkdir(parents=True)
    with pytest.raises(ArtifactWriteError, match="zero audio artifacts"):
        post_run_check_emitted_artifacts(tmp_path)


def test_empty_samples_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactWriteError, match="zero samples"):
        log_validation_step(
            out_dir=tmp_path,
            step=0,
            samples=[],
            scalar_metrics={},
        )


def test_disk_failure_surfaces_as_typed_error(tmp_path: Path) -> None:
    # Place a regular file at the path where the step dir would go, so
    # mkdir(parents=True, exist_ok=True) raises and the helper converts it
    # into an ArtifactWriteError. Works regardless of process privileges.
    (tmp_path / "validation_step_000000").write_text("not a directory")
    with pytest.raises(ArtifactWriteError):
        log_validation_step(
            out_dir=tmp_path,
            step=0,
            samples=[_make_sample(0)] * 4,
            scalar_metrics={"mean_reward": 0.0},
        )
