# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the post-run AC-8 fail-closed check in main.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verl_omni.trainer.qwen3_tts_grpo.main import _resolve_validation_dir
from verl_omni.utils.validation_audio_logger import (
    ArtifactWriteError,
    log_validation_step,
    post_run_check_emitted_artifacts,
)


def test_resolve_validation_dir_explicit(tmp_path: Path) -> None:
    cfg = SimpleNamespace(trainer=SimpleNamespace(validation_data_dir=str(tmp_path / "explicit"), default_local_dir=str(tmp_path / "default")))
    assert _resolve_validation_dir(cfg) == tmp_path / "explicit"


def test_resolve_validation_dir_default(tmp_path: Path) -> None:
    cfg = SimpleNamespace(trainer=SimpleNamespace(validation_data_dir=None, default_local_dir=str(tmp_path / "runs/exp")))
    assert _resolve_validation_dir(cfg) == tmp_path / "runs/exp" / "validation_audio"


def test_resolve_validation_dir_none(tmp_path: Path) -> None:
    cfg = SimpleNamespace(trainer=SimpleNamespace(validation_data_dir=None, default_local_dir=None))
    assert _resolve_validation_dir(cfg) is None


def test_post_run_check_passes_when_artifacts_present(tmp_path: Path) -> None:
    wav = np.zeros(24000, dtype=np.float32)
    sample = {"waveform": wav, "sample_rate": 24000, "ref_audio": wav}
    log_validation_step(
        out_dir=tmp_path,
        step=1,
        samples=[sample] * 4,
        scalar_metrics={"mean_reward": 0.5, "mean_cer": 0.1, "mean_duration_ratio": 1.0},
    )
    steps = post_run_check_emitted_artifacts(tmp_path)
    assert len(steps) == 1


def test_post_run_check_flags_empty_step(tmp_path: Path) -> None:
    (tmp_path / "validation_step_000001").mkdir()
    with pytest.raises(ArtifactWriteError, match="zero audio artifacts"):
        post_run_check_emitted_artifacts(tmp_path)
