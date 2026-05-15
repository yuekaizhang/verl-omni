# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the fail-closed _check_speaker_encoder helper (T1 contract)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import safetensors.torch as st

_SMOKE_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "qwen3_tts_smoke.py"
spec = importlib.util.spec_from_file_location("qwen3_tts_smoke", _SMOKE_PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)  # type: ignore[union-attr]


def test_hub_id_with_no_local_checkpoint_fails_closed() -> None:
    # A Hub ID that has never been downloaded must NOT report success.
    assert smoke._check_speaker_encoder("Qwen/Some-Model-Not-Downloaded") is False


def test_local_checkpoint_without_speaker_encoder_fails_closed(tmp_path: Path) -> None:
    weights = {"talker.layer.weight": torch.zeros(2, 2), "talker.layer.bias": torch.zeros(2)}
    st.save_file(weights, str(tmp_path / "model.safetensors"))
    assert smoke._check_speaker_encoder(str(tmp_path)) is False


def test_local_checkpoint_with_speaker_encoder_passes(tmp_path: Path) -> None:
    weights = {
        "speaker_encoder.layer.weight": torch.zeros(2, 2),
        "talker.head.weight": torch.zeros(2, 2),
    }
    st.save_file(weights, str(tmp_path / "model.safetensors"))
    assert smoke._check_speaker_encoder(str(tmp_path)) is True


def test_unreadable_safetensors_fails_closed(tmp_path: Path) -> None:
    # Drop a junk file claiming to be safetensors; safe_open will raise.
    (tmp_path / "model.safetensors").write_text("not a real safetensors file")
    assert smoke._check_speaker_encoder(str(tmp_path)) is False
