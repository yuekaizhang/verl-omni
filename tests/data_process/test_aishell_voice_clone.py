# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pure-Python tests for the AISHELL pairing script (no HF download)."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

# Load the data_process script as a module without requiring it to be a package.
_PAIRING_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "qwen3_tts_grpo_trainer"
    / "data_process"
    / "aishell_voice_clone.py"
)
spec = importlib.util.spec_from_file_location("aishell_voice_clone", _PAIRING_PATH)
aishell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aishell)  # type: ignore[union-attr]


def _row(speaker: str, utt_id: str, text: str = "hi", duration: float = 1.5) -> dict:
    return {
        "audio": {"path": f"/tmp/{utt_id}.wav", "duration": duration},
        "text": text,
        "speaker_id": speaker,
        "utt_id": utt_id,
    }


def test_pairs_are_same_speaker_and_different_utt() -> None:
    rows = [_row("S0", f"S0-{i}") for i in range(5)] + [_row("S1", f"S1-{i}") for i in range(5)]
    pairs = aishell._build_pairs(rows, pairs_per_speaker=4, rng=random.Random(0), data_source="aishell")
    assert len(pairs) > 0
    for p in pairs:
        assert p["speaker_id"] in {"S0", "S1"}
        assert p["ref_utt_id"] != p["target_utt_id"]


def test_pairs_skip_speakers_with_single_utt() -> None:
    rows = [_row("solo", "solo-0"), _row("S0", "S0-0"), _row("S0", "S0-1")]
    pairs = aishell._build_pairs(rows, pairs_per_speaker=2, rng=random.Random(0), data_source="aishell")
    speakers = {p["speaker_id"] for p in pairs}
    assert speakers == {"S0"}


def test_enforce_disjoint_rejects_overlap() -> None:
    train = [{"target_utt_id": "U1"}, {"target_utt_id": "U2"}]
    bad_eval = [{"target_utt_id": "U2"}, {"target_utt_id": "U3"}]
    with pytest.raises(RuntimeError, match="overlap"):
        aishell._enforce_disjoint(train, bad_eval)


def test_pairing_writes_required_columns() -> None:
    rows = [_row("S0", f"S0-{i}") for i in range(4)]
    pairs = aishell._build_pairs(rows, pairs_per_speaker=3, rng=random.Random(1), data_source="aishell")
    required = {
        "prompt_text",
        "ref_audio",
        "ref_text",
        "speaker_id",
        "ref_utt_id",
        "target_utt_id",
        "target_duration",
        "data_source",
    }
    for p in pairs:
        assert required.issubset(p.keys())
