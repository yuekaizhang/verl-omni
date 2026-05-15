# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for verl_omni.utils.dataset.qwen3_tts_dataset.Qwen3TTSDataset.

These tests run with no GPU and no remote services.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from verl_omni.utils.dataset.qwen3_tts_dataset import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    Qwen3TTSDataset,
)


def _good_row(i: int = 0, speaker: str = "S0", ref: int = 0, tgt: int = 1) -> dict:
    return {
        "prompt_text": f"text-{i}",
        "ref_audio": f"/tmp/ref-{i}.wav",
        "ref_text": f"ref-text-{i}",
        "speaker_id": speaker,
        "ref_utt_id": f"{speaker}-U{ref}",
        "target_utt_id": f"{speaker}-U{tgt}",
        "target_duration": 2.5 + i * 0.1,
        "data_source": "aishell-test-fixture",
        "target_audio": f"/tmp/tgt-{i}.wav",
    }


def _write_parquet(tmp_path: Path, rows: list[dict], name: str = "fixture.parquet") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_roundtrip_four_rows_preserves_all_fields(tmp_path: Path) -> None:
    rows = [_good_row(i, speaker=f"S{i // 2}", ref=i, tgt=i + 10) for i in range(4)]
    parquet = _write_parquet(tmp_path, rows)

    ds = Qwen3TTSDataset(parquet)

    assert len(ds) == 4
    for idx in range(4):
        item = ds[idx]
        for col in REQUIRED_COLUMNS:
            assert col in item, f"missing required column {col} at row {idx}"
            assert item[col] is not None
            if isinstance(item[col], str):
                assert item[col] != ""
        for col in OPTIONAL_COLUMNS:
            assert col in item  # fixture includes target_audio
        assert isinstance(item["target_duration"], float)
        assert item["ref_utt_id"] != item["target_utt_id"]


def test_optional_target_audio_omitted_when_absent(tmp_path: Path) -> None:
    rows = [_good_row(i) for i in range(2)]
    for row in rows:
        row.pop("target_audio")
    parquet = _write_parquet(tmp_path, rows)

    ds = Qwen3TTSDataset(parquet)

    item = ds[0]
    for col in REQUIRED_COLUMNS:
        assert col in item
    assert "target_audio" not in item


def test_null_speaker_id_rejected_at_load(tmp_path: Path) -> None:
    rows = [_good_row(0), _good_row(1)]
    rows[1]["speaker_id"] = None
    parquet = _write_parquet(tmp_path, rows)

    with pytest.raises(ValueError, match="null speaker_id"):
        Qwen3TTSDataset(parquet)


def test_same_utt_id_rejected_at_load(tmp_path: Path) -> None:
    rows = [_good_row(0), _good_row(1, ref=7, tgt=7)]
    parquet = _write_parquet(tmp_path, rows)

    with pytest.raises(ValueError, match="ref_utt_id == target_utt_id"):
        Qwen3TTSDataset(parquet)


def test_missing_required_column_rejected(tmp_path: Path) -> None:
    rows = [_good_row(0)]
    rows[0].pop("target_duration")
    parquet = _write_parquet(tmp_path, rows)

    with pytest.raises(ValueError, match="missing required column"):
        Qwen3TTSDataset(parquet)


def test_multiple_parquet_files_concatenate(tmp_path: Path) -> None:
    p1 = _write_parquet(tmp_path, [_good_row(i, speaker="S0") for i in range(3)], "a.parquet")
    p2 = _write_parquet(tmp_path, [_good_row(i, speaker="S1") for i in range(2)], "b.parquet")

    ds = Qwen3TTSDataset([p1, p2])

    assert len(ds) == 5
    speakers = {ds[i]["speaker_id"] for i in range(len(ds))}
    assert speakers == {"S0", "S1"}


def test_max_samples_caps_loaded_rows(tmp_path: Path) -> None:
    rows = [_good_row(i) for i in range(10)]
    parquet = _write_parquet(tmp_path, rows)

    ds = Qwen3TTSDataset(parquet, max_samples=3)
    assert len(ds) == 3


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Qwen3TTSDataset(tmp_path / "does-not-exist.parquet")


def test_empty_data_files_rejected() -> None:
    with pytest.raises(ValueError, match="at least one parquet path"):
        Qwen3TTSDataset([])
