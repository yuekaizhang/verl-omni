# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build same-speaker, different-utterance voice-cloning parquets from AISHELL.

For each speaker, sample ``max_pairs_per_speaker`` random ``(ref_utt, target_utt)``
pairs with ``ref_utt_id != target_utt_id``. The train and eval parquets are
guaranteed disjoint by ``target_utt_id`` because they are built from disjoint
splits of the source dataset.

Required columns in the output parquet match
:class:`verl_omni.utils.dataset.qwen3_tts_dataset.Qwen3TTSDataset` REQUIRED_COLUMNS:

    prompt_text, ref_audio, ref_text, speaker_id, ref_utt_id, target_utt_id,
    target_duration, data_source (+ optional target_audio).

Usage::

    python aishell_voice_clone.py \\
        --hf-dataset yuekai/aishell \\
        --output-dir /path/to/data \\
        --train-pairs-per-speaker 20 \\
        --eval-pairs-per-speaker 5
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_columns(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize column names from AISHELL-style rows to our canonical schema.

    AISHELL/yuekai/aishell typically exposes ``audio`` (dict or path),
    ``text`` (transcript), and a speaker label. Caller can adjust the
    mapping when running against a fork with different field names.
    """

    audio = row.get("audio") or row.get("audio_path") or row.get("path")
    if isinstance(audio, dict):
        audio_path = audio.get("path") or audio.get("file") or audio.get("array")
        duration = audio.get("duration")
    else:
        audio_path = audio
        duration = row.get("duration")

    text = row.get("text") or row.get("transcript") or row.get("sentence") or ""
    speaker = (
        row.get("speaker_id")
        or row.get("speaker")
        or row.get("spk_id")
        or row.get("spk")
    )
    utt_id = row.get("utt_id") or row.get("id") or row.get("audio_id") or row.get("filename")
    return {
        "audio_path": audio_path,
        "duration": float(duration) if duration is not None else None,
        "text": text,
        "speaker_id": str(speaker) if speaker is not None else None,
        "utt_id": str(utt_id) if utt_id is not None else None,
    }


def _build_pairs(
    rows: list[dict[str, Any]],
    *,
    pairs_per_speaker: int,
    rng: random.Random,
    data_source: str,
) -> list[dict[str, Any]]:
    """Build same-speaker, different-utterance pairs from a flat row list."""

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        rec = _resolve_columns(raw)
        if rec["speaker_id"] is None or rec["utt_id"] is None or rec["audio_path"] is None:
            continue
        by_speaker.setdefault(rec["speaker_id"], []).append(rec)

    out: list[dict[str, Any]] = []
    for speaker, recs in by_speaker.items():
        if len(recs) < 2:
            continue  # cannot form a different-utterance pair
        for _ in range(pairs_per_speaker):
            ref, target = rng.sample(recs, 2)
            if ref["utt_id"] == target["utt_id"]:
                continue
            out.append(
                {
                    "prompt_text": target["text"],
                    "ref_audio": ref["audio_path"],
                    "ref_text": ref["text"],
                    "speaker_id": speaker,
                    "ref_utt_id": ref["utt_id"],
                    "target_utt_id": target["utt_id"],
                    "target_duration": target["duration"] or 0.0,
                    "data_source": data_source,
                    "target_audio": target["audio_path"],
                }
            )
    return out


def _load_split(hf_dataset: str, split: str) -> list[dict[str, Any]]:
    """Load one HF split lazily so the script does not require HF when imported."""

    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split=split)
    return [dict(row) for row in ds]


def _enforce_disjoint(train_rows: Iterable[dict[str, Any]], eval_rows: Iterable[dict[str, Any]]) -> None:
    train_set = {row["target_utt_id"] for row in train_rows}
    eval_set = {row["target_utt_id"] for row in eval_rows}
    overlap = train_set & eval_set
    if overlap:
        raise RuntimeError(
            f"target_utt_id overlap between train and eval: {len(overlap)} ids. "
            "Refusing to write parquets; pick disjoint splits."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-dataset", default="yuekai/aishell", help="HF dataset name")
    parser.add_argument("--output-dir", required=True, help="Directory to write train.parquet/eval.parquet")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-pairs-per-speaker", type=int, default=20)
    parser.add_argument("--eval-pairs-per-speaker", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-source", default="yuekai/aishell", help="Stored in each row as the data_source label"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    logger.info("Loading %s train split ...", args.hf_dataset)
    train_rows = _load_split(args.hf_dataset, args.train_split)
    logger.info("Loading %s eval split ...", args.hf_dataset)
    eval_rows = _load_split(args.hf_dataset, args.eval_split)

    train_pairs = _build_pairs(
        train_rows, pairs_per_speaker=args.train_pairs_per_speaker, rng=rng, data_source=args.data_source
    )
    eval_pairs = _build_pairs(
        eval_rows, pairs_per_speaker=args.eval_pairs_per_speaker, rng=rng, data_source=args.data_source
    )
    _enforce_disjoint(train_pairs, eval_pairs)

    train_path = out_dir / "train.parquet"
    eval_path = out_dir / "eval.parquet"
    pd.DataFrame(train_pairs).to_parquet(train_path)
    pd.DataFrame(eval_pairs).to_parquet(eval_path)
    logger.info(
        "Wrote %d train rows -> %s and %d eval rows -> %s (disjoint by target_utt_id).",
        len(train_pairs),
        train_path,
        len(eval_pairs),
        eval_path,
    )


__all__ = ["_build_pairs", "_resolve_columns", "_enforce_disjoint", "main"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
