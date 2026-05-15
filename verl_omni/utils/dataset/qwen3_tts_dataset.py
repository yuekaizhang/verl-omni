# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Voice-cloning dataset for Qwen3-TTS GRPO training.

Each row carries the inputs that ``vllm-omni``'s Qwen3-TTS Base pipeline
expects (``ref_audio``, ``ref_text``, ``prompt_text``), plus the metadata
needed for same-speaker / different-utterance pairing checks and for
duration-based reward shaping (``speaker_id``, ``ref_utt_id``,
``target_utt_id``, ``target_duration``, optional ``target_audio``).

Audio is referenced by path; loading is deferred to the AR-TTS AgentLoop so
the dataset stays cheap to iterate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

REQUIRED_COLUMNS: tuple[str, ...] = (
    "prompt_text",
    "ref_audio",
    "ref_text",
    "speaker_id",
    "ref_utt_id",
    "target_utt_id",
    "target_duration",
    "data_source",
)

OPTIONAL_COLUMNS: tuple[str, ...] = ("target_audio",)


class Qwen3TTSDataset(Dataset):
    """Voice-cloning triples for Qwen3-TTS GRPO.

    Args:
        data_files: One parquet path or a list of parquet paths. Files are
            concatenated row-wise in the order provided.
        max_samples: Optional cap on the number of rows after loading.
            ``-1`` (default) keeps every row.

    Raises:
        FileNotFoundError: A path in ``data_files`` does not exist.
        ValueError: A required column is missing, ``speaker_id`` is null on
            any row, or any row has ``ref_utt_id == target_utt_id``.
    """

    def __init__(
        self,
        data_files: str | Path | list[str | Path],
        max_samples: int = -1,
    ) -> None:
        if isinstance(data_files, (str, Path)):
            paths = [Path(data_files)]
        else:
            paths = [Path(p) for p in data_files]

        if not paths:
            raise ValueError(
                "Qwen3TTSDataset requires at least one parquet path; received an empty data_files list."
            )

        frames: list[pd.DataFrame] = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Qwen3TTSDataset parquet not found: {path}")
            frames.append(pd.read_parquet(path))

        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        if max_samples is not None and max_samples >= 0:
            df = df.head(max_samples).reset_index(drop=True)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Qwen3TTSDataset parquet is missing required column(s): {missing}. "
                f"Required columns: {list(REQUIRED_COLUMNS)}."
            )

        null_speaker_mask = df["speaker_id"].isna()
        if null_speaker_mask.any():
            offending = df.index[null_speaker_mask].tolist()[:5]
            raise ValueError(
                "Qwen3TTSDataset rejected rows with null speaker_id "
                f"(first offending row indices: {offending}). Every row "
                "must carry a non-null speaker_id."
            )

        same_utt_mask = df["ref_utt_id"].astype(str) == df["target_utt_id"].astype(str)
        if same_utt_mask.any():
            offending = df.index[same_utt_mask].tolist()[:5]
            raise ValueError(
                "Qwen3TTSDataset rejected rows where ref_utt_id == target_utt_id "
                f"(first offending row indices: {offending}). Voice cloning "
                "requires the reference and target to be different utterances."
            )

        self._df: pd.DataFrame = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._df.iloc[index]
        out: dict[str, Any] = {col: row[col] for col in REQUIRED_COLUMNS}
        for col in OPTIONAL_COLUMNS:
            if col in self._df.columns:
                out[col] = row[col]
        out["target_duration"] = float(out["target_duration"])
        return out
