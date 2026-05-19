# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build a tiny voice-cloning smoke parquet for T14.

12 rows × {train, eval} of synthetic Chinese prompts, all pointing at a
single real reference wav. Output shape matches
``verl_omni.utils.dataset.qwen3_tts_dataset.Qwen3TTSDataset`` columns:

    REQUIRED: prompt_text, ref_audio, ref_text, speaker_id, ref_utt_id,
              target_utt_id, target_duration, data_source
    OPTIONAL: language ("Chinese" — threads through to Qwen3-TTS Base
              mode's ``additional_information``)

Lives in-repo (not /tmp) so it survives Slurm restarts that wipe /tmp.

Usage::

    .venv/bin/python scripts/build_aishell_one_smoke.py \\
        [--out-dir /tmp/aishell_one] [--ref-audio /path/to/ref.wav]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import soundfile as sf

PROMPTS = (
    "你好,世界。",
    "美联航在声明中也为此次事故道歉。",
    "今天天气非常好。",
    "这是一个语音合成测试。",
    "请帮我把窗户关上。",
    "晚饭准备好了吗。",
    "深度学习改变世界。",
    "我们一起去公园散步吧。",
    "明天会下雨吗。",
    "请把音量调小一点。",
    "这家餐厅的菜很好吃。",
    "我喜欢听古典音乐。",
)


def make(out_dir: Path, split: str, wav: str, duration: float, language: str) -> None:
    rows = []
    for i, txt in enumerate(PROMPTS):
        rows.append(dict(
            prompt_text=txt,
            ref_audio=wav,
            ref_text="一段参考音频。",
            speaker_id=f"spk_{i % 2}",
            ref_utt_id=f"ref_{split}_{i:03d}",
            target_utt_id=f"tgt_{split}_{i:03d}",
            target_duration=duration,
            data_source="aishell_one_smoke",
            language=language,
        ))
    df = pd.DataFrame(rows)
    path = out_dir / f"{split}.parquet"
    df.to_parquet(path, index=False)
    print(f"wrote {path}: {len(df)} rows, language={language!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ref-audio",
        default="/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/wavs/r12_baseline.wav",
        help="Path to a real wav file used as the voice-cloning reference.",
    )
    parser.add_argument(
        "--out-dir", default="/tmp/aishell_one",
        help="Directory to write train.parquet and eval.parquet.",
    )
    parser.add_argument(
        "--language", default="Chinese",
        help="Language hint passed to Qwen3-TTS Base mode via "
             "``additional_information['language']``. "
             "Use ``Chinese`` / ``English`` / ``Auto``.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ref_audio):
        raise FileNotFoundError(args.ref_audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, sr = sf.read(args.ref_audio)
    duration = float(len(data) / sr)
    print(f"ref_audio={args.ref_audio} sr={sr} dur={duration:.2f}s")

    make(out_dir, "train", args.ref_audio, duration, args.language)
    make(out_dir, "eval", args.ref_audio, duration, args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
