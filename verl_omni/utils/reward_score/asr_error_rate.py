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
"""Reward shaping helpers for the Qwen3-TTS GRPO recipe.

Pure functions; no I/O. The ASR client lives in
``verl_omni.reward_loop.reward_manager.asr_error_rate``.

Reward shape:

    reward = (1 - min(CER, CER_CAP))
           - empty_penalty
           - duration_penalty
           - repetition_penalty
    reward = clip(reward, REWARD_FLOOR, REWARD_CEILING)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Lazy jiwer import — only required when CER is computed.
try:  # pragma: no cover - import-time fallback
    import jiwer as _jiwer
except ImportError:  # pragma: no cover - exercised only when jiwer is missing
    _jiwer = None


# Default punctuation set to strip during normalization. Whitespace is preserved.
_PUNCT_CHARS = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "！？，。、；：「」『』（）【】《》〈〉"
    "—…·"
)
_PUNCT_TABLE = str.maketrans("", "", _PUNCT_CHARS)


@dataclass(frozen=True)
class RewardConfig:
    """Knobs for the four-term reward formula.

    All thresholds are documented in ``logs/plan.md`` DEC-7. Defaults are
    chosen so the smoke run stays bounded in ``[-1, 1]``.
    """

    cer_cap: float = 2.0
    empty_penalty: float = 1.0
    duration_penalty: float = 0.5
    repetition_penalty: float = 0.5
    duration_low_ratio: float = 0.5
    duration_high_ratio: float = 2.0
    short_audio_seconds: float = 0.3
    reward_floor: float = -1.0
    reward_ceiling: float = 1.0


def normalize_mandarin_text(text: str) -> str:
    """Normalize Mandarin transcripts for CER comparison.

    Steps: NFKC fold (collapses fullwidth/halfwidth and compatibility forms),
    drop ASCII + CJK punctuation, lowercase, collapse whitespace.
    """

    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_TABLE)
    text = text.lower()
    return " ".join(text.split())


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate via jiwer, after Mandarin normalization."""

    if _jiwer is None:
        raise RuntimeError(
            "jiwer is required for CER computation but is not installed. "
            "Add `jiwer` to requirements.txt and reinstall."
        )
    hyp = normalize_mandarin_text(hypothesis)
    ref = normalize_mandarin_text(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(_jiwer.cer(reference=ref, hypothesis=hyp))


def detect_audio_repetition(
    codec_tokens: list[int] | None,
    repeat_ngram: int = 5,
    repeat_threshold: int = 3,
) -> bool:
    """Heuristic loop detector on stage-0 codec tokens.

    Returns True when the same ``repeat_ngram``-gram appears
    ``repeat_threshold`` or more times back-to-back (non-overlapping).
    Catches the common "model loops the same chunk forever" failure mode
    without full audio-spectral analysis.
    """

    if not codec_tokens or len(codec_tokens) < repeat_ngram * repeat_threshold:
        return False
    n = len(codec_tokens)
    last_start = n - repeat_ngram * repeat_threshold
    for start in range(last_start + 1):
        ngram = tuple(codec_tokens[start : start + repeat_ngram])
        all_match = True
        for k in range(1, repeat_threshold):
            offset = start + k * repeat_ngram
            if tuple(codec_tokens[offset : offset + repeat_ngram]) != ngram:
                all_match = False
                break
        if all_match:
            return True
    return False


def compute_reward(
    *,
    cer: float,
    generated_duration: float,
    target_duration: float,
    codec_tokens: list[int] | None = None,
    config: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the clipped reward + per-term breakdown.

    Args:
        cer: Character error rate; expected in ``[0, +inf)`` (typically ``[0, 1]``).
        generated_duration: Synthesized waveform duration in seconds.
        target_duration: Reference target utterance duration in seconds.
        codec_tokens: Optional stage-0 token IDs for repetition detection.
        config: Reward config (defaults to :class:`RewardConfig`).

    Returns:
        ``(reward, breakdown)`` where ``breakdown`` carries each penalty value.
    """

    cfg = config or RewardConfig()
    base = 1.0 - min(cer, cfg.cer_cap)

    empty_pen = 0.0
    duration_pen = 0.0
    repetition_pen = 0.0

    if generated_duration <= cfg.short_audio_seconds:
        empty_pen = cfg.empty_penalty
    else:
        ratio = generated_duration / max(target_duration, 1e-3)
        if ratio < cfg.duration_low_ratio or ratio > cfg.duration_high_ratio:
            duration_pen = cfg.duration_penalty

    if detect_audio_repetition(codec_tokens):
        repetition_pen = cfg.repetition_penalty

    reward = base - empty_pen - duration_pen - repetition_pen
    reward = max(cfg.reward_floor, min(cfg.reward_ceiling, reward))

    breakdown = {
        "cer": float(cer),
        "base": float(base),
        "empty_penalty": float(empty_pen),
        "duration_penalty": float(duration_pen),
        "repetition_penalty": float(repetition_pen),
        "reward": float(reward),
    }
    return float(reward), breakdown


__all__ = [
    "RewardConfig",
    "compute_cer",
    "compute_reward",
    "detect_audio_repetition",
    "normalize_mandarin_text",
]
