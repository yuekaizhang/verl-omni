"""Regression test for the CER reward path's punctuation-stripping behavior.

Covers AC-8 from the plan. The Qwen3-TTS GRPO reward is ASR-CER with
punctuation stripped via `normalize_mandarin_text`'s
`str.translate(_PUNCT_TABLE)` call. If a future refactor removes that
step, the CER values below jump from `0.0` to non-zero and this test
catches it.

No model load or RL machinery required; tests only the pure-Python
reward-shape utilities in `verl_omni.utils.reward_score.asr_error_rate`.
"""

from __future__ import annotations

import importlib.util

import pytest

from verl_omni.utils.reward_score.asr_error_rate import (
    compute_cer,
    normalize_mandarin_text,
)

_HAS_JIWER = importlib.util.find_spec("jiwer") is not None


class TestNormalizeMandarinText:
    def test_strips_cjk_punctuation(self):
        assert normalize_mandarin_text("你好，世界。") == "你好世界"

    def test_strips_ascii_punctuation(self):
        assert normalize_mandarin_text("Hello, World!") == "hello world"

    def test_strips_mixed_punctuation(self):
        assert normalize_mandarin_text("你好世界。！？") == "你好世界"

    def test_strips_fullwidth_punctuation(self):
        # NFKC fold + strip both fullwidth and halfwidth forms.
        assert normalize_mandarin_text("你好（世界）") == "你好世界"

    def test_empty_input(self):
        assert normalize_mandarin_text("") == ""
        assert normalize_mandarin_text(None) == ""  # type: ignore[arg-type]

    def test_collapses_whitespace(self):
        assert normalize_mandarin_text("a   b\tc") == "a b c"

    def test_lowercases(self):
        assert normalize_mandarin_text("HELLO") == "hello"


@pytest.mark.skipif(not _HAS_JIWER, reason="jiwer not installed")
class TestComputeCer:
    """AC-8 regression cases. If any of these fail, a refactor has dropped
    the `str.translate(_PUNCT_TABLE)` call from `normalize_mandarin_text`.

    Requires `jiwer`; skipped automatically on hosts where it isn't
    installed (the `TestNormalizeMandarinText` cases above already cover
    the punctuation-stripping core without requiring jiwer)."""

    def test_cjk_trailing_period_equivalent(self):
        # Reference has no punctuation; hypothesis adds CJK period. Should
        # normalize to the same string -> CER == 0.
        assert compute_cer("你好。", "你好") == pytest.approx(0.0)

    def test_ascii_trailing_period_equivalent(self):
        assert compute_cer("hello world.", "hello world") == pytest.approx(0.0)

    def test_cjk_multiple_trailing_punctuation_equivalent(self):
        # Reference identical to hypothesis after stripping trailing CJK
        # punctuation cluster.
        assert compute_cer("你好世界", "你好世界。！？") == pytest.approx(0.0)

    def test_real_character_substitution_produces_nonzero_cer(self):
        # Sanity check that the comparator isn't degenerate: actual char
        # differences must still produce non-zero CER.
        cer = compute_cer("你好世界", "你好地球")
        assert cer > 0.0
        # Two of four chars differ -> CER >= 0.5 (up to jiwer's edit-distance
        # boundary; assert the looser lower bound).
        assert cer >= 0.5 - 1e-6

    def test_empty_reference(self):
        # Per implementation: empty reference + non-empty hypothesis = CER 1.
        # Empty reference + empty hypothesis = CER 0.
        assert compute_cer("", "") == pytest.approx(0.0)
        assert compute_cer("x", "") == pytest.approx(1.0)
