"""Regression test for the agent loop's `[T, N]` codec layout helpers.

`AutoRegressiveTTSAgentLoopWorker._extract_codec_2d` + `_pad_2d_codec`
together produce the structured `codec_ids` tensor `[B, T_pad, N]` that
`MultiCodebookDPActor.compute_log_prob` reads. The helpers accept both
flat `list[int]` (legacy cb0-only) and `list[list[int]]` (multi-codebook)
payloads.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

# The agent loop module imports tensordict + ray + vllm. Skip this whole
# file on hosts that don't have those (e.g. the test runner without verl
# fully installed); the structural change is small and the helpers are
# pure-python list/torch operations covered by py_compile already.
if importlib.util.find_spec("tensordict") is None:
    pytest.skip("tensordict not installed (full verl env required)", allow_module_level=True)

from verl_omni.agent_loop.autoregressive_tts_agent_loop import (  # noqa: E402
    AutoRegressiveTTSAgentLoopWorker,
)


_extract = AutoRegressiveTTSAgentLoopWorker._extract_codec_2d
_pad_2d = AutoRegressiveTTSAgentLoopWorker._pad_2d_codec


def test_extract_codec_2d_from_flat_list():
    """Legacy single-codebook payload: flat `[T]` -> `[T, 1]`."""
    flat = [10, 20, 30, 40]
    out = _extract(flat)
    assert out == [[10], [20], [30], [40]]


def test_extract_codec_2d_from_2d_list():
    """Multi-codebook payload: `[T, N]` passes through unchanged."""
    nested = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    out = _extract(nested)
    assert out == nested


def test_extract_codec_2d_handles_empty():
    assert _extract([]) == []
    assert _extract(None) == []


def test_pad_2d_codec_short_input_pads_with_zeros():
    """Input shorter than `length` gets right-padded along time."""
    values = [[10, 11], [20, 21]]
    out = _pad_2d(values, length=5, pad_value=0)
    assert out.shape == (5, 2)
    assert out.tolist() == [[10, 11], [20, 21], [0, 0], [0, 0], [0, 0]]


def test_pad_2d_codec_long_input_truncates():
    values = [[i, i + 100] for i in range(8)]
    out = _pad_2d(values, length=3, pad_value=0)
    assert out.shape == (3, 2)
    assert out.tolist() == [[0, 100], [1, 101], [2, 102]]


def test_pad_2d_codec_empty_input_returns_pad_value_tensor():
    """Empty rollout (no codec frames) yields a `[length, 1]` all-pad tensor."""
    out = _pad_2d([], length=3, pad_value=0)
    assert out.shape == (3, 1)
    assert out.tolist() == [[0], [0], [0]]


def test_pad_2d_codec_rejects_uneven_rows():
    """Defensive: mismatched per-frame N is a hard error."""
    import pytest

    values = [[10, 11, 12], [20, 21]]  # frame 1 missing one codebook
    with pytest.raises(ValueError, match="uniform"):
        _pad_2d(values, length=3, pad_value=0)


def test_extract_plus_pad_preserves_codebook_major_within_frame():
    """End-to-end: a multi-codebook payload + pad produces `[T_pad, N]`
    where row `t` = the codec ids at frame `t` (codebooks in order)."""
    payload = [[100, 101, 102], [200, 201, 202], [300, 301, 302]]
    parsed = _extract(payload)
    padded = _pad_2d(parsed, length=5, pad_value=0)
    assert padded.shape == (5, 3)
    # Frame 0: (cb0, cb1, cb2) = (100, 101, 102).
    assert padded[0].tolist() == [100, 101, 102]
    # Frame 2: (cb0, cb1, cb2) = (300, 301, 302).
    assert padded[2].tolist() == [300, 301, 302]
    # Padded frames: all zeros.
    assert padded[3].tolist() == [0, 0, 0]
    assert padded[4].tolist() == [0, 0, 0]
