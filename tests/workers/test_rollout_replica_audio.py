# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for AudioRolloutOutput / UnsupportedOutputTypeError (no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from verl_omni.workers.rollout.replica import (
    AudioRolloutOutput,
    CompletionAudio,
    UnsupportedOutputTypeError,
)


def _make_completion(index: int = 0) -> CompletionAudio:
    return CompletionAudio(
        sample_index=index,
        codec_tokens=[101, 202, 303, 404],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        waveform=np.zeros(8000, dtype=np.float32),
        finish_reason="stop",
    )


def test_audio_rollout_output_carries_grouped_samples() -> None:
    audio = AudioRolloutOutput(
        completions=[_make_completion(0), _make_completion(1)],
        sample_rate=24000,
        stop_reason="completed",
        num_preempted=0,
    )
    assert audio.sample_rate == 24000
    assert len(audio.completions) == 2
    for c in audio.completions:
        assert len(c.codec_tokens) == len(c.logprobs)
        assert c.waveform.shape[-1] > 0
        assert c.finish_reason == "stop"


def test_unsupported_output_type_is_typeerror_subclass() -> None:
    assert issubclass(UnsupportedOutputTypeError, TypeError)
    with pytest.raises(UnsupportedOutputTypeError):
        raise UnsupportedOutputTypeError("test")


def test_completion_audio_rejects_mismatched_logprob_length() -> None:
    # AC-1 requires len(logprobs) == len(codec_tokens). The model validator
    # enforces this invariant; callers cannot construct a misaligned object.
    with pytest.raises(ValueError, match="len\\(codec_tokens\\)"):
        CompletionAudio(
            sample_index=0,
            codec_tokens=[1, 2, 3],
            logprobs=[-0.1, -0.2],  # mismatched on purpose
            waveform=np.zeros(1, dtype=np.float32),
        )
