# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the ASR error-rate reward manager + scoring helpers (no GPU)."""

from __future__ import annotations

import math

import httpx
import numpy as np
import pytest

from verl_omni.reward_loop.reward_manager.asr_error_rate import (
    AsrEndpointConfig,
    AsrEndpointError,
    AsrErrorRateRewardManager,
)
from verl_omni.utils.reward_score.asr_error_rate import (
    RewardConfig,
    compute_cer,
    compute_reward,
    compute_wer,
    detect_audio_repetition,
    normalize_mandarin_text,
)


# ---------------------------------------------------------------------------- pure scoring


def test_normalize_strips_punctuation_and_fold_widths() -> None:
    text = "你好，世界！Hello, World."
    assert normalize_mandarin_text(text) == "你好世界hello world"


def test_compute_cer_zero_on_exact_match() -> None:
    assert compute_cer("你好世界", "你好世界") == 0.0


def test_compute_cer_substitutions() -> None:
    cer = compute_cer("你好今界", "你好世界")  # one of 4 chars wrong
    assert 0.0 < cer < 1.0


def test_compute_wer_requires_tokenize_fn() -> None:
    with pytest.raises(ValueError, match="tokenize_fn"):
        compute_wer("you go to school", "you go to school", None)


def test_compute_wer_with_whitespace_tokenizer() -> None:
    tokenize_fn = lambda s: s.split()
    assert compute_wer("you go to school", "you go to school", tokenize_fn) == 0.0
    assert compute_wer("you go to school today", "you go to school", tokenize_fn) > 0.0


def test_metric_branching_uses_configured_metric(monkeypatch) -> None:
    # Use a stub tokenizer that just splits on whitespace.
    import sys, types
    stub = types.ModuleType("fake_tok")
    stub.cut = lambda s: s.split()
    sys.modules["fake_tok"] = stub

    endpoint = AsrEndpointConfig(base_url="http://asr.example.com")
    rm = AsrErrorRateRewardManager(
        config={},
        endpoint=endpoint,
        metric="wer",
        chinese_tokenization="fake_tok.cut",
        transport=_mock_transcription_transport("hello world"),
    )
    # _compute_error_rate should call the WER path (not CER):
    rate = rm._compute_error_rate("hello world", "hello world")
    assert rate == 0.0


def test_detect_audio_repetition_catches_loops() -> None:
    looped = ([1, 2, 3, 4, 5] * 4)  # the same 5-gram repeats four times
    assert detect_audio_repetition(looped, repeat_ngram=5, repeat_threshold=3) is True


def test_detect_audio_repetition_clean_audio() -> None:
    assert detect_audio_repetition(list(range(50))) is False


def test_compute_reward_clean_sample() -> None:
    reward, breakdown = compute_reward(
        cer=0.05,
        generated_duration=2.0,
        target_duration=2.0,
        codec_tokens=list(range(30)),
    )
    assert -1.0 <= reward <= 1.0
    assert math.isfinite(reward)
    assert breakdown["empty_penalty"] == 0.0
    assert breakdown["duration_penalty"] == 0.0
    assert breakdown["repetition_penalty"] == 0.0


def test_compute_reward_empty_audio_only_empty_penalty() -> None:
    reward, breakdown = compute_reward(
        cer=1.0,
        generated_duration=0.0,
        target_duration=2.0,
    )
    assert math.isfinite(reward)
    assert breakdown["empty_penalty"] == 1.0
    assert breakdown["duration_penalty"] == 0.0


def test_compute_reward_short_nonempty_gets_duration_penalty_not_empty() -> None:
    _, breakdown = compute_reward(cer=0.1, generated_duration=0.1, target_duration=2.0)
    assert breakdown["empty_penalty"] == 0.0  # not truly empty
    assert breakdown["duration_penalty"] > 0.0


def test_compute_reward_truly_empty_gets_empty_penalty() -> None:
    _, breakdown = compute_reward(cer=1.0, generated_duration=0.0, target_duration=2.0)
    assert breakdown["empty_penalty"] == 1.0
    assert breakdown["duration_penalty"] == 0.0


def test_compute_reward_long_audio_gets_duration_penalty() -> None:
    _, breakdown = compute_reward(cer=0.1, generated_duration=10.0, target_duration=2.0)
    assert breakdown["duration_penalty"] > 0


def test_compute_reward_repetition_penalty_fires() -> None:
    looped = ([7, 8, 9, 10, 11] * 5)
    _, breakdown = compute_reward(
        cer=0.1, generated_duration=2.0, target_duration=2.0, codec_tokens=looped
    )
    assert breakdown["repetition_penalty"] > 0


def test_compute_reward_clipped_to_floor() -> None:
    reward, _ = compute_reward(
        cer=2.0,
        generated_duration=0.0,
        target_duration=2.0,
        config=RewardConfig(reward_floor=-1.0, reward_ceiling=1.0),
    )
    assert reward == -1.0


# ---------------------------------------------------------------------------- reward manager


def _mock_transcription_transport(transcript: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": transcript})

    return httpx.MockTransport(handler)


def _mock_unreachable_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated network error", request=request)

    return httpx.MockTransport(handler)


def test_co_located_mode_rejected_at_construction() -> None:
    endpoint = AsrEndpointConfig(base_url="http://asr.example.com", co_located=True)
    with pytest.raises(ValueError, match="co-located ASR mode"):
        AsrErrorRateRewardManager(config={}, endpoint=endpoint)


def test_wer_without_chinese_tokenization_rejected() -> None:
    endpoint = AsrEndpointConfig(base_url="http://asr.example.com")
    with pytest.raises(ValueError, match="chinese_tokenization"):
        AsrErrorRateRewardManager(config={}, endpoint=endpoint, metric="wer")


def test_invalid_metric_rejected() -> None:
    endpoint = AsrEndpointConfig(base_url="http://asr.example.com")
    with pytest.raises(ValueError, match="Unsupported metric"):
        AsrErrorRateRewardManager(config={}, endpoint=endpoint, metric="bleu")


def test_missing_base_url_rejected() -> None:
    with pytest.raises(ValueError, match="endpoint URL"):
        AsrErrorRateRewardManager(config={})


@pytest.mark.asyncio
async def test_unreachable_endpoint_returns_failure_outcome() -> None:
    endpoint = AsrEndpointConfig(base_url="http://asr.example.com")
    rm = AsrErrorRateRewardManager(
        config={},
        endpoint=endpoint,
        transport=_mock_unreachable_transport(),
    )
    outcome = await rm.score_sample(
        waveform=np.zeros(24000, dtype=np.float32),
        sample_rate=24000,
        prompt_text="你好世界",
        target_duration=1.0,
    )
    assert outcome.success is False
    assert math.isnan(outcome.reward)
    assert outcome.error is not None


@pytest.mark.asyncio
async def test_clean_sample_returns_high_reward() -> None:
    endpoint = AsrEndpointConfig(base_url="http://asr.example.com")
    rm = AsrErrorRateRewardManager(
        config={},
        endpoint=endpoint,
        transport=_mock_transcription_transport("你好世界"),
    )
    outcome = await rm.score_sample(
        waveform=np.zeros(24000, dtype=np.float32),
        sample_rate=24000,
        prompt_text="你好世界",
        target_duration=1.0,
    )
    assert outcome.success is True
    assert outcome.reward == 1.0
    assert outcome.transcript == "你好世界"
    assert math.isfinite(outcome.reward)


@pytest.mark.asyncio
async def test_run_single_wraps_score_sample_for_reward_loop() -> None:
    """run_single is the public reward-loop entry point — exercise it."""
    from verl.protocol import DataProto

    endpoint = AsrEndpointConfig(base_url="http://asr.example.com")
    rm = AsrErrorRateRewardManager(
        config={},
        endpoint=endpoint,
        transport=_mock_transcription_transport("你好世界"),
    )
    waveform_arr = np.empty(1, dtype=object)
    waveform_arr[0] = np.zeros(24000, dtype=np.float32)
    codec_arr = np.empty(1, dtype=object)
    codec_arr[0] = [1, 2, 3, 4, 5]
    non_tensor_batch = {
        "waveform": waveform_arr,
        "sample_rate": np.array([24000]),
        "prompt_text": np.array(["你好世界"]),
        "target_duration": np.array([1.0]),
        "codec_tokens": codec_arr,
    }
    data = DataProto(non_tensor_batch=non_tensor_batch)
    result = await rm.run_single(data)
    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"]["success"] is True
    assert result["reward_extra_info"]["transcript"] == "你好世界"
