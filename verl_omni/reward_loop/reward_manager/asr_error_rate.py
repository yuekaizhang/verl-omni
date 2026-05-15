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
"""ASR-based reward manager for the Qwen3-TTS GRPO recipe.

Calls a separately-served remote vLLM Qwen3-ASR endpoint over HTTP and
computes a clipped Mandarin CER reward. There is no co-located mode: a
configuration that asks for an in-process ASR server is rejected at
construction time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
import numpy as np

from verl_omni.utils.reward_score.asr_error_rate import (
    RewardConfig,
    compute_cer,
    compute_reward,
)

logger = logging.getLogger(__file__)


class AsrEndpointError(RuntimeError):
    """Raised when the remote ASR endpoint is unreachable or returns an error."""


@dataclass(frozen=True)
class AsrEndpointConfig:
    """Connection settings for the remote vLLM Qwen3-ASR STT endpoint."""

    base_url: str
    model: str = "qwen3-asr"
    timeout_s: float = 60.0
    language: str = "zh"
    # ``co_located`` is intentionally fixed to False; the constructor rejects True.
    co_located: bool = False


@dataclass
class RewardOutcome:
    """Per-sample reward output, NaN-safe with explicit success flag."""

    success: bool
    reward: float
    transcript: str
    breakdown: dict[str, float]
    error: str | None = None


class AsrErrorRateRewardManager:
    """HTTP-only client that scores TTS rollouts via a remote Qwen3-ASR endpoint.

    Each call returns a :class:`RewardOutcome` with ``success=False`` and a
    NaN-safe sentinel reward (``math.nan``) when the endpoint fails. The
    caller (the agent-loop worker or trainer) is expected to exclude failed
    samples from the GRPO advantage computation. Silent zero-reward fallback
    is forbidden — that would let endpoint failures look like real samples.
    """

    def __init__(
        self,
        *,
        endpoint: AsrEndpointConfig,
        reward_config: RewardConfig | None = None,
        metric: str = "cer",
        chinese_tokenization: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if endpoint.co_located:
            raise ValueError(
                "AsrErrorRateRewardManager does not support a co-located ASR mode. "
                "Launch the vLLM Qwen3-ASR server as a separate process and pass "
                "its base_url in AsrEndpointConfig."
            )
        if metric not in ("cer", "wer"):
            raise ValueError(f"Unsupported metric {metric!r}; choose 'cer' or 'wer'.")
        if metric == "wer" and not chinese_tokenization:
            raise ValueError(
                "WER on Mandarin requires an explicit chinese_tokenization config "
                "(e.g. 'jieba', 'pkuseg'). CER is the recommended default."
            )

        self._endpoint = endpoint
        self._reward_config = reward_config or RewardConfig()
        self._metric = metric
        self._chinese_tokenization = chinese_tokenization
        self._transport = transport

    # ------------------------------------------------------------------ HTTP

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._endpoint.base_url,
            timeout=self._endpoint.timeout_s,
            transport=self._transport,
        )

    async def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        """POST one waveform to the remote ASR endpoint and return the transcript."""

        if waveform is None or (hasattr(waveform, "size") and waveform.size == 0):
            return ""

        buf = BytesIO()
        try:
            import soundfile as sf

            sf.write(buf, waveform, sample_rate, format="WAV")
        except Exception as exc:  # pragma: no cover - depends on soundfile/codec presence
            raise AsrEndpointError(f"Failed to encode waveform for STT upload: {exc}") from exc
        buf.seek(0)

        files = {"file": ("audio.wav", buf.getvalue(), "audio/wav")}
        data = {
            "model": self._endpoint.model,
            "language": self._endpoint.language,
            "response_format": "json",
        }

        try:
            async with self._build_client() as client:
                response = await client.post(
                    "/v1/audio/transcriptions",
                    files=files,
                    data=data,
                )
        except httpx.HTTPError as exc:
            raise AsrEndpointError(
                f"Unreachable remote ASR endpoint {self._endpoint.base_url!r}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise AsrEndpointError(
                f"Remote ASR endpoint returned {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise AsrEndpointError(f"Remote ASR returned non-JSON body: {exc}") from exc

        transcript = payload.get("text") or ""
        return str(transcript)

    # --------------------------------------------------------------- scoring

    async def score_sample(
        self,
        *,
        waveform: np.ndarray | None,
        sample_rate: int,
        prompt_text: str,
        target_duration: float,
        codec_tokens: list[int] | None = None,
    ) -> RewardOutcome:
        """Score one rollout sample. Returns failure-marked outcome on endpoint errors."""

        generated_duration = (
            float(len(waveform)) / max(sample_rate, 1)
            if waveform is not None and hasattr(waveform, "__len__") and len(waveform) > 0
            else 0.0
        )

        try:
            transcript = await self.transcribe(waveform, sample_rate)
        except AsrEndpointError as exc:
            logger.warning("ASR endpoint failure: %s", exc)
            return RewardOutcome(
                success=False,
                reward=math.nan,
                transcript="",
                breakdown={"error": -1.0},
                error=str(exc),
            )

        cer = compute_cer(hypothesis=transcript, reference=prompt_text)
        reward, breakdown = compute_reward(
            cer=cer,
            generated_duration=generated_duration,
            target_duration=target_duration,
            codec_tokens=codec_tokens,
            config=self._reward_config,
        )
        breakdown["generated_duration_seconds"] = float(generated_duration)
        return RewardOutcome(
            success=True,
            reward=reward,
            transcript=transcript,
            breakdown=breakdown,
        )


__all__ = [
    "AsrEndpointConfig",
    "AsrEndpointError",
    "AsrErrorRateRewardManager",
    "RewardOutcome",
]
