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
computes a clipped Mandarin CER (default) or WER (opt-in with explicit
Chinese tokenization) reward. There is no co-located mode — a configuration
that asks for an in-process ASR server is rejected at construction time.

Subclasses :class:`verl.experimental.reward_loop.reward_manager.base.RewardManagerBase`
so the upstream reward-loop pool can drive it: the trainer's reward worker
calls :meth:`run_single` for each grouped completion.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Optional

import httpx
import numpy as np
import torch
from omegaconf import DictConfig
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.protocol import DataProto

from verl_omni.utils.reward_score.asr_error_rate import (
    RewardConfig,
    compute_cer,
    compute_reward,
    compute_wer,
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


def _resolve_tokenize_fn(spec: Optional[str]) -> Optional[Callable[[str], list[str]]]:
    """Resolve a ``module.attr`` tokenize_fn spec (e.g. ``jieba.lcut``) into a callable."""

    if spec is None:
        return None
    if "." not in spec:
        raise ValueError(
            f"chinese_tokenization must be a dotted module.attr path, got {spec!r}."
        )
    module_name, _, attr = spec.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"chinese_tokenization module {module_name!r} cannot be imported: {exc}"
        ) from exc
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ValueError(f"{spec!r} does not resolve to a callable.")
    return fn  # type: ignore[return-value]


class AsrErrorRateRewardManager(RewardManagerBase):
    """HTTP-only Qwen3-ASR reward manager wired into the reward-loop pool.

    Each :meth:`run_single` call processes one grouped sample emitted by the
    AR-TTS agent loop. Endpoint failures surface as ``success=False`` /
    ``reward=NaN`` so the trainer can exclude failed samples from the
    group-advantage computation rather than silently scoring them zero.
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: Any = None,
        compute_score: Any = None,
        *,
        endpoint: AsrEndpointConfig | None = None,
        reward_config: RewardConfig | None = None,
        metric: str | None = None,
        chinese_tokenization: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)

        reward_cfg = config.get("reward", {}) if hasattr(config, "get") else {}
        rm_cfg = reward_cfg.get("reward_model", {}) if hasattr(reward_cfg, "get") else {}
        if endpoint is None:
            base_url = rm_cfg.get("base_url") if hasattr(rm_cfg, "get") else None
            if not base_url:
                raise ValueError(
                    "AsrErrorRateRewardManager requires a remote ASR endpoint URL. "
                    "Pass endpoint=AsrEndpointConfig(base_url=...) or set "
                    "reward.reward_model.base_url in the recipe config."
                )
            endpoint = AsrEndpointConfig(
                base_url=base_url,
                model=rm_cfg.get("model", "qwen3-asr") if hasattr(rm_cfg, "get") else "qwen3-asr",
                timeout_s=float(rm_cfg.get("timeout_s", 60.0)) if hasattr(rm_cfg, "get") else 60.0,
                language=rm_cfg.get("language", "zh") if hasattr(rm_cfg, "get") else "zh",
                co_located=bool(rm_cfg.get("co_located", False)) if hasattr(rm_cfg, "get") else False,
            )
        if endpoint.co_located:
            raise ValueError(
                "AsrErrorRateRewardManager does not support a co-located ASR mode. "
                "Launch the vLLM Qwen3-ASR server as a separate process and pass "
                "its base_url."
            )

        metric = metric or (rm_cfg.get("metric", "cer") if hasattr(rm_cfg, "get") else "cer")
        if metric not in ("cer", "wer"):
            raise ValueError(f"Unsupported metric {metric!r}; choose 'cer' or 'wer'.")
        chinese_tokenization = chinese_tokenization or (
            rm_cfg.get("chinese_tokenization") if hasattr(rm_cfg, "get") else None
        )
        if metric == "wer" and not chinese_tokenization:
            raise ValueError(
                "WER on Mandarin requires an explicit chinese_tokenization config "
                "(e.g. 'jieba.lcut'). CER is the recommended default."
            )

        self._endpoint = endpoint
        self._reward_config = reward_config or RewardConfig()
        self._metric = metric
        self._chinese_tokenize_spec = chinese_tokenization
        self._chinese_tokenize_fn = _resolve_tokenize_fn(chinese_tokenization)
        self._transport = transport
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score) if self.compute_score else True

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

            wav = np.asarray(waveform)
            if wav.dtype.kind == "O":  # object dtype -> unwrap
                wav = np.asarray(wav.item() if wav.shape == () else wav[0])
            if wav.dtype not in (np.float32, np.float64, np.int16, np.int32):
                wav = wav.astype(np.float32)
            sf.write(buf, wav, int(sample_rate), format="WAV", subtype="PCM_16")
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

    def _compute_error_rate(self, transcript: str, prompt_text: str) -> float:
        if self._metric == "wer":
            return compute_wer(transcript, prompt_text, self._chinese_tokenize_fn)
        return compute_cer(transcript, prompt_text)

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

        rate = self._compute_error_rate(transcript, prompt_text)
        reward, breakdown = compute_reward(
            cer=rate,
            generated_duration=generated_duration,
            target_duration=target_duration,
            codec_tokens=codec_tokens,
            config=self._reward_config,
        )
        breakdown["metric"] = self._metric
        breakdown["generated_duration_seconds"] = float(generated_duration)
        return RewardOutcome(
            success=True,
            reward=reward,
            transcript=transcript,
            breakdown=breakdown,
        )

    # ------------------------------------------------------- reward-loop API

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Per-sample scalar rewards: ``rm_scores`` has shape ``(batch_size, 1)``.

        TTS rollouts produce one reward per grouped sample (one waveform ->
        one CER -> one scalar). The AR-token GRPO loss reshapes this back
        into the grouped tensor downstream.
        """

        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    async def run_single(self, data: DataProto) -> dict:
        """Score one grouped TTS sample handed in by the reward-loop worker.

        The agent-loop worker bundles the per-sample inputs into ``data``:

        - ``data.non_tensor_batch["waveform"]``      — synthesized audio array
        - ``data.non_tensor_batch["sample_rate"]``   — codec sample rate (int)
        - ``data.non_tensor_batch["prompt_text"]``   — text the talker was asked to speak
        - ``data.non_tensor_batch["target_duration"]`` — reference duration in seconds
        - ``data.non_tensor_batch["codec_tokens"]``  — stage-0 token IDs (for repetition detector)
        """

        assert len(data) == 1, "AsrErrorRateRewardManager.run_single processes one sample at a time."

        def _unwrap(value: Any) -> Any:
            # DataProto slicing returns per-row numpy arrays; unwrap to the
            # underlying scalar / object when the array is length-1.
            if isinstance(value, np.ndarray):
                if value.shape == ():
                    return value.item()
                if value.shape == (1,):
                    return value[0]
            return value

        non_tensor = data.non_tensor_batch
        waveform = _unwrap(non_tensor.get("waveform"))
        sample_rate = int(_unwrap(non_tensor.get("sample_rate", 24000)) or 24000)
        prompt_text = str(_unwrap(non_tensor.get("prompt_text", "")))
        target_duration = float(_unwrap(non_tensor.get("target_duration", 0.0)) or 0.0)
        codec_tokens = _unwrap(non_tensor.get("codec_tokens"))
        if codec_tokens is not None and not isinstance(codec_tokens, list):
            codec_tokens = list(codec_tokens)

        outcome = await self.score_sample(
            waveform=waveform,
            sample_rate=sample_rate,
            prompt_text=prompt_text,
            target_duration=target_duration,
            codec_tokens=codec_tokens,
        )

        return {
            "reward_score": outcome.reward,
            "reward_extra_info": {
                "success": outcome.success,
                "transcript": outcome.transcript,
                "error": outcome.error or "",
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in outcome.breakdown.items()},
            },
        }


__all__ = [
    "AsrEndpointConfig",
    "AsrEndpointError",
    "AsrErrorRateRewardManager",
    "RewardOutcome",
]
