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
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, model_validator
from verl.workers.rollout.replica import RolloutReplicaRegistry


class UnsupportedOutputTypeError(TypeError):
    """Raised when a request is routed through the wrong rollout path.

    Example: routing a TTS request through ``vLLMOmniHttpServer.generate``
    (the diffusion-only image/video path) instead of the TTS-specific
    ``generate_tts`` method.
    """


class DiffusionOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    diffusion_output: Any
    """generated image tensor (CHW format) / video tensor (TCHW format)"""
    log_probs: Optional[Any] = None
    """logprobs of generated image/video"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class AudioRolloutOutput(BaseModel):
    """Output produced by :meth:`generate_tts` for one rollout request.

    Stage 0 (``qwen3_tts_talker``) emits the AR-sampled codec token IDs and
    their per-token logprobs. Stage 1 (``qwen3_tts_code2wav``) emits the
    synthesized waveform. The TTS rollout path correlates the two stage
    emissions per ``(request_id, sample_index)`` and packs them into a list
    of ``CompletionAudio`` entries, one per grouped sample, plus the audio
    sample rate that stage 1 produces.

    All tensor-like fields are kept as plain Python lists (or numpy arrays)
    at this layer to keep the rollout server free of torch dependencies;
    the agent loop is responsible for tensorising and padding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    completions: list["CompletionAudio"]
    """One entry per grouped sample (``n=group_size`` items)."""
    sample_rate: int
    """Sample rate of the stage-1 waveform output (e.g. 24000)."""
    stop_reason: Optional[str] = None
    """Stop reason for the whole rollout: 'completed', 'aborted', or other."""
    num_preempted: Optional[int] = None
    """Preemption count surfaced by the engine for metrics."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class CompletionAudio(BaseModel):
    """Per-sample bundle correlating stage-0 tokens with stage-1 audio."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_index: int
    """Group index within the request (0 .. n-1)."""
    codec_tokens: list[int]
    """Stage-0 codec token IDs sampled by the talker (action sequence)."""
    logprobs: list[float]
    """Per-token logprobs aligned with ``codec_tokens`` (same length)."""
    waveform: Any
    """Stage-1 synthesized audio tensor / numpy array."""
    finish_reason: Optional[str] = None
    """Per-sample finish reason from the talker (e.g. 'stop', 'length')."""

    @model_validator(mode="after")
    def _check_token_logprob_alignment(self) -> "CompletionAudio":
        if len(self.codec_tokens) != len(self.logprobs):
            raise ValueError(
                "CompletionAudio invariant violated: len(codec_tokens) "
                f"({len(self.codec_tokens)}) != len(logprobs) ({len(self.logprobs)}). "
                "Per AC-1 the AR scheduler must populate one logprob per sampled codec token."
            )
        return self


AudioRolloutOutput.model_rebuild()


def _load_vllm_omni():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

    return vLLMOmniReplica


def _load_vllm_omni_tts():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_tts_async_server import vLLMOmniTTSReplica

    return vLLMOmniTTSReplica


RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)
RolloutReplicaRegistry.register("vllm_omni_tts", _load_vllm_omni_tts)
