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
"""AR-TTS agent loop, worker, and manager for the Qwen3-TTS GRPO recipe.

These three classes parallel the diffusion stack:
- :class:`AutoRegressiveTTSSingleTurnAgentLoop` — builds the rollout request
  from one dataset row and calls ``server.generate_tts``.
- :class:`AutoRegressiveTTSAgentLoopWorker` — runs the agent loop for a
  batch and packs grouped audio + logprobs into a :class:`DataProto`.
- :class:`AutoRegressiveTTSAgentLoopManager` — verl-omni-specific manager
  wired through ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``.

Naming is ``AutoRegressiveTTS*`` (not ``Qwen3TTS*``) because the
infrastructure is model-generic: any AR speech-token generator that fits
the ``(prompt_text, ref_audio, ref_text)`` interface can reuse it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopMetrics,
    DictConfigWrap,
    _agent_loop_registry,
    register,
)
from verl.protocol import DataProto
from verl.utils.profiler import simple_timer
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AutoRegressiveTTSAgentLoopOutput(BaseModel):
    """Output produced by one AR-TTS agent loop run (one dataset row).

    Holds the grouped audio rollout from ``generate_tts`` plus the metrics.
    The :class:`AutoRegressiveTTSAgentLoopWorker` flattens these into a
    :class:`DataProto` whose ``responses`` is a list of waveforms and whose
    extra fields carry the per-sample codec tokens and logprobs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_text: str
    """The text the talker was asked to speak."""
    completions: list[Any]
    """List of :class:`CompletionAudio` from :class:`AudioRolloutOutput`."""
    sample_rate: int
    """Codec sample rate of the synthesized audio."""
    reward_score: float | None = None
    """Reward score, filled in by the reward manager after rollout."""
    num_turns: int = 0
    """Number of dialogue turns (always 2 for single-turn TTS)."""
    metrics: AgentLoopMetrics
    """Runtime metrics from the rollout."""
    extra_fields: dict[str, Any] = {}
    """Bag for ref_audio / target_audio / speaker_id / etc to flow to reward."""


@register("autoregressive_tts_single_turn_agent")
class AutoRegressiveTTSSingleTurnAgentLoop(AgentLoopBase):
    """Single-turn AR-TTS agent loop.

    Reads ``prompt_text``, ``ref_audio``, ``ref_text`` from the dataset row,
    builds the ``generate_tts`` request, and packages the grouped audio +
    logprobs for the worker.
    """

    async def run(  # type: ignore[override]
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> AutoRegressiveTTSAgentLoopOutput:
        prompt_text = kwargs["prompt_text"]
        ref_audio = kwargs["ref_audio"]
        ref_text = kwargs["ref_text"]
        n = int(sampling_params.get("n", 2))

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate_tts(
                request_id=uuid4().hex,
                prompt_text=prompt_text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                sampling_params=sampling_params,
                n=n,
            )

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        extra_fields = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "speaker_id": kwargs.get("speaker_id"),
            "ref_utt_id": kwargs.get("ref_utt_id"),
            "target_utt_id": kwargs.get("target_utt_id"),
            "target_duration": kwargs.get("target_duration"),
            "target_audio": kwargs.get("target_audio"),
            "data_source": kwargs.get("data_source"),
            "stop_reason": output.stop_reason,
        }

        return AutoRegressiveTTSAgentLoopOutput(
            prompt_text=prompt_text,
            completions=list(output.completions),
            sample_rate=output.sample_rate,
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics) if not isinstance(metrics, AgentLoopMetrics) else metrics,
            extra_fields=extra_fields,
        )


class AutoRegressiveTTSAgentLoopWorker:
    """Per-batch AR-TTS rollout dispatcher.

    Iterates dataset rows, runs each through the AR-TTS agent loop in
    parallel, and packages the grouped audio + logprobs for the trainer.

    The pattern mirrors :class:`verl_omni.agent_loop.diffusion_agent_loop.DiffusionAgentLoopWorker`
    but threads audio-shaped outputs instead of image tensors.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
    ) -> None:
        self.config = config
        self.server_manager = llm_client
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self._sampling_n_default = int(config.actor_rollout_ref.rollout.get("n", 2))

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        sampling_params = self._build_sampling_params(batch)

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = self.config.actor_rollout_ref.rollout.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, **kwargs)))
        outputs = await asyncio.gather(*tasks)

        return self._postprocess(outputs)

    def _build_sampling_params(self, batch: DataProto) -> dict[str, Any]:
        rollout = self.config.actor_rollout_ref.rollout
        is_validate = batch.meta_info.get("validate", False)
        params: dict[str, Any] = {
            "n": int(rollout.get("n", self._sampling_n_default)),
            "temperature": float(rollout.get("temperature", 0.9)),
            "top_p": float(rollout.get("top_p", 1.0)),
            "top_k": int(rollout.get("top_k", 50)),
            "max_tokens": int(rollout.get("max_tokens", 4096)),
            "repetition_penalty": float(rollout.get("repetition_penalty", 1.05)),
            "logprobs": int(rollout.get("logprobs", 1)),
            "stop_token_ids": list(rollout.get("stop_token_ids", [2150])),
        }
        if is_validate:
            params["temperature"] = float(rollout.val_kwargs.get("temperature", 0.0))
            params["seed"] = rollout.val_kwargs.get("seed")
        return params

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        **kwargs: Any,
    ) -> AutoRegressiveTTSAgentLoopOutput:
        if agent_name not in _agent_loop_registry:
            raise KeyError(
                f"Agent loop {agent_name!r} is not registered. "
                f"Known agent loops: {sorted(_agent_loop_registry.keys())}. "
                "AR-TTS recipes must set default_agent_loop=autoregressive_tts_single_turn_agent."
            )
        agent_loop_config = _agent_loop_registry[agent_name]
        import hydra

        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
        )
        return await agent_loop.run(sampling_params, **kwargs)

    def _postprocess(self, outputs: list[AutoRegressiveTTSAgentLoopOutput]) -> DataProto:
        non_tensor_batch: dict[str, np.ndarray] = {
            "prompt_text": np.array([o.prompt_text for o in outputs], dtype=object),
            "completions": np.array([list(o.completions) for o in outputs], dtype=object),
            "sample_rate": np.array([o.sample_rate for o in outputs], dtype=np.int32),
        }
        all_extra_keys: set[str] = set()
        for o in outputs:
            all_extra_keys.update(o.extra_fields.keys())
        for key in all_extra_keys:
            buf = np.empty(len(outputs), dtype=object)
            buf[:] = [o.extra_fields.get(key) for o in outputs]
            non_tensor_batch[key] = buf
        meta_info: dict[str, Any] = {
            "metrics": [o.metrics.model_dump() if hasattr(o.metrics, "model_dump") else o.metrics for o in outputs],
        }
        return DataProto(non_tensor_batch=non_tensor_batch, meta_info=meta_info)


class AutoRegressiveTTSAgentLoopManager(AgentLoopManager):
    """Manager subclass that swaps in :class:`AutoRegressiveTTSAgentLoopWorker`.

    The trainer config sets ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``
    to the FQN of this class so upstream :func:`AgentLoopManager.create()` instantiates
    the AR-TTS worker instead of the default upstream worker.
    """

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "AutoRegressiveTTSAgentLoopManager":
        cls.agent_loop_workers_class = ray.remote(AutoRegressiveTTSAgentLoopWorker)
        return super().create(*args, **kwargs)


__all__ = [
    "AutoRegressiveTTSAgentLoopOutput",
    "AutoRegressiveTTSSingleTurnAgentLoop",
    "AutoRegressiveTTSAgentLoopWorker",
    "AutoRegressiveTTSAgentLoopManager",
]
