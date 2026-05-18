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

The diffusion stack uses three pieces (``DiffusionSingleTurnAgentLoop`` /
``DiffusionAgentLoopWorker`` / a custom manager). The AR-TTS recipe adds a
parallel trio:

- :class:`AutoRegressiveTTSSingleTurnAgentLoop` — per-row request builder
  that calls ``server_manager.generate_tts(...)`` on the AR-TTS server
  client. Inherits from ``AgentLoopBase`` for registry compatibility but
  skips the chat-template system-prompt init because TTS rollouts do not
  use chat tokenization.
- :class:`AutoRegressiveTTSAgentLoopWorker` — per-batch dispatcher that
  initializes tokenizer/processor/dataset state, runs grouped rollouts in
  parallel, dispatches reward computation per grouped sample through the
  upstream ``reward_loop_worker_handles``, and produces a trainer-ready
  :class:`DataProto` with padded ``prompts`` / ``responses`` /
  ``rollout_log_probs`` / ``attention_mask``.
- :class:`AutoRegressiveTTSAgentLoopManager` — manager subclass wired
  through ``agent.agent_loop_manager_class`` that injects our worker.

Naming is ``AutoRegressiveTTS*`` (not ``Qwen3TTS*``) because the
infrastructure is model-generic — any AR speech-token generator fitting
the ``(prompt_text, ref_audio, ref_text)`` interface can reuse it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopMetrics,
    DictConfigWrap,
    _agent_loop_registry,
    register,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.utils.profiler import simple_timer
from verl.workers.rollout.llm_server import LLMServerClient

from omegaconf import OmegaConf

from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------- output


class AutoRegressiveTTSAgentLoopOutput(BaseModel):
    """Single-sample output produced by :class:`AutoRegressiveTTSSingleTurnAgentLoop`.

    The worker flattens these into a :class:`DataProto` with padded
    ``prompts`` / ``responses`` / ``rollout_log_probs`` / ``attention_mask``
    tensors plus non-tensor metadata for the reward path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    prompt_text: str
    completions: list[Any]
    sample_rate: int
    reward_score: float | None = None
    num_turns: int = 0
    metrics: AgentLoopMetrics
    extra_fields: dict[str, Any] = {}


# ---------------------------------------------------------------------------- agent loop


@register("autoregressive_tts_single_turn_agent")
class AutoRegressiveTTSSingleTurnAgentLoop(AgentLoopBase):
    """Single-turn AR-TTS agent loop.

    Reads ``prompt_text`` / ``ref_audio`` / ``ref_text`` from the dataset
    row, builds the rollout request, and packages the grouped audio +
    logprobs.

    Overrides :meth:`__init__` to keep tokenizer optional — TTS prompts are
    raw text plus an audio reference, not a tokenized chat. Upstream
    ``AgentLoopBase.__init__`` calls ``initialize_system_prompt(tokenizer)``
    which would fail when the tokenizer is the talker's BPE rather than a
    chat-template tokenizer.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: Any,
        tokenizer: Any = None,
        processor: Any = None,
        dataset_cls: Any = None,
        data_config: Any = None,
        **kwargs: Any,
    ) -> None:
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config if data_config is not None else None
        # No system prompt for TTS — the talker reads (text, ref_audio, ref_text).
        self.system_prompt = None
        from verl.experimental.agent_loop.agent_loop import get_event_loop

        self.loop = get_event_loop()

    async def run(  # type: ignore[override]
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> AutoRegressiveTTSAgentLoopOutput:
        prompt_text = str(kwargs["prompt_text"])
        ref_audio = kwargs["ref_audio"]
        ref_text = str(kwargs["ref_text"])
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
        metrics.setdefault("tool_calls", 0.0)
        metrics.setdefault("compute_score", 0.0)

        prompt_ids = self._tokenize_prompt(prompt_text)
        completions = list(output.completions)

        extra_fields = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "speaker_id": kwargs.get("speaker_id"),
            "ref_utt_id": kwargs.get("ref_utt_id"),
            "target_utt_id": kwargs.get("target_utt_id"),
            "target_duration": float(kwargs.get("target_duration") or 0.0),
            "target_audio": kwargs.get("target_audio"),
            "data_source": kwargs.get("data_source"),
            "stop_reason": output.stop_reason,
            "sample_rate": output.sample_rate,
        }

        return AutoRegressiveTTSAgentLoopOutput(
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            completions=completions,
            sample_rate=output.sample_rate,
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics) if not isinstance(metrics, AgentLoopMetrics) else metrics,
            extra_fields=extra_fields,
        )

    def _tokenize_prompt(self, prompt_text: str) -> list[int]:
        """Tokenize the text prompt with the talker's tokenizer when available.

        Used downstream by the worker to populate ``batch["prompts"]`` so
        upstream ``AgentLoopManager._performance_metrics`` does not crash.
        Falls back to a single placeholder token id (``0``) when no
        tokenizer was injected.
        """

        if self.tokenizer is None or not prompt_text:
            return [0]
        try:
            ids = self.tokenizer(prompt_text, add_special_tokens=False).get("input_ids")
            if isinstance(ids, list) and ids:
                return list(ids)
        except Exception as exc:  # pragma: no cover - tokenizer differences
            logger.warning("Talker tokenizer failed on prompt_text; using placeholder. %s", exc)
        return [0]


# ---------------------------------------------------------------------------- worker


class AutoRegressiveTTSAgentLoopWorker:
    """Per-batch AR-TTS rollout dispatcher emitting a trainer-ready :class:`DataProto`.

    Mirrors the structure of :class:`verl_omni.agent_loop.diffusion_agent_loop.DiffusionAgentLoopWorker`:

    - Loads dataset class, tokenizer, processor, agent-loop registry config.
    - Builds the sampling params dict from ``rollout_config`` and runs each
      row through :class:`AutoRegressiveTTSSingleTurnAgentLoop` in parallel.
    - Dispatches reward computation per grouped sample through
      ``reward_loop_worker_handles`` when available; failed samples
      (``success=False``) carry ``reward_score=NaN`` and the trainer is
      expected to mask them out of the GRPO advantage computation.
    - Pads ``prompts`` / ``responses`` / ``rollout_log_probs`` /
      ``attention_mask`` into a tensor batch that upstream
      :meth:`AgentLoopManager._performance_metrics` accepts.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
    ) -> None:
        self.config = config
        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model
        self.rollout_config: DiffusionRolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: DiffusionModelConfig = omega_conf_to_dataclass(model_config)

        self.server_manager = llm_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        # Validation artifact logger: AC-8 requires generated/ref/(target) wavs
        # plus metrics.json per validation step. The trainer wires this into
        # the worker's runtime path; the per-step write happens inside
        # generate_sequences when batch.meta_info["validate"] is True.
        from verl_omni.utils.validation_audio_logger import (
            log_validation_step,
            post_run_check_emitted_artifacts,
        )

        self._log_validation_step = log_validation_step
        self._post_run_check_emitted_artifacts = post_run_check_emitted_artifacts
        validation_dir = getattr(
            getattr(config.trainer, "validation_data_dir", None),
            "__fspath__",
            None,
        )
        self._validation_output_dir = (
            Path(config.trainer.validation_data_dir)
            if getattr(config.trainer, "validation_data_dir", None)
            else Path(config.trainer.default_local_dir) / "validation_audio"
        ) if hasattr(config, "trainer") else None
        self._validation_step_counter = 0

        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        sampling_params = self._build_sampling_params(batch)
        is_validate = bool(batch.meta_info.get("validate", False))

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = self.config.actor_rollout_ref.rollout.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, **kwargs)))
        outputs = await asyncio.gather(*tasks)

        if is_validate and self._validation_output_dir is not None:
            self._emit_validation_artifacts(outputs)

        return self._postprocess(outputs)

    def _emit_validation_artifacts(
        self,
        outputs: list["AutoRegressiveTTSAgentLoopOutput"],
    ) -> None:
        """Persist AC-8 audio artifacts for one validation step."""

        samples: list[dict[str, Any]] = []
        for output in outputs:
            for completion in output.completions:
                samples.append(
                    {
                        "waveform": completion.waveform,
                        "sample_rate": output.sample_rate,
                        "ref_audio": output.extra_fields.get("ref_audio"),
                        "target_audio": output.extra_fields.get("target_audio"),
                    }
                )
        if not samples:
            return

        # AC-8 requires the per-step metrics.json to carry the AC-6 scalar set.
        # policy_loss / kl_loss are training-step quantities; for a rollout-only
        # validation step we emit them as null so the schema is stable but the
        # operator can see they were not produced this step.
        step_metrics: dict[str, Any] = {
            "mean_reward": None,
            "mean_cer": None,
            "mean_duration_ratio": None,
            "policy_loss": None,
            "kl_loss": None,
        }

        rewards = [o.reward_score for o in outputs if o.reward_score is not None]
        if rewards:
            step_metrics["mean_reward"] = float(sum(rewards) / len(rewards))

        cers: list[float] = []
        for o in outputs:
            info = o.extra_fields.get("reward_extra_info") if isinstance(o.extra_fields, dict) else None
            if isinstance(info, dict):
                per_cers = info.get("per_sample_cers")
                if isinstance(per_cers, (list, tuple)):
                    cers.extend(float(x) for x in per_cers if isinstance(x, (int, float)) and math.isfinite(x))
        if cers:
            step_metrics["mean_cer"] = float(sum(cers) / len(cers))

        duration_ratios: list[float] = []
        for o in outputs:
            target_duration = float(o.extra_fields.get("target_duration") or 0.0)
            if target_duration <= 0:
                continue
            sample_rate = max(int(o.sample_rate or 1), 1)
            for completion in o.completions:
                waveform = completion.waveform
                if waveform is None or not hasattr(waveform, "__len__"):
                    continue
                generated = float(len(waveform)) / sample_rate
                duration_ratios.append(generated / target_duration)
        if duration_ratios:
            step_metrics["mean_duration_ratio"] = float(sum(duration_ratios) / len(duration_ratios))

        self._validation_step_counter += 1
        try:
            self._log_validation_step(
                out_dir=self._validation_output_dir,
                step=self._validation_step_counter,
                samples=samples,
                scalar_metrics=step_metrics,
                num_samples=min(4, len(samples)),
            )
        except Exception as exc:  # pragma: no cover - logged for the operator
            logger.warning("Validation audio logging failed at step %s: %s", self._validation_step_counter, exc)
            raise

    def _build_sampling_params(self, batch: DataProto) -> dict[str, Any]:
        rollout = self.config.actor_rollout_ref.rollout
        is_validate = batch.meta_info.get("validate", False)
        params: dict[str, Any] = {
            "n": int(rollout.get("n", 2)),
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
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )
        output: AutoRegressiveTTSAgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
        await self._compute_score(output, kwargs=kwargs)
        return output

    async def _compute_score(
        self,
        output: AutoRegressiveTTSAgentLoopOutput,
        *,
        kwargs: dict[str, Any],
    ) -> None:
        """Score every grouped completion through ``reward_loop_worker_handles``."""

        if not self.reward_loop_worker_handles:
            return
        if output.reward_score is not None:
            return

        timing: dict[str, Any] = {}
        with simple_timer("compute_score", timing):
            scores: list[float] = []
            successes: list[bool] = []
            transcripts: list[str] = []
            cers: list[float] = []
            duration_penalties: list[float] = []
            for completion in output.completions:
                # Wrap the waveform via np.empty+assign so DataProto keeps the
                # underlying ndarray rather than collapsing it into a 2-D
                # object array (BL-20260515-dataproto-object-batch-wrap).
                waveform_arr = np.empty(1, dtype=object)
                waveform_arr[0] = completion.waveform
                codec_arr = np.empty(1, dtype=object)
                codec_arr[0] = list(completion.codec_tokens)
                non_tensor_batch = {
                    "waveform": waveform_arr,
                    "sample_rate": np.array([output.sample_rate]),
                    "prompt_text": np.array([output.prompt_text]),
                    "target_duration": np.array([output.extra_fields.get("target_duration", 0.0)]),
                    "codec_tokens": codec_arr,
                    "data_source": np.array([output.extra_fields.get("data_source", "qwen3_tts")]),
                }
                data = DataProto(non_tensor_batch=non_tensor_batch)
                handle = random.choice(self.reward_loop_worker_handles)
                result = await handle.compute_score.remote(data)
                scores.append(float(result["reward_score"]))
                info = result.get("reward_extra_info", {})
                successes.append(bool(info.get("success", True)))
                transcripts.append(str(info.get("transcript", "")))
                # Preserve the reward-manager breakdown so validation logging
                # can emit per-step mean_cer / mean_duration_penalty etc.
                cer_value = info.get("cer")
                if isinstance(cer_value, (int, float)) and math.isfinite(cer_value):
                    cers.append(float(cer_value))
                dp_value = info.get("duration_penalty")
                if isinstance(dp_value, (int, float)) and math.isfinite(dp_value):
                    duration_penalties.append(float(dp_value))

            # GRPO-mean over successful samples; failed (NaN) samples are excluded.
            finite = [s for s, ok in zip(scores, successes) if ok and math.isfinite(s)]
            if finite:
                output.reward_score = float(sum(finite) / len(finite))
            else:
                output.reward_score = math.nan
            output.extra_fields["reward_extra_info"] = {
                "per_sample_rewards": scores,
                "per_sample_success": successes,
                "per_sample_transcripts": transcripts,
                "per_sample_cers": cers,
                "per_sample_duration_penalties": duration_penalties,
            }
        output.metrics.compute_score = timing.get("compute_score", 0.0)

    # ---------------------------------------------------------------- postprocess

    def _postprocess(self, outputs: list[AutoRegressiveTTSAgentLoopOutput]) -> DataProto:
        prompt_pad = int(self.rollout_config.prompt_length or 64)
        response_pad = max(
            (len(c.codec_tokens) for o in outputs for c in o.completions),
            default=int(self.rollout_config.response_length or 1),
        )
        response_pad = max(response_pad, 1)

        rows_per_sample_prompt: list[torch.Tensor] = []
        rows_per_sample_response: list[torch.Tensor] = []
        rows_per_sample_logprobs: list[torch.Tensor] = []
        rows_per_sample_attention: list[torch.Tensor] = []
        rows_non_tensor: dict[str, list[Any]] = {
            "prompt_text": [],
            "completion_index": [],
            "sample_rate": [],
            "waveform": [],
            "codec_tokens": [],
            # Upstream RayPPOTrainer.training_step iterates
            # batch.non_tensor_batch["multi_modal_inputs"] unconditionally
            # (see verl.trainer.ppo.ray_trainer:1418). For audio recipes there
            # are no image/video inputs, so we emit an empty-dict placeholder
            # per row to keep the trainer's generic bookkeeping path safe.
            "multi_modal_inputs": [],
            "__num_turns__": [],
        }
        for key in (
            "ref_audio",
            "ref_text",
            "speaker_id",
            "ref_utt_id",
            "target_utt_id",
            "target_duration",
            "target_audio",
            "data_source",
            # ``transcript`` is declared in ``meta_info["reward_extra_keys"]``
            # below (so upstream verl's ``extract_reward`` can pick it up);
            # the actual per-row strings come from
            # ``reward_extra_info["per_sample_transcripts"]`` computed in
            # ``run`` after ASR scoring. Without populating it here the
            # trainer crashes at
            # ``verl/trainer/ppo/reward.py:166`` with
            # ``KeyError: 'transcript'`` when iterating reward_extra_keys.
            "transcript",
        ):
            rows_non_tensor[key] = []
        per_sample_rewards: list[float] = []
        per_sample_success: list[bool] = []

        for output in outputs:
            prompt_t = self._pad_1d(output.prompt_ids, prompt_pad, pad_value=0)
            extra = output.extra_fields
            reward_info = extra.get("reward_extra_info", {}) if isinstance(extra, dict) else {}
            per_rewards = reward_info.get("per_sample_rewards", [float("nan")] * len(output.completions))
            per_success = reward_info.get("per_sample_success", [True] * len(output.completions))
            per_transcripts = reward_info.get(
                "per_sample_transcripts", [""] * len(output.completions)
            )
            for i, completion in enumerate(output.completions):
                rows_per_sample_prompt.append(prompt_t)
                resp_t = self._pad_1d(list(completion.codec_tokens), response_pad, pad_value=0)
                rows_per_sample_response.append(resp_t)
                logp_t = self._pad_1d_float(list(completion.logprobs), response_pad, pad_value=0.0)
                rows_per_sample_logprobs.append(logp_t)
                response_mask = (resp_t != 0).long()
                attention = torch.cat([torch.ones_like(prompt_t), response_mask], dim=-1)
                rows_per_sample_attention.append(attention)
                rows_non_tensor["prompt_text"].append(output.prompt_text)
                rows_non_tensor["completion_index"].append(int(completion.sample_index))
                rows_non_tensor["sample_rate"].append(int(output.sample_rate))
                rows_non_tensor["waveform"].append(completion.waveform)
                rows_non_tensor["codec_tokens"].append(list(completion.codec_tokens))
                rows_non_tensor["multi_modal_inputs"].append({})
                rows_non_tensor["__num_turns__"].append(int(output.num_turns))
                for key in (
                    "ref_audio",
                    "ref_text",
                    "speaker_id",
                    "ref_utt_id",
                    "target_utt_id",
                    "target_duration",
                    "target_audio",
                    "data_source",
                ):
                    rows_non_tensor[key].append(extra.get(key))
                rows_non_tensor["transcript"].append(
                    str(per_transcripts[i]) if i < len(per_transcripts) else ""
                )
                per_sample_rewards.append(float(per_rewards[i]) if i < len(per_rewards) else float("nan"))
                per_sample_success.append(bool(per_success[i]) if i < len(per_success) else True)

        prompts_t = torch.stack(rows_per_sample_prompt, dim=0)
        responses_t = torch.stack(rows_per_sample_response, dim=0)
        logprobs_t = torch.stack(rows_per_sample_logprobs, dim=0)
        attention_t = torch.stack(rows_per_sample_attention, dim=0)
        rm_scores = torch.tensor(per_sample_rewards, dtype=torch.float32).unsqueeze(-1)
        success_mask = torch.tensor(per_sample_success, dtype=torch.bool).unsqueeze(-1)

        # Upstream verl's ``_compute_old_log_prob`` calls
        # ``left_right_2_no_padding`` which asserts the batch carries
        # ``input_ids``, ``attention_mask``, ``response_mask`` and
        # ``position_ids`` (see
        # ``verl/workers/utils/padding.py:39-42``). Synthesise them from
        # the prompts/responses we already have so the actor's log-prob
        # recompute path can run without changes.
        input_ids_t = torch.cat([prompts_t, responses_t], dim=-1)
        response_mask_t = (responses_t != 0).long()
        # position_ids are derived from attention_mask cumulative sum so
        # left-padded prompts get position 0 at the first real token.
        position_ids_t = (attention_t.long().cumsum(dim=-1) - 1).clamp(min=0)

        batch = TensorDict(
            {
                "prompts": prompts_t,
                "responses": responses_t,
                "rollout_log_probs": logprobs_t,
                "attention_mask": attention_t,
                "response_mask": response_mask_t,
                "input_ids": input_ids_t,
                "position_ids": position_ids_t,
                "rm_scores": rm_scores,
                "success_mask": success_mask,
            },
            batch_size=len(rows_per_sample_prompt),
        )

        non_tensor_batch: dict[str, np.ndarray] = {}
        for key, values in rows_non_tensor.items():
            arr = np.empty(len(values), dtype=object)
            arr[:] = values
            non_tensor_batch[key] = arr

        metrics = [
            o.metrics.model_dump() if hasattr(o.metrics, "model_dump") else o.metrics for o in outputs
        ]
        # Replicate per-row metrics across each sample for the upstream metric reducer.
        per_sample_metrics: list[dict[str, Any]] = []
        for output in outputs:
            for _ in output.completions:
                m = output.metrics.model_dump() if hasattr(output.metrics, "model_dump") else dict(output.metrics)
                m.setdefault("tool_calls", 0.0)
                m.setdefault("compute_score", 0.0)
                m.setdefault("num_preempted", -1)
                m.setdefault("generate_sequences", 0.0)
                per_sample_metrics.append(m)

        meta_info = {"metrics": per_sample_metrics, "reward_extra_keys": ["transcript"]}
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    @staticmethod
    def _pad_1d(values: list[int], length: int, pad_value: int) -> torch.Tensor:
        if not values:
            values = [pad_value]
        if len(values) >= length:
            return torch.tensor(values[:length], dtype=torch.long)
        out = torch.full((length,), pad_value, dtype=torch.long)
        out[: len(values)] = torch.tensor(values, dtype=torch.long)
        return out

    @staticmethod
    def _pad_1d_float(values: list[float], length: int, pad_value: float) -> torch.Tensor:
        if not values:
            values = [pad_value]
        if len(values) >= length:
            return torch.tensor(values[:length], dtype=torch.float32)
        out = torch.full((length,), pad_value, dtype=torch.float32)
        out[: len(values)] = torch.tensor(values, dtype=torch.float32)
        return out


# ---------------------------------------------------------------------------- manager


class AutoRegressiveTTSAgentLoopManager(AgentLoopManager):
    """Manager subclass that wires :class:`AutoRegressiveTTSAgentLoopWorker`.

    The trainer config sets ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``
    to the FQN of this class so upstream :func:`AgentLoopManager.create()`
    instantiates the AR-TTS worker instead of the default upstream worker.
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
