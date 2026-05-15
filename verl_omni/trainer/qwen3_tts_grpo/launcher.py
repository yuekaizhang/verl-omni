# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Recipe-side fail-fast validator for the Qwen3-TTS GRPO trainer.

Runs *before* upstream :func:`verl.trainer.main_ppo.run_ppo` so misconfigured
runs fail at launch — not mid-rollout with an opaque KeyError. Catches the
checks the upstream validators do not perform until runtime:

- ``default_agent_loop`` must be ``autoregressive_tts_single_turn_agent``.
- ``agent.agent_loop_manager_class`` must point at our AR-TTS manager.
- ``algorithm.adv_estimator`` must be ``grpo`` (or a documented synonym).
- ``actor_rollout_ref.rollout.n`` must be ``>= 2`` for group-relative GRPO.
- ``reward.reward_model.base_url`` must be set (remote ASR endpoint).
- ``reward.reward_model.co_located`` must be ``false``.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

EXPECTED_AGENT_LOOP = "autoregressive_tts_single_turn_agent"
EXPECTED_MANAGER_FQN = (
    "verl_omni.agent_loop.autoregressive_tts_agent_loop.AutoRegressiveTTSAgentLoopManager"
)
SUPPORTED_ADV_ESTIMATORS = {"grpo"}


class RecipeConfigError(ValueError):
    """Raised when the Qwen3-TTS recipe config is misconfigured."""


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    """Dotted-path getter that works on both DictConfig and plain dict."""

    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, DictConfig):
            try:
                cur = OmegaConf.select(cur, part, default=None)
            except Exception:
                cur = None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur if cur is not None else default


def validate_qwen3_tts_recipe_config(config: Any) -> None:
    """Validate the recipe config; raise :class:`RecipeConfigError` on any failure.

    Args:
        config: The parsed Hydra/OmegaConf config object passed to the trainer.
    """

    default_agent_loop = _get(config, "actor_rollout_ref.rollout.agent.default_agent_loop")
    if default_agent_loop != EXPECTED_AGENT_LOOP:
        raise RecipeConfigError(
            f"actor_rollout_ref.rollout.agent.default_agent_loop must be "
            f"{EXPECTED_AGENT_LOOP!r}, got {default_agent_loop!r}."
        )

    manager_fqn = _get(config, "actor_rollout_ref.rollout.agent.agent_loop_manager_class")
    if manager_fqn != EXPECTED_MANAGER_FQN:
        raise RecipeConfigError(
            f"actor_rollout_ref.rollout.agent.agent_loop_manager_class must be "
            f"{EXPECTED_MANAGER_FQN!r}, got {manager_fqn!r}."
        )

    adv_estimator = _get(config, "algorithm.adv_estimator")
    if adv_estimator not in SUPPORTED_ADV_ESTIMATORS:
        raise RecipeConfigError(
            f"algorithm.adv_estimator must be one of {sorted(SUPPORTED_ADV_ESTIMATORS)}, "
            f"got {adv_estimator!r}. flow_grpo and diffusion-shaped estimators are "
            f"not applicable to AR speech-token rollouts."
        )

    n = _get(config, "actor_rollout_ref.rollout.n")
    if n is None or not isinstance(n, int) or n < 2:
        raise RecipeConfigError(
            f"actor_rollout_ref.rollout.n must be an integer >= 2 for group-relative "
            f"GRPO sampling, got {n!r}."
        )

    base_url = _get(config, "reward.reward_model.base_url")
    if not base_url or base_url == "???":
        raise RecipeConfigError(
            "reward.reward_model.base_url is required — launch a remote vLLM "
            "Qwen3-ASR server and pass its URL via "
            "reward.reward_model.base_url=http://<host>:<port>."
        )

    co_located = bool(_get(config, "reward.reward_model.co_located", False))
    if co_located:
        raise RecipeConfigError(
            "reward.reward_model.co_located=true is forbidden. The Qwen3-TTS "
            "recipe runs the ASR reward server as a separate remote process."
        )

    logger.info(
        "Qwen3-TTS recipe config validated: agent_loop=%s manager=%s adv=%s n=%s asr=%s",
        default_agent_loop,
        manager_fqn,
        adv_estimator,
        n,
        base_url,
    )


__all__ = ["RecipeConfigError", "validate_qwen3_tts_recipe_config"]
