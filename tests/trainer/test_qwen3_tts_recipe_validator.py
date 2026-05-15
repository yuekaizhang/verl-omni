# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fail-fast tests for the Qwen3-TTS recipe validator (no GPU)."""

from __future__ import annotations

import copy

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.qwen3_tts_grpo import RecipeConfigError, validate_qwen3_tts_recipe_config


def _good_cfg() -> dict:
    return {
        "algorithm": {"adv_estimator": "grpo"},
        "actor_rollout_ref": {
            "rollout": {
                "n": 4,
                "agent": {
                    "default_agent_loop": "autoregressive_tts_single_turn_agent",
                    "agent_loop_manager_class": "verl_omni.agent_loop.autoregressive_tts_agent_loop.AutoRegressiveTTSAgentLoopManager",
                },
            },
        },
        "reward": {"reward_model": {"base_url": "http://asr:8001", "co_located": False}},
    }


def test_good_config_validates() -> None:
    validate_qwen3_tts_recipe_config(OmegaConf.create(_good_cfg()))


def test_wrong_agent_loop_rejected() -> None:
    cfg = _good_cfg()
    cfg["actor_rollout_ref"]["rollout"]["agent"]["default_agent_loop"] = "diffusion_single_turn_agent"
    with pytest.raises(RecipeConfigError, match="default_agent_loop"):
        validate_qwen3_tts_recipe_config(OmegaConf.create(cfg))


def test_wrong_manager_class_rejected() -> None:
    cfg = _good_cfg()
    cfg["actor_rollout_ref"]["rollout"]["agent"]["agent_loop_manager_class"] = "something.else.Manager"
    with pytest.raises(RecipeConfigError, match="agent_loop_manager_class"):
        validate_qwen3_tts_recipe_config(OmegaConf.create(cfg))


def test_flow_grpo_rejected() -> None:
    cfg = _good_cfg()
    cfg["algorithm"]["adv_estimator"] = "flow_grpo"
    with pytest.raises(RecipeConfigError, match="adv_estimator"):
        validate_qwen3_tts_recipe_config(OmegaConf.create(cfg))


def test_n_below_two_rejected() -> None:
    cfg = _good_cfg()
    cfg["actor_rollout_ref"]["rollout"]["n"] = 1
    with pytest.raises(RecipeConfigError, match="rollout.n"):
        validate_qwen3_tts_recipe_config(OmegaConf.create(cfg))


def test_missing_asr_endpoint_rejected() -> None:
    cfg = _good_cfg()
    cfg["reward"]["reward_model"]["base_url"] = ""
    with pytest.raises(RecipeConfigError, match="base_url"):
        validate_qwen3_tts_recipe_config(OmegaConf.create(cfg))


def test_unset_asr_endpoint_rejected() -> None:
    # OmegaConf interprets "???" as missing.
    cfg = _good_cfg()
    cfg["reward"]["reward_model"]["base_url"] = "???"
    with pytest.raises(RecipeConfigError, match="base_url"):
        validate_qwen3_tts_recipe_config(OmegaConf.create(cfg))


def test_co_located_asr_rejected() -> None:
    cfg = _good_cfg()
    cfg["reward"]["reward_model"]["co_located"] = True
    with pytest.raises(RecipeConfigError, match="co_located"):
        validate_qwen3_tts_recipe_config(OmegaConf.create(cfg))


def test_works_on_plain_dict() -> None:
    # Validator should work on plain dicts too, not just DictConfig.
    cfg = _good_cfg()
    validate_qwen3_tts_recipe_config(cfg)
