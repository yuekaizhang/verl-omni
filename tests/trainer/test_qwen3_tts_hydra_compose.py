# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hydra composition tests for the Qwen3-TTS recipe (no GPU).

Codex round-2 review correctly flagged that the previous validator tests
exercised a synthetic hand-built dict and missed real composition bugs.
These tests compose the shipped ``qwen3_tts_trainer.yaml`` through Hydra,
apply the launch-time overrides, and assert:

1. The composed config has the expected top-level shape (no
   ``actor.actor_rollout_ref`` double-nesting from misplaced
   ``# @package _global_`` directives).
2. The recipe validator accepts the composed config when required overrides
   are set.
3. The recipe validator rejects launches that are missing required overrides.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from verl_omni.trainer.qwen3_tts_grpo import RecipeConfigError, validate_qwen3_tts_recipe_config

CONFIG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "verl_omni"
    / "trainer"
    / "config"
)


def _compose(extra_overrides: list[str] | None = None) -> DictConfig:
    overrides = [
        "actor_rollout_ref.model.path=/tmp/qwen3-tts-stub",
        "data.train_files=/tmp/train.parquet",
        "data.val_files=/tmp/eval.parquet",
        "reward.reward_model.base_url=http://asr:8001",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="qwen3_tts/qwen3_tts_trainer", overrides=overrides)


def test_compose_succeeds_and_top_level_keys_are_correct() -> None:
    cfg = _compose()
    # Sanity: top-level groups exist where the trainer expects them, not
    # double-nested under their group names.
    assert "actor_rollout_ref" in cfg
    assert "rollout" in cfg.actor_rollout_ref
    assert "actor" in cfg.actor_rollout_ref
    assert "model" in cfg.actor_rollout_ref
    assert "ref" in cfg.actor_rollout_ref
    assert "reward" in cfg
    assert "algorithm" in cfg
    assert "trainer" in cfg
    assert "data" in cfg
    assert "ray_kwargs" in cfg


def test_compose_keeps_ar_tts_overrides() -> None:
    cfg = _compose()
    assert cfg.actor_rollout_ref.rollout.name == "vllm_omni_tts"
    assert cfg.actor_rollout_ref.rollout.agent.default_agent_loop == "autoregressive_tts_single_turn_agent"
    assert cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class == (
        "verl_omni.agent_loop.autoregressive_tts_agent_loop.AutoRegressiveTTSAgentLoopManager"
    )
    assert cfg.actor_rollout_ref.rollout.n >= 2
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.reward.reward_model.base_url == "http://asr:8001"
    assert cfg.reward.reward_model.co_located is False
    # The TTS recipe overrides DiffusionAlgoConfig with upstream AlgoConfig.
    assert cfg.algorithm._target_ == "verl.trainer.config.AlgoConfig"


def test_composed_config_passes_recipe_validator() -> None:
    cfg = _compose()
    validate_qwen3_tts_recipe_config(cfg)


def test_composed_config_rejects_missing_asr_url() -> None:
    overrides = [
        "actor_rollout_ref.model.path=/tmp/qwen3-tts-stub",
        "data.train_files=/tmp/train.parquet",
        "data.val_files=/tmp/eval.parquet",
        # Intentionally NOT setting reward.reward_model.base_url; the
        # Hydra default `???` should remain and be rejected.
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        with pytest.raises(Exception):  # MissingMandatoryValue from OmegaConf at access time
            cfg = compose(config_name="qwen3_tts/qwen3_tts_trainer", overrides=overrides)
            validate_qwen3_tts_recipe_config(cfg)


def test_composed_config_rejects_flow_grpo() -> None:
    cfg = _compose(extra_overrides=["algorithm.adv_estimator=flow_grpo"])
    with pytest.raises(RecipeConfigError, match="adv_estimator"):
        validate_qwen3_tts_recipe_config(cfg)


def test_composed_config_rejects_co_located_asr() -> None:
    cfg = _compose(extra_overrides=["reward.reward_model.co_located=true"])
    with pytest.raises(RecipeConfigError, match="co_located"):
        validate_qwen3_tts_recipe_config(cfg)


def test_composed_config_rejects_n_below_two() -> None:
    cfg = _compose(extra_overrides=["actor_rollout_ref.rollout.n=1"])
    with pytest.raises(RecipeConfigError, match="rollout.n"):
        validate_qwen3_tts_recipe_config(cfg)
