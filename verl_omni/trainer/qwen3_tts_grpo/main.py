# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hydra entry point for the Qwen3-TTS GRPO recipe.

Wraps upstream :func:`verl.trainer.main_ppo.run_ppo` with a recipe-side
fail-fast validator. Launch via::

    .venv/bin/python -m verl_omni.trainer.qwen3_tts_grpo.main \\
        --config-path=verl_omni/trainer/config/qwen3_tts \\
        --config-name=qwen3_tts_trainer \\
        data.train_files=... \\
        data.val_files=... \\
        actor_rollout_ref.model.path=Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        reward.reward_model.base_url=http://<asr-host>:<port>
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from verl_omni.utils.validation_audio_logger import (
    ArtifactWriteError,
    post_run_check_emitted_artifacts,
)

from .launcher import validate_qwen3_tts_recipe_config

logger = logging.getLogger(__name__)


def _resolve_validation_dir(config: DictConfig) -> Path | None:
    """Derive the per-step validation artifact dir the worker writes to.

    Mirrors the resolution in :class:`AutoRegressiveTTSAgentLoopWorker.__init__`
    so the post-run AC-8 check inspects the same location the worker used.
    """

    trainer = getattr(config, "trainer", None)
    if trainer is None:
        return None
    explicit = getattr(trainer, "validation_data_dir", None)
    if explicit:
        return Path(str(explicit))
    default_local = getattr(trainer, "default_local_dir", None)
    if not default_local:
        return None
    return Path(str(default_local)) / "validation_audio"


@hydra.main(
    config_path="../config",
    config_name="qwen3_tts/qwen3_tts_trainer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Recipe entry point.

    Resolves config from ``verl_omni/trainer/config`` so the ``defaults:``
    list in ``qwen3_tts/qwen3_tts_trainer.yaml`` can reuse the existing
    diffusion sub-configs (``diffusion/actor/...``, ``diffusion/rollout/...``,
    etc.) — the AR-TTS recipe inherits those and overrides only the fields
    that need to change.

    The canonical recipe layout (per AC-9) is
    ``verl_omni/trainer/config/qwen3_tts/qwen3_tts_trainer.yaml``.
    """

    validate_qwen3_tts_recipe_config(config)

    # Plumb a Ray worker setup hook that registers Qwen3-TTS with HF
    # AutoConfig/AutoModel inside each worker subprocess. Without this,
    # TaskRunner and FSDP WorkerDict actors fail with
    # ``KeyError: 'qwen3_tts'`` / ``Unrecognized configuration class``
    # because qwen-tts is not auto-registered on import. Doing it via
    # runtime_env (rather than a venv-wide ``.pth``) confines the heavy
    # qwen-tts import to actual worker processes — Ray's DashboardAgent,
    # RuntimeEnvAgent, raylet, etc. do not pay the cost, which would
    # otherwise stall ``ray.init`` connect for minutes.
    #
    # The hook resolves ``qwen3_tts_autoregister.setup`` via importlib,
    # so the module must be on the worker's PYTHONPATH. The driver
    # launches with ``python -m verl_omni.trainer.qwen3_tts_grpo.main``
    # from the project root, which puts the repo root on sys.path —
    # but Ray's default_worker.py is spawned by raylet without that
    # ``-m`` context, so we inject the repo root into the worker's
    # PYTHONPATH explicitly via ``runtime_env.env_vars``.
    OmegaConf.set_struct(config, False)
    repo_root = str(Path(__file__).resolve().parents[3])
    runtime_env = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env")
    if runtime_env is None:
        config.ray_kwargs.ray_init.runtime_env = OmegaConf.create(
            {
                "worker_process_setup_hook": "qwen3_tts_autoregister.setup",
                "env_vars": {"PYTHONPATH": repo_root},
            }
        )
    else:
        runtime_env["worker_process_setup_hook"] = "qwen3_tts_autoregister.setup"
        env_vars = runtime_env.get("env_vars") or {}
        existing_pp = env_vars.get("PYTHONPATH", "")
        env_vars["PYTHONPATH"] = (
            f"{repo_root}:{existing_pp}" if existing_pp else repo_root
        )
        runtime_env["env_vars"] = env_vars
    OmegaConf.set_struct(config, True)

    from verl.trainer.main_ppo import run_ppo

    try:
        run_ppo(config)
    finally:
        # AC-8 post-run fail-closed check: if any validation step ran, make sure
        # at least one generated audio artifact landed. A run that emitted zero
        # artifacts is flagged here rather than silently passing.
        validation_dir = _resolve_validation_dir(config)
        if validation_dir is not None and validation_dir.exists():
            try:
                steps = post_run_check_emitted_artifacts(validation_dir)
                logger.info(
                    "Post-run AC-8 check: %d validation step(s) emitted audio artifacts at %s",
                    len(steps),
                    validation_dir,
                )
            except ArtifactWriteError:
                logger.error("Post-run AC-8 check failed for %s", validation_dir)
                raise


if __name__ == "__main__":
    main()
