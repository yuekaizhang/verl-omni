"""Hydra entry point for the multi_codebook_tts_grpo recipe.

Generic by design: dispatches to the right `MultiCodebookTTSModel` adapter
via `model.name` so a future fish-speech recipe (deferred per DEC-2) plugs
in with a sibling adapter directory + a config switch only.

Launch via::

    .venv/bin/python -m verl_omni.trainer.multi_codebook_tts_grpo.main \\
        --config-path=../config/multi_codebook_tts \\
        --config-name=qwen3_tts_trainer \\
        actor_rollout_ref.model.path=Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        data.train_files=... \\
        data.val_files=... \\
        reward.reward_model.base_url=http://<asr-host>:<port>
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from verl_omni.utils.ray_runtime_env import build_runtime_env

from .launcher import validate_multi_codebook_tts_recipe_config

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../config",
    config_name="multi_codebook_tts/qwen3_tts_trainer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Recipe entry point.

    Fails fast in the driver process via
    `validate_multi_codebook_tts_recipe_config`, then plumbs the
    `worker_process_setup_hook` so Ray workers register Qwen3-TTS with HF
    Auto* before importing the model. Training execution itself is delegated
    to upstream `verl.trainer.main_ppo.run_ppo` (the multi-codebook actor
    wiring lives in a follow-up; see `verl_omni/workers/actor/multi_codebook_dp_actor.py`).
    """
    validate_multi_codebook_tts_recipe_config(config)

    # Centralized runtime_env builder. Sets `worker_process_setup_hook`
    # to the lightweight top-level `multi_codebook_tts_setup:setup` and
    # plumbs PYTHONPATH (repo root + optional VLLM_OMNI_VERL_DIR fork).
    runtime_env = build_runtime_env()

    OmegaConf.set_struct(config, False)
    existing = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env")
    if existing is None:
        config.ray_kwargs.ray_init.runtime_env = OmegaConf.create(runtime_env)
    else:
        existing["worker_process_setup_hook"] = runtime_env["worker_process_setup_hook"]
        env_vars = existing.get("env_vars") or {}
        existing_pp = env_vars.get("PYTHONPATH", "")
        new_pp = runtime_env["env_vars"]["PYTHONPATH"]
        env_vars["PYTHONPATH"] = f"{new_pp}:{existing_pp}" if existing_pp else new_pp
        existing["env_vars"] = env_vars
    OmegaConf.set_struct(config, True)

    # NOTE (round-1 scaffold): the upstream PPO driver does not natively
    # dispatch into `MultiCodebookDPActor`. Round-2 work wires the actor
    # into the worker; for now this entry point is callable and validates
    # the config so end-to-end smoke tests can iterate on the YAML shape.
    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


if __name__ == "__main__":
    main()
