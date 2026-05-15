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

import hydra
from omegaconf import DictConfig

from .launcher import validate_qwen3_tts_recipe_config


@hydra.main(
    config_path="../config/qwen3_tts",
    config_name="qwen3_tts_trainer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    validate_qwen3_tts_recipe_config(config)
    # Delegate to upstream PPO/GRPO trainer.
    from verl.trainer.main_ppo import run_ppo

    run_ppo(config)


if __name__ == "__main__":
    main()
