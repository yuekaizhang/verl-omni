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
"""Rollout adapter binding the Qwen3-TTS GRPO recipe to the vllm-omni TTS server.

The adapter holds the path to the verl-omni-side stage_config override
(``stage_configs/qwen3_tts.yaml``) and exposes the rollout-server name
registered in :mod:`verl_omni.workers.rollout.replica` so the trainer's Hydra
config can resolve the right server by name.
"""

from __future__ import annotations

from . import STAGE_CONFIG_PATH

ROLLOUT_REPLICA_NAME = "vllm_omni_tts"
"""Name registered with ``RolloutReplicaRegistry`` for this recipe."""

ARCHITECTURE = "Qwen3TTSPipeline"
"""Pipeline architecture key, used by Hydra to discriminate against other recipes."""

ALGORITHM = "grpo"
"""RL algorithm key — upstream ``verl.trainer.main_ppo`` with ``adv_estimator=grpo``."""


def get_stage_config_path() -> str:
    """Return the on-disk path to the verl-omni stage_config override.

    The override adds ``final_output: true`` and ``logprobs: 1`` on stage 0
    so per-token codec logprobs reach ``OmniRequestOutput`` (required by
    GRPO). See ``stage_configs/qwen3_tts.yaml`` in this package.
    """

    return str(STAGE_CONFIG_PATH)


__all__ = [
    "ROLLOUT_REPLICA_NAME",
    "ARCHITECTURE",
    "ALGORITHM",
    "get_stage_config_path",
]
