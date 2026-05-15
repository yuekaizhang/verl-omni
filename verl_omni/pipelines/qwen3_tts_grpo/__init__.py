"""Qwen3-TTS GRPO recipe (RL post-training of Qwen3-TTS via vllm-omni rollout + remote Qwen3-ASR reward)."""

from pathlib import Path

STAGE_CONFIG_PATH = Path(__file__).parent / "stage_configs" / "qwen3_tts.yaml"

# Lazy-import rollout adapter symbols so circular-import risk with
# verl_omni.workers.rollout.* stays contained when third parties import the
# package for its assets.
from .vllm_omni_rollout_adapter import (  # noqa: E402
    ALGORITHM,
    ARCHITECTURE,
    ROLLOUT_REPLICA_NAME,
    get_stage_config_path,
)

__all__ = [
    "STAGE_CONFIG_PATH",
    "ROLLOUT_REPLICA_NAME",
    "ARCHITECTURE",
    "ALGORITHM",
    "get_stage_config_path",
]
