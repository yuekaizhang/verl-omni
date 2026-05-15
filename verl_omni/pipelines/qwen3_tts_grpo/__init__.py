"""Qwen3-TTS GRPO recipe (RL post-training of Qwen3-TTS via vllm-omni rollout + remote Qwen3-ASR reward)."""

from pathlib import Path

STAGE_CONFIG_PATH = Path(__file__).parent / "stage_configs" / "qwen3_tts.yaml"

__all__ = ["STAGE_CONFIG_PATH"]
