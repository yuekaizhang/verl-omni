"""Qwen3-TTS GRPO trainer entry point (wraps upstream `verl.trainer.main_ppo`)."""

from .launcher import RecipeConfigError, validate_qwen3_tts_recipe_config

__all__ = ["RecipeConfigError", "validate_qwen3_tts_recipe_config"]
