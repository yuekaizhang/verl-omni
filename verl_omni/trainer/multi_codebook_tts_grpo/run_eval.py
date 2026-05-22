"""Evaluation entry for the multi_codebook_tts_grpo recipe.

Round-1 stub: delegates to upstream verl's eval path with the same Hydra
config. Round-2 work will replace this with a multi-codebook eval that
exercises the new `MultiCodebookDPActor.compute_log_prob` path.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .launcher import validate_multi_codebook_tts_recipe_config


@hydra.main(
    config_path="../config",
    config_name="multi_codebook_tts/qwen3_tts_trainer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    validate_multi_codebook_tts_recipe_config(config)
    # Stub: round-2 work will replace this with a real eval driver. For now
    # we just confirm the recipe config validates cleanly.
    print(
        "[multi_codebook_tts_grpo:run_eval] config validated; "
        "eval driver to be implemented in a follow-up round."
    )


if __name__ == "__main__":
    main()
