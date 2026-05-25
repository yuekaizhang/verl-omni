# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hydra-instantiable actor config for the multi_codebook_tts_grpo recipe.

Upstream `verl.workers.config.actor.FSDPActorConfig` is a `@dataclass`
with strict fields. When Hydra calls `omega_conf_to_dataclass(...)` on
our actor YAML it rejects unknown kwargs (`multi_codebook_loss`,
`w_cb0`, `w_cb_rest`, `cb0`, `cb_rest`) with `TypeError(...) got an
unexpected keyword argument`. This module defines a subclass that
accepts those extra fields so the recipe YAML can carry them through
the standard verl config pipeline.

Mirrors the pattern verl-omni already uses for its diffusion side:
`verl_omni.workers.config.diffusion.FSDPDiffusionActorConfig` extends
the upstream config with diffusion-specific fields and is referenced
via `_target_` in the diffusion actor YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verl.workers.config.actor import FSDPActorConfig

__all__ = [
    "MultiCodebookFSDPActorConfig",
    "MultiCodebookStreamConfig",
]


@dataclass
class MultiCodebookStreamConfig:
    """Per-stream actor-loss configuration block.

    Stored as a dataclass so Hydra can populate it from the YAML's
    `cb0:` / `cb_rest:` sub-blocks via `_target_` resolution. The
    runtime loss function (`multi_codebook_ppo_loss`) only reads it
    via `.get(...)` so any extra fields (e.g. future per-stream
    annealing knobs) pass through harmlessly.
    """

    # `policy_loss` carries `loss_mode` + `clip_ratio` (matches the
    # upstream `PolicyLossConfig` field surface but kept as a free-form
    # dict so additions don't break the contract).
    policy_loss: Any = field(default_factory=lambda: {"loss_mode": "vanilla", "clip_ratio": 0.2})
    loss_agg_mode: str = "token-mean"
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"


@dataclass
class MultiCodebookFSDPActorConfig(FSDPActorConfig):
    """FSDP actor config + Fish-S2 multi-codebook fields.

    Added fields:

    - `multi_codebook_loss`: bool gate. When `True`,
      `verl_omni.workers.engine_workers` dispatches `self.loss_fn` to
      `multi_codebook_ppo_loss` instead of upstream `ppo_loss`. False
      keeps the legacy single-codebook path active.
    - `w_cb0` / `w_cb_rest`: Fish-S2 weighted-sum coefficients
      (`w_cb_rest` = paper's Fast-AR `gamma`).
    - `cb0` / `cb_rest`: per-stream actor-loss config blocks. Each has
      `policy_loss.loss_mode`, `policy_loss.clip_ratio`, `loss_agg_mode`,
      `use_kl_loss`, `kl_loss_coef`, `kl_loss_type`.

    Everything inherited from `FSDPActorConfig` (strategy, fsdp_config,
    optim, ppo_mini_batch_size, etc.) keeps the same semantics.
    """

    multi_codebook_loss: bool = False
    w_cb0: float = 1.0
    w_cb_rest: float = 0.1
    cb0: Any = field(default_factory=MultiCodebookStreamConfig)
    cb_rest: Any = field(default_factory=MultiCodebookStreamConfig)
    # Codec adapter dispatch key — resolves against
    # `verl_omni.models.multi_codebook_tts.MULTI_CODEBOOK_ADAPTER_REGISTRY`.
    # The legacy `model.name` location collided with upstream's strict
    # `HFModelConfig`; keeping it on the actor sub-config side-steps that.
    codec_adapter: str = "qwen3_tts"
    # AC-3 gating flag. Lives on the actor sub-config (not rollout) so we
    # don't have to subclass + override the upstream `RolloutConfig` (which
    # is itself a strict dataclass + carries fields like `disaggregation`
    # whose struct schema would break if we replaced its `_target_`).
    diagnostic_logprobs: bool = False

    def __post_init__(self):
        super().__post_init__()
        # `multi_codebook_loss=True` requires `w_cb0 + w_cb_rest > 0`
        # else the weighted-sum loss is degenerate.
        if self.multi_codebook_loss and float(self.w_cb0) + float(self.w_cb_rest) <= 0.0:
            raise ValueError(
                f"`actor.multi_codebook_loss=True` but w_cb0 + w_cb_rest = "
                f"{self.w_cb0} + {self.w_cb_rest} <= 0. The weighted-sum "
                f"loss would be degenerate."
            )
