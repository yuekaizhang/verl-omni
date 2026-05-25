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
"""Rollout adapter for the `multi_codebook_tts_grpo` recipe.

Holds the path to the verl-omni-side stage_config override and exposes
the rollout-server name registered in :mod:`verl_omni.workers.rollout.replica`.
Also wires the `rollout.diagnostic_logprobs` gating flag with the
warn-on-absence behavior required by AC-3:

- When `diagnostic_logprobs: false` (the v1 default), the adapter does not
  touch the rollout payload for diagnostic fields.
- When `diagnostic_logprobs: true` BUT the vllm-omni-verl fork edit that
  emits `extra_logprobs.cb_rest` has NOT landed (the expected v1 state per
  DEC-5), the adapter logs `logger.warning` once per step and proceeds
  without placing diagnostic fields into the batch. Training is unaffected.
- When `diagnostic_logprobs: true` AND the diagnostic fields are present
  (future state after the fork edit lands), the adapter stuffs
  `vllm_logprob_cb0` and `vllm_logprob_cb_rest` into the batch for
  drift-metric computation in the actor. Gradient flow never touches
  these fields — they are diagnostic only (AC-2 / AC-4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Stage config: lives under this package (migrated from the deleted
# legacy pipelines path).
_STAGE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "stage_configs" / "qwen3_tts.yaml"
)

# Public alias for back-compat with the legacy import path
# `from verl_omni.pipelines.qwen3_tts_grpo import STAGE_CONFIG_PATH`.
STAGE_CONFIG_PATH = _STAGE_CONFIG_PATH

ROLLOUT_REPLICA_NAME = "vllm_omni_tts"
"""Name registered with ``RolloutReplicaRegistry`` for this recipe."""

ARCHITECTURE = "MultiCodebookTTSPipeline"
"""Pipeline architecture key, used by Hydra to discriminate against other recipes."""

ALGORITHM = "grpo"
"""RL algorithm key — upstream ``verl.trainer.main_ppo`` with ``adv_estimator=grpo``."""


def get_stage_config_path() -> str:
    """Return the on-disk path to the verl-omni stage_config override."""
    return str(_STAGE_CONFIG_PATH)


def apply_diagnostic_logprobs_to_batch(
    batch: dict[str, Any],
    rollout_payload: dict[str, Any] | None,
    *,
    diagnostic_logprobs_enabled: bool,
) -> dict[str, Any]:
    """Conditionally place `vllm_logprob_*` fields into the rollout batch.

    AC-3 contract:

    - `diagnostic_logprobs_enabled=False` (v1 default): no-op; the batch
      is returned unchanged.
    - `diagnostic_logprobs_enabled=True` AND `rollout_payload` contains
      both `extra_logprobs.cb0` and `extra_logprobs.cb_rest`: copy them
      into the batch as `vllm_logprob_cb0` + `vllm_logprob_cb_rest` for
      diagnostic / drift-metric use ONLY (AC-2: never read by gradients).
    - `diagnostic_logprobs_enabled=True` BUT the diagnostic fields are
      absent (expected v1 state because the vllm-omni-verl fork edit is
      deferred per DEC-5): log `logger.warning` once and proceed without
      placing diagnostic fields into the batch. Training is unaffected.

    Args:
        batch: the rollout batch dict (mutated in place AND returned).
        rollout_payload: the raw vllm-omni response payload for this
            request, or `None` if no rollout-level extras are available.
        diagnostic_logprobs_enabled: value of
            `actor_rollout_ref.rollout.diagnostic_logprobs` from the
            trainer config.

    Returns:
        The (possibly mutated) `batch` dict.
    """
    if not diagnostic_logprobs_enabled:
        return batch

    extra = (rollout_payload or {}).get("extra_logprobs") or {}
    cb0_lp = extra.get("cb0")
    cb_rest_lp = extra.get("cb_rest")
    if cb0_lp is None or cb_rest_lp is None:
        logger.warning(
            "[multi_codebook_tts_grpo] rollout.diagnostic_logprobs=True but "
            "the rollout payload does not carry both `extra_logprobs.cb0` "
            "and `extra_logprobs.cb_rest`. This is the expected v1 state "
            "(per DEC-5 the vllm-omni-verl fork edit is deferred). Drift "
            "metrics will not be emitted this step; training is unaffected."
        )
        return batch

    batch["vllm_logprob_cb0"] = cb0_lp
    batch["vllm_logprob_cb_rest"] = cb_rest_lp
    return batch


__all__ = [
    "ROLLOUT_REPLICA_NAME",
    "ARCHITECTURE",
    "ALGORITHM",
    "get_stage_config_path",
    "apply_diagnostic_logprobs_to_batch",
]
