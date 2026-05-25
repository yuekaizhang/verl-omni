"""Fail-fast config validator for the `multi_codebook_tts_grpo` recipe.

Runs in the driver process BEFORE `ray.init`, so misconfiguration surfaces
as a clear `ValueError` instead of an opaque worker traceback. Checks the
three invariants required by the plan:

1. `model.name` resolves to a registered `MultiCodebookTTSModel` adapter.
2. `w_cb0 + w_cb_rest > 0` (so the weighted-sum loss isn't degenerate).
3. `rollout.diagnostic_logprobs` is a boolean (per-AC-3 gating).

Optionally validates that `cb0` and `cb_rest` sub-configs reference the
same `model.name` to catch copy-paste errors in the actor YAML.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def _read(config: DictConfig, dotted_path: str, default=None):
    """Read `config.<dotted_path>` defensively, returning `default` when any
    segment is missing."""
    val = OmegaConf.select(config, dotted_path)
    return val if val is not None else default


def validate_multi_codebook_tts_recipe_config(config: DictConfig) -> None:
    """Fail-fast validator. Raises `ValueError` with a helpful message on
    any inconsistency."""
    # Defer the registry import: it triggers HF Auto* registration as a
    # side-effect, which we want to fire here so the driver process knows
    # `model.name == "qwen3_tts"` (or future codecs) is loadable.
    from verl_omni.models.multi_codebook_tts import (
        MULTI_CODEBOOK_ADAPTER_REGISTRY,
    )

    # 1. codec_adapter must be registered.
    # The dispatch key moved off `model.name` (which collides with
    # upstream's strict `HFModelConfig`) onto `actor.codec_adapter`
    # (which lives on `MultiCodebookFSDPActorConfig`, our subclass).
    model_name = _read(config, "actor_rollout_ref.actor.codec_adapter", default=None)
    if model_name is None:
        # Back-compat fall-backs (older configs that may still set
        # `model.name` or `model.codec_adapter`).
        model_name = (
            _read(config, "actor_rollout_ref.model.codec_adapter", default=None)
            or _read(config, "actor_rollout_ref.model.name", default=None)
            or _read(config, "model.name", default=None)
        )
    if model_name is None:
        raise ValueError(
            "multi_codebook_tts_grpo: `actor.codec_adapter` is required "
            "but not set in the trainer config. Choose one of the "
            f"registered adapters: {sorted(MULTI_CODEBOOK_ADAPTER_REGISTRY.keys())}."
        )
    # Trigger lazy registration of the named adapter so the check is honest.
    try:
        import importlib
        importlib.import_module(
            f"verl_omni.models.multi_codebook_tts.{model_name}"
        )
    except ImportError as exc:
        raise ValueError(
            f"multi_codebook_tts_grpo: model.name={model_name!r} but no "
            f"such adapter subpackage. Underlying import error: {exc}."
        ) from exc
    if model_name not in MULTI_CODEBOOK_ADAPTER_REGISTRY:
        raise ValueError(
            f"multi_codebook_tts_grpo: model.name={model_name!r} is not in "
            f"the multi-codebook adapter registry "
            f"({sorted(MULTI_CODEBOOK_ADAPTER_REGISTRY.keys())}). The adapter "
            f"module imported but did not register itself; check "
            f"`@register_adapter(...)` on the adapter class."
        )

    # 2. w_cb0 + w_cb_rest > 0.
    w_cb0 = _read(config, "actor_rollout_ref.actor.w_cb0", default=None)
    w_cb_rest = _read(config, "actor_rollout_ref.actor.w_cb_rest", default=None)
    if w_cb0 is None or w_cb_rest is None:
        raise ValueError(
            "multi_codebook_tts_grpo: both "
            "`actor_rollout_ref.actor.w_cb0` and "
            "`actor_rollout_ref.actor.w_cb_rest` are required (the Fish-S2 "
            f"weighted-sum coefficients). Got w_cb0={w_cb0}, "
            f"w_cb_rest={w_cb_rest}."
        )
    if float(w_cb0) + float(w_cb_rest) <= 0:
        raise ValueError(
            "multi_codebook_tts_grpo: w_cb0 + w_cb_rest must be > 0; got "
            f"{w_cb0=}, {w_cb_rest=}. A non-positive sum would make the "
            "weighted-sum loss degenerate."
        )

    # 3. actor.diagnostic_logprobs is a boolean. (Lives on the actor
    # sub-config — see qwen3_tts_trainer.yaml — to avoid touching the
    # upstream `RolloutConfig` schema.)
    diag = _read(
        config,
        "actor_rollout_ref.actor.diagnostic_logprobs",
        default=_read(config, "actor_rollout_ref.rollout.diagnostic_logprobs", default=False),
    )
    if not isinstance(diag, bool):
        raise ValueError(
            "multi_codebook_tts_grpo: "
            "`actor_rollout_ref.rollout.diagnostic_logprobs` must be a "
            f"bool; got {type(diag).__name__}={diag!r}."
        )

    # Optional: cb0 / cb_rest sub-configs exist and have a `policy_loss.loss_mode`.
    for stream in ("cb0", "cb_rest"):
        loss_mode = _read(
            config,
            f"actor_rollout_ref.actor.{stream}.policy_loss.loss_mode",
            default=None,
        )
        if loss_mode is None:
            raise ValueError(
                f"multi_codebook_tts_grpo: missing "
                f"`actor_rollout_ref.actor.{stream}.policy_loss.loss_mode` "
                f"in the trainer config (e.g. 'vanilla')."
            )


__all__ = ["validate_multi_codebook_tts_recipe_config"]
