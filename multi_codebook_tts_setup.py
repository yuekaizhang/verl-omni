"""Lightweight Ray-worker setup hook for the multi_codebook_tts_grpo recipe.

This module is at the project root (NOT inside ``verl_omni/``) so Ray's
``runtime_env.worker_process_setup_hook = "multi_codebook_tts_setup:setup"``
can invoke it from worker subprocesses. Replaces the old top-level
``qwen3_tts_autoregister.py`` (deleted in task15). All seven monkey-patch
points it covered are now in clean module-level code (see PROVENANCE.md
under the qwen3_tts adapter subpackage and the docstrings of the
relevant modules).

Fail-closed contract: Qwen3-TTS Auto* registration is the only REQUIRED
side-effect — the recipe cannot load the codec model without it. If the
required registration fails, ``setup()`` re-raises so workers crash
loudly with the actual import error rather than silently continuing to
later "Unrecognized model identifier" errors. The other side-effects
(pipelines rollout registry entry, attention-utils fallback) are
best-effort: their absence degrades specific features (e.g. flash-attn
fallback is only needed on hosts without flash_attn installed) but does
not prevent training.
"""

import os
import sys

_REGISTERED = False


def _pin_cuda_visible_devices_for_rank() -> None:
    """Pin this worker process to a single physical GPU based on its
    Ray-assigned ``RANK``.

    Why this is needed: verl's `_create_worker` allocates
    `num_gpus = 1 / max_colocate_count` (fractional, e.g. ~0.333 when
    actor+rollout+ref are colocated). For fractional GPU allocations
    Ray does NOT slice ``CUDA_VISIBLE_DEVICES`` per worker — every
    worker sees the full list. Workers default to physical device 0
    when they call ``torch.cuda.set_device(0)``, so FSDP rank 0 and
    rank 1 collide on the same GPU and NCCL raises "Duplicate GPU
    detected" during ``_sync_module_params_and_buffers``.

    Fix (verl-omni-side, no upstream verl change): in the worker
    setup hook, read ``RANK`` and ``RAY_LOCAL_WORLD_SIZE`` that verl
    already set on the worker, and narrow ``CUDA_VISIBLE_DEVICES`` to
    just this worker's assigned device BEFORE any torch import. The
    subsequent HF Auto* registration imports torch with the correct
    device visible, and downstream ``torch.cuda.set_device(0)``
    resolves to the right physical GPU.
    """
    rank_str = os.environ.get("RANK")
    if rank_str is None:
        # Not a Ray worker (e.g. driver process, unit-test invocation).
        return
    try:
        rank = int(rank_str)
    except ValueError:
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return  # Ray already sliced, or nothing to pin.
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    if not devices:
        return

    # Map global rank → local device. Each placement-group group (actor,
    # rollout, ref under colocation) sees `RANK` cycle within its own
    # world, but `RANK` is set per group by verl so `rank % len(devices)`
    # gives a stable device for each rank.
    local_idx = rank % len(devices)
    chosen = devices[local_idx]
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    # Diagnostic — emit once per worker so the smoke log shows the
    # pinning result.
    sys.stderr.write(
        f"[multi_codebook_tts_setup] pinning RANK={rank} -> "
        f"CUDA_VISIBLE_DEVICES={chosen} (was {visible!r}, "
        f"{len(devices)} visible)\n"
    )
    sys.stderr.flush()


def setup() -> None:
    """Idempotent worker registration. Called once per Ray worker subprocess
    via ``runtime_env.worker_process_setup_hook``.

    Raises:
        ImportError (or whatever the underlying import raised) when the
        REQUIRED Qwen3-TTS Auto* registration cannot run. Optional
        side-effects (rollout-registry, attention-utils fallback) failing
        does NOT raise — they are logged to stderr.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # MUST happen before any torch-importing chain runs (HF Auto*
    # registration below triggers torch via the model class). See the
    # docstring on `_pin_cuda_visible_devices_for_rank` for the
    # Ray/fractional-GPU rationale.
    _pin_cuda_visible_devices_for_rank()

    # Make sure the repo root is on sys.path. Ray's default worker spawn
    # does NOT inherit the launching shell's PYTHONPATH unless explicitly
    # set in runtime_env.env_vars; defending here is cheap.
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    # Optional fork PYTHONPATH (vllm-omni-verl, etc.). The recipe's
    # `verl_omni.utils.ray_runtime_env.build_runtime_env` already
    # appends the fork dir to env_vars["PYTHONPATH"] when
    # VLLM_OMNI_VERL_DIR is set in the driver's env; this is a
    # belt-and-suspenders fallback.
    fork = os.environ.get("VLLM_OMNI_VERL_DIR")
    if fork and fork not in sys.path:
        sys.path.insert(0, fork)

    # REQUIRED: HF Auto* registration for Qwen3-TTS. Imports the
    # vendored configuration + modeling and calls the three .register()
    # lines at module-import time. Re-raise on failure so workers fail
    # loudly rather than crashing later with a misleading
    # "Unrecognized model identifier" error.
    import verl_omni.models.multi_codebook_tts.qwen3_tts  # noqa: F401

    # Optional: _ROLLOUT_REGISTRY entry for the vllm_omni_tts async
    # server. Failure does not block training - it would only block
    # workers that need to resolve the rollout class by name, which
    # the driver also does. Log to stderr so it shows up in Ray logs.
    try:
        import verl_omni.pipelines.multi_codebook_tts_grpo  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic
        sys.stderr.write(
            f"[multi_codebook_tts_setup] rollout-registry wiring "
            f"failed: {type(exc).__name__}: {exc}\n"
        )

    # Optional: flash-attn fallback. Only relevant on hosts without
    # flash_attn installed. The import has the side-effect of patching
    # verl.utils.attention_utils if needed.
    try:
        from verl_omni.utils import attention_utils_fallback  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic
        sys.stderr.write(
            f"[multi_codebook_tts_setup] attention-utils fallback "
            f"installation failed: {type(exc).__name__}: {exc}\n"
        )

    # Mark registered ONLY after the required import succeeded.
    _REGISTERED = True
