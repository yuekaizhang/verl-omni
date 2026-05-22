"""Lightweight Ray-worker setup hook for the multi_codebook_tts_grpo recipe.

This module is at the project root (NOT inside ``verl_omni/``) so Ray's
``runtime_env.worker_process_setup_hook = "multi_codebook_tts_setup:setup"``
can invoke it without dragging in the heavy ``verl_omni/__init__.py``
import chain (which imports pipelines, diffusion adapters, rollout
backends, etc. and would stall worker spawn).

Replaces the old top-level ``qwen3_tts_autoregister.py`` (deleted in
task15). All seven monkey-patch points it covered are now folded into
clean module-level code:

1-3. HF ``AutoConfig`` / ``AutoModel`` / ``AutoModelForCausalLM``
   registration -> module-import side-effect of
   ``verl_omni.models.multi_codebook_tts.qwen3_tts``.
4. ``_ROLLOUT_REGISTRY[("vllm_omni_tts","async")] = ...`` -> module-import
   side-effect of ``verl_omni.pipelines.multi_codebook_tts_grpo``.
5. ``verl.utils.attention_utils._get_attention_functions`` flash-attn
   fallback -> ``verl_omni.utils.attention_utils_fallback`` (applied at
   import time).
6. ``Qwen3TTSConfig`` talker-field promotion + ``text_config`` alias ->
   inside vendored ``configuration_qwen3_tts.py::Qwen3TTSConfig.__init__``.
7. ``Qwen3TTSForConditionalGeneration.forward`` training shim -> replaced
   by a real ``forward_training`` method on the vendored class
   (pending task2; see PROVENANCE.md).

Each registration runs once per worker subprocess (the underlying calls
are idempotent via ``exist_ok=True`` / module-level guards).
"""

import os
import sys

_REGISTERED = False


def setup() -> None:
    """Idempotent worker registration. Called once per Ray worker subprocess
    via ``runtime_env.worker_process_setup_hook``."""
    global _REGISTERED
    if _REGISTERED:
        return

    # Make sure the repo root is on sys.path. Ray's default worker spawn
    # does NOT inherit the launching shell's PYTHONPATH unless explicitly
    # set in runtime_env.env_vars, but defending here is cheap.
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    # Optional fork PYTHONPATH (vllm-omni-verl, etc.). The recipe's
    # `verl_omni.utils.ray_runtime_env.build_runtime_env` already appends
    # the fork dir to env_vars["PYTHONPATH"] when VLLM_OMNI_VERL_DIR is
    # set in the driver's env; this is a belt-and-suspenders fallback.
    fork = os.environ.get("VLLM_OMNI_VERL_DIR")
    if fork and fork not in sys.path:
        sys.path.insert(0, fork)

    # Trigger the four side-effecting imports. Each is wrapped in
    # try/except so a missing optional dep (e.g., diffusers required by
    # verl_omni.pipelines._patch on the diffusion side) doesn't break
    # TTS-only workers.
    try:
        # HF Auto* registration for Qwen3-TTS. Imports the vendored
        # configuration + modeling and calls the three .register() lines
        # at module-import time. See
        # verl_omni/models/multi_codebook_tts/qwen3_tts/__init__.py.
        import verl_omni.models.multi_codebook_tts.qwen3_tts  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic
        sys.stderr.write(
            f"[multi_codebook_tts_setup] Qwen3-TTS Auto* registration "
            f"failed: {type(exc).__name__}: {exc}\n"
        )

    try:
        # _ROLLOUT_REGISTRY entry for the vllm_omni_tts async server.
        import verl_omni.pipelines.multi_codebook_tts_grpo  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic
        sys.stderr.write(
            f"[multi_codebook_tts_setup] rollout-registry wiring "
            f"failed: {type(exc).__name__}: {exc}\n"
        )

    try:
        # flash-attn fallback. The import has the side-effect of patching
        # verl.utils.attention_utils if flash_attn is missing.
        from verl_omni.utils import attention_utils_fallback  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic
        sys.stderr.write(
            f"[multi_codebook_tts_setup] attention-utils fallback "
            f"installation failed: {type(exc).__name__}: {exc}\n"
        )

    _REGISTERED = True
