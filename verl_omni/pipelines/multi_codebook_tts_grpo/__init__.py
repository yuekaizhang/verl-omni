"""Multi-codebook TTS GRPO pipeline (generic recipe; dispatches by
`model.name` to the active codec adapter).

Importing this module has a side-effect of registering the vllm-omni TTS
async server with verl's `_ROLLOUT_REGISTRY` so workers can resolve
`rollout.name=vllm_omni_tts` without depending on the (deleted)
`qwen3_tts_autoregister.py` top-level shim.

The recipe-side rollout adapter (`vllm_omni_rollout_adapter.py`, written
by task10) reads the `rollout.diagnostic_logprobs` config flag and
gracefully handles the case where the vllm-omni-verl fork edit has not
landed (per DEC-5, deferred): it logs a `logger.warning` and proceeds
without diagnostic logprob fields.
"""

# _ROLLOUT_REGISTRY wiring (side-effect-only). Wrapped in try/except so
# importing this package in a unit-test environment without verl present
# is not fatal.
try:
    from verl.workers.rollout.base import _ROLLOUT_REGISTRY
except ImportError:  # pragma: no cover - test-only path
    _ROLLOUT_REGISTRY = None

if _ROLLOUT_REGISTRY is not None:
    _ROLLOUT_REGISTRY[("vllm_omni_tts", "async")] = (
        "verl.workers.rollout.vllm_rollout.ServerAdapter"
    )

__all__: list[str] = []
