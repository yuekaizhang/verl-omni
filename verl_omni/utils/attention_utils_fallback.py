"""Pure-PyTorch fallback for `verl.utils.attention_utils` on hosts that
do not have `flash_attn` installed.

This module reproduces the side-effect previously performed by the
top-level `qwen3_tts_autoregister.py` (now deleted): swap verl's
flash-attn-only attention helpers for the transformers-provided
pure-PyTorch implementations so the FSDP log-prob recompute path
(`left_right_2_no_padding` -> `unpad_input`) works on flash-attn-free
hosts.

Importing this module performs the patch as a side-effect, so import it
once from `verl_omni/utils/__init__.py`. The patch is idempotent.
"""

from __future__ import annotations

_PATCH_APPLIED = False


def apply_flash_attn_fallback() -> None:
    """Swap verl's flash-attn attention helpers for transformers'
    pure-PyTorch fallbacks. Idempotent."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    try:
        from verl.utils import attention_utils as _verl_attn
        from transformers.modeling_flash_attention_utils import (
            _index_first_axis as _tf_index_first_axis,
            _pad_input as _tf_pad_input,
            _unpad_input as _tf_unpad_input,
        )
    except ImportError:
        # verl or transformers not present in this environment; nothing
        # to patch. A later `import verl.utils.attention_utils` will
        # raise on its own.
        return

    try:
        from einops import rearrange as _einops_rearrange
    except ImportError:
        _einops_rearrange = None

    def _patched_get_attention_functions():
        return (
            _tf_index_first_axis,
            _tf_pad_input,
            _einops_rearrange,
            _tf_unpad_input,
        )

    _verl_attn._get_attention_functions = _patched_get_attention_functions
    _verl_attn._index_first_axis = _tf_index_first_axis
    _verl_attn._pad_input = _tf_pad_input
    _verl_attn._rearrange = _einops_rearrange
    _verl_attn._unpad_input = _tf_unpad_input
    _PATCH_APPLIED = True


# Apply on import.
apply_flash_attn_fallback()
