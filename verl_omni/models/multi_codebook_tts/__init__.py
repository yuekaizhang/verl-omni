"""Multi-codebook codec-TTS model adapters.

This package hosts the codec-agnostic `MultiCodebookTTSModel` ABC and the
per-codec adapter directories (currently `qwen3_tts/`; future
`fish_speech/` is deferred to v2 per DEC-2). The trainer + actor talk to
adapters through the ABC by dispatching on `model.name` in the trainer
config.

Importing a specific adapter subpackage (e.g.
`verl_omni.models.multi_codebook_tts.qwen3_tts`) also registers its
backbone with the transformers Auto* classes as a module-level
side-effect; see that subpackage's `__init__.py` for details.
"""

from .base import (
    MultiCodebookForwardOutput,
    MultiCodebookTTSModel,
    MULTI_CODEBOOK_ADAPTER_REGISTRY,
    get_adapter,
    register_adapter,
)

__all__ = [
    "MultiCodebookForwardOutput",
    "MultiCodebookTTSModel",
    "MULTI_CODEBOOK_ADAPTER_REGISTRY",
    "get_adapter",
    "register_adapter",
]
