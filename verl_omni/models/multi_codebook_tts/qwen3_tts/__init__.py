"""Qwen3-TTS adapter: vendored model code + HF auto-registration.

Importing this module has the side-effect of registering the vendored
`Qwen3TTSConfig` / `Qwen3TTSForConditionalGeneration` with the
transformers Auto* classes so `AutoConfig.from_pretrained(path)` and
`AutoModelForCausalLM.from_pretrained(path)` resolve to them when the
checkpoint declares `"model_type": "qwen3_tts"`.

This replaces the seven runtime patches that used to live in the
top-level `qwen3_tts_autoregister.py` (now deleted):
1. `AutoConfig.register("qwen3_tts", ...)` - here.
2. `AutoModel.register(...)` - here.
3. `AutoModelForCausalLM.register(...)` - here.
4. `_ROLLOUT_REGISTRY[("vllm_omni_tts","async")] = ...` - lives in
   `verl_omni/pipelines/multi_codebook_tts_grpo/__init__.py`.
5. `verl.utils.attention_utils._get_attention_functions` flash-attn
   fallback - lives in `verl_omni/utils/attention_utils_fallback.py`.
6. `Qwen3TTSConfig` talker-field promotion + `text_config` alias - now
   inside the vendored `configuration_qwen3_tts.py:Qwen3TTSConfig.__init__`.
7. `Qwen3TTSForConditionalGeneration.forward` training shim - will be
   replaced by a real `forward_training` method on the vendored model
   class (task2; see PROVENANCE.md).
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
AutoModelForCausalLM.register(
    Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True
)

# Import the adapter for its `@register_adapter("qwen3_tts")` decorator
# side-effect: populates `MULTI_CODEBOOK_ADAPTER_REGISTRY["qwen3_tts"]`
# so the recipe trainer can dispatch by `model.name`.
from .adapter import Qwen3TTSAdapter  # noqa: E402, F401

__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSAdapter",
]
