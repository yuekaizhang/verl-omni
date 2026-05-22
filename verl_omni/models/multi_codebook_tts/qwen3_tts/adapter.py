"""Concrete `MultiCodebookTTSModel` adapter for Qwen3-TTS.

Registered under `model.name == "qwen3_tts"`. Wraps the vendored
`Qwen3TTSForConditionalGeneration` so the generic
`multi_codebook_tts_grpo` recipe trainer + `MultiCodebookDPActor` stay
codec-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ..base import (
    MultiCodebookForwardOutput,
    MultiCodebookTTSModel,
    register_adapter,
)
from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


@register_adapter("qwen3_tts")
class Qwen3TTSAdapter(MultiCodebookTTSModel):
    """Adapter binding the vendored Qwen3-TTS class to the codec-agnostic
    multi-codebook training contract."""

    # Cache of the config so `num_codebooks` / vocab-size properties can
    # answer without holding a full model reference. Populated on the
    # first `load_pretrained` call; tests can set it explicitly.
    _config: Qwen3TTSConfig | None = None

    def load_pretrained(self, path: str) -> nn.Module:
        """Load Qwen3-TTS via the HF Auto* machinery (the vendored
        `__init__.py` already registered our config + model class with
        `AutoConfig` / `AutoModelForCausalLM`)."""
        model = AutoModelForCausalLM.from_pretrained(path)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"Qwen3TTSAdapter.load_pretrained: expected an instance of "
                f"Qwen3TTSForConditionalGeneration but got {type(model).__name__}. "
                f"This usually means the HF Auto* registration in "
                f"verl_omni/models/multi_codebook_tts/qwen3_tts/__init__.py "
                f"did not run; check your worker setup hook."
            )
        Qwen3TTSAdapter._config = model.config
        return model

    def forward_training(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,
        codec_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> MultiCodebookForwardOutput:
        """Delegate to the vendored model's `forward_training` and wrap
        the `(talker_logits, cb_rest_logits)` tuple in the adapter ABC's
        return type."""
        talker_logits, cb_rest_logits = model.forward_training(
            input_ids=input_ids,
            codec_ids=codec_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            prompt_lens=prompt_lens,
        )
        return MultiCodebookForwardOutput(
            talker_logits=talker_logits,
            cb_rest_logits=cb_rest_logits,
        )

    @property
    def num_codebooks(self) -> int:
        cfg = self._require_config()
        return int(cfg.talker_config.num_code_groups)

    @property
    def cb0_vocab_size(self) -> int:
        """cb0 vocab size = talker `codec_head` output dim (the talker
        config's `vocab_size`)."""
        cfg = self._require_config()
        return int(cfg.talker_config.vocab_size)

    @property
    def cb_rest_vocab_size(self) -> int:
        """cb_rest vocab size = code_predictor `lm_head[i]` output dim.
        All residual codebooks share the same vocab size (per Qwen3-TTS
        config); if a future codec breaks this, override `forward_training`
        to return per-codebook vocab metadata."""
        cfg = self._require_config()
        return int(cfg.talker_config.code_predictor_config.vocab_size)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _require_config(cls) -> Qwen3TTSConfig:
        if cls._config is None:
            raise RuntimeError(
                "Qwen3TTSAdapter: config not set. Call `load_pretrained(...)` "
                "first, or set `Qwen3TTSAdapter._config = Qwen3TTSConfig(...)` "
                "explicitly in unit tests."
            )
        return cls._config


__all__ = ["Qwen3TTSAdapter"]
