"""Codec-agnostic abstract base class for multi-codebook TTS adapters.

The adapter pattern lets the trainer + actor stay codec-agnostic: the
generic `multi_codebook_tts_grpo` recipe dispatches on `model.name` to a
concrete adapter (e.g. `Qwen3TTSAdapter`) and calls its `forward_training`
during `MultiCodebookDPActor.compute_log_prob` / `.update_policy`. Future
codecs (e.g. fish-speech, deferred to v2 per DEC-2) plug in by adding a
sibling adapter directory and a `register_adapter("<name>")` line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class MultiCodebookForwardOutput:
    """Output of `MultiCodebookTTSModel.forward_training`.

    - `talker_logits`: shape `[B, T_total, V_cb0]`. Logits for codebook 0
      over the FULL prompt+codec sequence. The actor slices this to the
      response region (`prompt_len-1 : -1`) before computing per-frame
      log-probs.
    - `cb_rest_logits`: shape `[B, T_codec, N-1, V_cb_rest]`. Logits for
      residual codebooks cb1..cb_{N-1} over the response region only.
    """

    talker_logits: torch.Tensor
    cb_rest_logits: torch.Tensor


class MultiCodebookTTSModel(ABC):
    """Adapter interface every codec backbone implements."""

    @abstractmethod
    def load_pretrained(self, path: str) -> nn.Module:
        """Load the codec-TTS backbone from a checkpoint path and return
        the `nn.Module`. Adapters are free to call HF
        `AutoModelForCausalLM.from_pretrained(...)` here as long as the
        backbone's `forward_training` exists on the returned module.
        """

    @abstractmethod
    def forward_training(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,         # [B, T_total]  (prompt + codec)
        codec_ids: torch.LongTensor,         # [B, T_codec, N]  (response region only)
        attention_mask: torch.Tensor,        # [B, T_total]
        response_mask: torch.Tensor,         # [B, T_total]; 1 on generated frames
        prompt_lens: torch.Tensor,           # [B]; per-sample prompt length
    ) -> MultiCodebookForwardOutput:
        """Run a single training-time forward pass.

        The implementation must return:
        - `talker_logits` of shape `[B, T_total, V_cb0]` (full sequence; the
          actor will slice the response region itself).
        - `cb_rest_logits` of shape `[B, T_codec, N-1, V_cb_rest]` for the
          residual codebooks over the response region.
        """

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        """Total codebooks `N` in the codec (cb0 + N-1 residual)."""

    @property
    @abstractmethod
    def cb0_vocab_size(self) -> int:
        """Vocab size of codebook 0."""

    @property
    @abstractmethod
    def cb_rest_vocab_size(self) -> int:
        """Vocab size of the residual codebooks (must be uniform across
        cb1..cb_{N-1}; if a codec has per-codebook vocab sizes, override
        `forward_training` to return a per-stream vocab tuple)."""


# ----------------------------------------------------------------------
# Adapter registry. `register_adapter("<name>")` adds an entry; the
# recipe trainer's launcher looks the adapter up by `model.name`.
# ----------------------------------------------------------------------

MULTI_CODEBOOK_ADAPTER_REGISTRY: dict[str, MultiCodebookTTSModel] = {}


def register_adapter(
    name: str,
) -> Callable[[type[MultiCodebookTTSModel]], type[MultiCodebookTTSModel]]:
    """Decorator: register a concrete adapter class under `name`.

    The adapter is instantiated lazily (no-arg constructor) and stored in
    the registry. Re-registering the same name replaces the previous
    entry.
    """

    def _wrap(cls: type[MultiCodebookTTSModel]) -> type[MultiCodebookTTSModel]:
        MULTI_CODEBOOK_ADAPTER_REGISTRY[name] = cls()
        return cls

    return _wrap


def get_adapter(name: str) -> MultiCodebookTTSModel:
    """Look up an adapter by `model.name`. Raises `KeyError` with a
    helpful message listing the registered names if missing."""
    if name not in MULTI_CODEBOOK_ADAPTER_REGISTRY:
        registered = sorted(MULTI_CODEBOOK_ADAPTER_REGISTRY.keys())
        raise KeyError(
            f"multi-codebook TTS adapter {name!r} is not registered; "
            f"known adapters: {registered}. Did you forget to import "
            f"`verl_omni.models.multi_codebook_tts.{name}`?"
        )
    return MULTI_CODEBOOK_ADAPTER_REGISTRY[name]
