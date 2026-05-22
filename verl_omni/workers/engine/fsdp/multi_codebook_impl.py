# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FSDP engine for the `multi_codebook_tts_grpo` recipe.

Subclasses upstream `FSDPEngineWithLMHead` minimally:

- `prepare_model_inputs` is extended to forward `codec_ids` (and the
  associated `response_mask` / `prompt_lens`) into the model's
  `forward(...)` call so the vendored
  `Qwen3TTSForConditionalGeneration.forward` takes the multi-codebook
  branch and returns both cb0 logits AND cb_rest logits.
- `prepare_model_outputs` extracts `log_probs_cb_rest` from the
  cb_rest logits the model packed into the `hidden_states` slot of
  `CausalLMOutputWithPast`. The standard cb0 `log_probs` extraction is
  delegated to the upstream method.

Registered under `model_type="multi_codebook_tts"` via
`@EngineRegistry.register(...)` so `engine_workers.EngineRegistry.new(...)`
resolves to this class when the trainer YAML sets
`actor_rollout_ref.model.model_type: multi_codebook_tts`. The recipe's
new actor YAML pins this; see
`verl_omni/trainer/config/multi_codebook_tts/qwen3_tts_trainer.yaml`.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

logger = logging.getLogger(__name__)

# One-shot log latches so the per-step training loop doesn't spam logs.
_LOGGED_MISSING_CODEC_IDS = False
_LOGGED_MISSING_CB_REST_HIDDEN = False


@EngineRegistry.register(
    model_type="multi_codebook_tts",
    backend=["fsdp", "fsdp2"],
    device=["cuda", "npu"],
)
class MultiCodebookTTSFSDPEngine(FSDPEngineWithLMHead):
    """FSDP engine that produces per-stream `log_probs` (cb0 + cb_rest).

    `prepare_model_inputs` forwards the multi-codebook fields
    (`codec_ids`, `response_mask`, `prompt_lens`) into the model's
    `forward(...)` kwargs so the vendored Qwen3-TTS forward takes its
    multi-codebook branch (returns cb0 logits + cb_rest logits).

    `prepare_model_outputs` returns the upstream `log_probs` (cb0)
    plus a new `log_probs_cb_rest` `[B, T_codec * (N-1)]` extracted
    from the cb_rest logits the model packed into the
    `hidden_states[0]` slot of `CausalLMOutputWithPast`. Codebook-
    major-within-frame layout (matches AC-5.2 + the test fixtures in
    `tests/multi_codebook_tts/test_cb_rest_flatten_order.py`).
    """

    # ------------------------------------------------------------------
    # Inputs: pass codec_ids + response_mask + prompt_lens through to the
    # model so the vendored forward() takes the multi-codebook branch.
    # ------------------------------------------------------------------

    def prepare_model_inputs(self, micro_batch: TensorDict):
        global _LOGGED_MISSING_CODEC_IDS

        model_inputs, output_args = super().prepare_model_inputs(micro_batch=micro_batch)

        codec_ids = micro_batch.get("codec_ids")
        if codec_ids is None:
            if not _LOGGED_MISSING_CODEC_IDS:
                logger.warning(
                    "[MultiCodebookTTSFSDPEngine] micro_batch has no "
                    "`codec_ids` field. The agent loop must be updated "
                    "(see task8) for the multi-codebook training path to "
                    "activate. Falling back to cb0-only single-stream "
                    "forward. Suppressing subsequent warnings."
                )
                _LOGGED_MISSING_CODEC_IDS = True
            return model_inputs, output_args

        model_inputs["codec_ids"] = codec_ids
        response_mask = micro_batch.get("response_mask")
        if response_mask is not None:
            model_inputs["response_mask"] = response_mask
        # prompt_lens: if not in batch, the model derives it from
        # input_ids / codec_ids shapes (uniform across the batch).
        prompt_lens = micro_batch.get("prompt_lens")
        if prompt_lens is not None:
            model_inputs["prompt_lens"] = prompt_lens
        return model_inputs, output_args

    # ------------------------------------------------------------------
    # Outputs: extract log_probs_cb_rest from the cb_rest logits the
    # model packed into hidden_states[0].
    # ------------------------------------------------------------------

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        global _LOGGED_MISSING_CB_REST_HIDDEN

        # Standard cb0 path.
        model_output = super().prepare_model_outputs(
            output=output,
            output_args=output_args,
            micro_batch=micro_batch,
            logits_processor_func=logits_processor_func,
        )

        codec_ids = micro_batch.get("codec_ids")
        if codec_ids is None:
            return model_output

        # The vendored `forward()` packs cb_rest_logits into the
        # `hidden_states[0]` slot of `CausalLMOutputWithPast`. If it's
        # absent something upstream went wrong; warn (once) and skip
        # cb_rest extraction. The multi_codebook_ppo_loss falls back to
        # cb0-only when `log_probs_cb_rest` is missing.
        hidden_states = getattr(output, "hidden_states", None)
        if not hidden_states:
            if not _LOGGED_MISSING_CB_REST_HIDDEN:
                logger.warning(
                    "[MultiCodebookTTSFSDPEngine] `output.hidden_states` is "
                    "empty; cb_rest_logits not propagated by the model "
                    "forward. The recipe's vendored "
                    "`Qwen3TTSForConditionalGeneration.forward` must take "
                    "the multi-codebook branch when `codec_ids` is "
                    "supplied. Suppressing subsequent warnings."
                )
                _LOGGED_MISSING_CB_REST_HIDDEN = True
            return model_output

        cb_rest_logits = hidden_states[0]
        # Expected shape: [B, T_codec, N-1, V_cb_rest]
        if cb_rest_logits.dim() != 4:
            logger.warning(
                "[MultiCodebookTTSFSDPEngine] cb_rest_logits has unexpected "
                "shape %s; expected `[B, T_codec, N-1, V_cb_rest]`. "
                "Skipping cb_rest extraction for this step.",
                tuple(cb_rest_logits.shape),
            )
            return model_output

        # Gather log-probs at the target residual codec ids.
        # codec_ids: [B, T_codec, N]. residual slice = codec_ids[..., 1:]
        # = [B, T_codec, N-1]. Clamp defensively to the cb_rest vocab.
        V_cb_rest = cb_rest_logits.size(-1)
        target_cb_rest = codec_ids[..., 1:].clamp(0, V_cb_rest - 1)
        log_softmax = F.log_softmax(cb_rest_logits, dim=-1)
        per_position = log_softmax.gather(
            dim=-1, index=target_cb_rest.unsqueeze(-1),
        ).squeeze(-1)  # [B, T_codec, N-1]

        # AC-5.2 flatten: codebook-major-within-frame
        # (position `t*(N-1) + (k-1)` -> residual codebook `k` at frame `t`).
        log_probs_cb_rest = per_position.flatten(1, 2).contiguous()
        model_output["log_probs_cb_rest"] = log_probs_cb_rest
        return model_output
