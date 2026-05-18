"""Lazy Qwen3-TTS HF AutoConfig/AutoModel registration.

This is a *top-level* module (project root, not inside ``verl_omni/``) so
Ray's ``runtime_env.worker_process_setup_hook = "qwen3_tts_autoregister.setup"``
can invoke it inside each Ray actor / worker without dragging in the
heavy ``verl_omni`` package (whose ``__init__`` imports pipelines,
diffusion adapters, rollout backends, etc. and costs ~20s of startup).

The Qwen3-TTS-12Hz-0.6B-Base checkpoint has ``model_type: qwen3_tts``
but ``auto_map`` is not shipped in its config.json. Every Python process
that calls ``AutoConfig.from_pretrained`` (verl's HFModelConfig
validation, run inside TaskRunner) or ``AutoModel.from_pretrained``
(FSDP weight load, run inside WorkerDict) needs the Qwen3-TTS classes
registered.

The driver process registers via ``verl_omni/__init__.py``'s call to
:func:`setup`. Ray-spawned worker subprocesses get registration via the
``worker_process_setup_hook`` runtime_env entry plumbed in by
``verl_omni.trainer.qwen3_tts_grpo.main`` (the recipe's Hydra entry
point) before ``run_ppo`` is invoked.

If ``qwen-tts`` is not pip-installed, ``QWEN3_TTS_SOURCE_DIR`` may point
at a local clone of the upstream qwen-tts repo (the dir containing the
``qwen_tts/`` package).
"""

import os
import sys


_REGISTERED = False


def setup() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    src = os.environ.get("QWEN3_TTS_SOURCE_DIR")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        from qwen_tts.core.models import (
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    except ImportError:
        return
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
    AutoModelForCausalLM.register(
        Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True
    )

    # Also wire ``rollout.name=vllm_omni_tts`` into verl's
    # ``_ROLLOUT_REGISTRY`` so that ``get_rollout_class("vllm_omni_tts",
    # "async")`` resolves inside Ray WorkerDict actors. The driver
    # process already does this via ``verl_omni.workers.rollout.base``;
    # mirroring it here keeps the worker setup self-contained without
    # forcing the full ~20s ``verl_omni`` package import on every Ray
    # subprocess.
    try:
        from verl.workers.rollout.base import _ROLLOUT_REGISTRY
    except ImportError:
        pass
    else:
        _ROLLOUT_REGISTRY[("vllm_omni_tts", "async")] = (
            "verl.workers.rollout.vllm_rollout.ServerAdapter"
        )

    # verl's ``attention_utils._get_attention_functions`` hard-imports
    # ``flash_attn.bert_padding`` on CUDA (see
    # ``verl/utils/attention_utils.py:30``). On environments without
    # flash_attn installed, ``_compute_old_log_prob`` →
    # ``left_right_2_no_padding`` → ``unpad_input`` then crashes with
    # ``ModuleNotFoundError: No module named 'flash_attn'`` *after*
    # rollouts succeed. Transformers ships pure-PyTorch fallbacks at
    # ``transformers.modeling_flash_attention_utils._{unpad_input,
    # pad_input, index_first_axis}`` — swap them in here so the FSDP
    # log-prob recompute path works without flash_attn.
    try:
        from verl.utils import attention_utils as _verl_attn
        from transformers.modeling_flash_attention_utils import (
            _index_first_axis as _tf_index_first_axis,
            _pad_input as _tf_pad_input,
            _unpad_input as _tf_unpad_input,
        )
        try:
            from einops import rearrange as _einops_rearrange
        except ImportError:
            _einops_rearrange = None
    except ImportError:
        pass
    else:
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

    # Qwen3TTSConfig holds all the standard transformer hyperparameters
    # (hidden_size, num_attention_heads, etc.) under ``talker_config``,
    # not at the top level. Upstream verl's FSDP / monkey-patch code
    # reads ``config.num_attention_heads`` and
    # ``config.text_config.hidden_size`` directly and raises
    # AttributeError on Qwen3-TTS. Bridge the two layouts here:
    #
    # 1. wrap ``__init__`` so freshly-constructed Qwen3TTSConfig
    #    instances copy ``talker_config``'s standard fields to the
    #    top-level config.
    # 2. expose a ``text_config`` property aliased to ``talker_config``
    #    so the VLM-style fallback in verl works.
    if not getattr(Qwen3TTSConfig, "_verl_layout_bridge_applied", False):
        _orig_init = Qwen3TTSConfig.__init__
        _bridged_fields = (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "vocab_size",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "head_dim",
            "attention_bias",
            "attention_dropout",
        )

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            talker = getattr(self, "talker_config", None)
            if talker is None:
                return
            for field in _bridged_fields:
                if getattr(self, field, None) is None:
                    value = getattr(talker, field, None)
                    if value is not None:
                        setattr(self, field, value)

        Qwen3TTSConfig.__init__ = _patched_init
        Qwen3TTSConfig.text_config = property(
            lambda self: self.talker_config,
            doc=(
                "Alias used by verl's VLM-style config inspection paths "
                "(see ``verl.models.transformers.monkey_patch``)."
            ),
        )
        Qwen3TTSConfig._verl_layout_bridge_applied = True

    # ``Qwen3TTSForConditionalGeneration`` is designed to be driven via
    # ``generate()`` only — it inherits the ``nn.Module._forward_unimplemented``
    # stub at the top level. verl's PPO/GRPO actor calls
    # ``model(input_ids=..., attention_mask=..., position_ids=...,
    # labels=...)`` during ``update_actor`` (gradient step), which
    # raises ``TypeError: _forward_unimplemented() got an unexpected
    # keyword argument 'input_ids'``.
    #
    # The internal ``self.talker`` has a forward, but it expects
    # ref-audio-derived ``inputs_embeds`` (prefill branch at
    # ``modeling_qwen3_tts.py:1665``) or ``past_hidden /
    # trailing_text_hidden / tts_pad_embed`` (generate branch at line
    # 1669) — neither of which verl supplies. Delegating directly to
    # ``self.talker(*args, **kwargs)`` falls through to the generate
    # branch and crashes on ``past_hidden=None``.
    #
    # The shim below bypasses both of the talker's custom branches and
    # drives the underlying ``self.talker.model`` (the plain Qwen3
    # decoder) + ``self.talker.codec_head`` directly. Tokens that fall
    # outside the codec vocab (i.e. the HF-tokenized prompt portion of
    # ``input_ids = cat(prompts, responses)``) are clamped to
    # ``codec_pad_id`` so the codec embedding lookup stays in range;
    # labels for those tokens are masked to ``-100``.
    #
    # This is sufficient for verl's GRPO plumbing — policy_loss can be
    # computed against the codec response_mask region — but it is NOT
    # a faithful Qwen3-TTS training-time forward: the prompt-embed
    # reconstruction (which depends on ref_audio + text) is replaced
    # by a no-op codec_pad placeholder, so gradients on the prefill
    # region are uninformative. The right long-term fix is to thread
    # ``additional_information`` through ``model_kwargs`` and let the
    # shim invoke ``self.talker._build_prompt_embeds`` per-row.
    if not getattr(Qwen3TTSForConditionalGeneration, "_verl_forward_shim_applied", False):
        from transformers.modeling_outputs import CausalLMOutputWithPast as _CausalLMOutputWithPast
        import torch as _torch
        import torch.nn.functional as _F

        def _talker_training_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            inputs_embeds=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            **kwargs,
        ):
            talker = self.talker
            talker_cfg = talker.config

            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError(
                        "Qwen3-TTS training-shape forward requires input_ids or inputs_embeds."
                    )
                codec_vocab = int(getattr(talker_cfg, "vocab_size", 3072))
                pad_id = int(getattr(talker_cfg, "codec_pad_id", 0))
                safe_input_ids = _torch.where(
                    (input_ids >= 0) & (input_ids < codec_vocab),
                    input_ids,
                    _torch.full_like(input_ids, pad_id),
                )
                inputs_embeds = talker.get_input_embeddings()(safe_input_ids)

            outputs = talker.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            hidden_states = outputs.last_hidden_state
            logits = talker.codec_head(hidden_states)

            loss = None
            if labels is not None:
                codec_vocab = int(getattr(talker_cfg, "vocab_size", logits.shape[-1]))
                safe_labels = _torch.where(
                    (labels >= 0) & (labels < codec_vocab),
                    labels,
                    _torch.full_like(labels, -100),
                )
                # Standard HF causal-LM loss: shift by one and cross-entropy.
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = safe_labels[..., 1:].contiguous()
                loss = _F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            return _CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        Qwen3TTSForConditionalGeneration.forward = _talker_training_forward
        Qwen3TTSForConditionalGeneration._verl_forward_shim_applied = True

    _REGISTERED = True
