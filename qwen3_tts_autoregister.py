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
    # stub at the top level (verified via
    # ``inspect.signature(...) == (self, *input)``). verl's PPO/GRPO
    # actor calls ``model(input_ids=..., attention_mask=...,
    # position_ids=...)`` for both ``_compute_old_log_prob`` and
    # ``update_actor`` (the gradient step), which raises
    # ``TypeError: _forward_unimplemented() got an unexpected keyword
    # argument 'input_ids'``. The internal ``self.talker`` IS a
    # standard HF CausalLM with a real forward, so delegate.
    if not getattr(Qwen3TTSForConditionalGeneration, "_verl_forward_shim_applied", False):
        def _talker_forward_shim(self, *args, **kwargs):
            return self.talker(*args, **kwargs)

        Qwen3TTSForConditionalGeneration.forward = _talker_forward_shim
        Qwen3TTSForConditionalGeneration._verl_forward_shim_applied = True

    _REGISTERED = True
