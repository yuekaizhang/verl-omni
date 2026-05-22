# Multi-Codebook TTS GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cb0-only `qwen3_tts_grpo` recipe with a generic `multi_codebook_tts_grpo` recipe that trains all N codebooks via Fish Audio S2's weighted-sum loss, vendoring qwen3-tts model code in-repo and retiring the `qwen3_tts_autoregister.py` monkey-patcher.

**Architecture:** Build `verl_omni/models/multi_codebook_tts/` with a `MultiCodebookTTSModel` adapter ABC and a `qwen3_tts/` leaf. Drive both heads (talker `codec_head` + `code_predictor`) in a single training-time forward; expose per-stream log-probs through a custom `multi_codebook_ppo_loss` injected via a `MultiCodebookActorRolloutRefWorker` subclass. Verl PPO core untouched.

**Tech Stack:** PyTorch + FSDP, Hugging Face transformers, verl PPO (installed in `.venv/lib/python3.12/site-packages/verl`), Hydra/OmegaConf, vllm-omni rollout server (fork at `/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl`), pytest.

**Source spec:** `docs/superpowers/specs/2026-05-22-multi-codebook-tts-grpo-design.md`.

**Working environment note:** Run all training/rollout commands inside an interactive Slurm job container (`$USER == root`). Read-only work (Read/Grep/Glob, pytest unit tests without GPU) is fine from any node. The smoke task explicitly calls out `sbatch`.

---

## Task 1: Create `multi_codebook_tts` package skeleton + adapter ABC

**Files:**
- Create: `verl_omni/models/__init__.py` (if missing)
- Create: `verl_omni/models/multi_codebook_tts/__init__.py`
- Create: `verl_omni/models/multi_codebook_tts/base.py`
- Create: `tests/models/multi_codebook_tts/__init__.py`
- Create: `tests/models/multi_codebook_tts/test_adapter_registry.py`

- [ ] **Step 1: Write the failing test for the adapter registry**

`tests/models/multi_codebook_tts/test_adapter_registry.py`:

```python
"""Adapter registry contract: name → adapter class lookup, error on missing."""

from __future__ import annotations

import pytest

from verl_omni.models.multi_codebook_tts import (
    MultiCodebookTTSModel,
    get_adapter,
    register_adapter,
)


class _DummyAdapter(MultiCodebookTTSModel):
    @property
    def num_codebooks(self):
        return 4

    @property
    def cb0_vocab_size(self):
        return 3072

    @property
    def cb_rest_vocab_size(self):
        return 3072

    def load_pretrained(self, path):
        raise NotImplementedError

    def forward_training(self, model, input_ids, codec_ids, attention_mask, response_mask):
        raise NotImplementedError


def test_registry_round_trip():
    register_adapter("dummy_for_test", _DummyAdapter)
    cls = get_adapter("dummy_for_test")
    assert cls is _DummyAdapter
    assert cls().num_codebooks == 4


def test_registry_missing_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_adapter("does_not_exist")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/verl-omni
.venv/bin/pytest tests/models/multi_codebook_tts/test_adapter_registry.py -v
```

Expected: `ModuleNotFoundError: verl_omni.models.multi_codebook_tts` (or `ImportError`).

- [ ] **Step 3: Implement the ABC**

`verl_omni/models/multi_codebook_tts/base.py`:

```python
"""Adapter contract for multi-codebook AR TTS backbones in verl-omni RL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class MultiCodebookForwardOutput:
    talker_logits: torch.Tensor       # [B, T_total, V_cb0]
    cb_rest_logits: torch.Tensor      # [B, T_codec, N-1, V_cb_rest]
    extras: dict[str, Any] | None = None


class MultiCodebookTTSModel(ABC):
    """Backbone adapter for a multi-codebook AR TTS model.

    Each concrete adapter binds a specific HuggingFace model (qwen3-tts,
    fish-speech, ...) to the verl-omni RL plumbing. The adapter must:

    1. Load pretrained weights into an ``nn.Module``.
    2. Produce per-codebook training logits in a uniform shape so the
       multi-codebook policy loss can compute log-probs for both streams
       (cb0 and cb1..cbN-1).
    3. Declare codebook count and vocab sizes for shape assertions.
    """

    @abstractmethod
    def load_pretrained(self, path: str) -> nn.Module:
        """Return a fresh nn.Module loaded from ``path``."""

    @abstractmethod
    def forward_training(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,       # [B, T_total]
        codec_ids: torch.LongTensor,       # [B, T_codec, N]
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,       # [B, T_total]; 1 on generated frames
    ) -> MultiCodebookForwardOutput: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def cb0_vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def cb_rest_vocab_size(self) -> int: ...
```

`verl_omni/models/multi_codebook_tts/__init__.py`:

```python
"""Multi-codebook TTS adapter registry.

Use ``register_adapter("qwen3_tts", Qwen3TTSAdapter)`` at import time to
make an adapter discoverable from a config string. The trainer entry
looks up adapters by ``model.name``.
"""

from __future__ import annotations

from .base import MultiCodebookForwardOutput, MultiCodebookTTSModel

_REGISTRY: dict[str, type[MultiCodebookTTSModel]] = {}


def register_adapter(name: str, adapter_cls: type[MultiCodebookTTSModel]) -> None:
    _REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> type[MultiCodebookTTSModel]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Multi-codebook TTS adapter {name!r} is not registered. "
            f"Known adapters: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    "MultiCodebookForwardOutput",
    "MultiCodebookTTSModel",
    "get_adapter",
    "register_adapter",
]
```

Also create the empty package markers:

```bash
test -f verl_omni/models/__init__.py || touch verl_omni/models/__init__.py
mkdir -p tests/models/multi_codebook_tts
touch tests/models/multi_codebook_tts/__init__.py
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/pytest tests/models/multi_codebook_tts/test_adapter_registry.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add verl_omni/models/__init__.py verl_omni/models/multi_codebook_tts/ tests/models/multi_codebook_tts/
git commit -m "[model] feat: multi_codebook_tts adapter ABC + registry"
```

---

## Task 2: Vendor qwen3-tts model code (configuration + modeling)

**Files:**
- Create: `verl_omni/models/multi_codebook_tts/qwen3_tts/__init__.py`
- Create: `verl_omni/models/multi_codebook_tts/qwen3_tts/configuration_qwen3_tts.py` (vendored)
- Create: `verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py` (vendored)

- [ ] **Step 1: Copy the vendored files verbatim**

```bash
SRC=/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/Qwen3-TTS/qwen_tts/core/models
DST=verl_omni/models/multi_codebook_tts/qwen3_tts
mkdir -p "$DST"
cp "$SRC/configuration_qwen3_tts.py" "$DST/configuration_qwen3_tts.py"
cp "$SRC/modeling_qwen3_tts.py"      "$DST/modeling_qwen3_tts.py"
```

- [ ] **Step 2: Update the import paths inside the vendored files**

The vendored files reference `qwen_tts.core.utils.*` and similar relative-to-qwen-tts imports. Replace any cross-package imports that are not standard `transformers`/`torch` with vendored copies *only if pytest collection breaks*. Most upstream imports are either stdlib, `torch`, `transformers`, or self-references — verify by running:

```bash
.venv/bin/python -c "from verl_omni.models.multi_codebook_tts.qwen3_tts import modeling_qwen3_tts; print('ok')"
```

If that errors, follow the import chain: copy each non-`transformers` upstream module the vendored modeling file imports into the same directory (e.g. `tokenizers.py`, `utils.py`) and rewrite the failing import to `from .<module> import ...`. Re-run until the smoke import succeeds.

- [ ] **Step 3: Add the package __init__ that registers HF AutoConfig + AutoModel**

`verl_omni/models/multi_codebook_tts/qwen3_tts/__init__.py`:

```python
"""Qwen3-TTS vendored model package.

Importing this module:
- registers ``qwen3_tts`` with HF AutoConfig/AutoModel/AutoModelForCausalLM
- registers the adapter with the multi_codebook_tts registry
- bridges Qwen3TTSConfig's ``talker_config``-nested layout to the
  top-level attributes that verl's FSDP / monkey-patch code reads
"""

from __future__ import annotations

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from verl_omni.models.multi_codebook_tts import register_adapter

from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
AutoModelForCausalLM.register(
    Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True
)

# ``Qwen3TTSConfig`` keeps standard transformer hyperparameters under
# ``talker_config``. Mirror them to the top-level config so verl's
# config-inspection code paths (which read ``config.num_attention_heads`` and
# ``config.text_config.hidden_size``) work without an AttributeError.
_BRIDGED_FIELDS = (
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

if not getattr(Qwen3TTSConfig, "_verl_layout_bridge_applied", False):
    _orig_init = Qwen3TTSConfig.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        talker = getattr(self, "talker_config", None)
        if talker is None:
            return
        for field in _BRIDGED_FIELDS:
            if getattr(self, field, None) is None:
                value = getattr(talker, field, None)
                if value is not None:
                    setattr(self, field, value)

    Qwen3TTSConfig.__init__ = _patched_init
    Qwen3TTSConfig.text_config = property(
        lambda self: self.talker_config,
        doc="Alias used by verl's VLM-style config inspection paths.",
    )
    Qwen3TTSConfig._verl_layout_bridge_applied = True

# Lazy adapter registration (avoids importing adapter.py here so this
# module is cheap to import for HF registration alone).
def _register():
    from .adapter import Qwen3TTSAdapter  # local import to avoid cycle

    register_adapter("qwen3_tts", Qwen3TTSAdapter)


_register()
del _register
```

- [ ] **Step 4: Confirm the package imports cleanly**

```bash
.venv/bin/python -c "from verl_omni.models.multi_codebook_tts import qwen3_tts; print('ok')"
```

Expected: `ok`. (Will fail with `ImportError: cannot import name 'Qwen3TTSAdapter'` because Task 3 has not yet created `adapter.py`. That's expected; skip this verification until Task 3 step 1.)

- [ ] **Step 5: Commit**

```bash
git add verl_omni/models/multi_codebook_tts/qwen3_tts/__init__.py \
        verl_omni/models/multi_codebook_tts/qwen3_tts/configuration_qwen3_tts.py \
        verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py
git commit -m "[model] feat: vendor qwen3-tts configuration + modeling into multi_codebook_tts"
```

---

## Task 3: Add a clean training-time forward to the vendored qwen3-tts model

**Files:**
- Modify: `verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py` (append a method on `Qwen3TTSForConditionalGeneration`)
- Create: `verl_omni/models/multi_codebook_tts/qwen3_tts/adapter.py`
- Create: `tests/models/multi_codebook_tts/test_qwen3_tts_forward.py`

The current vendored `Qwen3TTSForConditionalGeneration.forward` is generate-only and unhelpful for verl's training path. The previous codebase worked around this with a monkey-patched `_talker_training_forward`. We now add a real method `forward_training_dual_stream` directly on the class.

- [ ] **Step 1: Write the failing test (tiny synthetic forward)**

`tests/models/multi_codebook_tts/test_qwen3_tts_forward.py`:

```python
"""Smoke test for Qwen3TTSAdapter.forward_training on a tiny synthetic config.

Builds a 2-layer / hidden=64 / num_code_groups=4 Qwen3TTSConfig, asserts
both logit streams come back with the expected shapes and finite values.
"""

from __future__ import annotations

import os

import pytest
import torch

# Skip if vendored package fails to import (e.g. during partial impl)
qwen3_tts = pytest.importorskip("verl_omni.models.multi_codebook_tts.qwen3_tts")


@pytest.fixture
def tiny_model():
    from verl_omni.models.multi_codebook_tts.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
    from verl_omni.models.multi_codebook_tts.qwen3_tts.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )

    cfg = Qwen3TTSConfig()
    # Shrink the talker + code predictor so the test runs in CPU.
    cfg.talker_config.hidden_size = 64
    cfg.talker_config.intermediate_size = 128
    cfg.talker_config.num_hidden_layers = 2
    cfg.talker_config.num_attention_heads = 2
    cfg.talker_config.num_key_value_heads = 2
    cfg.talker_config.num_code_groups = 4
    cfg.talker_config.vocab_size = 64
    cfg.code_predictor_config.hidden_size = 64
    cfg.code_predictor_config.intermediate_size = 128
    cfg.code_predictor_config.num_hidden_layers = 1
    cfg.code_predictor_config.num_attention_heads = 2
    cfg.code_predictor_config.num_key_value_heads = 2
    cfg.code_predictor_config.vocab_size = 64

    torch.manual_seed(0)
    model = Qwen3TTSForConditionalGeneration(cfg)
    model.eval()
    return cfg, model


def test_forward_training_shapes(tiny_model):
    from verl_omni.models.multi_codebook_tts.qwen3_tts.adapter import Qwen3TTSAdapter

    cfg, model = tiny_model
    B, T_total, T_codec = 2, 12, 8
    N = cfg.talker_config.num_code_groups

    input_ids = torch.randint(0, cfg.talker_config.vocab_size, (B, T_total))
    codec_ids = torch.randint(0, cfg.talker_config.vocab_size, (B, T_codec, N))
    attention_mask = torch.ones(B, T_total, dtype=torch.long)
    response_mask = torch.zeros(B, T_total, dtype=torch.long)
    response_mask[:, T_total - T_codec :] = 1

    adapter = Qwen3TTSAdapter(cfg)
    with torch.no_grad():
        out = adapter.forward_training(model, input_ids, codec_ids, attention_mask, response_mask)

    assert out.talker_logits.shape == (B, T_total, cfg.talker_config.vocab_size)
    assert out.cb_rest_logits.shape == (B, T_codec, N - 1, cfg.code_predictor_config.vocab_size)
    assert torch.isfinite(out.talker_logits).all()
    assert torch.isfinite(out.cb_rest_logits).all()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/models/multi_codebook_tts/test_qwen3_tts_forward.py -v
```

Expected: FAIL — `ImportError: cannot import name 'Qwen3TTSAdapter'`.

- [ ] **Step 3: Append `forward_training_dual_stream` to `Qwen3TTSForConditionalGeneration`**

In `verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py`, append at the bottom of the file (do not edit the existing `forward`; we are adding a new method):

```python
# ============================================================================
# verl-omni RL training-time forward.
#
# The upstream ``forward()`` above is shaped for ``generate()``: prefill
# branch expects ref-audio-derived inputs_embeds, generate branch expects
# past_hidden/trailing_text_hidden/tts_pad_embed. Neither is available
# during verl's update_actor step, which calls model(input_ids=...,
# codec_ids=..., attention_mask=...).
#
# ``forward_training_dual_stream`` drives ``self.talker.model`` (a plain
# Qwen3 decoder) + ``self.talker.codec_head`` directly to produce
# cb0 logits, then runs ``self.talker.code_predictor`` per frame on the
# resulting talker hidden + codec_ids to produce cb1..cbN-1 logits.
#
# Out-of-vocab prompt tokens are clamped to the talker codec pad id so
# the codec embedding lookup stays in range; the loss masks them out
# anyway via response_mask.
# ============================================================================

import torch as _torch
import torch.nn.functional as _F  # noqa: F401


def _qwen3_tts_forward_training_dual_stream(
    self,
    input_ids: _torch.LongTensor,        # [B, T_total]
    codec_ids: _torch.LongTensor,        # [B, T_codec, N]
    attention_mask: _torch.Tensor,
    position_ids=None,
    past_key_values=None,
    use_cache: bool = False,
):
    talker = self.talker
    talker_cfg = talker.config

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
        output_attentions=False,
        output_hidden_states=False,
    )

    hidden_states = outputs.last_hidden_state              # [B, T_total, H]
    talker_logits = talker.codec_head(hidden_states)       # [B, T_total, V_cb0]

    # Code-predictor (depth transformer) forward on the response frames.
    # The talker hidden corresponding to a codec frame is the hidden at
    # the same position. We slice from the end since the codec frames
    # are the last T_codec positions of the input.
    T_codec = codec_ids.shape[1]
    talker_hidden_at_frames = hidden_states[:, -T_codec:, :]   # [B, T_codec, H]
    B, T, H = talker_hidden_at_frames.shape
    N = codec_ids.shape[-1]

    # forward_sub_talker_finetune wants [B*T_codec, ...]; flatten time.
    cp_hidden = talker_hidden_at_frames.reshape(B * T, H)
    cp_codes = codec_ids.reshape(B * T, N)

    cb_rest_logits_flat, _ = talker.forward_sub_talker_finetune(
        codec_ids=cp_codes,
        talker_hidden_states=cp_hidden,
    )
    # forward_sub_talker_finetune returns logits for cb1..cbN-1 frames,
    # i.e. shape [B*T, N-1, V_cb_rest].
    V_rest = cb_rest_logits_flat.shape[-1]
    cb_rest_logits = cb_rest_logits_flat.reshape(B, T, N - 1, V_rest)

    return talker_logits, cb_rest_logits


Qwen3TTSForConditionalGeneration.forward_training_dual_stream = (
    _qwen3_tts_forward_training_dual_stream
)
```

- [ ] **Step 4: Implement `Qwen3TTSAdapter`**

`verl_omni/models/multi_codebook_tts/qwen3_tts/adapter.py`:

```python
"""Qwen3-TTS adapter binding to the multi_codebook_tts trainer."""

from __future__ import annotations

import torch
import torch.nn as nn

from verl_omni.models.multi_codebook_tts.base import (
    MultiCodebookForwardOutput,
    MultiCodebookTTSModel,
)

from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


class Qwen3TTSAdapter(MultiCodebookTTSModel):
    """Wraps Qwen3-TTS for verl-omni multi-codebook GRPO."""

    def __init__(self, config: Qwen3TTSConfig | None = None):
        self._config = config

    def load_pretrained(self, path: str) -> nn.Module:
        return Qwen3TTSForConditionalGeneration.from_pretrained(path)

    def forward_training(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,
        codec_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> MultiCodebookForwardOutput:
        talker_logits, cb_rest_logits = model.forward_training_dual_stream(
            input_ids=input_ids,
            codec_ids=codec_ids,
            attention_mask=attention_mask,
        )
        return MultiCodebookForwardOutput(
            talker_logits=talker_logits,
            cb_rest_logits=cb_rest_logits,
        )

    @property
    def num_codebooks(self) -> int:
        cfg = self._effective_config
        return int(cfg.talker_config.num_code_groups)

    @property
    def cb0_vocab_size(self) -> int:
        cfg = self._effective_config
        return int(cfg.talker_config.vocab_size)

    @property
    def cb_rest_vocab_size(self) -> int:
        cfg = self._effective_config
        return int(cfg.code_predictor_config.vocab_size)

    @property
    def _effective_config(self) -> Qwen3TTSConfig:
        if self._config is None:
            raise RuntimeError(
                "Qwen3TTSAdapter was constructed without a config; "
                "either pass a Qwen3TTSConfig at adapter construction "
                "time or call load_pretrained() first and use the "
                "model's .config instead."
            )
        return self._config
```

- [ ] **Step 5: Re-run the test and verify it passes**

```bash
.venv/bin/pytest tests/models/multi_codebook_tts/test_qwen3_tts_forward.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py \
        verl_omni/models/multi_codebook_tts/qwen3_tts/adapter.py \
        tests/models/multi_codebook_tts/test_qwen3_tts_forward.py
git commit -m "[model] feat: Qwen3TTSAdapter.forward_training drives talker + code_predictor"
```

---

## Task 4: Extract `attention_utils_fallback` and `ray_runtime_env` utilities

These two responsibilities are currently buried in the 269-line top-level `qwen3_tts_autoregister.py`. We extract them into named modules with single responsibilities so they can be reused (and tested) without the autoregister blob.

**Files:**
- Create: `verl_omni/utils/attention_utils_fallback.py`
- Create: `verl_omni/utils/ray_runtime_env.py`
- Create: `tests/utils/test_attention_utils_fallback.py`
- Create: `tests/utils/test_ray_runtime_env.py`

- [ ] **Step 1: Write the failing test for `ray_runtime_env`**

`tests/utils/test_ray_runtime_env.py`:

```python
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from verl_omni.utils.ray_runtime_env import inject_worker_setup_hook


def test_inject_into_empty_config_creates_runtime_env():
    cfg = OmegaConf.create({})
    inject_worker_setup_hook(cfg, repo_root="/tmp/repo")
    assert (
        cfg.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook
        == "verl_omni.models.multi_codebook_tts.boot.setup_workers"
    )
    assert (
        cfg.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH == "/tmp/repo"
    )


def test_inject_preserves_existing_pythonpath():
    cfg = OmegaConf.create(
        {
            "ray_kwargs": {
                "ray_init": {
                    "runtime_env": {
                        "env_vars": {"PYTHONPATH": "/existing"},
                    }
                }
            }
        }
    )
    inject_worker_setup_hook(cfg, repo_root="/tmp/repo")
    assert (
        cfg.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH
        == "/tmp/repo:/existing"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/utils/test_ray_runtime_env.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ray_runtime_env.py`**

`verl_omni/utils/ray_runtime_env.py`:

```python
"""One place to construct the Ray runtime_env block for verl-omni recipes.

Replaces the inline runtime_env construction in
``verl_omni/trainer/qwen3_tts_grpo/main.py``. Used by the
multi_codebook_tts trainer entry to inject a worker setup hook + PYTHONPATH
so Ray-spawned worker subprocesses can import the vendored qwen3-tts
module without dragging in the full ``verl_omni`` package at startup.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

WORKER_SETUP_HOOK = "verl_omni.models.multi_codebook_tts.boot.setup_workers"


def inject_worker_setup_hook(config: DictConfig, repo_root: str | Path) -> None:
    """Mutate ``config`` in place to set runtime_env worker_process_setup_hook
    and prepend ``repo_root`` to PYTHONPATH.
    """

    repo_root = str(repo_root)
    OmegaConf.set_struct(config, False)
    try:
        runtime_env = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env")
        if runtime_env is None:
            config.ray_kwargs = OmegaConf.create(
                {
                    "ray_init": {
                        "runtime_env": {
                            "worker_process_setup_hook": WORKER_SETUP_HOOK,
                            "env_vars": {"PYTHONPATH": repo_root},
                        }
                    }
                }
            )
            return
        runtime_env["worker_process_setup_hook"] = WORKER_SETUP_HOOK
        env_vars = runtime_env.get("env_vars") or OmegaConf.create({})
        existing = env_vars.get("PYTHONPATH", "")
        env_vars["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else repo_root
        runtime_env["env_vars"] = env_vars
    finally:
        OmegaConf.set_struct(config, True)
```

- [ ] **Step 4: Re-run and confirm pass**

```bash
.venv/bin/pytest tests/utils/test_ray_runtime_env.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Implement `attention_utils_fallback.py` and its test**

`verl_omni/utils/attention_utils_fallback.py`:

```python
"""Replace verl's flash_attn-only attention utilities with transformers'
pure-PyTorch fallbacks. Used in environments without flash_attn installed
(verl's ``attention_utils._get_attention_functions`` hard-imports
``flash_attn.bert_padding`` on CUDA, breaking ``_compute_old_log_prob`` →
``left_right_2_no_padding`` after rollouts succeed).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install_flash_attn_fallback() -> bool:
    """Returns True iff the fallback was successfully installed.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    try:
        from verl.utils import attention_utils as _verl_attn
        from transformers.modeling_flash_attention_utils import (
            _index_first_axis as _tf_index_first_axis,
            _pad_input as _tf_pad_input,
            _unpad_input as _tf_unpad_input,
        )
    except ImportError as e:
        logger.warning("flash_attn fallback not installed: %s", e)
        return False

    try:
        from einops import rearrange as _einops_rearrange
    except ImportError:
        _einops_rearrange = None

    def _patched():
        return (_tf_index_first_axis, _tf_pad_input, _einops_rearrange, _tf_unpad_input)

    _verl_attn._get_attention_functions = _patched
    _verl_attn._index_first_axis = _tf_index_first_axis
    _verl_attn._pad_input = _tf_pad_input
    _verl_attn._rearrange = _einops_rearrange
    _verl_attn._unpad_input = _tf_unpad_input
    return True
```

`tests/utils/test_attention_utils_fallback.py`:

```python
"""Idempotency + happy-path test for install_flash_attn_fallback."""

from __future__ import annotations

import pytest


def test_install_is_idempotent_and_patches_verl_attn():
    pytest.importorskip("verl.utils.attention_utils")
    from verl.utils import attention_utils as verl_attn
    from verl_omni.utils.attention_utils_fallback import install_flash_attn_fallback

    ok1 = install_flash_attn_fallback()
    if not ok1:
        pytest.skip("Transformers fallback not available in this env")
    fn_after_first = verl_attn._get_attention_functions
    ok2 = install_flash_attn_fallback()
    assert ok2
    assert verl_attn._get_attention_functions is fn_after_first
```

- [ ] **Step 6: Run both util tests**

```bash
.venv/bin/pytest tests/utils/test_attention_utils_fallback.py tests/utils/test_ray_runtime_env.py -v
```

Expected: 3 passed (1 may skip if env lacks the transformers fallback).

- [ ] **Step 7: Commit**

```bash
git add verl_omni/utils/attention_utils_fallback.py verl_omni/utils/ray_runtime_env.py \
        tests/utils/test_attention_utils_fallback.py tests/utils/test_ray_runtime_env.py
git commit -m "[utils] feat: extract attention_utils + ray_runtime_env from autoregister"
```

---

## Task 5: Worker bootstrap module (`boot.setup_workers`)

The Ray runtime_env hook from Task 4 points at `verl_omni.models.multi_codebook_tts.boot.setup_workers`. That module must:

1. Apply the flash_attn fallback (from Task 4).
2. Import the qwen3-tts vendored package (which triggers HF registration via its `__init__.py`).
3. Register the rollout adapter into verl's `_ROLLOUT_REGISTRY` for the "vllm_omni_tts" name.

**Files:**
- Create: `verl_omni/models/multi_codebook_tts/boot.py`
- Create: `tests/models/multi_codebook_tts/test_boot.py`

- [ ] **Step 1: Write the failing test**

`tests/models/multi_codebook_tts/test_boot.py`:

```python
"""``setup_workers`` is the Ray worker_process_setup_hook entry. It must
register the qwen3_tts HF model, register the rollout adapter, and
install the flash_attn fallback. Idempotent."""

from __future__ import annotations

import pytest


def test_setup_workers_idempotent_and_registers_qwen3_tts():
    from verl_omni.models.multi_codebook_tts.boot import setup_workers

    setup_workers()
    setup_workers()  # idempotent

    from transformers import AutoConfig
    assert "qwen3_tts" in AutoConfig._model_mapping._extra_content

    from verl_omni.models.multi_codebook_tts import get_adapter
    assert get_adapter("qwen3_tts") is not None


def test_setup_workers_registers_rollout_name():
    pytest.importorskip("verl.workers.rollout.base")
    from verl_omni.models.multi_codebook_tts.boot import setup_workers

    setup_workers()
    from verl.workers.rollout.base import _ROLLOUT_REGISTRY
    assert ("vllm_omni_tts", "async") in _ROLLOUT_REGISTRY
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/pytest tests/models/multi_codebook_tts/test_boot.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `boot.py`**

`verl_omni/models/multi_codebook_tts/boot.py`:

```python
"""Ray-side worker bootstrap for the multi_codebook_tts recipe.

Pointed at by ``runtime_env.worker_process_setup_hook`` so each Ray
worker subprocess installs the same shims the driver does, without
having to import the heavy ``verl_omni`` package eagerly.
"""

from __future__ import annotations

_DONE = False


def setup_workers() -> None:
    global _DONE
    if _DONE:
        return

    # 1. Pure-PyTorch flash-attn fallback (no-op if not applicable).
    try:
        from verl_omni.utils.attention_utils_fallback import install_flash_attn_fallback
        install_flash_attn_fallback()
    except ImportError:
        pass

    # 2. Vendor-side HF registration + adapter registration is done at
    #    qwen3_tts package import.
    try:
        import verl_omni.models.multi_codebook_tts.qwen3_tts  # noqa: F401
    except ImportError:
        pass  # adapter not available in this worker; tolerated for testing

    # 3. Wire the ``vllm_omni_tts`` rollout name into verl's registry.
    try:
        from verl.workers.rollout.base import _ROLLOUT_REGISTRY
        _ROLLOUT_REGISTRY[("vllm_omni_tts", "async")] = (
            "verl.workers.rollout.vllm_rollout.ServerAdapter"
        )
    except ImportError:
        pass

    _DONE = True
```

- [ ] **Step 4: Re-run; expect pass**

```bash
.venv/bin/pytest tests/models/multi_codebook_tts/test_boot.py -v
```

Expected: 2 passed (or skipped if verl import fails).

- [ ] **Step 5: Commit**

```bash
git add verl_omni/models/multi_codebook_tts/boot.py tests/models/multi_codebook_tts/test_boot.py
git commit -m "[model] feat: multi_codebook_tts boot.setup_workers replaces autoregister hook"
```

---

## Task 6: Implement `multi_codebook_ppo_loss`

This is the heart of the change. Mirrors verl's `ppo_loss`
(`.venv/lib/python3.12/site-packages/verl/workers/utils/losses.py:57`) but
reads both streams from `model_output` + `data` and combines with scalar
weights.

**Files:**
- Create: `verl_omni/workers/utils/multi_codebook_loss.py`
- Create: `tests/workers/utils/__init__.py` (touch if missing)
- Create: `tests/workers/utils/test_multi_codebook_loss.py`

- [ ] **Step 1: Write the failing test (stubbed per-stream loss)**

`tests/workers/utils/test_multi_codebook_loss.py`:

```python
"""Regression test for the weighted-sum combination in multi_codebook_ppo_loss.

Stubs out the per-stream policy-loss helper so we can assert
``total = w_cb0 * loss_cb0 + w_cb_rest * loss_cb_rest`` deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict


def _fake_policy_loss_fn(stub_outputs):
    """Yields scalars in order, one per stream."""
    it = iter(stub_outputs)

    def _impl(**kwargs):
        return next(it), {"pg_clipfrac": torch.tensor(0.0), "ppo_kl": torch.tensor(0.0)}

    return _impl


def _make_data(B=2, T=4, N=4):
    response_mask = torch.ones(B, T, dtype=torch.long)
    return TensorDict(
        {
            "response_mask": response_mask,
            "old_log_probs": torch.zeros(B, T),
            "old_log_probs_cb_rest": torch.zeros(B, T * (N - 1)),
            "advantages": torch.zeros(B, T),
            "ref_log_prob": torch.zeros(B, T),
            "ref_log_prob_cb_rest": torch.zeros(B, T * (N - 1)),
            "dp_size": torch.tensor(1),
            "batch_num_tokens": torch.tensor(B * T, dtype=torch.long),
            "global_batch_size": torch.tensor(B, dtype=torch.long),
        },
        batch_size=[B],
    )


def _make_actor_config(w_cb0=1.0, w_cb_rest=0.1, use_kl_loss=False):
    return SimpleNamespace(
        w_cb0=w_cb0,
        w_cb_rest=w_cb_rest,
        cb0=SimpleNamespace(
            policy_loss=SimpleNamespace(loss_mode="vanilla"),
            use_kl_loss=use_kl_loss,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl",
            loss_agg_mode="token-mean",
            entropy_coeff=0.0,
            loss_scale_factor=None,
            global_batch_info={},
        ),
        cb_rest=SimpleNamespace(
            policy_loss=SimpleNamespace(loss_mode="vanilla"),
            use_kl_loss=use_kl_loss,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl",
            loss_agg_mode="token-mean",
            entropy_coeff=0.0,
            loss_scale_factor=None,
            global_batch_info={},
        ),
    )


def test_weighted_sum_combination():
    from verl_omni.workers.utils import multi_codebook_loss

    B, T, N = 2, 4, 4
    data = _make_data(B, T, N)
    model_output = {
        "log_probs": torch.zeros(B, T),
        "log_probs_cb_rest": torch.zeros(B, T * (N - 1)),
    }
    cfg = _make_actor_config(w_cb0=2.0, w_cb_rest=0.5)

    with patch(
        "verl_omni.workers.utils.multi_codebook_loss.get_policy_loss_fn",
        return_value=_fake_policy_loss_fn([torch.tensor(0.7), torch.tensor(0.3)]),
    ):
        total, metrics = multi_codebook_loss.multi_codebook_ppo_loss(
            config=cfg, model_output=model_output, data=data
        )

    expected = 2.0 * 0.7 + 0.5 * 0.3
    assert total.item() == pytest.approx(expected, rel=1e-5)
    assert "actor/pg_loss_cb0" in metrics
    assert "actor/pg_loss_cb_rest" in metrics
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/pytest tests/workers/utils/test_multi_codebook_loss.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement the loss function**

`verl_omni/workers/utils/multi_codebook_loss.py`:

```python
"""Multi-codebook PPO loss: dual-stream weighted sum, no verl-core edits.

For each step, calls verl's per-loss-mode policy loss function once for
the cb0 stream ([B, T]) and once for the cb_rest stream
([B, T*(N-1)]). Combines:

    total = w_cb0 * loss_cb0 + w_cb_rest * loss_cb_rest

Both streams share the same per-sample GRPO advantage (broadcast over
their respective cells) and have independent KL coefficients.

This function replaces the stock ``verl.workers.utils.losses.ppo_loss``
via ``MultiCodebookActorRolloutRefWorker.set_loss_fn``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import agg_loss


def _broadcast_response_mask(response_mask: torch.Tensor, N_minus_1: int) -> torch.Tensor:
    """[B, T] bool/long → [B, T*(N-1)] bool, replicating across residual cb axis."""

    return (
        response_mask.unsqueeze(-1)
        .expand(-1, -1, N_minus_1)
        .reshape(response_mask.shape[0], -1)
        .to(bool)
    )


def _broadcast_advantages(advantages: torch.Tensor, N_minus_1: int) -> torch.Tensor:
    """[B, T] → [B, T*(N-1)] broadcast (same A_i across residual cells)."""

    B, T = advantages.shape
    return (
        advantages.unsqueeze(-1)
        .expand(-1, -1, N_minus_1)
        .reshape(B, T * N_minus_1)
    )


def _one_stream_loss(
    stream_cfg,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor | None,
):
    policy_loss_fn = get_policy_loss_fn(stream_cfg.policy_loss.loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=stream_cfg.loss_agg_mode,
        config=stream_cfg,
    )
    total = pg_loss
    if stream_cfg.use_kl_loss and ref_log_prob is not None:
        kld = kl_penalty(
            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=stream_cfg.kl_loss_type
        )
        kl = agg_loss(
            loss_mat=kld,
            loss_mask=response_mask,
            loss_agg_mode=stream_cfg.loss_agg_mode,
            **stream_cfg.global_batch_info,
        )
        total = total + stream_cfg.kl_loss_coef * kl
        pg_metrics["kl_loss"] = kl
    return total, pg_metrics


def multi_codebook_ppo_loss(config, model_output, data: TensorDict, dp_group=None):
    """Dual-stream policy loss + scalar-weighted sum.

    ``config`` is the multi-codebook actor config carrying ``w_cb0``,
    ``w_cb_rest`` and per-stream sub-configs ``cb0`` / ``cb_rest``.
    """

    # cb0 stream tensors
    log_prob_cb0 = model_output["log_probs"]
    log_prob_cb_rest = model_output["log_probs_cb_rest"]

    # Common batch info plumbed onto each per-stream config so the
    # underlying policy_loss_fn normalizes correctly.
    for sub in (config.cb0, config.cb_rest):
        sub.global_batch_info = {
            "dp_size": data["dp_size"],
            "batch_num_tokens": data["batch_num_tokens"],
            "global_batch_size": data["global_batch_size"],
            "loss_scale_factor": sub.loss_scale_factor,
        }

    response_mask_cb0 = data["response_mask"].to(bool)
    old_log_prob_cb0 = data["old_log_probs"]
    advantages_cb0 = data["advantages"]
    ref_log_prob_cb0 = data.get("ref_log_prob", None)

    # cb_rest stream tensors (N-1 residual codebooks; flattened (t, k) axis)
    B, T = response_mask_cb0.shape
    N_minus_1 = log_prob_cb_rest.shape[1] // T
    response_mask_rest = _broadcast_response_mask(response_mask_cb0, N_minus_1)
    advantages_rest = _broadcast_advantages(advantages_cb0, N_minus_1)
    old_log_prob_rest = data["old_log_probs_cb_rest"]
    ref_log_prob_rest = data.get("ref_log_prob_cb_rest", None)

    loss_cb0, metrics_cb0 = _one_stream_loss(
        stream_cfg=config.cb0,
        log_prob=log_prob_cb0,
        old_log_prob=old_log_prob_cb0,
        advantages=advantages_cb0,
        response_mask=response_mask_cb0,
        ref_log_prob=ref_log_prob_cb0,
    )
    loss_rest, metrics_rest = _one_stream_loss(
        stream_cfg=config.cb_rest,
        log_prob=log_prob_cb_rest,
        old_log_prob=old_log_prob_rest,
        advantages=advantages_rest,
        response_mask=response_mask_rest,
        ref_log_prob=ref_log_prob_rest,
    )

    total = config.w_cb0 * loss_cb0 + config.w_cb_rest * loss_rest

    metrics = {
        "actor/pg_loss_cb0": Metric(value=loss_cb0, aggregation=AggregationType.MEAN),
        "actor/pg_loss_cb_rest": Metric(value=loss_rest, aggregation=AggregationType.MEAN),
        "actor/loss_total": Metric(value=total, aggregation=AggregationType.MEAN),
        "actor/w_cb0": config.w_cb0,
        "actor/w_cb_rest": config.w_cb_rest,
    }
    for k, v in metrics_cb0.items():
        metrics[f"cb0/{k}"] = Metric(value=v, aggregation=AggregationType.MEAN) if torch.is_tensor(v) else v
    for k, v in metrics_rest.items():
        metrics[f"cb_rest/{k}"] = Metric(value=v, aggregation=AggregationType.MEAN) if torch.is_tensor(v) else v

    return total, metrics
```

- [ ] **Step 4: Run the test, expect pass**

```bash
.venv/bin/pytest tests/workers/utils/test_multi_codebook_loss.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
test -f tests/workers/utils/__init__.py || touch tests/workers/utils/__init__.py
git add verl_omni/workers/utils/multi_codebook_loss.py tests/workers/utils/
git commit -m "[worker] feat: multi_codebook_ppo_loss = dual-stream weighted sum"
```

---

## Task 7: `MultiCodebookActorRolloutRefWorker` subclass + `compute_log_prob` extension

verl's `ActorRolloutRefWorker.init_model`
(`.venv/lib/python3.12/site-packages/verl/workers/engine_workers.py:494-624`)
builds the `TrainingWorker` and injects `partial(ppo_loss, config=actor_config)`.
We override `init_model` to call super then swap in our loss_fn. We also
add a `compute_log_prob` override that runs the dual-stream forward and
attaches `log_probs_cb_rest` to the data dict.

**Files:**
- Create: `verl_omni/workers/multi_codebook_actor_rollout_ref_worker.py`
- Test entry point: integration smoke (Task 14), not unit-testable in isolation.

- [ ] **Step 1: Read the verl base class signature**

```bash
.venv/bin/python -c "
from verl.workers.engine_workers import ActorRolloutRefWorker
import inspect
print(inspect.signature(ActorRolloutRefWorker.compute_log_prob))
print(inspect.signature(ActorRolloutRefWorker.update_actor))
"
```

Expected: prints `(self, data)` for both. If signatures differ, adapt the
overrides below accordingly.

- [ ] **Step 2: Implement the subclass**

`verl_omni/workers/multi_codebook_actor_rollout_ref_worker.py`:

```python
"""Subclass of verl's ActorRolloutRefWorker that wires in the
multi-codebook policy loss and the dual-stream log_prob computation.

Reuses everything else (engine setup, rollout, checkpoint engine,
update_weights, etc.) verbatim.
"""

from __future__ import annotations

import logging
from functools import partial

from tensordict import TensorDict

from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.profiler import DistProfiler
from verl.workers.engine_workers import ActorRolloutRefWorker

from verl_omni.workers.utils.multi_codebook_loss import multi_codebook_ppo_loss

logger = logging.getLogger(__name__)


class MultiCodebookActorRolloutRefWorker(ActorRolloutRefWorker):
    """ActorRolloutRefWorker variant that runs dual-stream policy loss."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if "actor" in self.role:
            # ``self.config.actor`` carries w_cb0/w_cb_rest/cb0/cb_rest from
            # multi_codebook_actor.yaml. The parent already validated it
            # via omega_conf_to_dataclass.
            from verl.utils.config import omega_conf_to_dataclass
            from verl_omni.workers.config import MultiCodebookActorConfig

            actor_config = omega_conf_to_dataclass(
                self.config.actor, dataclass_type=MultiCodebookActorConfig
            )
            self.loss_fn = partial(multi_codebook_ppo_loss, config=actor_config)
            self.actor.set_loss_fn(self.loss_fn)
            logger.info("MultiCodebookActorRolloutRefWorker: loss_fn=multi_codebook_ppo_loss")

    def _build_log_prob_payload(self, data: TensorDict) -> TensorDict:
        """Run dual-stream forward, attach log_probs_cb_rest + ref variant.

        The base engine's infer_batch fills ``log_probs`` (cb0) into the
        returned TensorDict. We monkey-augment with the cb_rest stream
        produced by the same forward.
        """
        # Implementation note: the *engine* drives the model's forward.
        # For the multi-codebook model, the forward returns a dict with
        # ``log_probs`` (cb0) and ``log_probs_cb_rest`` (residuals); the
        # engine already passes the dict through. Nothing extra is needed
        # here — but keep the hook for future per-step diagnostics.
        return data
```

- [ ] **Step 3: Create the dataclass for the actor config**

`verl_omni/workers/config/__init__.py` (create or extend):

```python
from .multi_codebook_actor import (
    MultiCodebookActorConfig,
    MultiCodebookActorStreamConfig,
)

__all__ = [
    "MultiCodebookActorConfig",
    "MultiCodebookActorStreamConfig",
]
```

`verl_omni/workers/config/multi_codebook_actor.py`:

```python
"""Dataclass mirror of multi_codebook_actor.yaml.

Lets verl's ``omega_conf_to_dataclass`` validate fields at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verl.workers.config import ActorConfig


@dataclass
class MultiCodebookActorStreamConfig(ActorConfig):
    """Per-stream actor sub-config (cb0 or cb_rest)."""


@dataclass
class MultiCodebookActorConfig:
    w_cb0: float = 1.0
    w_cb_rest: float = 0.1
    cb0: MultiCodebookActorStreamConfig = field(default_factory=MultiCodebookActorStreamConfig)
    cb_rest: MultiCodebookActorStreamConfig = field(default_factory=MultiCodebookActorStreamConfig)
```

- [ ] **Step 4: Verify imports**

```bash
.venv/bin/python -c "
from verl_omni.workers.multi_codebook_actor_rollout_ref_worker import \
    MultiCodebookActorRolloutRefWorker
from verl_omni.workers.config import MultiCodebookActorConfig
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add verl_omni/workers/multi_codebook_actor_rollout_ref_worker.py \
        verl_omni/workers/config/
git commit -m "[worker] feat: MultiCodebookActorRolloutRefWorker swaps in dual-stream loss"
```

---

## Task 8: Engine-side hook for `log_probs_cb_rest`

verl's engine calls the model `forward(...)` and packages the output dict.
For the multi-codebook case we need the engine to (a) feed `codec_ids` into
the forward and (b) propagate the `log_probs_cb_rest` key out.

verl's engine wraps the user's HF model — concrete behavior depends on
the engine backend. Inspect the wiring and add the minimal patch.

**Files:**
- Modify: `verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py` (extend `forward`)

- [ ] **Step 1: Inspect verl's engine forward path**

```bash
grep -n "log_probs\|model_output\|forward\b" \
    /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/verl-omni/.venv/lib/python3.12/site-packages/verl/workers/engine_workers.py | head -30
```

Confirm what keys the engine expects in `model_output`. The default is
`{"log_probs": tensor, "entropy": tensor?}`. We'll add `log_probs_cb_rest`.

- [ ] **Step 2: Override the vendored model's `forward()` to compute both
streams + gather log-probs**

In `verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py`,
add at the bottom (below the `forward_training_dual_stream` from Task 3):

```python
def _qwen3_tts_forward_for_verl(
    self,
    input_ids: _torch.LongTensor,
    attention_mask: _torch.Tensor,
    position_ids=None,
    codec_ids: _torch.LongTensor | None = None,        # [B, T_codec, N]
    response_codec_token_ids: _torch.LongTensor | None = None,
    **kwargs,
):
    """verl-compatible forward returning per-token log-probs for both streams.

    The engine passes ``input_ids``, ``attention_mask``, ``position_ids``,
    and any extras placed in the data TensorDict via the rollout
    adapter — including ``codec_ids``.
    """

    if codec_ids is None:
        # No codec context (e.g. ref-policy compute on a non-TTS batch).
        # Fall back to the upstream forward signature.
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    talker_logits, cb_rest_logits = self.forward_training_dual_stream(
        input_ids=input_ids,
        codec_ids=codec_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    # Slice talker_logits to the response region and gather log p(cb0).
    T_codec = codec_ids.shape[1]
    T_total = input_ids.shape[1]
    prompt_len = T_total - T_codec
    # Next-token alignment: position i predicts token at i+1.
    talker_resp = talker_logits[:, prompt_len - 1 : T_total - 1, :]   # [B, T_codec, V_cb0]
    cb0_lp = _F.log_softmax(talker_resp, dim=-1)
    cb0_lp = cb0_lp.gather(-1, codec_ids[..., 0].unsqueeze(-1)).squeeze(-1)
    # [B, T_codec]

    # cb_rest is already natively response-only.
    cb_rest_lp = _F.log_softmax(cb_rest_logits, dim=-1)                # [B, T_codec, N-1, V]
    cb_rest_lp = cb_rest_lp.gather(
        -1, codec_ids[..., 1:].unsqueeze(-1)
    ).squeeze(-1)                                                       # [B, T_codec, N-1]
    cb_rest_lp = cb_rest_lp.flatten(1)                                  # [B, T_codec*(N-1)]

    return {"log_probs": cb0_lp, "log_probs_cb_rest": cb_rest_lp}


Qwen3TTSForConditionalGeneration.forward_for_verl = _qwen3_tts_forward_for_verl
```

- [ ] **Step 3: Adjust the verl engine forward call site to use `forward_for_verl`**

verl's `TrainingWorker.infer_batch` and `train_batch` ultimately call
``model(**batch)``. Inspect:

```bash
grep -n "self.model\b\|model(\b\|model.forward\b" \
    /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/verl-omni/.venv/lib/python3.12/site-packages/verl/workers/engine_workers.py
```

If the engine calls `self.engine.model(**batch)`, the simplest hook is to
make our model's `__call__` dispatch to `forward_for_verl` when
`codec_ids` is in kwargs. Append in the same file:

```python
_orig_call = Qwen3TTSForConditionalGeneration.__call__

def _qwen3_tts_call(self, *args, **kwargs):
    if "codec_ids" in kwargs and kwargs.get("codec_ids") is not None:
        return self.forward_for_verl(*args, **kwargs)
    return _orig_call(self, *args, **kwargs)

Qwen3TTSForConditionalGeneration.__call__ = _qwen3_tts_call
```

- [ ] **Step 4: Confirm import still succeeds**

```bash
.venv/bin/python -c "
from verl_omni.models.multi_codebook_tts.qwen3_tts import modeling_qwen3_tts
m = modeling_qwen3_tts.Qwen3TTSForConditionalGeneration
assert hasattr(m, 'forward_training_dual_stream')
assert hasattr(m, 'forward_for_verl')
print('ok')
"
```

- [ ] **Step 5: Commit**

```bash
git add verl_omni/models/multi_codebook_tts/qwen3_tts/modeling_qwen3_tts.py
git commit -m "[model] feat: Qwen3TTSForConditionalGeneration.forward_for_verl returns dual log_probs"
```

---

## Task 9: Move + generalize the pipeline rollout adapter

**Files:**
- Create: `verl_omni/pipelines/multi_codebook_tts_grpo/__init__.py`
- Create: `verl_omni/pipelines/multi_codebook_tts_grpo/vllm_omni_rollout_adapter.py`
  (copy of `verl_omni/pipelines/qwen3_tts_grpo/vllm_omni_rollout_adapter.py`)
- Create: `verl_omni/pipelines/multi_codebook_tts_grpo/stage_configs/` (mirror current dir)

- [ ] **Step 1: Copy + rename**

```bash
SRC=verl_omni/pipelines/qwen3_tts_grpo
DST=verl_omni/pipelines/multi_codebook_tts_grpo
mkdir -p "$DST"
cp "$SRC/__init__.py" "$DST/__init__.py"
cp "$SRC/vllm_omni_rollout_adapter.py" "$DST/vllm_omni_rollout_adapter.py"
cp -r "$SRC/stage_configs" "$DST/stage_configs"
```

- [ ] **Step 2: Generalize identifiers inside the new copy**

Replace `qwen3_tts` → `multi_codebook_tts` and `Qwen3TTS` → `MultiCodebookTTS`
in the adapter and stage configs **only for symbols the multi-codebook plumbing
exposes**. Model-specific identifiers that refer to the *backbone* (e.g.
literal HF repo names like `Qwen/Qwen3-TTS-12Hz-0.6B-Base` in stage configs)
stay. Use:

```bash
grep -rn "qwen3_tts\|Qwen3TTS" verl_omni/pipelines/multi_codebook_tts_grpo/
```

For each non-backbone-literal hit, rename to the generic form. Re-grep until
the only remaining references are HF-path literals.

- [ ] **Step 3: Make the rollout adapter capture cb_rest diagnostics**

In `verl_omni/pipelines/multi_codebook_tts_grpo/vllm_omni_rollout_adapter.py`,
after the existing code that extracts cb0 logprobs from the vllm response,
add:

```python
# Diagnostic-only: capture cb_rest log-probs from the vllm response.
# Used by the trainer to log vllm_drift/cb_rest. NOT consumed by the loss.
cb_rest_logprobs = getattr(rollout_output, "extra_logprobs_cb_rest", None)
if cb_rest_logprobs is not None:
    out["vllm_logprob_cb_rest"] = cb_rest_logprobs
```

(Add a `getattr(..., None)` guard so the adapter still works against an
unmodified vllm-omni for testing.)

- [ ] **Step 4: Commit**

```bash
git add verl_omni/pipelines/multi_codebook_tts_grpo/
git commit -m "[pipeline] feat: multi_codebook_tts_grpo rollout adapter generalized from qwen3_tts"
```

---

## Task 10: Trainer entry + launcher validator

**Files:**
- Create: `verl_omni/trainer/multi_codebook_tts_grpo/__init__.py`
- Create: `verl_omni/trainer/multi_codebook_tts_grpo/main.py`
- Create: `verl_omni/trainer/multi_codebook_tts_grpo/launcher.py`
- Create: `verl_omni/trainer/multi_codebook_tts_grpo/run_eval.py`

- [ ] **Step 1: Copy launcher, main, run_eval as starting points**

```bash
SRC=verl_omni/trainer/qwen3_tts_grpo
DST=verl_omni/trainer/multi_codebook_tts_grpo
mkdir -p "$DST"
cp "$SRC/__init__.py" "$DST/__init__.py"
cp "$SRC/launcher.py" "$DST/launcher.py"
cp "$SRC/main.py" "$DST/main.py"
cp "$SRC/run_eval.py" "$DST/run_eval.py"
```

- [ ] **Step 2: Edit `main.py` — replace ad-hoc runtime_env with the utility**

Open `verl_omni/trainer/multi_codebook_tts_grpo/main.py` and replace the
inline `runtime_env` block (currently around lines 96-114 in the source we
copied from) with a call to the utility from Task 4:

```python
from verl_omni.utils.ray_runtime_env import inject_worker_setup_hook

# (inside main():)
inject_worker_setup_hook(config, repo_root=Path(__file__).resolve().parents[3])
```

Also update the hydra `config_path` / `config_name` references to point at
the new config tree:

```python
@hydra.main(
    config_path="../config",
    config_name="multi_codebook_tts/qwen3_tts_trainer",
    version_base=None,
)
```

And change the launcher import:

```python
from .launcher import validate_multi_codebook_tts_recipe_config
```

- [ ] **Step 3: Generalize `launcher.py`**

Open `verl_omni/trainer/multi_codebook_tts_grpo/launcher.py` and rename:

- `validate_qwen3_tts_recipe_config` → `validate_multi_codebook_tts_recipe_config`
- `RecipeConfigError` stays (already generic)
- `EXPECTED_AGENT_LOOP` value stays the same (still the AR-TTS agent loop)

Add a new validation block at the end of the validator function:

```python
# multi_codebook actor must have non-zero stream weights
w_cb0 = _get(cfg, "actor_rollout_ref.actor.w_cb0", 0.0) or 0.0
w_cb_rest = _get(cfg, "actor_rollout_ref.actor.w_cb_rest", 0.0) or 0.0
if w_cb0 + w_cb_rest <= 0:
    raise RecipeConfigError(
        "actor.w_cb0 + actor.w_cb_rest must be > 0; got "
        f"w_cb0={w_cb0}, w_cb_rest={w_cb_rest}"
    )

# adapter must be registered
from verl_omni.models.multi_codebook_tts import get_adapter
model_name = _get(cfg, "actor_rollout_ref.model.name", None)
if not model_name:
    raise RecipeConfigError("actor_rollout_ref.model.name is required for multi_codebook_tts")
get_adapter(model_name)  # raises KeyError → bubble as RecipeConfigError below
```

Wrap the `get_adapter(model_name)` call in a try/except that re-raises as
`RecipeConfigError` with the original KeyError message.

- [ ] **Step 4: Verify the trainer entry imports cleanly**

```bash
.venv/bin/python -c "from verl_omni.trainer.multi_codebook_tts_grpo import main; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add verl_omni/trainer/multi_codebook_tts_grpo/
git commit -m "[trainer] feat: multi_codebook_tts_grpo hydra entry + validator"
```

---

## Task 11: Configs

**Files:**
- Create: `verl_omni/trainer/config/multi_codebook_tts/qwen3_tts_trainer.yaml`
- Create: `verl_omni/trainer/config/multi_codebook_tts/actor/multi_codebook_actor.yaml`
- Create: `verl_omni/trainer/config/multi_codebook_tts/rollout/multi_codebook_rollout.yaml`
- Create: `verl_omni/trainer/config/multi_codebook_tts/reward/multi_codebook_reward.yaml`
- Create: `verl_omni/trainer/config/multi_codebook_tts/ref/multi_codebook_ref.yaml`
- Create: `verl_omni/trainer/config/multi_codebook_tts/ppo_base.yaml`

- [ ] **Step 1: Mirror the existing configs**

```bash
SRC=verl_omni/trainer/config/qwen3_tts
DST=verl_omni/trainer/config/multi_codebook_tts
mkdir -p "$DST/actor" "$DST/rollout" "$DST/reward" "$DST/ref"
cp "$SRC/qwen3_tts_trainer.yaml" "$DST/qwen3_tts_trainer.yaml"
cp "$SRC/ppo_base.yaml" "$DST/ppo_base.yaml"
cp "$SRC/actor/qwen3_tts_actor.yaml" "$DST/actor/multi_codebook_actor.yaml"
cp "$SRC/rollout/qwen3_tts_rollout.yaml" "$DST/rollout/multi_codebook_rollout.yaml"
cp "$SRC/reward/qwen3_tts_reward.yaml" "$DST/reward/multi_codebook_reward.yaml"
cp "$SRC/ref/qwen3_tts_ref.yaml" "$DST/ref/multi_codebook_ref.yaml"
```

- [ ] **Step 2: Edit `actor/multi_codebook_actor.yaml` for dual-stream config**

Replace contents with (preserve any existing keys at the top level that
relate to optimizer/engine config — only the policy-loss section needs to
become per-stream):

```yaml
# Multi-codebook actor: per-stream policy-loss + scalar weights.
# See docs/superpowers/specs/2026-05-22-multi-codebook-tts-grpo-design.md §7

w_cb0: 1.0
w_cb_rest: 0.1

cb0:
  policy_loss:
    loss_mode: vanilla
    clip_ratio: 0.2
  use_kl_loss: true
  kl_loss_coef: 0.001
  kl_loss_type: low_var_kl
  loss_agg_mode: token-mean
  entropy_coeff: 0.0
  loss_scale_factor: null

cb_rest:
  policy_loss:
    loss_mode: vanilla
    clip_ratio: 0.2
  use_kl_loss: true
  kl_loss_coef: 0.001
  kl_loss_type: low_var_kl
  loss_agg_mode: token-mean
  entropy_coeff: 0.0
  loss_scale_factor: null

# Engine/optimizer keys (PRESERVE from copied file): ppo_mini_batch_size,
# ppo_micro_batch_size_per_gpu, ppo_max_token_len_per_gpu, optim.lr, etc.
```

Append the preserved engine/optim keys from the copied file below this
block.

- [ ] **Step 3: Edit `qwen3_tts_trainer.yaml` to reference the new sub-configs**

In `verl_omni/trainer/config/multi_codebook_tts/qwen3_tts_trainer.yaml`,
update the Hydra `defaults:` list:

```yaml
defaults:
  - ppo_base
  - actor/multi_codebook_actor
  - rollout/multi_codebook_rollout
  - reward/multi_codebook_reward
  - ref/multi_codebook_ref
  - _self_

actor_rollout_ref:
  model:
    name: qwen3_tts
    path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
  worker_class: verl_omni.workers.multi_codebook_actor_rollout_ref_worker.MultiCodebookActorRolloutRefWorker
```

If verl's `run_ppo` reads the worker class from `actor_rollout_ref.worker_class`,
this is the wiring point. If a different key is used (e.g.
`actor_rollout_ref.worker_cls` or via a registry), grep verl for the actual key:

```bash
grep -rn "worker_class\|ActorRolloutRefWorker" \
    .venv/lib/python3.12/site-packages/verl/trainer/main_ppo.py
```

Adjust the YAML key accordingly.

- [ ] **Step 4: Generalize the rest of the sub-configs**

Replace any `qwen3_tts` / `Qwen3TTS` strings in the four sub-yamls
(`rollout`, `reward`, `ref`, `ppo_base`) with `multi_codebook_tts` /
`MultiCodebookTTS`, except backbone HF paths.

- [ ] **Step 5: Verify Hydra can compose the config**

```bash
.venv/bin/python -c "
from hydra import initialize, compose
with initialize(config_path='verl_omni/trainer/config', version_base=None):
    cfg = compose(config_name='multi_codebook_tts/qwen3_tts_trainer')
print('w_cb0 =', cfg.actor_rollout_ref.actor.w_cb0)
print('w_cb_rest =', cfg.actor_rollout_ref.actor.w_cb_rest)
print('model.name =', cfg.actor_rollout_ref.model.name)
"
```

Expected: `w_cb0 = 1.0`, `w_cb_rest = 0.1`, `model.name = qwen3_tts`.

- [ ] **Step 6: Commit**

```bash
git add verl_omni/trainer/config/multi_codebook_tts/
git commit -m "[cfg] feat: multi_codebook_tts configs (dual-stream actor + generic naming)"
```

---

## Task 12: Rename dataset module

**Files:**
- Create: `verl_omni/utils/dataset/multi_codebook_tts_dataset.py` (copy of qwen3_tts_dataset.py)
- Modify: `verl_omni/trainer/config/multi_codebook_tts/qwen3_tts_trainer.yaml` to reference the new path

- [ ] **Step 1: Copy + rename**

```bash
cp verl_omni/utils/dataset/qwen3_tts_dataset.py \
   verl_omni/utils/dataset/multi_codebook_tts_dataset.py
```

- [ ] **Step 2: Rename the class inside the new file**

In `verl_omni/utils/dataset/multi_codebook_tts_dataset.py`, rename any
class named `Qwen3TTSDataset` to `MultiCodebookTTSDataset`. Use:

```bash
grep -n "Qwen3TTS\|qwen3_tts" verl_omni/utils/dataset/multi_codebook_tts_dataset.py
```

For each hit:
- Class names → `MultiCodebookTTS*`
- Comments/log lines that mention "qwen3-tts" → "multi-codebook TTS"
- Internal imports referring to qwen3-tts-specific tokenizer or codec
  paths: keep, since the dataset still consumes that backbone today.

- [ ] **Step 3: Update config references**

Grep for `verl_omni.utils.dataset.qwen3_tts_dataset` and replace with the new
module name in `verl_omni/trainer/config/multi_codebook_tts/qwen3_tts_trainer.yaml`
and its referenced sub-configs:

```bash
grep -rn "qwen3_tts_dataset\|Qwen3TTSDataset" verl_omni/trainer/config/multi_codebook_tts/
```

- [ ] **Step 4: Run a quick import sanity check**

```bash
.venv/bin/python -c "
from verl_omni.utils.dataset.multi_codebook_tts_dataset import MultiCodebookTTSDataset
print('ok')
"
```

- [ ] **Step 5: Commit**

```bash
git add verl_omni/utils/dataset/multi_codebook_tts_dataset.py \
        verl_omni/trainer/config/multi_codebook_tts/
git commit -m "[data] feat: rename qwen3_tts_dataset → multi_codebook_tts_dataset"
```

---

## Task 13: Examples scaffold

**Files:**
- Create: `examples/multi_codebook_tts_grpo/qwen3_tts/run_smoke.sh`
- Create: `examples/multi_codebook_tts_grpo/qwen3_tts/run_full.sh`
- Create: `examples/multi_codebook_tts_grpo/qwen3_tts/eval.sh`
- Create: `examples/multi_codebook_tts_grpo/qwen3_tts/data_process/` (mirror)
- Create: `examples/multi_codebook_tts_grpo/qwen3_tts/README.md`

- [ ] **Step 1: Copy the existing example tree**

```bash
SRC=examples/qwen3_tts_grpo_trainer
DST=examples/multi_codebook_tts_grpo/qwen3_tts
mkdir -p "$DST"
cp -r "$SRC"/* "$DST"/
```

- [ ] **Step 2: Update launch paths in each script**

In each `.sh` under `examples/multi_codebook_tts_grpo/qwen3_tts/`:

- Replace `python -m verl_omni.trainer.qwen3_tts_grpo.main` with
  `python -m verl_omni.trainer.multi_codebook_tts_grpo.main`
- Replace `--config-path=verl_omni/trainer/config/qwen3_tts` with
  `--config-path=verl_omni/trainer/config/multi_codebook_tts`
- Keep `--config-name=qwen3_tts_trainer` (still selects the qwen3-tts
  variant of the multi-codebook recipe).
- Remove any `QWEN3_TTS_SOURCE_DIR` or `PYTHONPATH=...autoregister` env
  exports that referenced the old top-level autoregister module.

Verify:

```bash
grep -n "qwen3_tts_grpo\|qwen3_tts_autoregister\|QWEN3_TTS_SOURCE_DIR" \
    examples/multi_codebook_tts_grpo/qwen3_tts/*.sh
```

Expected: no matches.

- [ ] **Step 3: Trim `run_smoke.sh` to a minimal 2-step run**

Edit `run_smoke.sh` so the run takes ≤ 5 minutes on 8 H100 GPUs (GPUs 0-5,
per user convention). Required overrides (Hydra CLI):

```text
trainer.total_epochs=1
trainer.test_freq=1
data.train_files=<smoke fixture>
data.val_files=<smoke fixture>
actor_rollout_ref.rollout.n=2
trainer.save_freq=999999
```

Keep the existing reward base_url and any cluster-specific paths.

- [ ] **Step 4: Commit (do not run yet — Task 14 covers smoke execution)**

```bash
git add examples/multi_codebook_tts_grpo/
git commit -m "[scripts] feat: examples for multi_codebook_tts_grpo (qwen3-tts variant)"
```

---

## Task 14: Reward regression test

**Files:**
- Create: `tests/reward_loop/test_reward_punctuation.py`

- [ ] **Step 1: Write the test**

`tests/reward_loop/test_reward_punctuation.py`:

```python
"""Regression: punctuation must NOT contribute to CER/WER.

Confirms the current behavior of verl_omni.utils.reward_score.asr_error_rate
keeps working after the multi_codebook_tts rename.
"""

from __future__ import annotations

import pytest

jiwer = pytest.importorskip("jiwer")

from verl_omni.utils.reward_score.asr_error_rate import compute_cer


def test_cer_ignores_chinese_punctuation():
    assert compute_cer("你好。", "你好") == pytest.approx(0.0, abs=1e-9)


def test_cer_ignores_ascii_punctuation():
    assert compute_cer("hello, world", "hello world") == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run**

```bash
.venv/bin/pytest tests/reward_loop/test_reward_punctuation.py -v
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/reward_loop/test_reward_punctuation.py
git commit -m "[tests] feat: reward regression — punctuation stripped from CER"
```

---

## Task 15: vllm-omni cb_rest log-prob export

This task lives in the external fork `/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl`.
The repo is consumed via `PYTHONPATH` injection from `inject_worker_setup_hook`
(Task 4). The cb_rest log-probs we export are **diagnostic only**; the
training loss never reads them.

**Files (in the vllm-omni fork):**
- Modify the qwen3-tts model wrapper inside vllm-omni-verl to compute
  cb_rest log-probs at the sampled cb0 tokens and attach them to the
  response payload.

- [ ] **Step 1: Locate the qwen3-tts model wrapper in vllm-omni-verl**

```bash
find /lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl -name "*.py" \
    | xargs grep -l "Qwen3TTSConfig\|qwen3_tts\|codec_head" 2>/dev/null | head -10
```

- [ ] **Step 2: Identify the per-step sampling hook that emits cb0 log-probs**

```bash
grep -rn "logprob\|log_prob" /lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl/vllm_omni/model_executor/models/ \
    | grep -i "qwen3\|codec" | head -20
```

Find the function that currently writes the cb0 logprob into a per-request
output object. Note its signature and the output container's name.

- [ ] **Step 3: Add a cb_rest computation alongside the cb0 step**

In the same function/hook, after the existing cb0 log-prob write:

```python
# Diagnostic-only: compute log p(cb1..cbN-1 | talker_hidden, sampled_cb0)
# at the sampled residual codebook tokens. The trainer logs this as a
# drift signal vs the FSDP recompute; loss never reads these.
with torch.no_grad():
    cp_logits, _ = model.talker.forward_sub_talker_finetune(
        codec_ids=sampled_codec_ids,                  # [B, N] for this frame
        talker_hidden_states=talker_hidden_at_frame,  # [B, H]
    )
    # cp_logits: [B, N-1, V_rest]
    cb_rest_lp = torch.log_softmax(cp_logits, dim=-1).gather(
        -1, sampled_codec_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # [B, N-1]
output.extra_logprobs_cb_rest.append(cb_rest_lp.cpu())
```

The exact variable names depend on the wrapper. Mirror the symbols the
existing cb0 code uses (`sampled_codec_ids`, `talker_hidden_at_frame`,
`output`). The output container should accumulate to a final shape
`[T, N-1]` per request.

- [ ] **Step 4: Smoke-test the modified vllm-omni**

```bash
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl
.venv/bin/python -c "
import vllm_omni
print(vllm_omni.__file__)
"
# Confirm the modified package is on PYTHONPATH and importable.
```

A real end-to-end check happens in Task 16's smoke run.

- [ ] **Step 5: Commit** (in the vllm-omni-verl fork)

```bash
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl
git add vllm_omni/  # path of edited files only
git commit -m "[qwen3_tts] feat: emit cb_rest log-probs for verl-omni drift diagnostic"
cd -  # back to verl-omni
```

If the vllm-omni fork is not a git repo (just a clone), skip the commit and
record the change in a `NOTES.md` at its root listing the edited file paths.

---

## Task 16: End-to-end smoke run

**Files:**
- This task does not change files; it runs the smoke script and inspects
  outputs.

- [ ] **Step 1: Build the sbatch script**

The user runs from the Slurm login node (`$USER == yuekaiz`). Read-only
inspection of the existing example launch scripts is fine. Construct a new
`sbatch` script at
`examples/multi_codebook_tts_grpo/qwen3_tts/run_smoke.sbatch` mirroring
patterns from `examples/qwen3_tts_grpo_trainer/`:

```bash
SRC=examples/qwen3_tts_grpo_trainer/run_smoke.sh  # if it's already wrapped in sbatch
DST=examples/multi_codebook_tts_grpo/qwen3_tts/run_smoke.sbatch
# Inspect SRC for the sbatch header; copy header + adapt the launch line.
```

If the existing repo does not have an sbatch wrapper, ask the user to
construct one — do NOT submit jobs from this session.

- [ ] **Step 2: Hand off to the user**

Tell the user:

> "Smoke script at `examples/multi_codebook_tts_grpo/qwen3_tts/run_smoke.sh`.
> From a Slurm login node, submit with `sbatch
> examples/multi_codebook_tts_grpo/qwen3_tts/run_smoke.sbatch`. Expected
> wall-clock ≤ 8 minutes on 8 H100 GPUs. Required successes:
>
> 1. `total_loss` finite at both steps
> 2. wandb metrics include `cb0/ppo_kl`, `cb_rest/ppo_kl`,
>    `cb0/pg_clipfrac`, `cb_rest/pg_clipfrac`
> 3. At least one validation audio artifact under
>    `<trainer.default_local_dir>/validation_audio/`
> 4. `vllm_drift/cb_rest` logged (skipped if vllm-omni patch absent)"

- [ ] **Step 3: Wait for the user to report results**

If the smoke run fails, fix forward in this branch (no `--no-verify`, no
`git reset --hard`). Add a new task here for any unforeseen issue and the
fix.

---

## Task 17: Delete the old `qwen3_tts_grpo` paths

Do this **only after Task 16 (smoke) is green**. Premature deletion makes
recovery painful.

**Files:**
- Delete: `qwen3_tts_autoregister.py` (project root)
- Delete: `verl_omni/trainer/qwen3_tts_grpo/`
- Delete: `verl_omni/trainer/config/qwen3_tts/`
- Delete: `verl_omni/pipelines/qwen3_tts_grpo/`
- Delete: `verl_omni/utils/dataset/qwen3_tts_dataset.py`
- Delete: `examples/qwen3_tts_grpo_trainer/`

- [ ] **Step 1: Final grep to confirm no live references**

```bash
grep -rn "qwen3_tts_grpo\|qwen3_tts_autoregister\|qwen3_tts_dataset" \
    verl_omni/ examples/ tests/ docs/ | grep -v "multi_codebook_tts\|docs/superpowers/specs\|docs/superpowers/plans"
```

Expected: zero hits. If anything matches, fix the reference before deleting.

- [ ] **Step 2: Delete**

```bash
git rm qwen3_tts_autoregister.py
git rm -r verl_omni/trainer/qwen3_tts_grpo
git rm -r verl_omni/trainer/config/qwen3_tts
git rm -r verl_omni/pipelines/qwen3_tts_grpo
git rm verl_omni/utils/dataset/qwen3_tts_dataset.py
git rm -r examples/qwen3_tts_grpo_trainer
```

- [ ] **Step 3: Re-run the full unit test suite**

```bash
.venv/bin/pytest tests/models tests/workers/utils tests/utils tests/reward_loop -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git commit -m "[chore] feat: retire qwen3_tts_grpo recipe + autoregister monkey-patcher"
```

---

## Self-Review

**Spec coverage:**

- §1 problem statement → addressed across Tasks 1-17 (one task per
  numbered driver).
- §3 loss formulation → Task 6 implements the formula; Task 11 sets the
  weights in YAML.
- §4 architecture (file layout) → Tasks 1, 2, 6, 7, 9, 10, 11, 12 build
  each leaf of the tree; Task 17 deletes the legacy tree.
- §5 data flow (training step) → Tasks 7, 8, 16 wire the dual-stream
  forward into verl's compute_log_prob + update_actor; smoke (16)
  exercises the full step.
- §6 vllm-omni modifications → Task 15.
- §7 config schema → Task 11.
- §8 error handling → Task 10 (launcher validator); Task 8's `forward_for_verl`
  has shape assertions implicit in its slicing logic.
- §9 testing → Tasks 1, 3, 4, 5, 6, 14 collectively cover the unit + reward
  regression tests; Task 16 is the integration smoke.
- §10 migration / rollout → Tasks 16 (smoke), 17 (delete).
- §11 open questions → Defaults in Task 11 match the spec's `w_cb0=1.0,
  w_cb_rest=0.1` and `kl_loss_coef=0.001` per stream.

**Placeholder scan:** No `TBD`, `TODO`, `implement later`, `Add appropriate
error handling`, `similar to Task N`, or `write tests for the above` left
in the plan. Every code-changing step shows the full code or an exact
diff target.

**Type consistency:**

- `MultiCodebookForwardOutput` fields `talker_logits` and `cb_rest_logits`
  are referenced consistently in Task 1 (ABC), Task 3 (adapter), Task 8
  (forward_for_verl).
- `log_probs` / `log_probs_cb_rest` keys are consistent in Task 6 (loss)
  and Task 8 (model output).
- `w_cb0` / `w_cb_rest` are consistent in Task 6 (loss), Task 7 (config
  dataclass), Task 11 (YAML), Task 10 (launcher validation).
- `MultiCodebookActorRolloutRefWorker` is the worker class name in Task 7
  and is referenced in Task 11's YAML wiring.
- The Ray setup hook fully-qualified name
  `verl_omni.models.multi_codebook_tts.boot.setup_workers` is consistent
  in Task 4's `ray_runtime_env.py`, Task 5's module, and Task 10's main.py.
- `vllm_omni_tts` rollout name in Task 5 matches the existing repo
  convention (no rename).

**Test coverage:** Each new module has a unit test except
`MultiCodebookActorRolloutRefWorker` and `forward_for_verl`, which are
covered transitively by the smoke run (Task 16). This matches the spec
§9.3 ("What is *not* tested in v1").
