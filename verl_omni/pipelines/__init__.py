# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""verl_omni.pipelines package init.

Optional subpackages (each importing diffusion deps that the TTS path
does not need) are imported best-effort so a TTS-only worker image
without `diffusers` can still `import verl_omni.pipelines` without
crashing. The `_patch` Ulysses-mask shim is also gated since it
imports `diffusers` at module load.
"""

import logging
from types import ModuleType

_logger = logging.getLogger(__name__)

__all__: list[str] = []

# Best-effort: Ulysses mask fix for diffusers' Qwen-Image transformer.
try:
    from . import _patch  # noqa: F401 — apply Ulysses mask fix
except ImportError as _exc:
    _logger.debug(
        "verl_omni.pipelines: skipping Ulysses mask patch (%s: %s); "
        "this is expected on TTS-only environments without diffusers.",
        type(_exc).__name__, _exc,
    )

# Diffusion recipes (require diffusers). TTS recipes do not.
for _subpkg_name in ("qwen_image_flow_grpo", "qwen_image_mix_grpo"):
    try:
        _subpkg: ModuleType = __import__(
            f"verl_omni.pipelines.{_subpkg_name}", fromlist=["__all__"],
        )
    except ImportError as _exc:
        _logger.debug(
            "verl_omni.pipelines: skipping diffusion subpackage %r (%s: %s).",
            _subpkg_name, type(_exc).__name__, _exc,
        )
        continue
    globals()[_subpkg_name] = _subpkg
    __all__.extend(getattr(_subpkg, "__all__", []))

# TTS recipes (no diffusers needed). Both the legacy qwen3_tts_grpo path
# (deleted in task15) and the new multi_codebook_tts_grpo path are
# imported best-effort so a partial deletion mid-migration doesn't break
# package init.
for _subpkg_name in ("qwen3_tts_grpo", "multi_codebook_tts_grpo"):
    try:
        _subpkg = __import__(
            f"verl_omni.pipelines.{_subpkg_name}", fromlist=["__all__"],
        )
    except ImportError as _exc:
        _logger.debug(
            "verl_omni.pipelines: skipping TTS subpackage %r (%s: %s).",
            _subpkg_name, type(_exc).__name__, _exc,
        )
        continue
    globals()[_subpkg_name] = _subpkg
    __all__.extend(getattr(_subpkg, "__all__", []))
