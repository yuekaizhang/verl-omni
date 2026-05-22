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
"""verl_omni package init.

Kept lightweight on purpose: importing `verl_omni` (e.g. so a Ray-worker
setup hook can reach `verl_omni.models.multi_codebook_tts.qwen3_tts`)
must NOT require diffusion deps that the TTS path does not need. Heavy
subpackage side-effects (pipeline / reward_loop / worker auto-registration)
are now wrapped in `try/except ImportError` so a missing optional dep
(e.g. `diffusers` on a TTS-only worker image) degrades gracefully rather
than failing the whole import.

Recipe-specific HF Auto* registration is the responsibility of each
recipe's adapter subpackage; see e.g.
`verl_omni/models/multi_codebook_tts/qwen3_tts/__init__.py`.
"""
import logging
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


_logger = logging.getLogger(__name__)


# Best-effort subpackage auto-registration. Each side-effect import is
# isolated so a missing optional dep in one subpackage doesn't break
# the others. The TTS recipe's lightweight Ray-worker setup hook
# (`multi_codebook_tts_setup.setup`) does its own targeted imports and
# does NOT rely on these blanket auto-imports.
for _subpkg in (
    "verl_omni.pipelines",
    "verl_omni.reward_loop",
    "verl_omni.workers.engine",
    "verl_omni.workers.rollout",
):
    try:
        __import__(_subpkg)
    except ImportError as _exc:
        _logger.debug(
            "verl_omni: optional subpackage %r is unavailable (%s: %s); "
            "skipping its auto-registration. This is fine for TTS-only "
            "workers that do not ship diffusion deps.",
            _subpkg, type(_exc).__name__, _exc,
        )
