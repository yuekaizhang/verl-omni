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

# Import-time side-effect: install the pure-PyTorch fallback for verl's
# flash-attn-only attention helpers. The patch is idempotent and only
# applies if verl + transformers are importable. See
# `verl_omni/utils/attention_utils_fallback.py` for rationale.
from . import attention_utils_fallback  # noqa: F401
