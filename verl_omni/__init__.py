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
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


# Qwen3-TTS isn't shipped with stock transformers; the `qwen-tts` PyPI
# package supplies `Qwen3TTSConfig` / `Qwen3TTSForConditionalGeneration`.
# Register them with HF auto-classes in this (driver) process here.
# Ray-spawned worker subprocesses (TaskRunner, WorkerDict, ...) get
# registration via the `worker_process_setup_hook` runtime_env entry
# plumbed in by `verl_omni.trainer.qwen3_tts_grpo.main` — see that file
# for the wiring. Using the standalone `qwen3_tts_autoregister` module
# (not this package) for the worker hook avoids dragging the verl_omni
# package __init__ chain (~20s) into every Ray worker.
import qwen3_tts_autoregister  # noqa: E402

qwen3_tts_autoregister.setup()


# Import pipelines / rollout / reward loop / engines to auto-register them
import verl_omni.pipelines  # noqa: E402, F401
import verl_omni.reward_loop  # noqa: E402, F401
import verl_omni.workers.engine  # noqa: E402, F401
import verl_omni.workers.rollout  # noqa: E402, F401
