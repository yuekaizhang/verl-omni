"""Single source of truth for building the Ray worker `runtime_env` dict.

Workers spawn outside the driver's Python import context and need an
explicit `PYTHONPATH` containing the repo root + a `worker_process_setup_hook`
that registers Qwen3-TTS (and future codec) backbones with the
transformers Auto* classes before any model load happens.

The qwen3-tts vLLM-Omni fork (currently consumed via PYTHONPATH, not
upstream-installed) is appended to the worker `PYTHONPATH` here when
`VLLM_OMNI_VERL_DIR` is set in the env. Once the upstream fork edits land
(deferred per DEC-5), the rollout adapter starts consuming
`extra_logprobs.cb_rest`; until then this just makes the import path
consistent across driver + workers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _repo_root() -> str:
    """Resolve the verl-omni repo root from this file's location."""
    # this file lives at verl-omni/verl_omni/utils/ray_runtime_env.py
    return str(Path(__file__).resolve().parents[2])


def build_runtime_env(
    *,
    setup_hook: str = "multi_codebook_tts_setup.setup",
    extra_env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a Ray `runtime_env` dict for `multi_codebook_tts_grpo` workers.

    Args:
        setup_hook: dotted path to a `setup()` function executed inside each
            worker subprocess BEFORE the worker imports the trainer's heavy
            verl_omni deps. The default points at the top-level
            `multi_codebook_tts_setup` module (a lightweight replacement for
            the old `qwen3_tts_autoregister.py` — see task15/task3 in the
            plan). The module must do all `verl_omni`-side imports lazily
            inside `setup()` so the hook target stays cheap to import.
        extra_env_vars: optional additional env vars to inject (merged
            into the worker `env_vars`; PYTHONPATH is always set/merged).

    Returns:
        A dict suitable for `ray.init(runtime_env=...)` or for assignment
        to `config.ray_kwargs.ray_init.runtime_env` in the Hydra config.
    """
    repo_root = _repo_root()

    # Build PYTHONPATH. Always include repo root. Optionally include the
    # vllm-omni-verl fork via VLLM_OMNI_VERL_DIR.
    pythonpath_parts: list[str] = [repo_root]
    fork_dir = os.environ.get("VLLM_OMNI_VERL_DIR")
    if fork_dir:
        pythonpath_parts.append(fork_dir)

    env_vars: dict[str, str] = {"PYTHONPATH": ":".join(pythonpath_parts)}
    if extra_env_vars:
        for k, v in extra_env_vars.items():
            if k == "PYTHONPATH":
                # Append, don't replace, so both the repo root and any
                # caller-provided paths survive.
                env_vars["PYTHONPATH"] = ":".join(
                    [env_vars["PYTHONPATH"], v]
                )
            else:
                env_vars[k] = v

    return {
        "worker_process_setup_hook": setup_hook,
        "env_vars": env_vars,
    }


__all__ = ["build_runtime_env"]
