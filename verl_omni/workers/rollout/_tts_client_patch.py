# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Monkey-patch upstream ``LLMServerManager.get_client`` to return the AR-TTS client.

Upstream ``verl.workers.rollout.llm_server.LLMServerManager.get_client`` is
hard-coded to return :class:`LLMServerClient` or :class:`FullyLLMServerClient`.
Neither exposes ``generate_tts``. There is no upstream hook (e.g. a
``llm_server_client_class`` config field) to swap the client.

We patch ``get_client`` so that, when the rollout config's ``name`` is
``vllm_omni_tts`` (the rollout registered by this recipe), it returns an
:class:`AutoRegressiveTTSServerClient`. All other rollout names fall through
to the original implementation, so the diffusion / vanilla LLM paths are
untouched.

The patch is idempotent and import-triggered: it applies the first time
``verl_omni.workers.rollout.replica`` (or any code that imports it) loads.
"""

from __future__ import annotations

import logging
from typing import Any

from verl.workers.rollout.llm_server import (
    FullyLLMServerClient,
    LLMServerClient,
    LLMServerManager,
)

from verl_omni.workers.rollout.autoregressive_tts_server_client import (
    AutoRegressiveTTSServerClient,
)

logger = logging.getLogger(__name__)

_TTS_ROLLOUT_NAME = "vllm_omni_tts"
_PATCH_ATTR = "_verl_omni_tts_patch_applied"


def _patched_get_client(self: LLMServerManager, fully_async: bool = False) -> Any:
    rollout_name = getattr(self.rollout_config, "name", None)
    if rollout_name == _TTS_ROLLOUT_NAME:
        servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
        return AutoRegressiveTTSServerClient(
            config=self.config,
            servers=servers,
            load_balancer_handle=self.global_load_balancer,
        )
    return _original_get_client(self, fully_async=fully_async)


_original_get_client = LLMServerManager.get_client


def apply() -> None:
    """Idempotent installer for the patch.

    Safe to call multiple times; only patches once per process.
    """

    if getattr(LLMServerManager, _PATCH_ATTR, False):
        return
    LLMServerManager.get_client = _patched_get_client  # type: ignore[assignment]
    setattr(LLMServerManager, _PATCH_ATTR, True)
    logger.info(
        "LLMServerManager.get_client patched to return AutoRegressiveTTSServerClient "
        "when rollout.name=%s.",
        _TTS_ROLLOUT_NAME,
    )


__all__ = ["apply"]
