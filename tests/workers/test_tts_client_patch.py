# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the LLMServerManager.get_client monkey-patch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager

# Trigger the patch via the canonical import path.
import verl_omni.workers.rollout.replica  # noqa: F401
from verl_omni.workers.rollout.autoregressive_tts_server_client import (
    AutoRegressiveTTSServerClient,
)


def _make_manager(rollout_name: str) -> LLMServerManager:
    # Construct a bare-bones LLMServerManager-like instance without actually
    # spinning anything up. The patched get_client only touches
    # self.rollout_config.name, self.server_addresses, self.server_handles,
    # and self.global_load_balancer.
    mgr = LLMServerManager.__new__(LLMServerManager)
    mgr.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace())
    mgr.rollout_config = SimpleNamespace(name=rollout_name, nnodes=1)
    mgr.server_addresses = ["host:1"]
    mgr.server_handles = [SimpleNamespace(name="dummy_handle")]
    mgr.global_load_balancer = SimpleNamespace(name="dummy_lb")
    return mgr


def test_patch_installed_idempotent() -> None:
    assert getattr(LLMServerManager, "_verl_omni_tts_patch_applied", False) is True


def test_get_client_returns_tts_client_for_vllm_omni_tts() -> None:
    mgr = _make_manager("vllm_omni_tts")
    client = mgr.get_client()
    assert isinstance(client, AutoRegressiveTTSServerClient)
    assert hasattr(client, "generate_tts")


def test_get_client_falls_through_for_other_rollouts() -> None:
    mgr = _make_manager("vllm")
    client = mgr.get_client()
    # Non-TTS rollout names must fall through to the upstream client.
    assert isinstance(client, LLMServerClient)
    assert not isinstance(client, AutoRegressiveTTSServerClient)


def test_tts_client_exposes_generate_tts_method() -> None:
    mgr = _make_manager("vllm_omni_tts")
    client = mgr.get_client()
    method = getattr(client, "generate_tts", None)
    assert callable(method)
