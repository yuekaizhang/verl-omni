# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Integration test: AR-TTS worker actually invokes the validation logger.

This test wires a fake worker through ``_emit_validation_artifacts`` to
prove the recipe runtime (not just the standalone helper) writes the
AC-8 artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verl_omni.agent_loop.autoregressive_tts_agent_loop import (
    AutoRegressiveTTSAgentLoopOutput,
    AutoRegressiveTTSAgentLoopWorker,
)
from verl_omni.utils.validation_audio_logger import post_run_check_emitted_artifacts
from verl_omni.workers.rollout.replica import CompletionAudio


def _make_completion(idx: int) -> CompletionAudio:
    return CompletionAudio(
        sample_index=idx,
        codec_tokens=[1, 2, 3, 4],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        waveform=np.zeros(24000, dtype=np.float32),
        finish_reason="stop",
    )


def _make_output(idx: int, num_completions: int = 2) -> AutoRegressiveTTSAgentLoopOutput:
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

    metrics = AgentLoopMetrics(
        generate_sequences=0.0,
        tool_calls=0.0,
        compute_score=0.0,
        num_preempted=0,
    )
    return AutoRegressiveTTSAgentLoopOutput(
        prompt_ids=[0],
        prompt_text=f"text-{idx}",
        completions=[_make_completion(i) for i in range(num_completions)],
        sample_rate=24000,
        reward_score=0.5,
        num_turns=2,
        metrics=metrics,
        extra_fields={"ref_audio": None, "target_audio": None},
    )


def test_emit_validation_artifacts_writes_step_dir(tmp_path: Path) -> None:
    # Build a bare worker: bypass full __init__ since we only exercise the
    # validation-emission path.
    worker = AutoRegressiveTTSAgentLoopWorker.__new__(AutoRegressiveTTSAgentLoopWorker)
    from verl_omni.utils.validation_audio_logger import (
        log_validation_step,
        post_run_check_emitted_artifacts as _check,
    )

    worker._log_validation_step = log_validation_step
    worker._post_run_check_emitted_artifacts = _check
    worker._validation_output_dir = tmp_path
    worker._validation_step_counter = 0

    outputs = [_make_output(i) for i in range(3)]
    worker._emit_validation_artifacts(outputs)

    step_dirs = sorted(tmp_path.glob("validation_step_*"))
    assert len(step_dirs) == 1
    generated = list(step_dirs[0].glob("generated_*.wav"))
    assert len(generated) >= 4  # AC-8 minimum

    # AC-6 scalar metrics carried in metrics.json (AC-8 requirement that
    # Codex round-4 explicitly flagged).
    import json

    metrics = json.loads((step_dirs[0] / "metrics.json").read_text())
    for key in ("mean_reward", "mean_cer", "mean_duration_ratio", "policy_loss", "kl_loss"):
        assert key in metrics, f"AC-6 metric {key!r} missing from validation metrics.json"
    # mean_reward is always populated for non-empty outputs.
    assert metrics["mean_reward"] is not None
    # policy_loss / kl_loss are null at validation steps (no training update).
    assert metrics["policy_loss"] is None
    assert metrics["kl_loss"] is None

    good = post_run_check_emitted_artifacts(tmp_path)
    assert good == step_dirs


def test_emit_validation_artifacts_zero_samples_no_op(tmp_path: Path) -> None:
    worker = AutoRegressiveTTSAgentLoopWorker.__new__(AutoRegressiveTTSAgentLoopWorker)
    from verl_omni.utils.validation_audio_logger import (
        log_validation_step,
        post_run_check_emitted_artifacts as _check,
    )

    worker._log_validation_step = log_validation_step
    worker._post_run_check_emitted_artifacts = _check
    worker._validation_output_dir = tmp_path
    worker._validation_step_counter = 0

    # No outputs -> emit should return early without raising and without
    # creating an empty validation_step_* dir.
    worker._emit_validation_artifacts([])
    assert list(tmp_path.glob("validation_step_*")) == []
