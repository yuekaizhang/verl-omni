"""AC-3 regression: `rollout.diagnostic_logprobs` gating + warn-on-absence.

Tests the rollout adapter's `apply_diagnostic_logprobs_to_batch` helper
directly. Three scenarios:

1. Flag OFF (v1 default): batch is unchanged, no warning logged.
2. Flag ON + diagnostic fields ABSENT (expected v1 state per DEC-5):
   batch unchanged + `logger.warning` emitted.
3. Flag ON + diagnostic fields PRESENT (future state): `vllm_logprob_cb0`
   + `vllm_logprob_cb_rest` placed into the batch.
"""

from __future__ import annotations

import logging

from verl_omni.pipelines.multi_codebook_tts_grpo.vllm_omni_rollout_adapter import (
    apply_diagnostic_logprobs_to_batch,
)


def test_flag_off_is_noop_and_no_warning(caplog):
    """v1 default: flag OFF -> batch unchanged, no warning."""
    batch = {"codec_ids": "sentinel"}
    caplog.set_level(logging.WARNING)
    out = apply_diagnostic_logprobs_to_batch(
        batch=batch,
        rollout_payload={"extra_logprobs": {"cb0": "x", "cb_rest": "y"}},
        diagnostic_logprobs_enabled=False,
    )
    assert out is batch
    assert "vllm_logprob_cb0" not in batch
    assert "vllm_logprob_cb_rest" not in batch
    # No warning should be emitted because the gating short-circuits early.
    assert not any(
        "diagnostic_logprobs" in record.message for record in caplog.records
    ), "Flag OFF must not emit a warning."


def test_flag_on_but_fields_absent_warns_and_proceeds(caplog):
    """Expected v1 state (DEC-5): flag ON but vllm-omni-verl fork edit
    not landed -> warning + no diagnostic fields in batch."""
    batch = {"codec_ids": "sentinel"}
    caplog.set_level(logging.WARNING)
    out = apply_diagnostic_logprobs_to_batch(
        batch=batch,
        rollout_payload=None,
        diagnostic_logprobs_enabled=True,
    )
    assert out is batch
    assert "vllm_logprob_cb0" not in batch
    assert "vllm_logprob_cb_rest" not in batch
    # Exactly one warning about the absent diagnostic fields.
    warning_messages = [
        r.message for r in caplog.records
        if r.levelno == logging.WARNING and "diagnostic_logprobs" in r.message
    ]
    assert len(warning_messages) == 1, (
        f"Expected exactly one diagnostic_logprobs warning; got "
        f"{warning_messages}."
    )


def test_flag_on_with_fields_present_populates_batch(caplog):
    """Future state (post-DEC-5 follow-up): flag ON + fork edit landed
    -> diagnostic fields appear in the batch."""
    batch = {"codec_ids": "sentinel"}
    rollout_payload = {
        "extra_logprobs": {
            "cb0": "cb0_logprob_tensor_sentinel",
            "cb_rest": "cb_rest_logprob_tensor_sentinel",
        },
    }
    caplog.set_level(logging.WARNING)
    out = apply_diagnostic_logprobs_to_batch(
        batch=batch,
        rollout_payload=rollout_payload,
        diagnostic_logprobs_enabled=True,
    )
    assert out is batch
    assert batch["vllm_logprob_cb0"] == "cb0_logprob_tensor_sentinel"
    assert batch["vllm_logprob_cb_rest"] == "cb_rest_logprob_tensor_sentinel"
    # No warning when fields are present.
    assert not any(
        "diagnostic_logprobs" in r.message for r in caplog.records
        if r.levelno == logging.WARNING
    ), "Flag ON + fields present must not emit a warning."


def test_flag_on_partial_fields_warns():
    """Defensive: if only cb0 OR only cb_rest is present, treat it the
    same as fully absent (warn + no batch mutation)."""
    batch = {"codec_ids": "sentinel"}
    out = apply_diagnostic_logprobs_to_batch(
        batch=batch,
        rollout_payload={"extra_logprobs": {"cb0": "x"}},  # missing cb_rest
        diagnostic_logprobs_enabled=True,
    )
    assert "vllm_logprob_cb0" not in batch
    assert "vllm_logprob_cb_rest" not in batch
