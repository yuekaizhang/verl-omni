"""AC-5.1 regression: `_gather_log_probs_cb0` slices `talker_logits` to the
response region per-sample under variable prompt lengths.

For sample `b` with prompt length `prompt_lens[b]`, the gathered cb0
log-prob at response frame `t` must equal a hand-computed
`log_softmax(talker_logits[b, prompt_lens[b] - 1 + t, :])[codec_ids_cb0[b, t]]`.

A one-token shift in the slice (e.g. `prompt_len:-1` instead of
`prompt_len-1:-1`) would predict the wrong codec frame and this test
catches it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from verl_omni.workers.actor.multi_codebook_dp_actor import (
    _gather_log_probs_cb0,
)


def test_logit_token_alignment_variable_prompt_lengths():
    # 3 samples with prompt lengths 5, 7, 11 (per AC-5.1 example).
    B = 3
    prompt_lens = torch.tensor([5, 7, 11], dtype=torch.long)
    T_codec = 4  # response frames per sample (could be padded with mask).
    V_cb0 = 32

    # Total sequence length must cover the longest prompt + response.
    T_total = int(prompt_lens.max().item()) + T_codec + 2  # +slack

    torch.manual_seed(0)
    talker_logits = torch.randn(B, T_total, V_cb0)
    target_cb0 = torch.randint(0, V_cb0, (B, T_codec))

    gathered = _gather_log_probs_cb0(
        talker_logits=talker_logits,
        target_cb0=target_cb0,
        prompt_lens=prompt_lens,
        T_codec=T_codec,
    )
    assert gathered.shape == (B, T_codec)

    # Hand-compute the expected log-prob at the first and last response
    # frame of each sample. Per next-token-aligned slicing convention:
    # gathered[b, t] = log_softmax(talker_logits[b, prompt_lens[b]-1+t, :])[target_cb0[b, t]]
    for b in range(B):
        for t in (0, T_codec - 1):
            sliced_logits = talker_logits[b, int(prompt_lens[b]) - 1 + t, :]
            expected = F.log_softmax(sliced_logits, dim=-1)[int(target_cb0[b, t])]
            actual = gathered[b, t]
            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5), (
                f"AC-5.1 mismatch at sample {b}, frame {t}: "
                f"gathered={actual.item()}, expected={expected.item()}. "
                f"This is exactly the regression an off-by-one slice would produce."
            )


def test_off_by_one_slice_catches_regression():
    """Sanity check: if a buggy implementation used
    `prompt_lens[b]` (one too high) as the offset, the gathered value
    would NOT match the expected log-prob. We simulate the bug here
    and confirm the test would catch it."""
    B = 2
    prompt_lens = torch.tensor([3, 5], dtype=torch.long)
    T_codec = 3
    V_cb0 = 16
    T_total = 12

    torch.manual_seed(42)
    talker_logits = torch.randn(B, T_total, V_cb0)
    target_cb0 = torch.randint(0, V_cb0, (B, T_codec))

    # Correct gather (using prompt_lens[b] - 1 + t as the offset):
    correct = _gather_log_probs_cb0(
        talker_logits=talker_logits,
        target_cb0=target_cb0,
        prompt_lens=prompt_lens,
        T_codec=T_codec,
    )

    # Simulated off-by-one: pretend the implementation used `prompt_lens` (no -1).
    # Hand-compute the BUGGY value and confirm it differs from `correct`.
    for b in range(B):
        for t in range(T_codec):
            buggy_logits = talker_logits[b, int(prompt_lens[b]) + t, :]
            buggy_expected = F.log_softmax(buggy_logits, dim=-1)[int(target_cb0[b, t])]
            assert not torch.allclose(correct[b, t], buggy_expected, atol=1e-6), (
                f"AC-5.1 sentinel: at sample {b}, frame {t}, the correct "
                f"and off-by-one values happen to coincide (numerically). "
                f"This is unlikely with random inputs; rerun with a different seed."
            )
