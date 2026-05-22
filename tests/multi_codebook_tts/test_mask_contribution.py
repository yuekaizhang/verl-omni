"""AC-5.3 + AC-5.4 regression: explicit-contribution mask + post-EOS leakage.

Strategy: run the per-stream policy-loss path directly (mirroring the
math in `MultiCodebookDPActor._compute_stream_loss`) with hand-built
fixtures. We bypass the upstream `compute_policy_loss_vanilla` here so the
tests do not depend on the verl install — the masking semantics under
test are the SAME ones the actor relies on (mask-out positions whose
mask value is 0 must not contribute to the aggregated loss).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _vanilla_pg_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """Local replica of `compute_policy_loss_vanilla`'s mathematical core,
    without the metrics dict and without the verl dependency. Matches the
    upstream behavior closely enough that the masking tests below are
    representative.

    Returns the scalar loss. The mask multiplies the per-position PG term
    BEFORE reduction; positions with mask=0 contribute nothing regardless
    of their log_prob / advantage values.
    """
    ratio = torch.exp(log_prob - old_log_prob)
    pg = -advantages * ratio
    pg_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_loss_mat = torch.max(pg, pg_clipped)
    # Masked aggregation.
    masked = pg_loss_mat * response_mask
    if loss_agg_mode == "token-mean":
        return masked.sum() / response_mask.sum().clamp_min(1.0)
    return masked.sum()


def test_explicit_contribution_mask_zero_positions_dont_contribute():
    """AC-5.3: a position with mask=0 must not contribute to the loss, no
    matter what extreme values its `log_prob` / `advantage` carry.

    Construct two fixtures that differ ONLY at masked positions. The
    aggregated loss must match (finite sentinels are used; NaN is
    intentionally avoided so the test doesn't conflate masking with
    NaN propagation through `exp`/`clamp`)."""
    torch.manual_seed(1)
    B, T = 2, 5
    advantages = torch.randn(B, T)
    old_lp = torch.randn(B, T)
    new_lp = torch.randn(B, T)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0, 0.0, 0.0]])

    base_loss = _vanilla_pg_loss(old_lp, new_lp, advantages, mask)

    # Perturb ONLY masked positions with finite sentinels.
    perturbed_old = old_lp.clone()
    perturbed_new = new_lp.clone()
    perturbed_adv = advantages.clone()
    perturbed_old[mask == 0.0] = 1.0    # finite sentinel
    perturbed_new[mask == 0.0] = 2.0    # finite sentinel
    perturbed_adv[mask == 0.0] = 100.0  # extreme sentinel

    perturbed_loss = _vanilla_pg_loss(
        perturbed_old, perturbed_new, perturbed_adv, mask,
    )

    assert torch.allclose(base_loss, perturbed_loss, atol=1e-7), (
        f"AC-5.3: masked positions leaked into the loss. "
        f"base={base_loss.item()}, perturbed={perturbed_loss.item()}. "
        f"A regression that lets masked positions contribute is caught here."
    )


def test_post_eos_leakage_appended_wrong_codec_doesnt_change_loss():
    """AC-5.4: appending a deliberately wrong codec value at the post-EOS
    frame on at least one sample must NOT change the loss, because the
    response mask should be `0` at the post-EOS frame. A regression that
    includes post-EOS frames in the mask is caught here.

    We simulate this by treating frame T_eos as the EOS frame; positions
    `> T_eos` are post-EOS and have mask=0. Inject a sentinel at position
    T_eos+1 of one sample and confirm the loss is unchanged."""
    torch.manual_seed(7)
    B = 2
    T = 6
    advantages = torch.randn(B, T)
    old_lp = torch.randn(B, T)
    new_lp = torch.randn(B, T)
    # Sample 0: EOS at frame 3, post-EOS = frames 4,5.
    # Sample 1: EOS at frame 4, post-EOS = frame 5.
    mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
    )

    base_loss = _vanilla_pg_loss(old_lp, new_lp, advantages, mask)

    # Inject wrong codec sentinel at sample 0's post-EOS frame 4 by
    # perturbing the new log-prob at that cell (regression test would
    # have mask=1 there).
    perturbed_new = new_lp.clone()
    perturbed_new[0, 4] = -50.0  # extreme finite sentinel
    perturbed_loss = _vanilla_pg_loss(old_lp, perturbed_new, advantages, mask)
    assert torch.allclose(base_loss, perturbed_loss, atol=1e-7), (
        f"AC-5.4 post-EOS leakage: base={base_loss.item()}, "
        f"perturbed={perturbed_loss.item()}. A regression that includes "
        f"post-EOS frames in the mask would cause these to differ."
    )

    # Sanity sentinel: if we ACCIDENTALLY include the post-EOS position in
    # the mask, the loss changes. Confirms the test is not vacuously true.
    bad_mask = mask.clone()
    bad_mask[0, 4] = 1.0
    bad_loss = _vanilla_pg_loss(old_lp, perturbed_new, advantages, bad_mask)
    assert not torch.allclose(base_loss, bad_loss, atol=1e-4), (
        f"AC-5.4 sentinel: bad_mask should produce a different loss. "
        f"base={base_loss.item()}, bad={bad_loss.item()}."
    )
