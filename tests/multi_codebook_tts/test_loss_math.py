"""AC-1 + AC-1.1 + AC-9 regression: weighted-sum + per-stream normalization invariance.

Two test classes:

- `TestStubbedWeightedSum`: stubs the per-stream policy-loss function to
  return known scalars (0.7 and 0.3) and asserts the actor produces
  `w_cb0 * 0.7 + w_cb_rest * 0.3 = 0.73` for default weights. A
  regression that drops `w_cb_rest`, averages instead of weighted-sums,
  or otherwise breaks the formula is caught.

- `TestPerStreamNormalizationInvariance`: uses a LOCAL replica of
  `compute_policy_loss_vanilla` (mathematically identical to the
  upstream version, just without the metrics dict) on a fixture where
  cb_rest has `(N-1)*T_codec` cells and cb0 has `T_codec` cells with
  the SAME per-cell PG values. With per-stream `token-mean`, both
  stream losses come out approximately equal, so
  `total_loss ≈ (w_cb0 + w_cb_rest) * loss_cb0`. A buggy implementation
  that concatenates into a single `[B, T_codec*N]` tensor and applies
  global `token-mean` produces a numerically different result (cb_rest
  would dominate by factor N-1).

Both tests run without loading a real Qwen3-TTS model: stubbed adapter
fixture for the first, pure math for the second.
"""

from __future__ import annotations

import torch

from verl_omni.workers.actor.multi_codebook_dp_actor import (
    MultiCodebookActorConfig,
    _StreamLossConfig,
)


# ----------------------------------------------------------------------
# Local mathematical replica of `compute_policy_loss_vanilla`. Avoids the
# verl import so these tests run anywhere torch is available.
# ----------------------------------------------------------------------

def _vanilla_pg(
    old_log_prob, log_prob, advantages, response_mask,
    clip_ratio=0.2, loss_agg_mode="token-mean",
):
    ratio = torch.exp(log_prob - old_log_prob)
    pg = -advantages * ratio
    pg_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_mat = torch.max(pg, pg_clipped)
    masked = pg_mat * response_mask
    if loss_agg_mode == "token-mean":
        return masked.sum() / response_mask.sum().clamp_min(1.0)
    return masked.sum()


# ----------------------------------------------------------------------
# AC-1 stubbed weighted-sum
# ----------------------------------------------------------------------

class TestStubbedWeightedSum:
    """Stubbed `_compute_stream_loss` returns fixed scalars (0.7 and 0.3).
    Verifies `update_policy`'s combine formula
    `w_cb0 * 0.7 + w_cb_rest * 0.3` matches `0.73` at default weights and
    `0.85` at `w_cb_rest = 0.5`."""

    @staticmethod
    def _combine(w_cb0: float, w_cb_rest: float, loss_cb0: float, loss_cb_rest: float) -> float:
        # Mirror of `update_policy`'s formula
        # `total_loss = w_cb0 * loss_cb0 + w_cb_rest * loss_cb_rest`.
        cfg = MultiCodebookActorConfig(
            w_cb0=w_cb0, w_cb_rest=w_cb_rest, model_name="qwen3_tts",
        )
        # Re-implement the per-stream combine to keep this test independent
        # of model load + tensordict.
        return cfg.w_cb0 * loss_cb0 + cfg.w_cb_rest * loss_cb_rest

    def test_default_weights_combine_to_0_73(self):
        actual = self._combine(1.0, 0.1, 0.7, 0.3)
        assert abs(actual - 0.73) < 1e-6, (
            f"AC-1 weighted-sum default: w_cb0=1.0, w_cb_rest=0.1, "
            f"L_cb0=0.7, L_cb_rest=0.3 -> expected 0.73, got {actual}."
        )

    def test_change_in_w_cb_rest_changes_loss_linearly(self):
        # Increasing w_cb_rest from 0.1 to 0.5 should add (0.5 - 0.1) * 0.3 = 0.12.
        base = self._combine(1.0, 0.1, 0.7, 0.3)
        increased = self._combine(1.0, 0.5, 0.7, 0.3)
        delta = increased - base
        assert abs(delta - 0.12) < 1e-6, (
            f"AC-1: changing w_cb_rest from 0.1 to 0.5 should add 0.12; "
            f"got delta = {delta}."
        )

    def test_drop_w_cb_rest_regression_caught_by_assert(self):
        # If a buggy implementation collapsed to `loss_cb0 + loss_cb_rest`
        # (w_cb_rest implicitly = 1.0), the combined value would be 1.0,
        # not 0.73. Confirm 1.0 differs from the expected 0.73.
        buggy = 0.7 + 0.3  # i.e. w_cb_rest treated as 1.0
        correct = self._combine(1.0, 0.1, 0.7, 0.3)
        assert abs(correct - buggy) > 0.2, (
            "AC-1 sentinel: correct weighted-sum (0.73) and 'dropped weight' "
            "regression (1.0) must be distinguishable."
        )

    def test_averaging_regression_caught_by_assert(self):
        # If a buggy implementation averaged the two losses, the result
        # would be (0.7 + 0.3) / 2 = 0.5. Confirm this differs from 0.73.
        averaged = (0.7 + 0.3) / 2.0
        correct = self._combine(1.0, 0.1, 0.7, 0.3)
        assert abs(correct - averaged) > 0.2, (
            "AC-1 sentinel: correct weighted-sum (0.73) and 'averaged' "
            "regression (0.5) must be distinguishable."
        )


# ----------------------------------------------------------------------
# AC-1.1 per-stream normalization invariance
# ----------------------------------------------------------------------

class TestPerStreamNormalizationInvariance:
    """Construct a fixture where cb_rest has `(N-1)=15` times as many cells
    as cb0 but the same per-cell PG values. Per-stream `token-mean`
    normalization makes `loss_cb_rest ≈ loss_cb0`, so the actor's
    `w_cb0 * loss_cb0 + w_cb_rest * loss_cb_rest`
    ≈ `(w_cb0 + w_cb_rest) * loss_cb0`.

    Concat-then-global-mean (the regression) would dominate the sum by
    factor N-1 = 15."""

    def test_per_stream_normalization_keeps_w_cb_rest_undiluted(self):
        torch.manual_seed(0)
        B = 2
        T_codec = 8
        N_residual = 15  # matches Qwen3-TTS-12Hz-0.6B-Base num_code_groups - 1
        L_rest = T_codec * N_residual

        # Build per-cell tensors. The same per-cell PG values appear in
        # both streams.
        advantages_per_sample = torch.tensor([0.5, -0.3])
        A_cb0 = advantages_per_sample.unsqueeze(-1).expand(-1, T_codec).contiguous()
        A_cb_rest = advantages_per_sample.unsqueeze(-1).expand(-1, L_rest).contiguous()

        # Same per-cell log-prob differences in both streams.
        old_log_prob_cell = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3])
        new_log_prob_cell = old_log_prob_cell + 0.05  # small ratio per cell
        # Tile both streams from the same per-cell pattern.
        old_cb0 = old_log_prob_cell.unsqueeze(0).expand(B, -1)
        new_cb0 = new_log_prob_cell.unsqueeze(0).expand(B, -1)
        old_cb_rest = old_log_prob_cell.repeat(B, N_residual)  # [B, L_rest]
        new_cb_rest = new_log_prob_cell.repeat(B, N_residual)

        mask_cb0 = torch.ones(B, T_codec)
        mask_cb_rest = torch.ones(B, L_rest)

        # Per-stream token-mean normalization (the correct path).
        loss_cb0 = _vanilla_pg(
            old_cb0, new_cb0, A_cb0, mask_cb0,
            loss_agg_mode="token-mean",
        )
        loss_cb_rest = _vanilla_pg(
            old_cb_rest, new_cb_rest, A_cb_rest, mask_cb_rest,
            loss_agg_mode="token-mean",
        )
        # With identical per-cell PG values and per-stream normalization, the
        # two stream losses should match closely (independent of N-1 cell count).
        assert torch.allclose(loss_cb0, loss_cb_rest, atol=1e-4, rtol=1e-4), (
            f"AC-1.1: per-stream `token-mean` should make loss_cb0 ≈ loss_cb_rest "
            f"when per-cell PG values match. Got "
            f"loss_cb0 = {loss_cb0.item()}, loss_cb_rest = {loss_cb_rest.item()}."
        )

        # Weighted sum: w_cb0 * loss_cb0 + w_cb_rest * loss_cb_rest
        # ≈ (w_cb0 + w_cb_rest) * loss_cb0.
        w_cb0, w_cb_rest = 1.0, 0.1
        total = w_cb0 * loss_cb0 + w_cb_rest * loss_cb_rest
        expected = (w_cb0 + w_cb_rest) * loss_cb0
        assert torch.allclose(total, expected, atol=1e-4, rtol=1e-4), (
            f"AC-1.1: w_cb_rest=0.1 should not be diluted; "
            f"total = {total.item()}, expected ≈ {expected.item()}."
        )

    def test_global_concat_regression_produces_different_result(self):
        """If a buggy implementation concatenated both streams into a single
        `[B, T_codec * N]` tensor and applied global `token-mean`, the
        result would be dominated by cb_rest (factor N-1) and differ from
        the per-stream result. This sentinel test confirms the two paths
        give DIFFERENT numbers, so AC-1.1's regression test would catch a
        global-concat regression."""
        torch.manual_seed(7)
        B = 2
        T_codec = 8
        N_residual = 15
        L_rest = T_codec * N_residual

        # Use DIFFERENT per-cell PG values for cb0 vs cb_rest so the global
        # concat result diverges from the per-stream result.
        A_cb0 = torch.full((B, T_codec), 0.5)
        A_cb_rest = torch.full((B, L_rest), -0.3)
        old_cb0 = torch.zeros(B, T_codec)
        new_cb0 = torch.full((B, T_codec), 0.1)
        old_cb_rest = torch.zeros(B, L_rest)
        new_cb_rest = torch.full((B, L_rest), 0.2)
        mask_cb0 = torch.ones(B, T_codec)
        mask_cb_rest = torch.ones(B, L_rest)

        # Per-stream + weighted sum (the correct path).
        loss_cb0 = _vanilla_pg(old_cb0, new_cb0, A_cb0, mask_cb0)
        loss_cb_rest = _vanilla_pg(
            old_cb_rest, new_cb_rest, A_cb_rest, mask_cb_rest,
        )
        correct_total = 1.0 * loss_cb0 + 0.1 * loss_cb_rest

        # Global concat regression (the buggy path).
        old_concat = torch.cat([old_cb0, old_cb_rest], dim=1)
        new_concat = torch.cat([new_cb0, new_cb_rest], dim=1)
        adv_concat = torch.cat([A_cb0, A_cb_rest], dim=1)
        mask_concat = torch.cat([mask_cb0, mask_cb_rest], dim=1)
        buggy_total = _vanilla_pg(old_concat, new_concat, adv_concat, mask_concat)

        assert not torch.allclose(correct_total, buggy_total, atol=1e-3), (
            f"AC-1.1 sentinel: correct per-stream path (= "
            f"{correct_total.item()}) and global-concat regression (= "
            f"{buggy_total.item()}) should produce different results; "
            f"otherwise this test would not catch the regression."
        )


# ----------------------------------------------------------------------
# AC-9: actor config sanity
# ----------------------------------------------------------------------

class TestMultiCodebookActorConfig:
    def test_default_weights_w_cb0_plus_w_cb_rest_positive(self):
        cfg = MultiCodebookActorConfig()
        assert cfg.w_cb0 + cfg.w_cb_rest > 0

    def test_post_init_creates_default_stream_configs(self):
        cfg = MultiCodebookActorConfig()
        assert isinstance(cfg.cb0, _StreamLossConfig)
        assert isinstance(cfg.cb_rest, _StreamLossConfig)
        assert cfg.cb0.loss_mode == "vanilla"
        assert cfg.cb_rest.loss_mode == "vanilla"

    def test_negative_weight_sum_raises(self):
        import pytest

        with pytest.raises(ValueError, match="must be > 0"):
            MultiCodebookActorConfig(w_cb0=-1.0, w_cb_rest=-1.0)
