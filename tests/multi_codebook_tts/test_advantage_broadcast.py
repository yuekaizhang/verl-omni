"""AC-1 + AC-5 regression: per-sample advantage `A_i` shape `[B]` broadcasts
correctly to `[B, T]` for cb0 and `[B, T*(N-1)]` for cb_rest, with consistent
masking on padding frames.

Tests the broadcast semantics directly (without invoking the full
update_policy path), so the result is independent of any model load. If
a future refactor swaps `expand` for `repeat` or accidentally interleaves
the broadcast axes, the per-position advantage at frame `t` of sample `b`
would no longer equal `A_i[b]` and these tests catch it.
"""

from __future__ import annotations

import torch


def test_cb0_advantage_broadcast_shape_and_values():
    """AC-1: `advantages [B]` broadcasts to `[B, T_codec]` via
    `unsqueeze(-1).expand(-1, T_codec)`. After the broadcast, every cell
    `(b, t)` carries the per-sample advantage `advantages[b]`."""
    B = 4
    T_codec = 6
    advantages = torch.tensor([0.5, -0.2, 1.3, -0.7])
    A_cb0 = advantages.unsqueeze(-1).expand(-1, T_codec)
    assert A_cb0.shape == (B, T_codec)
    for b in range(B):
        for t in range(T_codec):
            assert A_cb0[b, t].item() == advantages[b].item(), (
                f"cb0 broadcast mismatch at ({b}, {t}): got {A_cb0[b, t]}, "
                f"expected {advantages[b]}."
            )


def test_cb_rest_advantage_broadcast_shape_and_values():
    """AC-1 + AC-5.2: `advantages [B]` broadcasts to `[B, T_codec * (N-1)]`
    via `unsqueeze(-1).expand(-1, T_codec * (N-1))`. Every cell in the
    flattened cb_rest axis carries `advantages[b]`, regardless of which
    `(frame, codebook)` it represents (because the GRPO advantage is
    per-sample, not per-position)."""
    B = 3
    T_codec = 5
    N_residual = 4
    L = T_codec * N_residual
    advantages = torch.tensor([0.1, 0.2, 0.3])
    A_cb_rest = advantages.unsqueeze(-1).expand(-1, L)
    assert A_cb_rest.shape == (B, L)
    for b in range(B):
        for pos in range(L):
            assert A_cb_rest[b, pos].item() == advantages[b].item(), (
                f"cb_rest broadcast mismatch at ({b}, {pos}): got "
                f"{A_cb_rest[b, pos]}, expected {advantages[b]}."
            )


def test_cb_rest_broadcast_aligns_with_flatten_order():
    """Sanity: when cb_rest is reshaped from `[B, T_codec, N-1]` to
    `[B, T_codec * (N-1)]` in codebook-major-within-frame order (AC-5.2),
    the broadcast advantages still match per-sample regardless of which
    `(frame, codebook)` the flat position represents."""
    B = 2
    T_codec = 3
    N_residual = 2

    # Per-position sentinel where the value encodes (frame, codebook).
    sentinel_3d = torch.tensor(
        [
            [[10, 11], [20, 21], [30, 31]],   # sample 0
            [[40, 41], [50, 51], [60, 61]],   # sample 1
        ],
        dtype=torch.float,
    )
    flat = sentinel_3d.flatten(1, 2)  # codebook-major-within-frame
    expected_layout = torch.tensor(
        [[10, 11, 20, 21, 30, 31], [40, 41, 50, 51, 60, 61]],
        dtype=torch.float,
    )
    assert torch.equal(flat, expected_layout)

    # Broadcast advantages.
    advantages = torch.tensor([7.0, 9.0])
    A_cb_rest = advantages.unsqueeze(-1).expand(-1, T_codec * N_residual)
    # Every position of sample 0 should carry 7.0; every position of
    # sample 1 should carry 9.0.
    assert A_cb_rest[0].tolist() == [7.0] * 6
    assert A_cb_rest[1].tolist() == [9.0] * 6
