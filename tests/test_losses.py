"""Tests for the loss functions."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from losses import CombinedLoss, balanced_bce, soft_dice_loss  # noqa: E402


def _recover_pos_weight(target, valid):
    """With zero logits, BCE(0, y=1) = w*log2 and BCE(0, y=0) = log2, so the
    mean loss pins down the positive weight the function actually used."""
    loss = balanced_bce(torch.zeros_like(target), target, valid).item()
    pos = target.mean().item()
    return ((loss / math.log(2)) - (1 - pos)) / pos


def test_positive_weight_is_per_channel_not_pooled():
    """Regression: pooling all boundary channels into one positive weight
    over-weights the dense channel and under-weights the sparse ones.

    On a representative set coastline is ~2.6% of valid pixels, limb ~0.66%,
    terminator ~0.36%. Pooled, every channel gets ~81 when coastline wants 37
    - a 2.2x over-weight that drives it to over-predict, producing thick
    blobby edges instead of thin lines.
    """
    B, C, H, W = 1, 3, 32, 32
    target = torch.zeros(B, C, H, W)
    valid = torch.ones(B, C, H, W)
    target[:, 0, ::4, :] = 1     # 25%
    target[:, 1, ::8, :] = 1     # 12.5%
    target[:, 2, ::16, :] = 1    # 6.25%

    for c, expected_frac in ((0, 0.25), (1, 0.125), (2, 0.0625)):
        t, v = target[:, c:c + 1], valid[:, c:c + 1]
        assert t.mean().item() == pytest.approx(expected_frac)
        got = _recover_pos_weight(t, v)
        want = (1 - expected_frac) / expected_frac
        assert got == pytest.approx(want, rel=1e-3), f"channel {c}: {got} != {want}"


def test_positive_weight_is_capped():
    """An almost-empty channel must not produce an unbounded weight."""
    t = torch.zeros(1, 1, 64, 64)
    t[0, 0, 0, 0] = 1
    v = torch.ones_like(t)
    assert _recover_pos_weight(t, v) <= 100.0 + 1e-3


def test_invalid_pixels_are_excluded():
    t = torch.zeros(1, 2, 16, 16)
    t[:, :, ::4, :] = 1
    v = torch.ones_like(t)
    v[:, :, :, 8:] = 0

    base = balanced_bce(torch.zeros_like(t), t, v)
    noisy = torch.zeros_like(t)
    noisy[:, :, :, 8:] = -50.0          # confident garbage, all of it masked out
    assert balanced_bce(noisy, t, v) == pytest.approx(base.item(), rel=1e-5)


def test_perfect_prediction_beats_inverted():
    t = torch.zeros(1, 3, 16, 16)
    t[:, :, ::4, :] = 1
    v = torch.ones_like(t)
    good = (t * 2 - 1) * 20
    assert balanced_bce(good, t, v) < 0.01 < balanced_bce(-good, t, v)


def test_dice_rewards_overlap():
    logits = torch.zeros(1, 3, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    logits[:, 0] = 10.0                       # confidently predicts class 0
    assert soft_dice_loss(logits, target, 3) < 0.5
    target_wrong = torch.ones(1, 8, 8, dtype=torch.long)
    assert soft_dice_loss(logits, target_wrong, 3) > 0.5


def test_combined_loss_runs_and_reports_parts():
    crit = CombinedLoss(num_classes=5)
    seg_out = torch.randn(1, 5, 32, 32)
    edge_out = torch.randn(1, 3, 32, 32)
    seg = torch.randint(0, 5, (1, 32, 32))
    edges = torch.zeros(1, 3, 32, 32)
    edges[:, :, ::8, :] = 1
    valid = torch.ones(1, 3, 32, 32)

    total, parts = crit(seg_out, edge_out, seg, edges, valid, ())
    assert torch.isfinite(total)
    for k in ("total", "seg", "seg_ce", "seg_dice", "edge", "w_seg", "w_edge"):
        assert k in parts and torch.isfinite(parts[k])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
