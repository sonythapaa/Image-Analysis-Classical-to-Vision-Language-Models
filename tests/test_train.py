"""Unit tests for train.py — Dice/IoU/loss functions, hand-computed on
tiny fixed tensors so the expected value can be verified by hand."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuclei_pipeline.train import dice_coefficient, iou_score, dice_loss, bce_loss, combined_loss


def make_perfect_match_logits_and_target():
    """Logits with large magnitude so sigmoid saturates near 0/1, matching
    the target exactly -> Dice and IoU should both be ~1.0."""
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 1:3, 1:3] = 1.0
    logits = (target * 2 - 1) * 20.0  # target=1 -> +20 (sigmoid~1), target=0 -> -20 (sigmoid~0)
    return logits, target


def make_disjoint_logits_and_target():
    """Prediction and target don't overlap at all -> Dice and IoU should be 0."""
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0:2, 0:2] = 1.0
    logits = torch.full((1, 1, 4, 4), -20.0)
    logits[:, :, 2:4, 2:4] = 20.0  # predicts the opposite corner
    return logits, target


def test_dice_coefficient_perfect_match_near_one():
    logits, target = make_perfect_match_logits_and_target()
    assert dice_coefficient(logits, target) > 0.99


def test_iou_score_perfect_match_near_one():
    logits, target = make_perfect_match_logits_and_target()
    assert iou_score(logits, target) > 0.99


def test_dice_coefficient_disjoint_is_zero():
    logits, target = make_disjoint_logits_and_target()
    assert dice_coefficient(logits, target) < 0.01


def test_iou_score_disjoint_is_zero():
    logits, target = make_disjoint_logits_and_target()
    assert iou_score(logits, target) < 0.01


def test_dice_loss_perfect_match_near_zero():
    logits, target = make_perfect_match_logits_and_target()
    assert dice_loss(logits, target).item() < 0.01


def test_bce_loss_is_nonnegative():
    logits, target = make_perfect_match_logits_and_target()
    assert bce_loss(logits, target).item() >= 0.0


def test_combined_loss_equals_sum_of_parts():
    logits, target = make_perfect_match_logits_and_target()
    expected = bce_loss(logits, target) + dice_loss(logits, target)
    assert torch.isclose(combined_loss(logits, target), expected)


def test_dice_and_iou_ordering_iou_leq_dice():
    # For any non-trivial overlap, IoU <= Dice (a known algebraic identity:
    # IoU = Dice / (2 - Dice) for Dice in [0,1]).
    target = torch.zeros(1, 1, 6, 6)
    target[:, :, 0:4, 0:4] = 1.0
    logits = torch.full((1, 1, 6, 6), -20.0)
    logits[:, :, 0:3, 0:3] = 20.0  # partial overlap
    d = dice_coefficient(logits, target)
    i = iou_score(logits, target)
    assert i <= d + 1e-6
