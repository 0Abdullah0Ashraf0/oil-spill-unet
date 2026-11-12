"""Losses. Four kinds so the U2 change can be decomposed into its parts:

    ce        plain cross-entropy                       (U1 baseline)
    wce       class-weighted cross-entropy              (isolates the effect of weights)
    ce_dice   cross-entropy + soft Dice, no weights     (isolates the effect of Dice)
    wce_dice  weighted cross-entropy + soft Dice        (U2: both together)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEEDS_WEIGHTS = ("wce", "wce_dice")


def class_weights_from_counts(counts, method="inv_sqrt"):
    """Weights from TRAINING-split pixel counts only (never val/test).

    inv_sqrt:    w_c ~ 1/sqrt(freq_c)  (milder, usually more stable)
    median_freq: w_c = median(freq) / freq_c  (Eigen & Fergus style, stronger)
    Normalised to mean 1 so the loss scale stays comparable to plain CE.
    """
    freq = np.asarray(counts, dtype=np.float64)
    freq = freq / freq.sum()
    freq = np.maximum(freq, 1e-8)  # a class absent from train would otherwise give inf
    w = 1.0 / np.sqrt(freq) if method == "inv_sqrt" else np.median(freq) / freq
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class SoftDiceLoss(nn.Module):
    """1 - mean per-class soft Dice over the batch. eps smooths classes absent from a batch."""

    def __init__(self, n_classes=5, eps=1.0):
        super().__init__()
        self.n, self.eps = n_classes, eps

    def forward(self, logits, target):
        p = F.softmax(logits.float(), dim=1)
        t = F.one_hot(target, self.n).permute(0, 3, 1, 2).float()
        inter = (p * t).sum((0, 2, 3))
        denom = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
        return 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()


class CEDice(nn.Module):
    def __init__(self, weight=None, dice_w=1.0, n_classes=5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice, self.dice_w = SoftDiceLoss(n_classes), dice_w

    def forward(self, logits, target):
        return self.ce(logits.float(), target) + self.dice_w * self.dice(logits, target)


def build_loss(kind, weights=None, dice_w=1.0):
    if kind in NEEDS_WEIGHTS and weights is None:
        raise ValueError(f"loss '{kind}' needs class weights")
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "wce":
        return nn.CrossEntropyLoss(weight=weights)
    if kind == "ce_dice":
        return CEDice(weight=None, dice_w=dice_w)
    if kind == "wce_dice":
        return CEDice(weight=weights, dice_w=dice_w)
    raise ValueError(kind)
