"""Segmentation metrics from a confusion matrix (rows = ground truth, cols = prediction).

Absent-class policy: a class with no ground-truth AND no predicted pixels has an undefined
IoU/Dice (0/0). It is reported as null and EXCLUDED from the mIoU mean (nanmean), rather
than counted as 0 or 1. Say this in the report.
"""
import numpy as np
import torch

NAMES = ["sea", "oil_spill", "look_alike", "ship", "land"]


def confusion(pred, target, n=5):
    idx = target.reshape(-1) * n + pred.reshape(-1)
    return torch.bincount(idx, minlength=n * n).reshape(n, n)


def summarize(cm, names=NAMES):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        dice = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    f = lambda v: None if np.isnan(v) else float(v)
    out = {"mIoU": f(np.nanmean(iou)), "pixel_acc": float(tp.sum() / max(cm.sum(), 1))}
    for i, n in enumerate(names):
        out[f"IoU_{n}"], out[f"Dice_{n}"] = f(iou[i]), f(dice[i])
    out["mDice"] = f(np.nanmean(dice))
    return out
