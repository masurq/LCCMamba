"""Segmentation metrics derived from a confusion matrix."""
from __future__ import annotations

import numpy as np


def metrics_from_confusion(confusion: np.ndarray, metric_classes: list[int]) -> dict:
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    total = confusion.sum()
    diagonal = np.diag(confusion)
    row_sum = confusion.sum(axis=1)
    col_sum = confusion.sum(axis=0)
    union = row_sum + col_sum - diagonal
    denominator_f1 = row_sum + col_sum
    iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union > 0)
    f1 = np.divide(2 * diagonal, denominator_f1, out=np.zeros_like(diagonal), where=denominator_f1 > 0)

    selected = np.asarray(metric_classes, dtype=int)
    if np.any(selected < 0) or np.any(selected >= confusion.shape[0]):
        raise ValueError("metric_classes contains an invalid class index")
    if not np.allclose(iou[selected], f1[selected] / (2.0 - f1[selected]), atol=1e-12):
        raise AssertionError("Per-class IoU/F1 identity failed")

    oa = diagonal.sum() / total if total else 0.0
    expected = (row_sum * col_sum).sum() / (total * total) if total else 0.0
    kappa = (oa - expected) / (1.0 - expected) if expected != 1.0 else 0.0
    return {
        "OA": float(oa),
        "mIoU": float(iou[selected].mean()),
        "mF1": float(f1[selected].mean()),
        "Kappa": float(kappa),
        "per_class_IoU": iou.tolist(),
        "per_class_F1": f1.tolist(),
    }
