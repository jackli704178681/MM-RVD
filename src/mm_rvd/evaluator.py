from __future__ import annotations

import numpy as np


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_count: int = 8) -> float:
    recalls = []
    for cls in range(class_count):
        mask = np.asarray(y_true) == cls
        if mask.any():
            recalls.append(float((np.asarray(y_pred)[mask] == cls).mean()))
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def chance_normalized_balanced_accuracy(balanced_acc: float, class_count: int = 8) -> float:
    chance = 1.0 / float(class_count)
    return float((balanced_acc - chance) / (1.0 - chance))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int = 8) -> np.ndarray:
    mat = np.zeros((class_count, class_count), dtype=np.int64)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        if 0 <= t < class_count and 0 <= p < class_count:
            mat[t, p] += 1
    return mat


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_count: int = 8) -> float:
    vals = []
    cm = confusion_matrix(y_true, y_pred, class_count)
    for c in range(class_count):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        vals.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(vals))


def five_missing_summary(condition_scores: dict[str, float]) -> dict[str, float]:
    keys = ["U30", "SW-U30", "T5", "B5", "J30-5"]
    vals = [float(condition_scores[k]) for k in keys]
    return {"five_missing_mean": float(np.mean(vals)), "five_missing_worst": float(np.min(vals))}
