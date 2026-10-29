"""Classification metrics for the held-out EEG trials."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from mindmove.preprocessing import CLASS_NAMES


def majority_class(labels: np.ndarray) -> int:
    """Most common training label. Ties resolve to the smaller class index (left)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    return int(np.argmax(counts))


def compute_metrics(y_true, y_pred, subjects) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    subjects = np.asarray(subjects)
    if len(y_true) == 0:
        raise ValueError("Cannot score an empty set of trials.")
    if not (len(y_true) == len(y_pred) == len(subjects)):
        raise ValueError("Predictions, labels, and subject ids have different lengths.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    per_subject = []
    for subject in sorted(int(item) for item in np.unique(subjects)):
        mask = subjects == subject
        true_s = y_true[mask]
        pred_s = y_pred[mask]
        both = len(np.unique(true_s)) > 1
        per_subject.append(
            {
                "subject": int(subject),
                "n_trials": int(mask.sum()),
                "n_left": int(np.sum(true_s == 0)),
                "n_right": int(np.sum(true_s == 1)),
                "accuracy": float(accuracy_score(true_s, pred_s)),
                "balanced_accuracy": (
                    float(balanced_accuracy_score(true_s, pred_s)) if both else None
                ),
            }
        )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
    return {
        "n_trials": int(len(y_true)),
        "n_subjects": int(len(np.unique(subjects))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "per_subject": per_subject,
    }
