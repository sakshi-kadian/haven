"""
metrics.py
----------
Evaluation metrics for the HAVEN framework.

Implements:
    - Per-principle Accuracy, Precision, Recall, F1
    - McNemar's test for statistical significance between two classifiers
    - Summary report generation

Usage:
    from src.metrics import compute_metrics, mcnemar_test, generate_report
"""

import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import chi2


def compute_metrics(y_true: list, y_pred: list, principle: str) -> dict:
    """
    Compute classification metrics for a single principle.

    Args:
        y_true:    True binary labels.
        y_pred:    Predicted binary labels.
        principle: Principle name (for labeling the output dict).

    Returns:
        Dict with keys: principle, accuracy, precision, recall, f1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "principle": principle,
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def mcnemar_test(y_true: list, y_pred_a: list, y_pred_b: list) -> dict:
    """
    Run McNemar's test to check whether model A and model B differ
    significantly in their errors.

    Null hypothesis: Both models make the same proportion of errors.

    Uses the continuity-corrected McNemar statistic:
        chi2 = (|b - c| - 1)^2 / (b + c)
    where b = A correct, B wrong; c = A wrong, B correct.

    Args:
        y_true:    True binary labels.
        y_pred_a:  Predictions from model A (e.g., HAVEN).
        y_pred_b:  Predictions from model B (e.g., monolithic baseline).

    Returns:
        Dict with keys: statistic, p_value, significant (bool, threshold p<0.05).
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong
    b = np.sum(correct_a & ~correct_b)
    # c: A wrong, B correct
    c = np.sum(~correct_a & correct_b)

    if (b + c) == 0:
        # Both models make identical errors — no test possible
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b": int(b),
            "c": int(c),
            "note": "b+c=0: models are identical on this dataset",
        }

    # Continuity-corrected McNemar statistic
    statistic = float((abs(b - c) - 1) ** 2 / (b + c))

    # Chi-squared distribution with 1 degree of freedom
    p_value = float(1 - chi2.cdf(statistic, df=1))

    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "b": int(b),
        "c": int(c),
    }


def generate_report(metrics_per_principle: list, output_path: str) -> None:
    """
    Save a full metrics report to JSON.

    Args:
        metrics_per_principle: List of dicts from compute_metrics().
        output_path:           Path to save metrics.json.
    """
    report = {
        "per_principle": metrics_per_principle,
        "macro_avg": {
            "accuracy": round(float(np.mean([m["accuracy"] for m in metrics_per_principle])), 4),
            "precision": round(float(np.mean([m["precision"] for m in metrics_per_principle])), 4),
            "recall": round(float(np.mean([m["recall"] for m in metrics_per_principle])), 4),
            "f1": round(float(np.mean([m["f1"] for m in metrics_per_principle])), 4),
        },
    }

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Metrics report saved to {output_path}")
