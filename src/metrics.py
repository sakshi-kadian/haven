"""
metrics.py
----------
Evaluation metrics for the HAVEN framework.

Implements:
    - Per-principle Accuracy, Precision, Recall, F1
    - McNemar's test for statistical significance between two classifiers
    - Summary report generation

Usage:
    from src.metrics import compute_metrics, mcnemar_test
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
    raise NotImplementedError("TODO: Implement this method.")


def mcnemar_test(y_true: list, y_pred_a: list, y_pred_b: list) -> dict:
    """
    Run McNemar's test to check whether model A and model B differ
    significantly in their errors.

    Null hypothesis: Both models make the same proportion of errors.

    Args:
        y_true:    True binary labels.
        y_pred_a:  Predictions from model A (e.g., HAVEN).
        y_pred_b:  Predictions from model B (e.g., monolithic baseline).

    Returns:
        Dict with keys: statistic, p_value, significant (bool, threshold p<0.05).
    """
    raise NotImplementedError("TODO: Implement this method.")


def generate_report(metrics_per_principle: list, output_path: str) -> None:
    """
    Save a full metrics report to JSON.

    Args:
        metrics_per_principle: List of dicts from compute_metrics().
        output_path:           Path to save metrics.json.
    """
    raise NotImplementedError("TODO: Implement this method.")
