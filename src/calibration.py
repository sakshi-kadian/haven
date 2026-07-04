"""
calibration.py
--------------
Calibration analysis for constitutional principle classifiers.

Implements:
    - Expected Calibration Error (ECE) with M=10 equal-width bins
    - Reliability diagram plotting (predicted confidence vs. actual accuracy)
    - Temperature Scaling: learns scalar T on validation set to minimize NLL

Reference:
    Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.

Usage:
    from src.calibration import compute_ece, temperature_scale, plot_reliability_diagram
"""


def compute_ece(confidences: list, labels: list, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: List of predicted probabilities in [0, 1].
        labels:      List of true binary labels {0, 1}.
        n_bins:      Number of equal-width bins (default: 10).

    Returns:
        ECE as a float.
    """
    raise NotImplementedError("TODO: Implement this method.")


def temperature_scale(logits: list, labels: list) -> float:
    """
    Learn a scalar temperature T that minimizes NLL on the validation set.
    Calibrated probability: q = sigmoid(logit / T)

    Args:
        logits: Raw model logits (before sigmoid).
        labels: True binary labels.

    Returns:
        Optimal temperature T (float).
    """
    raise NotImplementedError("TODO: Implement this method.")


def plot_reliability_diagram(confidences: list, labels: list,
                              principle: str, output_path: str) -> None:
    """
    Plot a reliability diagram (confidence vs. accuracy per bin).

    Args:
        confidences:  Predicted probabilities.
        labels:       True binary labels.
        principle:    Name of the principle (used in plot title).
        output_path:  Path to save the figure PNG.
    """
    raise NotImplementedError("TODO: Implement this method.")
