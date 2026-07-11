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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

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
    confidences = np.array(confidences)
    labels = np.array(labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include exactly 1.0 in the last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        prob_in_bin = in_bin.mean()
        
        if prob_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prob_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
    return float(ece)


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
    logits = np.array(logits)
    labels = np.array(labels)
    
    def nll(T):
        # Prevent division by zero or extreme T
        T = np.clip(T, 1e-3, 100)
        # Compute calibrated probabilities: q = sigmoid(logit / T)
        q = 1 / (1 + np.exp(-logits / T))
        # Clip to prevent log(0)
        q = np.clip(q, 1e-7, 1 - 1e-7)
        # Binary Cross Entropy (NLL)
        loss = -np.mean(labels * np.log(q) + (1 - labels) * np.log(1 - q))
        return loss

    # Optimize T starting from T=1.0
    result = minimize(nll, x0=np.array([1.0]), method='L-BFGS-B', bounds=[(0.05, 10.0)])
    optimal_T = float(result.x[0])
    
    return optimal_T


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
    confidences = np.array(confidences)
    labels = np.array(labels)
    n_bins = 10
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        if np.any(in_bin):
            accuracies.append(labels[in_bin].mean())
        else:
            accuracies.append(np.nan)
            
    plt.figure(figsize=(6, 6))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Plot model reliability
    plt.plot(bin_centers, accuracies, marker='o', linewidth=2, color='mediumpurple', label='Model')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram: {principle.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
