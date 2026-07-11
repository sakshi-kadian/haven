"""
conflict_detector.py
--------------------
Detects principle conflicts at the sample level.

Two-step approach:
    Step 1 (Global pre-check): Compute Pearson correlation matrix across
            all principle pairs on training data. Used as a one-time diagnostic
            to identify which pairs are CAPABLE of conflicting.

    Step 2 (Sample-level detection): For each (prompt, response) output o,
            a conflict between principle i and principle j is detected when:
                Conflict(o, pi, pj) = 1  iff  s(o, pi) < TAU_LOW  AND  s(o, pj) > TAU_HIGH

            Thresholds: TAU_LOW = 0.3, TAU_HIGH = 0.7

Usage:
    from src.conflict_detector import ConflictDetector
    detector = ConflictDetector()
    conflicts = detector.detect(score_vector)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

TAU_LOW = 0.3
TAU_HIGH = 0.7


class ConflictDetector:
    """
    Detects sample-level constitutional principle conflicts.

    Args:
        candidate_pairs: List of (pi, pj) tuples to check for conflicts.
                         Defaults to the pre-defined canonical conflict pairs.
    """

    def __init__(self, candidate_pairs=None):
        if candidate_pairs is None:
            # The classic tension points in LLM alignment
            self.candidate_pairs = [
                ("harmlessness", "helpfulness"),
                ("honesty", "helpfulness"),
                ("harmlessness", "truthfulness"),
            ]
        else:
            self.candidate_pairs = candidate_pairs

    def compute_correlation_matrix(self, scores_df) -> None:
        """
        Step 1: Compute Pearson correlation matrix across all principle
        score pairs over the training set. Saves as a heatmap figure.

        Args:
            scores_df: DataFrame with columns = principle names, rows = samples.
        """
        corr = scores_df.corr(method="pearson")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr, 
            annot=True, 
            cmap="Purples", 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5
        )
        plt.title("Pearson Correlation Between Principle Scores")
        
        os.makedirs("results/figures", exist_ok=True)
        plt.savefig("results/figures/correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("Correlation matrix saved to results/figures/correlation_matrix.png")

    def detect(self, score_vector: dict) -> list:
        """
        Step 2: Given a score vector for one sample, return all detected
        conflict pairs.

        Args:
            score_vector: Dict mapping principle name -> float score in [0, 1].
                          e.g. {'harmlessness': 0.2, 'helpfulness': 0.85, ...}

        Returns:
            List of conflicting (pi, pj) tuples. Empty list if no conflict.
        """
        detected_conflicts = []
        
        for pi, pj in self.candidate_pairs:
            if pi not in score_vector or pj not in score_vector:
                continue
                
            si = score_vector[pi]
            sj = score_vector[pj]
            
            # Condition 1: pi is low (violates), pj is high (satisfies)
            if si < TAU_LOW and sj > TAU_HIGH:
                detected_conflicts.append((pi, pj))
            # Condition 2: pj is low (violates), pi is high (satisfies)
            elif sj < TAU_LOW and si > TAU_HIGH:
                detected_conflicts.append((pi, pj))
                
        return detected_conflicts
