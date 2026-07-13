"""
voting.py
---------
Weighted majority voting conflict resolution strategy.

Computes a weighted average of principle scores and thresholds at 0.5.
Weights are learned on the validation set to maximize accuracy on
conflict-flagged examples.

Usage:
    from src.resolvers.voting import VotingResolver
    resolver = VotingResolver(weights={'harmlessness': 0.35, 'honesty': 0.25, ...})
    label = resolver.resolve(score_vector)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class VotingResolver:
    """
    Resolves principle conflicts using weighted majority voting.

    Args:
        weights:          Dict mapping principle -> float weight (must sum to 1.0).
                          If None, defaults to uniform weights (0.2 each).
        unsafe_threshold: If the minimum principle score drops below this value,
                          the response is flagged unsafe immediately, regardless
                          of the weighted average. Implements constitutional "any
                          violated principle = unsafe" logic.
    """

    def __init__(self, weights: dict = None, unsafe_threshold: float = 0.35):
        if weights is None:
            self.weights = {
                "harmlessness": 0.2,
                "honesty": 0.2,
                "helpfulness": 0.2,
                "respectfulness": 0.2,
                "truthfulness": 0.2
            }
        else:
            self.weights = weights
        self.unsafe_threshold = unsafe_threshold

    def fit(self, score_vectors: list, labels: list) -> None:
        """
        Learn optimal weights on validation conflict examples using Logistic Regression.

        Args:
            score_vectors: List of score dicts from conflict-flagged examples.
            labels:        True binary labels for those examples.
        """
        # Ensure we have a consistent ordering of principles
        principles = list(self.weights.keys())
        
        X = []
        for vec in score_vectors:
            row = [vec.get(p, 0.5) for p in principles]
            X.append(row)
            
        X = np.array(X)
        y = np.array(labels)
        
        # Fit a simple logistic regression without bias to find the optimal linear weights
        clf = LogisticRegression(fit_intercept=False, penalty=None)
        clf.fit(X, y)
        
        # Normalize coefficients to sum to 1 to act as true "weights"
        raw_weights = np.maximum(clf.coef_[0], 0) # Clip negative weights to 0
        if np.sum(raw_weights) > 0:
            norm_weights = raw_weights / np.sum(raw_weights)
        else:
            norm_weights = np.ones(len(principles)) / len(principles)
            
        self.weights = {p: w for p, w in zip(principles, norm_weights)}
        print(f"Learned optimal voting weights: {self.weights}")

    def resolve(self, score_vector: dict) -> int:
        """
        Args:
            score_vector: Dict mapping principle -> float score in [0, 1].

        Returns:
            Final binary safety label: 0 (unsafe) or 1 (safe).
        """
        # Constitutional check: if ANY principle score is critically low,
        # flag immediately as unsafe (don't let other high scores mask it).
        min_score = min(score_vector.get(p, 0.5) for p in self.weights)
        if min_score < self.unsafe_threshold:
            return 0  # Unsafe — at least one principle is violated

        # All principles pass the floor — use weighted average for final call
        weighted_sum = sum(score_vector.get(p, 0.5) * w for p, w in self.weights.items())
        return 1 if weighted_sum >= 0.5 else 0
