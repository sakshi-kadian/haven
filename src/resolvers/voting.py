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


class VotingResolver:
    """
    Resolves principle conflicts using weighted majority voting.

    Args:
        weights: Dict mapping principle -> float weight (must sum to 1.0).
                 If None, defaults to uniform weights (0.2 each).
    """

    def __init__(self, weights: dict = None):
        raise NotImplementedError("Implement on Day 7.")

    def fit(self, score_vectors: list, labels: list) -> None:
        """
        Learn optimal weights on validation conflict examples.

        Args:
            score_vectors: List of score dicts from conflict-flagged examples.
            labels:        True binary labels for those examples.
        """
        raise NotImplementedError("Implement on Day 7.")

    def resolve(self, score_vector: dict) -> int:
        """
        Args:
            score_vector: Dict mapping principle -> float score in [0, 1].

        Returns:
            Final binary safety label: 0 (unsafe) or 1 (safe).
        """
        raise NotImplementedError("Implement on Day 7.")
