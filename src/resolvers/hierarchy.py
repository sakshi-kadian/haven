"""
hierarchy.py
------------
Priority-based conflict resolution strategy.

When a conflict is detected between two principles, the higher-priority
principle's verdict takes precedence.

Priority order (descending):
    Harmlessness > Honesty > Truthfulness > Respectfulness > Helpfulness

Rationale: Safety-critical properties (harm prevention) override
           user-preference properties (helpfulness).

Usage:
    from src.resolvers.hierarchy import HierarchyResolver
    resolver = HierarchyResolver()
    label = resolver.resolve(score_vector, conflicts)
"""

PRIORITY_ORDER = ["harmlessness", "honesty", "truthfulness", "respectfulness", "helpfulness"]


class HierarchyResolver:
    """Resolves principle conflicts using a fixed priority hierarchy."""

    def resolve(self, score_vector: dict, conflicts: list) -> int:
        """
        Args:
            score_vector: Dict mapping principle -> float score in [0, 1].
            conflicts:    List of conflicting (pi, pj) tuples.

        Returns:
            Final binary safety label: 0 (unsafe) or 1 (safe).
        """
        raise NotImplementedError("TODO: Implement this method.")
