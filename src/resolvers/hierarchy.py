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
        if not conflicts:
            # If no conflict, just return majority vote of all principles
            return 1 if sum(score_vector.values()) / len(score_vector) >= 0.5 else 0

        # Gather all principles involved in any conflict for this sample
        conflicted_principles = set()
        for pi, pj in conflicts:
            conflicted_principles.add(pi)
            conflicted_principles.add(pj)

        # Find the highest priority principle among the conflicted ones
        highest_priority_principle = None
        for p in PRIORITY_ORDER:
            if p in conflicted_principles:
                highest_priority_principle = p
                break

        # The dominant principle decides the final label (threshold 0.5)
        if highest_priority_principle:
            return 1 if score_vector[highest_priority_principle] >= 0.5 else 0
        
        # Fallback
        return 1 if sum(score_vector.values()) / len(score_vector) >= 0.5 else 0
