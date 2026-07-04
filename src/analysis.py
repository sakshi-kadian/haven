"""
analysis.py
-----------
Error analysis and ablation studies for the HAVEN framework.

Implements:
    - Error categorization (sarcasm, cultural context, ambiguity, edge cases)
    - Ablation study: accuracy vs. number of active principles (1 to 5)
    - Qualitative case study extraction (top-10 hardest examples per principle)

Usage:
    from src.analysis import run_ablation, categorize_errors, extract_case_studies
"""


def run_ablation(predictions_path: str, output_path: str) -> None:
    """
    Run ablation study: evaluate HAVEN with 1, 2, 3, 4, 5 active principles.
    Measures the accuracy impact of progressive decomposition.

    Args:
        predictions_path: Path to results/predictions.csv
        output_path:      Path to save ablation_results.json
    """
    raise NotImplementedError("TODO: Implement this method.")


def categorize_errors(predictions_path: str, output_path: str) -> None:
    """
    Classify prediction errors into categories:
        - sarcasm / irony
        - cultural context dependency
        - inherent ambiguity (reasonable disagreement)
        - edge cases (near-threshold scores)

    Args:
        predictions_path: Path to results/predictions.csv
        output_path:      Path to save error_analysis.md
    """
    raise NotImplementedError("TODO: Implement this method.")


def extract_case_studies(predictions_path: str, n: int = 10) -> list:
    """
    Extract the n hardest examples per principle (highest model uncertainty
    or incorrect high-confidence predictions).

    Args:
        predictions_path: Path to results/predictions.csv
        n:                Number of examples to extract per principle.

    Returns:
        List of dicts: [{'principle', 'prompt', 'response', 'true_label',
                          'pred_label', 'confidence'}, ...]
    """
    raise NotImplementedError("TODO: Implement this method.")
