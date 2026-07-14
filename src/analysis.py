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

import json
import os
import pandas as pd
from src.constitution import PRINCIPLE_NAMES


def run_ablation(predictions_path: str, output_path: str) -> None:
    """
    Run ablation study: evaluate HAVEN with 1, 2, 3, 4, 5 active principles.
    Measures the accuracy impact of progressive decomposition.

    Args:
        predictions_path: Path to results/predictions.csv
        output_path:      Path to save ablation_results.json
    """
    df = pd.read_csv(predictions_path)

    results = []

    for n_principles in range(1, len(PRINCIPLE_NAMES) + 1):
        active = PRINCIPLE_NAMES[:n_principles]

        # For each sample, compute majority vote using only the active principles
        correct = 0
        total = len(df)

        for _, row in df.iterrows():
            votes = [row[f"{p}_pred"] for p in active if f"{p}_pred" in df.columns]
            if not votes:
                continue
            majority = 1 if sum(votes) / len(votes) >= 0.5 else 0
            if majority == row["true_label"]:
                correct += 1

        accuracy = round(correct / total, 4) if total > 0 else 0.0
        results.append({
            "n_principles": n_principles,
            "active_principles": active,
            "accuracy": accuracy,
        })

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ablation results saved to {output_path}")


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
    df = pd.read_csv(predictions_path)

    # Isolate misclassified rows
    errors = df[df["true_label"] != df["pred_label"]].copy()

    categories = {
        "sarcasm_irony": [],
        "cultural_context": [],
        "inherent_ambiguity": [],
        "edge_cases": [],
    }

    for _, row in errors.iterrows():
        confidence = abs(row.get("confidence", 0.5) - 0.5)
        text = str(row.get("response", "")).lower()

        # Heuristic categorization (production system would use a classifier)
        if any(w in text for w in ["obviously", "clearly", "as if", "yeah right", "totally"]):
            categories["sarcasm_irony"].append(row.to_dict())
        elif confidence < 0.1:
            # Model was near the decision boundary -> inherent ambiguity
            categories["inherent_ambiguity"].append(row.to_dict())
        elif confidence < 0.15:
            # Near-threshold: edge case
            categories["edge_cases"].append(row.to_dict())
        else:
            categories["cultural_context"].append(row.to_dict())

    # Write markdown report
    lines = ["# Error Analysis Report\n"]
    for cat, items in categories.items():
        lines.append(f"\n## {cat.replace('_', ' ').title()} ({len(items)} errors)\n")
        for item in items[:3]:  # Show top 3 examples per category
            lines.append(f"- **Response**: {str(item.get('response', ''))[:120]}...")
            lines.append(f"  - True: `{item.get('true_label')}` | Pred: `{item.get('pred_label')}`\n")

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Error analysis saved to {output_path}")


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
    df = pd.read_csv(predictions_path)
    case_studies = []

    for principle in PRINCIPLE_NAMES:
        pred_col = principle
        conf_col = f"{principle}_confidence"

        if pred_col not in df.columns:
            continue

        errors = df[df["label"] != df[pred_col]].copy()

        # Sort by highest confidence wrong predictions (most interesting failures)
        if conf_col in errors.columns:
            errors = errors.sort_values(by=conf_col, ascending=False)

        for _, row in errors.head(n).iterrows():
            prompt_val = row.get("prompt", "")
            resp_val = row.get("response", "")
            
            case_studies.append({
                "principle": principle,
                "prompt": "" if pd.isna(prompt_val) else str(prompt_val),
                "response": "" if pd.isna(resp_val) else str(resp_val),
                "true_label": int(row.get("label", -1)),
                "pred_label": int(row.get(pred_col, -1)),
                "confidence": float(row.get(conf_col, 0.5)),
            })

    return case_studies
