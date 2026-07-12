"""
run_comparison.py
-----------------
Master experiment runner for the HAVEN framework.

Orchestrates the full evaluation pipeline on Kaggle:
    1. Load test dataset from data/test.csv
    2. Run all 4 classifiers (rule-based, RoBERTa, monolithic, HAVEN)
    3. Compute per-principle and overall metrics
    4. Run McNemar significance tests vs. each baseline
    5. Compute ECE and temperature-scaled calibration
    6. Save all results to results/

Usage (on Kaggle T4 GPU):
    python experiments/run_comparison.py --checkpoint_dir checkpoints/ --data_path data/test.csv
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.constitution import PRINCIPLE_NAMES
from src.evaluator import HavenEvaluator
from src.metrics import compute_metrics, mcnemar_test, generate_report
from src.calibration import compute_ece
from src.baselines.rule_based import RuleBasedClassifier
from src.baselines.roberta_baseline import RobertaBaseline
from src.analysis import extract_case_studies


def parse_args():
    parser = argparse.ArgumentParser(description="HAVEN vs. Baseline Comparison")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/",
                        help="Directory with trained principle adapter checkpoints.")
    parser.add_argument("--data_path", type=str, default="data/test.csv",
                        help="Path to the test CSV file.")
    parser.add_argument("--resolver", type=str, default="gating",
                        choices=["hierarchy", "voting", "gating"],
                        help="Conflict resolution strategy.")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save all output files.")
    return parser.parse_args()


def run_haven(df: pd.DataFrame, checkpoint_dir: str, resolver: str) -> list:
    """Run the full HAVEN pipeline on all test rows. Returns list of reports."""
    evaluator = HavenEvaluator(
        checkpoint_dir=checkpoint_dir,
        resolver_strategy=resolver,
    )
    reports = []
    for _, row in df.iterrows():
        report = evaluator.evaluate(
            prompt=str(row.get("prompt", "")),
            response=str(row.get("response", "")),
        )
        report["true_label"] = int(row.get("label", -1))
        reports.append(report)
    return reports


def run_rule_based(df: pd.DataFrame) -> list:
    """Run the rule-based baseline on all test rows."""
    clf = RuleBasedClassifier()
    return clf.predict(df).tolist()


def run_roberta(df: pd.DataFrame) -> list:
    """Run the RoBERTa baseline."""
    import os
    roberta_checkpoint = "checkpoints/roberta/label"
    if not os.path.exists(roberta_checkpoint):
        print("  RoBERTa baseline checkpoint not found. Fine-tuning it now (takes ~2 mins on GPU)...")
        train_path = "data/haven_bench_train.csv"
        val_path = "data/haven_bench_val.csv"
        if os.path.exists(train_path) and os.path.exists(val_path):
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            clf = RobertaBaseline(principle="label")
            clf.train(train_df, val_df)
        else:
            print("  Warning: Train/Val files not found. Using untrained RoBERTa.")
            clf = RobertaBaseline(principle="label")
    else:
        clf = RobertaBaseline(principle="label")
    return clf.predict(df)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    y_true = df["label"].tolist()

    # Step 1: Run HAVEN
    print("\n[1/4] Running HAVEN evaluator...")
    haven_reports = run_haven(df, args.checkpoint_dir, args.resolver)
    haven_preds = [r["resolved_label"] for r in haven_reports]

    # Step 2: Run baselines
    print("[2/4] Running rule-based baseline...")
    rule_preds = run_rule_based(df)

    print("[3/4] Running RoBERTa baseline...")
    roberta_preds = run_roberta(df)

    # Step 3: Compute metrics for all models
    print("[4/4] Computing metrics and significance tests...")

    haven_metrics   = compute_metrics(y_true, haven_preds,   "haven_overall")
    rule_metrics    = compute_metrics(y_true, rule_preds,    "rule_based")
    roberta_metrics = compute_metrics(y_true, roberta_preds, "roberta")

    all_metrics = [haven_metrics, rule_metrics, roberta_metrics]
    generate_report(all_metrics, os.path.join(args.output_dir, "metrics.json"))

    # Step 4: McNemar significance tests (HAVEN vs. each baseline)
    mcnemar_results = {
        "haven_vs_rule_based": mcnemar_test(y_true, haven_preds, rule_preds),
        "haven_vs_roberta":    mcnemar_test(y_true, haven_preds, roberta_preds),
    }
    mcnemar_path = os.path.join(args.output_dir, "mcnemar.json")
    with open(mcnemar_path, "w") as f:
        json.dump(mcnemar_results, f, indent=2)
    print(f"  McNemar results saved to {mcnemar_path}")

    # Step 5: Calibration — ECE per principle
    ece_results = {}
    for principle in PRINCIPLE_NAMES:
        conf_col = f"{principle}_confidence"
        if conf_col in df.columns and "label" in df.columns:
            confs  = df[conf_col].tolist()
            labels = df["label"].tolist()
            ece_results[principle] = compute_ece(confs, labels)

    ece_path = os.path.join(args.output_dir, "ece.json")
    with open(ece_path, "w") as f:
        json.dump(ece_results, f, indent=2)
    print(f"  ECE results saved to {ece_path}")

    # Step 6: Extract qualitative case studies
    preds_path = os.path.join(args.output_dir, "predictions.csv")
    df["pred_label"] = haven_preds
    df.to_csv(preds_path, index=False)

    case_studies = extract_case_studies(preds_path, n=10)
    cs_path = os.path.join(args.output_dir, "case_studies.json")
    with open(cs_path, "w") as f:
        json.dump(case_studies, f, indent=2)
    print(f"  Case studies saved to {cs_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  HAVEN    — Accuracy: {haven_metrics['accuracy']:.4f}  F1: {haven_metrics['f1']:.4f}")
    print(f"  RuleBased— Accuracy: {rule_metrics['accuracy']:.4f}  F1: {rule_metrics['f1']:.4f}")
    print(f"  RoBERTa  — Accuracy: {roberta_metrics['accuracy']:.4f}  F1: {roberta_metrics['f1']:.4f}")
    print(f"\n  McNemar HAVEN vs RuleBased: p={mcnemar_results['haven_vs_rule_based']['p_value']} "
          f"significant={mcnemar_results['haven_vs_rule_based']['significant']}")
    print(f"  McNemar HAVEN vs RoBERTa:   p={mcnemar_results['haven_vs_roberta']['p_value']} "
          f"significant={mcnemar_results['haven_vs_roberta']['significant']}")
    print("=" * 60)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
