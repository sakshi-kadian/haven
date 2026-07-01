"""
roberta_baseline.py
-------------------
RoBERTa fine-tuned baseline for safety evaluation.

Fine-tunes roberta-base independently on each of the 5 principles
using standard sequence classification (no QLoRA, full fine-tune).

This is Baseline 2. Expected accuracy: ~70%.
Demonstrates that a standard encoder model trained on individual
principles performs better than keyword matching but worse than HAVEN.

Run on Kaggle (requires GPU).

Usage:
    from src.baselines.roberta_baseline import RobertaBaseline
    baseline = RobertaBaseline(principle="harmlessness")
    baseline.train(train_df, val_df)
    predictions = baseline.predict(test_df)
"""


class RobertaBaseline:
    """
    Fine-tuned RoBERTa binary classifier for a single principle.

    Args:
        principle:  One of the 5 constitutional principles.
        model_name: HuggingFace model ID (default: 'roberta-base').
    """

    def __init__(self, principle: str, model_name: str = "roberta-base"):
        raise NotImplementedError("Implement on Day 4.")

    def train(self, train_df, val_df) -> None:
        """Fine-tune on HAVEN-Bench training split."""
        raise NotImplementedError("Implement on Day 4.")

    def predict(self, df) -> list:
        """Run inference and return binary predictions."""
        raise NotImplementedError("Implement on Day 4.")
