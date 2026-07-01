"""
zeroshot_llm.py
---------------
Zero-shot LLM-as-a-judge baseline.

Uses the un-finetuned Phi-3-Mini-4K-Instruct model with the EXACT same
prompt template as the fine-tuned constitutional encoder. No training required.

This is Baseline 4 (the 2026 industry standard baseline).
Expected accuracy: ~65-70%.

The gap between this baseline and HAVEN proves that QLoRA fine-tuning on
HAVEN-Bench provides a non-trivial, measurable improvement over zero-shot
prompting alone.

Run on Kaggle (requires GPU for reasonable speed).

Usage:
    from src.baselines.zeroshot_llm import ZeroShotLLMBaseline
    baseline = ZeroShotLLMBaseline()
    predictions = baseline.evaluate(df, principle="harmlessness")
"""


class ZeroShotLLMBaseline:
    """
    Zero-shot Phi-3 Mini instruction-following evaluator.
    Uses the base instruction-tuned model with no fine-tuning.

    Args:
        model_name: HuggingFace model ID for the base instruction model.
    """

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        raise NotImplementedError("Implement on Day 4.")

    def evaluate(self, df, principle: str) -> list:
        """
        Run zero-shot evaluation on a dataframe for a given principle.

        Args:
            df:        DataFrame with columns [prompt, response].
            principle: Constitutional principle to evaluate.

        Returns:
            List of binary predictions {0, 1}.
        """
        raise NotImplementedError("Implement on Day 4.")
