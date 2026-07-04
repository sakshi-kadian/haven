"""
trainer.py
----------
QLoRA fine-tuning loop for the Phi-3 Mini constitutional encoder.

Trains one binary classifier per constitutional principle using
Parameter-Efficient Fine-Tuning (PEFT) with 4-bit quantization.

Config:
    - Base model: microsoft/Phi-3-mini-4k-instruct
    - Quantization: 4-bit NF4 (bitsandbytes)
    - LoRA rank: r=16, alpha=32, dropout=0.05
    - Learning rate: 2e-4
    - Epochs: 3
    - Batch size: 4 (gradient accumulation steps=4 for effective batch=16)

Usage (on Kaggle):
    python src/trainer.py --principle harmlessness --output_dir checkpoints/phi3_harmlessness
"""

import argparse


def get_qlora_config():
    """Return the LoRA configuration for PEFT."""
    raise NotImplementedError("TODO: Implement this method.")


def get_quantization_config():
    """Return BitsAndBytes 4-bit quantization config."""
    raise NotImplementedError("TODO: Implement this method.")


def train(principle: str, output_dir: str) -> None:
    """
    Full training loop for a single principle classifier.

    Args:
        principle:  One of the 5 constitutional principles.
        output_dir: Directory to save the adapter weights.
    """
    raise NotImplementedError("TODO: Implement this method.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a constitutional principle classifier.")
    parser.add_argument("--principle", type=str, required=True,
                        choices=["harmlessness", "honesty", "helpfulness",
                                 "respectfulness", "truthfulness"])
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    train(args.principle, args.output_dir)
