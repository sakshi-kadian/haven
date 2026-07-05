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
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.constitution import PRINCIPLES, PROMPT_TEMPLATE


def get_qlora_config() -> LoraConfig:
    """Return the LoRA configuration for PEFT."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_quantization_config() -> BitsAndBytesConfig:
    """Return BitsAndBytes 4-bit quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def train(principle: str, output_dir: str) -> None:
    """
    Full training loop for a single principle classifier.

    Loads the HAVEN-Bench train/val CSVs, formats them into
    constitutional prompts, applies QLoRA, and saves the adapter
    weights to output_dir.

    Args:
        principle:  One of the 5 constitutional principles.
        output_dir: Directory to save the adapter weights.
    """
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    principle_data = PRINCIPLES[principle]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_quantization_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, get_qlora_config())
    model.print_trainable_parameters()

    train_df = pd.read_csv("data/haven_bench_train.csv")
    val_df = pd.read_csv("data/haven_bench_val.csv")

    # Filter to rows that have a valid label for this principle
    train_df = train_df[train_df[principle] != -1].reset_index(drop=True)
    val_df = val_df[val_df[principle] != -1].reset_index(drop=True)

    def format_row(row):
        prompt = PROMPT_TEMPLATE.format(
            principle_name=principle_data["name"],
            principle_description=principle_data["description"],
            prompt=row["prompt"],
            response=row["response"],
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"text": input_text + str(int(row[principle]))}

    train_ds = Dataset.from_pandas(train_df).map(format_row)
    val_ds = Dataset.from_pandas(val_df).map(format_row)

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter weights saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a constitutional principle classifier.")
    parser.add_argument(
        "--principle",
        type=str,
        required=True,
        choices=["harmlessness", "honesty", "helpfulness", "respectfulness", "truthfulness"],
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    train(args.principle, args.output_dir)
