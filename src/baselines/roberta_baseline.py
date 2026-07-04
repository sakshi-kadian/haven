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
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.principle = principle
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Binary classification (0=unsafe, 1=safe)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def train(self, train_df, val_df, output_dir: str = "checkpoints/roberta") -> None:
        """Fine-tune on HAVEN-Bench training split."""
        import torch
        from transformers import Trainer, TrainingArguments
        from datasets import Dataset
        import os
        
        # Prepare datasets
        def preprocess(examples):
            # Combine prompt and response
            texts = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(examples['prompt'], examples['response'])]
            return self.tokenizer(texts, truncation=True, padding="max_length", max_length=256)
            
        train_ds = Dataset.from_pandas(train_df[['prompt', 'response', self.principle]])
        train_ds = train_ds.rename_column(self.principle, "label")
        train_ds = train_ds.map(preprocess, batched=True)
        
        val_ds = Dataset.from_pandas(val_df[['prompt', 'response', self.principle]])
        val_ds = val_ds.rename_column(self.principle, "label")
        val_ds = val_ds.map(preprocess, batched=True)
        
        # Set up training arguments
        out_path = os.path.join(output_dir, self.principle)
        training_args = TrainingArguments(
            output_dir=out_path,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer
        )
        
        self.trainer.train()

    def predict(self, df) -> list:
        """Run inference and return binary predictions."""
        from datasets import Dataset
        import numpy as np
        
        def preprocess(examples):
            texts = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(examples['prompt'], examples['response'])]
            return self.tokenizer(texts, truncation=True, padding="max_length", max_length=256)
            
        test_ds = Dataset.from_pandas(df[['prompt', 'response']])
        test_ds = test_ds.map(preprocess, batched=True)
        
        predictions = self.trainer.predict(test_ds)
        preds = np.argmax(predictions.predictions, axis=-1)
        return preds.tolist()
