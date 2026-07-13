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

    def __init__(self, principle: str = "label", model_name: str = "roberta-base"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.principle = principle
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Binary classification (0=unsafe, 1=safe)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
        if self.principle != "label":
            train_ds = train_ds.rename_column(self.principle, "label")
        train_ds = train_ds.map(preprocess, batched=True)
        
        val_ds = Dataset.from_pandas(val_df[['prompt', 'response', self.principle]])
        if self.principle != "label":
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
        self.trainer.save_model(out_path)  # Save the final model directly to out_path

    def predict(self, df) -> list:
        """Run inference and return binary predictions."""
        from datasets import Dataset
        import numpy as np
        import os
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint
        
        # Load fine-tuned weights if they exist and we haven't trained in this session
        if not hasattr(self, "trainer"):
            ckpt_path = os.path.join("checkpoints/roberta", self.principle)
            if os.path.exists(ckpt_path):
                from transformers import AutoModelForSequenceClassification
                
                # Determine where to load from: root or a checkpoint-XXX subdirectory
                last_checkpoint = get_last_checkpoint(ckpt_path)
                load_path = ckpt_path
                
                if not os.path.exists(os.path.join(ckpt_path, "config.json")):
                    if last_checkpoint is not None:
                        load_path = last_checkpoint
                    else:
                        checkpoints = [os.path.join(ckpt_path, d) for d in os.listdir(ckpt_path) if d.startswith("checkpoint-")]
                        if checkpoints:
                            load_path = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
                
                self.model = AutoModelForSequenceClassification.from_pretrained(load_path).to(self.device)
            
            # Create a dummy trainer for prediction
            training_args = TrainingArguments(output_dir="checkpoints/temp", report_to="none")
            self.trainer = Trainer(model=self.model, args=training_args)
        
        def preprocess(examples):
            texts = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(examples['prompt'], examples['response'])]
            return self.tokenizer(texts, truncation=True, padding="max_length", max_length=256)
            
        test_ds = Dataset.from_pandas(df[['prompt', 'response']])
        test_ds = test_ds.map(preprocess, batched=True)
        
        predictions = self.trainer.predict(test_ds)
        preds = np.argmax(predictions.predictions, axis=-1)
        return preds.tolist()

