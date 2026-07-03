"""
dataset.py
----------
PyTorch Dataset class for HAVEN-Bench.

Loads the harmonized CSV files and prepares tokenized inputs for
the Phi-3 Mini constitutional encoder.

Usage:
    from src.data.dataset import HavenBenchDataset
    train_ds = HavenBenchDataset("data/haven_bench_train.csv", tokenizer, principle="harmlessness")
"""

from torch.utils.data import Dataset


class HavenBenchDataset(Dataset):
    """
    PyTorch Dataset for a single principle's binary classification task.

    Args:
        csv_path:  Path to the HAVEN-Bench split CSV file.
        tokenizer: HuggingFace tokenizer for Phi-3 Mini.
        principle: One of ['harmlessness', 'honesty', 'helpfulness',
                           'respectfulness', 'truthfulness'].
        max_length: Maximum token length for truncation (default: 512).
    """

    def __init__(self, csv_path: str, tokenizer, principle: str, max_length: int = 512):
        import pandas as pd
        from src.constitution import PRINCIPLES, PROMPT_TEMPLATE
        
        if principle not in PRINCIPLES:
            raise ValueError(f"Unknown principle: {principle}")
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.principle_name = PRINCIPLES[principle]["name"]
        self.principle_desc = PRINCIPLES[principle]["description"]
        self.prompt_template = PROMPT_TEMPLATE
        
        # Load data and filter out rows where the principle label is missing (-1)
        df = pd.read_csv(csv_path)
        self.data = df[df[principle] != -1].copy().reset_index(drop=True)
        self.labels = self.data[principle].tolist()
        self.prompts = self.data['prompt'].tolist()
        self.responses = self.data['response'].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys: input_ids, attention_mask, labels
        """
        prompt = self.prompts[idx]
        response = self.responses[idx]
        label = self.labels[idx]
        
        # Format the full input string using the constitutional template
        full_text = self.prompt_template.format(
            principle_name=self.principle_name,
            principle_description=self.principle_desc,
            prompt=prompt,
            response=response
        )
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Squeeze the batch dimension added by return_tensors="pt"
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        
        import torch
        item["labels"] = torch.tensor(label, dtype=torch.long)
        
        return item
