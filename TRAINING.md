# Haven: Kaggle Training & Evaluation Guide


This guide provides the end-to-end pipeline for the Haven project. By following these steps in a Kaggle notebook, you can execute the entire research workflow: from dataset harmonization and QLoRA fine-tuning, to statistical evaluation and chart generation.

---

## Requirements

- Create a **Kaggle** account
- Create a new notebook: click **Create > Notebook**
- Enable GPU: inside the notebook, click **Settings > Accelerator > GPU T4 x2**

---

## Step 1: Clone the Repository (Kaggle cell)

```python
import subprocess
result = subprocess.run(
    ["git", "clone", "https://github.com/YOUR_USERNAME/haven.git"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
%cd haven
```

---

## Step 2: Install Dependencies (Kaggle cell)

```python
!pip install -q -r requirements.txt
print("All dependencies installed.")
```

---

## Step 3: Build Haven-Bench

Harmonizes four public safety datasets (PKU-SafeRLHF, BBQ, HaluEval, TruthfulQA)
into a single benchmark with a 70/15/15 train/val/test split.

```python
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.insert(0, '.')

from src.data.harmonizer import (
    harmonize_pku,
    harmonize_bbq,
    harmonize_halueval,
    harmonize_truthfulqa,
)

# PKU
print("Loading PKU-SafeRLHF...")
pku_raw = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train[:3000]")
pku_df = harmonize_pku(pku_raw.to_pandas())
print(f"  PKU done: {len(pku_df)} rows")

# BBQ
print("Loading BBQ (Directly from Source)...")
bbq_raw_df = pd.read_json("https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Age.jsonl", lines=True).head(1000)
bbq_df = harmonize_bbq(bbq_raw_df)
print(f"  BBQ done: {len(bbq_df)} rows")

# HaluEval
print("Loading HaluEval...")
halu_raw = load_dataset("pminervini/HaluEval", "qa", split="data[:2000]")
halu_df = harmonize_halueval(halu_raw.to_pandas())
print(f"  HaluEval done: {len(halu_df)} rows")

# TruthfulQA
print("Loading TruthfulQA...")
tqa_raw = load_dataset("truthful_qa", "generation", split="validation")
tqa_df = harmonize_truthfulqa(tqa_raw.to_pandas())
print(f"  TruthfulQA done: {len(tqa_df)} rows")

# Merge all datasets into a single Haven-Bench dataframe
haven_bench = pd.concat([pku_df, bbq_df, halu_df, tqa_df], ignore_index=True)
print(f"\nTotal HAVEN-Bench size: {len(haven_bench)}")

# 70/15/15 split
train_df, temp_df = train_test_split(haven_bench, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/haven_bench_train.csv", index=False)
val_df.to_csv("data/haven_bench_val.csv", index=False)
test_df.to_csv("data/haven_bench_test.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print("HAVEN-Bench saved to data/")
```

---

## Step 4: Train the 5 Principle Classifiers

Each principle fine-tunes a QLoRA adapter on top of Phi-3 Mini (3.8B parameters).
Run all 5 in sequence, or run each in a separate cell to avoid losing progress if a session times out.

```python
import subprocess

principles = ["harmlessness", "helpfulness", "honesty", "respectfulness", "truthfulness"]

for principle in principles:
    print(f"\n{'='*50}")
    print(f"Training: {principle.upper()}")
    print(f"{'='*50}")
    result = subprocess.run(
        ["python", "-m", "src.trainer",
         "--principle", principle,
         "--output_dir", f"checkpoints/phi3_{principle}"],
        capture_output=False  # Show live output
    )
    print(f"Done: {principle}")
```

To run a single principle individually:
```python
import subprocess
print("Training Harmlessness...")
subprocess.run(
    ["python", "-m", "src.trainer", "--principle", "harmlessness", "--output_dir", "checkpoints/phi3_harmlessness"],
    capture_output=False
)
print("Harmlessness completed.")
```

---

## Step 5: Run Full Evaluation

Runs Haven against all baselines (Rule-Based, RoBERTa), computes accuracy,
F1, ECE calibration, and McNemar significance tests.

```python
import subprocess
print("Running overall evaluation comparison...")
result = subprocess.run(
    ["python", "experiments/run_comparison.py", "--data_path", "data/haven_bench_test.csv"],
    capture_output=False
)
print("Evaluation completed.")
```

Results are saved to `results/`.

---

## Step 6: Generate Figures

```python
!jupyter nbconvert --to notebook --execute --inplace notebooks/05_figures.ipynb
print("Real figures generated!")
```

---

## Step 7: Verify Outputs

```python
import os, json

# Check all expected output files exist
expected = [
    "results/predictions.csv",
    "results/metrics.json",
    "results/mcnemar.json",
    "results/ece.json",
    "results/figures/ablation_study.png",
    "results/figures/correlation_matrix.png",
    "results/figures/reliability_harmlessness.png",
]

print("Output verification:")
for f in expected:
    status = "PASS" if os.path.exists(f) else "MISSING"
    print(f"  {status}  {f}")

# Print a summary of results
if os.path.exists("results/metrics.json"):
    with open("results/metrics.json") as f:
        metrics = json.load(f)
    print("\n=== HAVEN Results ===")
    print(json.dumps(metrics, indent=2))
```

---

## Step 8: Download Results (Optional)
```python
import shutil
shutil.make_archive('haven_results', 'zip', 'results')

from IPython.display import HTML
import base64

def create_download_link(title, filename):
    with open(filename, "rb") as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    payload = f'<a download="{filename.split("/")[-1]}" href="data:application/octet-stream;base64,{b64}" target="_blank" style="display:inline-block; padding:10px 20px; background-color:slategray; color:black; text-decoration:none; border-radius:5px; margin:5px; font-weight:bold;">{title}</a>'
    display(HTML(payload))

# Create the download buttons!
create_download_link("Download Results ZIP", "haven_results.zip")
create_download_link("Download Notebook", "notebooks/05_figures.ipynb")
```

---

## Notes

- Training all 5 principles takes approximately **5-8 hours** on a single **Kaggle T4 GPU (16 GB VRAM)** using Kaggle's free tier.
- Checkpoints are saved to `checkpoints/phi3_<principle>/`.
- All final outputs (metrics, figures, predictions) are saved to the `results/` folder.
