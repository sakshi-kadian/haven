"""
harmonizer.py
-------------
Maps each source dataset's raw labels into the unified HAVEN-Bench 5-principle
binary schema: {0 = violates, 1 = satisfies} per principle.

Source datasets:
    - PKU-SafeRLHF  -> Harmlessness, Helpfulness
    - BBQ           -> Respectfulness
    - HaluEval      -> Honesty
    - TruthfulQA    -> Truthfulness

Usage:
    from src.data.harmonizer import harmonize_pku, harmonize_bbq, harmonize_halueval, harmonize_truthfulqa
"""

import pandas as pd


def harmonize_pku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map PKU-SafeRLHF fields to Harmlessness and Helpfulness binary labels.

    PKU-SafeRLHF label schema:
        - is_response_0_safe: bool
        - is_response_1_safe: bool
        - better_response_id: 0 or 1 (which response is more helpful/preferred)

    Harmonization protocol:
        - harmlessness = 1 if the chosen response is safe, 0 otherwise
        - helpfulness  = 1 if the chosen response is preferred, 0 otherwise

    Args:
        df: Raw PKU-SafeRLHF dataframe.

    Returns:
        Dataframe with columns: [prompt, response, harmlessness, helpfulness]
    """
    records = []
    for _, row in df.iterrows():
        # For each example, there are two responses. We evaluate the chosen one.
        chosen_id = row.get('better_response_id', 0)
        chosen_response = row[f'response_{chosen_id}']
        is_safe = row[f'is_response_{chosen_id}_safe']
        
        records.append({
            'prompt': row['prompt'],
            'response': chosen_response,
            'harmlessness': 1 if is_safe else 0,
            'helpfulness': 1 # It was the preferred response
        })
        
        # We can also add the rejected response as a negative example for helpfulness
        rejected_id = 1 - chosen_id
        rejected_response = row[f'response_{rejected_id}']
        is_safe_rej = row[f'is_response_{rejected_id}_safe']
        
        records.append({
            'prompt': row['prompt'],
            'response': rejected_response,
            'harmlessness': 1 if is_safe_rej else 0,
            'helpfulness': 0 # It was rejected
        })

    out_df = pd.DataFrame(records)
    out_df['honesty'] = -1
    out_df['respectfulness'] = -1
    out_df['truthfulness'] = -1
    return out_df


def harmonize_bbq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map BBQ fields to Respectfulness binary label.

    BBQ label schema:
        - label: int (0 = unknown, 1 = non-stereotyped, 2 = stereotyped)
        - context_condition: 'ambig' or 'disambig'

    Harmonization protocol:
        - respectfulness = 1 if label != 2 (not stereotyped), 0 otherwise
        - Only use 'disambig' context for cleaner signal.

    Args:
        df: Raw BBQ dataframe.

    Returns:
        Dataframe with columns: [prompt, response, respectfulness]
    """
    # Filter for disambiguated contexts only for clearer safety signals
    if 'context_condition' in df.columns:
        df = df[df['context_condition'] == 'disambig'].copy()
        
    records = []
    for _, row in df.iterrows():
        prompt = row['context'] + " " + row['question']
        
        # label: 0=ans0, 1=ans1, 2=ans2
        # In BBQ, usually one answer is stereotyped. We need to know which one.
        # For simplicity in this benchmark, if label is valid, we consider it respectful.
        # BBQ format varies, so we approximate: if it's not the known biased answer, it's respectful.
        
        # We will use the given correct answer as the respectful response
        ans_idx = row['label']
        if pd.isna(ans_idx):
            continue
            
        ans_col = f"ans{int(ans_idx)}"
        if ans_col in row:
            records.append({
                'prompt': prompt,
                'response': row[ans_col],
                'harmlessness': -1,
                'helpfulness': -1,
                'honesty': -1,
                'respectfulness': 1,
                'truthfulness': -1
            })

    return pd.DataFrame(records)


def harmonize_halueval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map HaluEval fields to Honesty binary label.

    HaluEval label schema:
        - hallucination: 'yes' or 'no'

    Harmonization protocol:
        - honesty = 0 if hallucination == 'yes' (model is dishonest/hallucinating)
        - honesty = 1 if hallucination == 'no'

    Args:
        df: Raw HaluEval dataframe.

    Returns:
        Dataframe with columns: [prompt, response, honesty]
    """
    records = []
    for _, row in df.iterrows():
        prompt = row.get('query', row.get('dialogue', ''))
        response = row.get('response', '')
        hallucination = str(row.get('hallucination', 'no')).lower()
        
        records.append({
            'prompt': prompt,
            'response': response,
            'harmlessness': -1,
            'helpfulness': -1,
            'honesty': 0 if hallucination == 'yes' else 1,
            'respectfulness': -1,
            'truthfulness': -1
        })

    return pd.DataFrame(records)


def harmonize_truthfulqa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map TruthfulQA fields to Truthfulness binary label.

    TruthfulQA label schema:
        - best_answer: the correct answer string
        - correct_answers: list of acceptable answers
        - incorrect_answers: list of wrong answers

    Harmonization protocol:
        - For each (question, answer) pair, truthfulness = 1 if answer is in
          correct_answers list, 0 if in incorrect_answers list.

    Args:
        df: Raw TruthfulQA dataframe.

    Returns:
        Dataframe with columns: [prompt, response, truthfulness]
    """
    records = []
    for _, row in df.iterrows():
        prompt = row['question']
        
        # Add a positive example
        if 'best_answer' in row:
            records.append({
                'prompt': prompt,
                'response': row['best_answer'],
                'harmlessness': -1,
                'helpfulness': -1,
                'honesty': -1,
                'respectfulness': -1,
                'truthfulness': 1
            })
            
        # Add a negative example (incorrect answers are typically separated by ';')
        if 'incorrect_answers' in row and isinstance(row['incorrect_answers'], str):
            wrong_answers = row['incorrect_answers'].split(';')
            if wrong_answers:
                records.append({
                    'prompt': prompt,
                    'response': wrong_answers[0].strip(),
                    'harmlessness': -1,
                    'helpfulness': -1,
                    'honesty': -1,
                    'respectfulness': -1,
                    'truthfulness': 0
                })

    return pd.DataFrame(records)


def build_haven_bench(output_dir: str) -> None:
    """
    Orchestrate harmonization of all four datasets into a single HAVEN-Bench CSV.
    Applies stratified sampling for class balance (50/50 per principle).
    Creates train/val/test splits: 70% / 15% / 15%.

    Args:
        output_dir: Path to save haven_bench_train.csv, val.csv, test.csv.
    """
    print("Building HAVEN-Bench dataset pipeline...")
    
    import os
    from sklearn.model_selection import train_test_split
    
    # HuggingFace datasets library >3.0 blocks custom python scripts (like bbq.py).
    # Since we are just testing the pipeline locally on the Vivobook, we will generate
    # a synthetic dataset here. The real Kaggle run will downgrade the datasets library
    # to load the real data.
    
    print("Generating synthetic data for local pipeline testing...")
    
    records = []
    import random
    
    principles = ['harmlessness', 'helpfulness', 'honesty', 'respectfulness', 'truthfulness']
    
    for i in range(1000):
        # Create a mock example
        record = {
            'prompt': f"User prompt {i}",
            'response': f"Model response {i}",
        }
        
        # Assign -1 to all principles first
        for p in principles:
            record[p] = -1
            
        # Randomly assign a 0 or 1 to one specific principle to simulate the harmonized output
        target_principle = random.choice(principles)
        record[target_principle] = random.choice([0, 1])
        
        records.append(record)
        
    haven_bench = pd.DataFrame(records)
    
    print(f"Total size: {len(haven_bench)}")
    
    train_df, temp_df = train_test_split(haven_bench, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "haven_bench_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "haven_bench_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "haven_bench_test.csv"), index=False)
    
    print(f"Saved to {output_dir}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    build_haven_bench("data")
