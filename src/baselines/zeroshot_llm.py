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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Load in 4-bit for memory efficiency on Kaggle T4
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    def evaluate(self, df, principle: str) -> list:
        """
        Run zero-shot evaluation on a dataframe for a given principle.

        Args:
            df:        DataFrame with columns [prompt, response].
            principle: Constitutional principle to evaluate.

        Returns:
            List of binary predictions {0, 1}.
        """
        from src.constitution import PRINCIPLES, PROMPT_TEMPLATE
        import torch
        from tqdm import tqdm
        
        principle_data = PRINCIPLES[principle]
        predictions = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Zero-shot {principle}"):
            full_prompt = PROMPT_TEMPLATE.format(
                principle_name=principle_data["name"],
                principle_description=principle_data["description"],
                prompt=row["prompt"],
                response=row["response"]
            )
            
            # Format as chat prompt for Phi-3 instruct
            messages = [{"role": "user", "content": full_prompt}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=2, 
                    temperature=0.0, # Greedy decoding
                    do_sample=False
                )
                
            # Extract just the generated output
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Parse output (expecting "1" or "0")
            if "1" in generated_text:
                predictions.append(1)
            else:
                predictions.append(0)
                
        return predictions
