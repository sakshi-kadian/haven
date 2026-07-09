"""
evaluator.py
------------
Main inference pipeline for the HAVEN constitutional evaluator.

Loads all 5 principle classifiers, runs the full pipeline:
    Input -> Constitutional Encoder (x5) -> Conflict Detector
          -> Resolution Layer -> Calibration -> Safety Report

Usage:
    from src.evaluator import HavenEvaluator
    evaluator = HavenEvaluator(checkpoint_dir="checkpoints/")
    report = evaluator.evaluate(prompt="...", response="...")
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.constitution import PRINCIPLES, PRINCIPLE_NAMES, PROMPT_TEMPLATE
from src.conflict_detector import ConflictDetector

from src.resolvers.hierarchy import HierarchyResolver
from src.resolvers.voting import VotingResolver
from src.resolvers.gating import GatingResolver


class HavenEvaluator:
    """
    Full HAVEN pipeline: from raw text to calibrated principle scores.

    Args:
        checkpoint_dir:    Directory containing the 5 adapter weight folders.
        resolver_strategy: One of ['hierarchy', 'voting', 'gating'].
        temperature:       Dict of per-principle temperature values for calibration.
                           If None, defaults to T=1.0 (no calibration adjustment).
    """

    def __init__(self, checkpoint_dir: str, resolver_strategy: str = "gating",
                 temperature: dict = None):
        self.checkpoint_dir = checkpoint_dir
        self.resolver_strategy = resolver_strategy
        self.temperature = temperature or {p: 1.0 for p in PRINCIPLE_NAMES}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load per-principle tokenizers and classifier heads
        self.tokenizers = {}
        self.models = {}
        for principle in PRINCIPLE_NAMES:
            ckpt_path = os.path.join(checkpoint_dir, f"phi3_{principle}")
            if os.path.exists(ckpt_path):
                self.tokenizers[principle] = AutoTokenizer.from_pretrained(ckpt_path)
                self.models[principle] = (
                    AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=2)
                    .to(self.device)
                    .eval()
                )

        # Conflict detector using canonical pairs from constitution
        self.conflict_detector = ConflictDetector()

        # Resolution strategy
        if resolver_strategy == "hierarchy":
            self.resolver = HierarchyResolver()
        elif resolver_strategy == "voting":
            self.resolver = VotingResolver()
        elif resolver_strategy == "gating":
            phi3_hidden_dim = 3072  # Phi-3 Mini hidden state dimension
            self.resolver = GatingResolver(input_dim=phi3_hidden_dim)
        else:
            raise ValueError(
                f"Unknown resolver_strategy '{resolver_strategy}'. "
                f"Choose from: 'hierarchy', 'voting', 'gating'."
            )

    def score(self, prompt: str, response: str) -> dict:
        """
        Run all 5 classifiers and return raw principle scores.

        Returns:
            Dict mapping principle -> float score in [0, 1].
        """
        scores = {}

        for principle, info in PRINCIPLES.items():
            if principle not in self.models:
                # Fallback: if checkpoint not yet trained, return neutral score
                scores[principle] = 0.5
                continue

            # Build the constitutional prompt using the shared template
            input_text = PROMPT_TEMPLATE.format(
                principle_name=info["name"],
                principle_description=info["description"],
                prompt=prompt,
                response=response,
            )

            tokenizer = self.tokenizers[principle]
            model = self.models[principle]

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Apply temperature calibration before softmax
            T = self.temperature.get(principle, 1.0)
            calibrated_logits = logits / T
            probs = torch.softmax(calibrated_logits, dim=-1)

            # Score = probability of class 1 (satisfies principle)
            scores[principle] = probs[0, 1].item()

        return scores

    def evaluate(self, prompt: str, response: str) -> dict:
        """
        Full pipeline: score -> conflict detection -> resolution -> calibration.

        Returns:
            Dict with keys: scores, conflicts, resolved_label, calibrated_scores.
        """
        # Step 1: Run constitutional classifiers
        scores = self.score(prompt, response)

        # Step 2: Detect principle conflicts
        conflicts = self.conflict_detector.detect(scores)

        # Step 3: Resolve conflicts to get a single safety label
        if self.resolver_strategy == "gating":
            # GatingResolver needs a prompt embedding — use mean of score values
            # as a lightweight proxy when full hidden states are unavailable
            proxy_embedding = torch.tensor(
                [scores.get(p, 0.5) for p in PRINCIPLE_NAMES],
                dtype=torch.float32
            )
            # Pad to match the MLP's expected input_dim (3072)
            pad_size = 3072 - len(proxy_embedding)
            proxy_embedding = torch.nn.functional.pad(proxy_embedding, (0, pad_size))
            resolved_label = self.resolver.resolve(proxy_embedding, scores, conflicts)
        elif self.resolver_strategy == "voting":
            resolved_label = self.resolver.resolve(scores)
        else:
            resolved_label = self.resolver.resolve(scores, conflicts)

        # Step 4: Build safety report
        report = {
            "scores": scores,
            "conflicts": conflicts,
            "resolved_label": resolved_label,
            "is_safe": bool(resolved_label),
            "resolver_used": self.resolver_strategy,
        }

        return report
