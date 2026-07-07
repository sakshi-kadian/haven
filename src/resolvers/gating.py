"""
gating.py
---------
Context-conditioned gating conflict resolution strategy (key novel component).

A small gating classifier reads the prompt embedding and selects the
most appropriate resolution strategy (hierarchy or voting) for that
specific input context.

Architecture:
    - Input: prompt embedding (from Phi-3 Mini hidden states)
    - Gating network: 2-layer MLP
    - Output: soft selection weights for [hierarchy, voting]
    - Final score: weighted combination of the two resolvers' outputs

This is the HAVEN framework's key novel component (Strategy C).

Usage:
    from src.resolvers.gating import GatingResolver
    resolver = GatingResolver(checkpoint_path="checkpoints/gating_mlp.pt")
    label = resolver.resolve(prompt_embedding, score_vector, conflicts)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.resolvers.hierarchy import HierarchyResolver
from src.resolvers.voting import VotingResolver


class GatingMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2), # 2 outputs: hierarchy weight, voting weight
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class GatingResolver:
    """
    Context-conditioned gating resolver. Selects resolution strategy
    dynamically based on the prompt embedding.

    Args:
        input_dim:         Dimension of the prompt embedding.
        checkpoint_path:   Path to saved gating MLP weights (if any).
    """

    def __init__(self, input_dim: int = 3072, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = GatingMLP(input_dim).to(self.device)
        self.hierarchy = HierarchyResolver()
        self.voting = VotingResolver()
        self.checkpoint_path = checkpoint_path

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.mlp.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Loaded gating MLP weights from {checkpoint_path}")

    def fit(self, prompt_embeddings: list, score_vectors: list, labels: list) -> None:
        """
        Train the gating MLP on conflict-flagged validation examples.

        Args:
            prompt_embeddings: List of prompt embedding tensors.
            score_vectors:     List of principle score dicts.
            labels:            True binary labels.
        """
        if not prompt_embeddings:
            return

        optimizer = optim.AdamW(self.mlp.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.mlp.train()
        print(f"Training Gating MLP on {len(prompt_embeddings)} examples...")

        X = torch.stack(prompt_embeddings).to(self.device)

        # Build per-example targets: 0 = use hierarchy, 1 = use voting
        # Whichever resolver matches the true label gets the target assignment
        targets = []
        for sv, lbl in zip(score_vectors, labels):
            hier_correct = int(self.hierarchy.resolve(sv, []) == lbl)
            vote_correct = int(self.voting.resolve(sv) == lbl)
            # Prefer hierarchy (0) on a tie to match our safety-first design
            targets.append(0 if hier_correct >= vote_correct else 1)

        y = torch.tensor(targets, dtype=torch.long).to(self.device)

        for epoch in range(10):
            optimizer.zero_grad()
            gate_weights = self.mlp(X)
            loss = criterion(gate_weights, y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/10 — loss: {loss.item():.4f}")

        print("Gating MLP training complete.")

        if self.checkpoint_path:
            parent = os.path.dirname(self.checkpoint_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.mlp.state_dict(), self.checkpoint_path)

    def resolve(self, prompt_embedding, score_vector: dict, conflicts: list) -> int:
        """
        Args:
            prompt_embedding: Tensor of prompt hidden states.
            score_vector:     Dict mapping principle -> float score.
            conflicts:        List of conflicting (pi, pj) tuples.

        Returns:
            Final binary safety label: 0 (unsafe) or 1 (safe).
        """
        self.mlp.eval()
        
        with torch.no_grad():
            if not isinstance(prompt_embedding, torch.Tensor):
                prompt_embedding = torch.tensor(prompt_embedding, dtype=torch.float32)
                
            prompt_embedding = prompt_embedding.to(self.device)
            if prompt_embedding.dim() == 1:
                prompt_embedding = prompt_embedding.unsqueeze(0)
                
            gate_weights = self.mlp(prompt_embedding).squeeze(0)
            
        w_hier, w_vote = gate_weights[0].item(), gate_weights[1].item()
        
        out_hier = self.hierarchy.resolve(score_vector, conflicts)
        out_vote = self.voting.resolve(score_vector)
        
        # Soft voting based on gating network
        final_score = (w_hier * out_hier) + (w_vote * out_vote)
        
        return 1 if final_score >= 0.5 else 0
