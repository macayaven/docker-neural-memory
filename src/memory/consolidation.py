"""
Memory consolidation - compressing recent learning into stable patterns.

Like sleep for neural memory: strengthens important patterns,
prunes noise, and prevents catastrophic forgetting.
"""

import torch
import torch.nn as nn
from torch import Tensor


class MemoryConsolidator:
    """
    Consolidates memory by compressing recent learning into stable patterns.

    Uses Elastic Weight Consolidation (EWC) to protect important weights.
    """

    def __init__(self, ewc_lambda: float = 0.1):
        """
        Initialize consolidator.

        Args:
            ewc_lambda: Weight for EWC penalty term
        """
        self.ewc_lambda = ewc_lambda
        self.fisher: dict[str, Tensor] = {}
        self.optimal: dict[str, Tensor] = {}

    def compute_fisher(self, model: nn.Module, data_loader) -> None:
        """
        Compute Fisher information matrix for EWC.

        The Fisher matrix measures how important each weight is
        for the current task.
        """
        self.fisher = {}
        self.optimal = {}

        # Store current optimal weights
        for name, param in model.named_parameters():
            self.optimal[name] = param.clone().detach()
            self.fisher[name] = torch.zeros_like(param)

        model.eval()
        for x in data_loader:
            model.zero_grad()
            output = model(x)
            # Use reconstruction loss as proxy for importance
            loss = output.sum()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data.pow(2)

        # Normalize Fisher values
        for name in self.fisher:
            self.fisher[name] /= len(data_loader)

    def ewc_loss(self, model: nn.Module, current_loss: Tensor) -> Tensor:
        """
        Add EWC penalty to protect important weights from past tasks.

        Args:
            model: The neural memory model
            current_loss: Current task loss

        Returns:
            Loss with EWC penalty added
        """
        if not self.fisher:
            return current_loss

        ewc_penalty = torch.tensor(0.0, device=current_loss.device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                ewc_penalty += (
                    self.fisher[name] * (param - self.optimal[name]).pow(2)
                ).sum()

        return current_loss + self.ewc_lambda * ewc_penalty

    def consolidate(self, model: nn.Module, recent_observations: list[Tensor]) -> dict:
        """
        Perform consolidation pass.

        Args:
            model: Neural memory model
            recent_observations: Recent data to identify important patterns

        Returns:
            Consolidation metrics
        """
        # Create simple data loader from observations
        data_loader = recent_observations

        # Compute new Fisher information
        old_fisher = self.fisher.copy() if self.fisher else {}
        self.compute_fisher(model, data_loader)

        # Merge with existing Fisher (weighted average)
        patterns_merged = 0
        if old_fisher:
            for name in self.fisher:
                if name in old_fisher:
                    self.fisher[name] = 0.5 * (self.fisher[name] + old_fisher[name])
                    patterns_merged += 1

        # Compute compression metric (how much the Fisher changed)
        compression = 0.0
        if old_fisher:
            for name in self.fisher:
                if name in old_fisher:
                    diff = (self.fisher[name] - old_fisher[name]).abs().mean()
                    compression += diff.item()

        return {
            "patterns_merged": patterns_merged,
            "memory_compressed_by": compression,
            "stability_score": 1.0 / (1.0 + compression),
        }
