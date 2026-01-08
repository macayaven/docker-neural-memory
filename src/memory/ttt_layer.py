"""
Test-Time Training (TTT) Layer.

The hidden state is a machine learning model.
The update rule is a step of self-supervised learning.

Based on: https://arxiv.org/abs/2407.04620
"""

import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional


class TTTLayer(nn.Module):
    """
    Test-Time Training layer.

    The hidden state is itself a learnable model that updates
    via gradient descent during the forward pass.
    """

    def __init__(self, dim: int, variant: str = "linear"):
        """
        Initialize TTT layer.

        Args:
            dim: Input/output dimension
            variant: "linear" for TTT-Linear, "mlp" for TTT-MLP
        """
        super().__init__()
        self.dim = dim
        self.variant = variant

        if variant == "linear":
            # TTT-Linear: Hidden state is a linear model
            self.hidden_model = nn.Linear(dim, dim, bias=False)
        elif variant == "mlp":
            # TTT-MLP: Hidden state is a two-layer MLP
            self.hidden_model = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Linear(dim * 4, dim),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'linear' or 'mlp'.")

        # Project input to key/value for self-supervised learning
        self.to_kv = nn.Linear(dim, dim * 2)

        # Learnable learning rate
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Process sequence with test-time training.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        _batch, seq_len, _dim = x.shape

        # Clone hidden model for this sequence (mini-batch gradient descent)
        hidden_state = copy.deepcopy(self.hidden_model)

        outputs = []
        for t in range(seq_len):
            # Current token
            x_t = x[:, t : t + 1, :]

            # Self-supervised target: reconstruct from key-value
            kv = self.to_kv(x_t)
            k, v = kv.chunk(2, dim=-1)

            # Forward through hidden state
            y_t = hidden_state(x_t)

            # Compute loss and update hidden state
            loss = functional.mse_loss(y_t, v)

            # Compute gradients
            grads = torch.autograd.grad(
                loss, hidden_state.parameters(), create_graph=False
            )

            # Update hidden state weights
            with torch.no_grad():
                for param, grad in zip(hidden_state.parameters(), grads, strict=True):
                    param -= self.eta * grad

            outputs.append(y_t.detach())

        return torch.cat(outputs, dim=1)


class TTTLinear(TTTLayer):
    """TTT-Linear: Hidden state is a linear model (faster)."""

    def __init__(self, dim: int):
        super().__init__(dim, variant="linear")


class TTTMLP(TTTLayer):
    """TTT-MLP: Hidden state is a two-layer MLP (more expressive)."""

    def __init__(self, dim: int):
        super().__init__(dim, variant="mlp")
