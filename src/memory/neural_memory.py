"""
Titans-style neural long-term memory.

Key insight: The hidden state IS a neural network.
Updates happen via self-supervised learning during inference.

Based on: https://arxiv.org/abs/2501.00663
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from ..config import MemoryConfig


class NeuralMemory(nn.Module):
    """
    Titans-style neural long-term memory.

    The memory is a small neural network that updates its weights
    during inference via gradient descent (test-time training).

    Example:
        >>> config = MemoryConfig(dim=256)
        >>> memory = NeuralMemory(config)
        >>> result = memory.observe("Python uses indentation")
        >>> print(f"Surprise: {result['surprise']:.3f}")
    """

    def __init__(self, config: MemoryConfig | int | None = None, **kwargs: Any) -> None:
        super().__init__()

        # Handle both config object and legacy positional args
        if config is None:
            config = MemoryConfig(**kwargs)
        elif isinstance(config, int):
            # Legacy: NeuralMemory(dim=256) or NeuralMemory(256)
            config = MemoryConfig(dim=config, **kwargs)

        self.config = config
        self.dim = config.dim

        # The memory IS a neural network
        self.memory_net = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.LayerNorm(config.dim * 4),
            nn.Linear(config.dim * 4, config.dim),
        )

        # Target projection for self-supervised learning
        self.target_proj = nn.Linear(config.dim, config.dim)

        # Learnable learning rate (meta-learning)
        self.lr = nn.Parameter(torch.tensor(config.learning_rate))

        # Observation counter
        self._observation_count = 0
        self._recent_surprises: list[float] = []

        # Move to device
        self.to(config.device)

    def _encode_text(self, text: str) -> Tensor:
        """
        Encode text to tensor representation.

        Uses a simple but deterministic encoding for demo purposes.
        In production, would use a proper encoder (e.g., sentence-transformers).
        """
        # Create deterministic embedding from text
        text_bytes = text.encode("utf-8")
        hash_bytes = hashlib.sha256(text_bytes).digest()

        # Expand hash to fill dimension
        values = []
        for i in range(self.dim):
            byte_idx = i % len(hash_bytes)
            bit_offset = (i // len(hash_bytes)) % 8
            val = ((hash_bytes[byte_idx] >> bit_offset) & 1) * 2 - 1  # -1 or 1
            values.append(val * 0.1)

        # Add variation based on character positions
        for i, char in enumerate(text[: self.dim]):
            idx = i % self.dim
            values[idx] += (ord(char) / 255.0 - 0.5) * 0.2

        tensor = torch.tensor(values, dtype=torch.float32, device=self.config.device)
        # Shape: [1, seq_len, dim] - treat each character as a "token"
        seq_len = min(len(text), 64)  # Cap sequence length
        tensor = tensor.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1).clone()

        # Add positional variation
        for i in range(seq_len):
            if i < len(text):
                tensor[0, i, :] += torch.randn(self.dim, device=self.config.device) * 0.01
                tensor[0, i, i % self.dim] += ord(text[i]) / 255.0

        return tensor

    def forward(self, x: Tensor, learn: bool = True) -> Tensor:
        """
        Process input and optionally update memory weights.

        Args:
            x: Input tensor [batch, seq, dim]
            learn: Whether to update memory weights (test-time training)

        Returns:
            Memory-augmented representation
        """
        # Ensure requires_grad for learning
        if learn:
            x = x.detach().requires_grad_(False)
            for param in self.memory_net.parameters():
                param.requires_grad_(True)

        # Query the memory
        memory_output: Tensor = self.memory_net(x)

        if learn and x.shape[1] > 1:
            # Self-supervised objective: predict next token representation
            loss = self._compute_surprise_tensor(x, memory_output)

            if loss.requires_grad:
                # Update memory weights (this is the key innovation)
                self._update_weights(loss)

        return memory_output

    def _compute_surprise_tensor(self, x: Tensor, pred: Tensor) -> Tensor:
        """
        Compute surprise as prediction error (returns tensor for gradients).
        """
        if x.shape[1] <= 1:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Target: shifted input projected
        target = self.target_proj(x[:, 1:, :])
        prediction = pred[:, :-1, :]

        return functional.mse_loss(prediction, target)

    def _compute_surprise(self, x: Tensor, pred: Tensor) -> float:
        """
        Compute surprise score (0 to 1 range).
        """
        with torch.no_grad():
            if x.shape[1] <= 1:
                return 0.5

            target = self.target_proj(x[:, 1:, :])
            prediction = pred[:, :-1, :]
            mse = functional.mse_loss(prediction, target).item()

            # Convert to 0-1 range using sigmoid-like scaling
            surprise = 2.0 / (1.0 + torch.exp(torch.tensor(-mse * 10)).item()) - 1.0
            return float(max(0.0, min(1.0, surprise)))

    def _update_weights(self, loss: Tensor) -> None:
        """The key innovation: gradient descent during forward pass."""
        try:
            grads = torch.autograd.grad(
                loss, list(self.memory_net.parameters()), create_graph=False, allow_unused=True
            )

            with torch.no_grad():
                for param, grad in zip(self.memory_net.parameters(), grads):
                    if grad is not None:
                        param -= self.lr * grad
        except RuntimeError:
            # Gradient computation failed, skip update
            pass

    def observe(self, content: str | Tensor, learning_rate: float | None = None) -> dict[str, Any]:
        """
        Feed content to memory, triggering test-time learning.

        Args:
            content: Text string or tensor to learn from
            learning_rate: Optional override for learning rate

        Returns:
            dict with surprise score, weight delta, and metadata
        """
        # Handle learning rate override
        original_lr = None
        if learning_rate is not None:
            original_lr = self.lr.data.clone()
            self.lr.data = torch.tensor(learning_rate, device=self.config.device)

        # Encode if string
        x = self._encode_text(content) if isinstance(content, str) else content

        # Store initial weights for delta calculation
        initial_weights = {
            name: param.clone() for name, param in self.memory_net.named_parameters()
        }

        # Forward with learning
        output = self.forward(x, learn=True)

        # Calculate metrics
        surprise = self._compute_surprise(x, output)
        weight_delta = sum(
            (param - initial_weights[name]).abs().sum().item()
            for name, param in self.memory_net.named_parameters()
        )

        # Restore learning rate
        if original_lr is not None:
            self.lr.data = original_lr

        # Update stats
        self._observation_count += 1
        self._recent_surprises.append(surprise)
        if len(self._recent_surprises) > 100:
            self._recent_surprises.pop(0)

        return {
            "surprise": surprise,
            "weight_delta": weight_delta,
            "patterns_activated": [f"pattern_{self._observation_count}"],
            "learned": weight_delta > 1e-6,
        }

    def infer(self, query: str | Tensor, temperature: float = 1.0) -> dict[str, Any]:
        """
        Query memory using learned representations (no learning).

        Args:
            query: Text string or tensor to query
            temperature: Not used currently, for API compatibility

        Returns:
            dict with response tensor and confidence
        """
        del temperature  # Unused, kept for API compatibility
        x = self._encode_text(query) if isinstance(query, str) else query

        with torch.no_grad():
            output = self.forward(x, learn=False)
            confidence = 1.0 - self._compute_surprise(x, output)

        return {
            "response": output,
            "confidence": max(0.0, min(1.0, confidence)),
            "attention_weights": output[0, 0, :10].tolist() if output.dim() >= 3 else [],
        }

    def surprise(self, content: str | Tensor) -> float:
        """
        Measure how surprising/novel content is WITHOUT learning.

        Args:
            content: Text string or tensor to evaluate

        Returns:
            Surprise score between 0 (familiar) and 1 (novel)
        """
        x = self._encode_text(content) if isinstance(content, str) else content

        with torch.no_grad():
            output = self.memory_net(x)
            return self._compute_surprise(x, output)

    def get_weight_hash(self) -> str:
        """
        Get hash of current weights for change detection.

        Returns:
            16-character hex hash of weights
        """
        with torch.no_grad():
            state = self.memory_net.state_dict()
            flat = torch.cat([v.flatten().cpu() for v in state.values()])
            # Use string representation instead of numpy to avoid numpy dependency
            data_str = str(flat.tolist())
            hash_bytes = hashlib.sha256(data_str.encode()).digest()
            return hash_bytes[:8].hex()

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_observations": self._observation_count,
            "weight_parameters": sum(p.numel() for p in self.memory_net.parameters()),
            "avg_surprise": (
                sum(self._recent_surprises) / len(self._recent_surprises)
                if self._recent_surprises
                else 0.0
            ),
            "learning_rate": self.lr.item(),
            "dimension": self.dim,
        }
