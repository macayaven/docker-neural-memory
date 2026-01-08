"""
Tests to verify that the neural memory actually learns.

Key validation: surprise should decrease on repeated patterns.
"""

import torch

from src.memory.neural_memory import NeuralMemory
from src.memory.ttt_layer import TTTMLP, TTTLayer, TTTLinear


class TestNeuralMemoryLearning:
    """Test that NeuralMemory learns from observations."""

    def test_surprise_decreases_on_repetition(self):
        """Repeated patterns should have decreasing surprise."""
        memory = NeuralMemory(dim=64)

        # Create a pattern
        pattern = torch.randn(1, 10, 64)

        # Observe the pattern multiple times
        surprises = []
        for _ in range(5):
            result = memory.observe(pattern)
            surprises.append(result["surprise"])

        # Surprise should generally decrease (allowing some noise)
        assert surprises[-1] < surprises[0], (
            f"Surprise should decrease: first={surprises[0]:.4f}, last={surprises[-1]:.4f}"
        )

    def test_weights_update_during_observation(self):
        """Weights should change during observe()."""
        memory = NeuralMemory(dim=64)

        # Get initial weights
        initial_weights = [p.clone() for p in memory.memory_net.parameters()]

        # Observe a pattern
        pattern = torch.randn(1, 10, 64)
        result = memory.observe(pattern)

        # Check weights changed
        weight_changed = False
        for p, init in zip(memory.memory_net.parameters(), initial_weights, strict=True):
            if not torch.allclose(p, init):
                weight_changed = True
                break

        assert weight_changed, "Weights should update during observation"
        assert result["weight_delta"] > 0, "Weight delta should be positive"

    def test_infer_does_not_update_weights(self):
        """Infer should not modify weights."""
        memory = NeuralMemory(dim=64)

        # Get initial weights
        initial_weights = [p.clone() for p in memory.memory_net.parameters()]

        # Infer (not observe)
        pattern = torch.randn(1, 10, 64)
        memory.infer(pattern)

        # Check weights unchanged
        for p, init in zip(memory.memory_net.parameters(), initial_weights, strict=True):
            assert torch.allclose(p, init), "Weights should not change during infer"

    def test_different_patterns_have_different_surprise(self):
        """Different patterns should have different surprise levels."""
        memory = NeuralMemory(dim=64)

        # Learn one pattern
        pattern1 = torch.randn(1, 10, 64)
        for _ in range(10):
            memory.observe(pattern1)

        # Check surprise for learned vs new pattern
        surprise_learned = memory.surprise(pattern1)
        surprise_new = memory.surprise(torch.randn(1, 10, 64))

        assert surprise_new > surprise_learned, (
            f"New pattern should be more surprising: "
            f"learned={surprise_learned:.4f}, new={surprise_new:.4f}"
        )


class TestTTTLayer:
    """Test TTT layer variants."""

    def test_ttt_linear_output_shape(self):
        """TTT-Linear should maintain input shape."""
        layer = TTTLinear(dim=64)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == x.shape

    def test_ttt_mlp_output_shape(self):
        """TTT-MLP should maintain input shape."""
        layer = TTTMLP(dim=64)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == x.shape

    def test_ttt_layer_processes_sequence(self):
        """TTT should process each token in sequence."""
        layer = TTTLayer(dim=64, variant="linear")
        x = torch.randn(1, 5, 64)
        y = layer(x)

        # Output should be different from input (processing happened)
        assert not torch.allclose(x, y, atol=0.1)
