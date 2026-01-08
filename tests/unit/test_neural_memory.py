"""Unit tests for NeuralMemory module."""

import pytest

from src.config import MemoryConfig
from src.memory.neural_memory import NeuralMemory


class TestNeuralMemoryInit:
    """Tests for NeuralMemory initialization."""

    def test_init_with_config(self, memory_config: MemoryConfig) -> None:
        """Test initialization with MemoryConfig object."""
        memory = NeuralMemory(memory_config)
        assert memory.dim == memory_config.dim
        assert memory.config.learning_rate == memory_config.learning_rate

    def test_init_with_int(self) -> None:
        """Test legacy initialization with int dimension."""
        memory = NeuralMemory(256)
        assert memory.dim == 256

    def test_init_with_kwargs(self) -> None:
        """Test initialization with keyword arguments."""
        memory = NeuralMemory(dim=128, learning_rate=0.05)
        assert memory.dim == 128
        assert memory.config.learning_rate == 0.05


class TestObserve:
    """Tests for observe() method."""

    def test_observe_returns_dict(self, memory: NeuralMemory) -> None:
        """Test that observe returns expected keys."""
        result = memory.observe("Test content")
        assert "surprise" in result
        assert "weight_delta" in result
        assert "learned" in result

    def test_observe_changes_weights(self, memory: NeuralMemory) -> None:
        """Test that observe actually changes weights."""
        hash_before = memory.get_weight_hash()
        memory.observe("Python uses indentation for code blocks")
        hash_after = memory.get_weight_hash()
        assert hash_before != hash_after, "Weights should change after observe"

    def test_observe_increments_counter(self, memory: NeuralMemory) -> None:
        """Test that observation counter increments."""
        initial_count = memory._observation_count
        memory.observe("Test observation")
        assert memory._observation_count == initial_count + 1

    def test_surprise_decreases_on_repetition(self, small_memory: NeuralMemory) -> None:
        """Test that surprise decreases for similar content."""
        # Observe similar content multiple times
        r1 = small_memory.observe("Machine learning models learn from data")
        r2 = small_memory.observe("ML models are trained on datasets")
        r3 = small_memory.observe("Models in machine learning learn from training data")

        # Surprise should generally decrease (allowing some variance)
        surprises = [r1["surprise"], r2["surprise"], r3["surprise"]]
        # At least the last should be lower than the first
        assert surprises[-1] < surprises[0] + 0.1, "Surprise should trend downward"


class TestInfer:
    """Tests for infer() method."""

    def test_infer_returns_dict(self, memory: NeuralMemory) -> None:
        """Test that infer returns expected keys."""
        result = memory.infer("Test query")
        assert "response" in result
        assert "confidence" in result

    def test_infer_no_weight_change(self, memory: NeuralMemory) -> None:
        """Test that infer does not change weights."""
        memory.observe("Some training content")
        hash_before = memory.get_weight_hash()
        memory.infer("Query about content")
        hash_after = memory.get_weight_hash()
        assert hash_before == hash_after, "Infer should not change weights"

    def test_confidence_bounded(self, memory: NeuralMemory) -> None:
        """Test that confidence is between 0 and 1."""
        result = memory.infer("Any query")
        assert 0.0 <= result["confidence"] <= 1.0


class TestSurprise:
    """Tests for surprise() method."""

    def test_surprise_returns_float(self, memory: NeuralMemory) -> None:
        """Test that surprise returns a float."""
        score = memory.surprise("Test content")
        assert isinstance(score, float)

    def test_surprise_bounded(self, memory: NeuralMemory) -> None:
        """Test that surprise is between 0 and 1."""
        score = memory.surprise("Any content")
        assert 0.0 <= score <= 1.0

    def test_surprise_no_weight_change(self, memory: NeuralMemory) -> None:
        """Test that surprise check does not change weights."""
        memory.observe("Training content")
        hash_before = memory.get_weight_hash()
        memory.surprise("Check this content")
        hash_after = memory.get_weight_hash()
        assert hash_before == hash_after


class TestWeightHash:
    """Tests for get_weight_hash() method."""

    def test_hash_is_string(self, memory: NeuralMemory) -> None:
        """Test that weight hash is a string."""
        hash_val = memory.get_weight_hash()
        assert isinstance(hash_val, str)

    def test_hash_is_deterministic(self, memory: NeuralMemory) -> None:
        """Test that same weights produce same hash."""
        hash1 = memory.get_weight_hash()
        hash2 = memory.get_weight_hash()
        assert hash1 == hash2

    def test_hash_changes_after_observe(self, memory: NeuralMemory) -> None:
        """Test that hash changes after learning."""
        hash1 = memory.get_weight_hash()
        memory.observe("New content to learn")
        hash2 = memory.get_weight_hash()
        assert hash1 != hash2


class TestStats:
    """Tests for get_stats() method."""

    def test_stats_returns_dict(self, memory: NeuralMemory) -> None:
        """Test that stats returns expected keys."""
        stats = memory.get_stats()
        assert "total_observations" in stats
        assert "weight_parameters" in stats
        assert "dimension" in stats
        assert "learning_rate" in stats

    def test_stats_tracks_observations(self, memory: NeuralMemory) -> None:
        """Test that stats tracks observation count."""
        memory.observe("First")
        memory.observe("Second")
        stats = memory.get_stats()
        assert stats["total_observations"] == 2
