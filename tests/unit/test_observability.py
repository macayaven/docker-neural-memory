"""Unit tests for observability module."""

import pytest

from src.config import MemoryConfig
from src.memory.neural_memory import NeuralMemory
from src.observability.metrics import MemoryObserver, MetricsSnapshot


@pytest.fixture
def observer() -> MemoryObserver:
    """Create an observer without Langfuse (local metrics only)."""
    memory = NeuralMemory(MemoryConfig(dim=64, learning_rate=0.02, device="cpu"))
    return MemoryObserver(memory, langfuse=None)


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        snapshot = MetricsSnapshot(
            timestamp="2024-01-01T00:00:00Z",
            observation_count=5,
            surprise=0.7,
            weight_delta=0.001,
            weight_hash="abc123",
            latency_ms=50.0,
            patterns_activated=["p1", "p2"],
            learned=True,
        )
        d = snapshot.to_dict()
        assert d["surprise"] == 0.7
        assert d["learned"] is True
        assert len(d["patterns_activated"]) == 2


class TestMemoryObserver:
    """Tests for MemoryObserver class."""

    def test_observe_tracks_metrics(self, observer: MemoryObserver) -> None:
        """Test that observe tracks metrics locally."""
        result = observer.observe("Test content")
        assert "surprise" in result
        assert "latency_ms" in result
        assert "snapshot" in result
        assert len(observer._observations) == 1

    def test_infer_tracks_metrics(self, observer: MemoryObserver) -> None:
        """Test that infer tracks metrics locally."""
        observer.observe("Training content")
        result = observer.infer("Query")
        assert "confidence" in result
        assert "latency_ms" in result
        assert len(observer._inferences) == 1

    def test_surprise_returns_float(self, observer: MemoryObserver) -> None:
        """Test surprise method returns float."""
        score = observer.surprise("Test content")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_get_summary(self, observer: MemoryObserver) -> None:
        """Test summary includes all sections."""
        observer.observe("Content 1")
        observer.observe("Content 2")
        observer.infer("Query")

        summary = observer.get_summary()
        assert "training" in summary
        assert "inference" in summary
        assert "memory" in summary
        assert summary["training"]["total_observations"] == 2
        assert summary["inference"]["total_queries"] == 1

    def test_session_id_generated(self, observer: MemoryObserver) -> None:
        """Test that session ID is generated."""
        assert observer.session_id.startswith("session_")

    def test_custom_session_id(self) -> None:
        """Test custom session ID."""
        memory = NeuralMemory(64)
        observer = MemoryObserver(memory, session_id="custom_123")
        assert observer.session_id == "custom_123"
