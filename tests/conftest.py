"""
Pytest configuration and fixtures for Docker Neural Memory tests.
"""

import tempfile
from pathlib import Path

import pytest
import torch

# Ensure reproducibility
torch.manual_seed(42)


@pytest.fixture
def tmp_path():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def memory_dim():
    """Standard dimension for test memory modules."""
    return 256


@pytest.fixture
def small_memory_dim():
    """Small dimension for fast tests."""
    return 64


@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Python uses indentation for code blocks",
        "Machine learning models learn from data",
        "Docker containers package applications",
        "Neural networks have layers of neurons",
        "Test-time training updates weights during inference",
    ]


@pytest.fixture
def similar_texts():
    """Similar texts for pattern recognition testing."""
    return [
        "Python uses whitespace for structure",
        "In Python, indentation defines blocks",
        "Python relies on indentation instead of braces",
        "Whitespace matters in Python code",
        "Python syntax uses indentation for scoping",
    ]


@pytest.fixture
def diverse_texts():
    """Diverse texts for novelty testing."""
    return [
        "Python programming language",
        "Quantum physics experiments",
        "Medieval European history",
        "Organic chemistry compounds",
        "Jazz music improvisation",
    ]


@pytest.fixture
def memory_config():
    """Create a test memory configuration."""
    from src.config import MemoryConfig

    return MemoryConfig(dim=128, learning_rate=0.02, device="cpu")


@pytest.fixture
def memory(memory_config):
    """Create a test neural memory instance."""
    from src.memory.neural_memory import NeuralMemory

    return NeuralMemory(memory_config)


@pytest.fixture
def small_memory():
    """Create a small memory for fast tests."""
    from src.config import MemoryConfig
    from src.memory.neural_memory import NeuralMemory

    return NeuralMemory(MemoryConfig(dim=64, learning_rate=0.05, device="cpu"))
