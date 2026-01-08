"""
Tests for state persistence - verifying learned state survives restart.
"""

import tempfile

import torch

from src.memory.neural_memory import NeuralMemory
from src.state.checkpoint import CheckpointManager
from src.state.versioning import VersionManager


class TestCheckpointPersistence:
    """Test checkpoint save/restore functionality."""

    def test_checkpoint_and_restore(self):
        """Checkpointed state should restore exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            memory = NeuralMemory(dim=64)

            # Learn something
            pattern = torch.randn(1, 10, 64)
            for _ in range(10):
                memory.observe(pattern)

            # Get surprise before checkpoint
            surprise_before = memory.surprise(pattern)

            # Checkpoint
            info = manager.checkpoint(memory, "test-v1", "Test checkpoint")
            assert info.tag == "test-v1"
            assert info.size_mb > 0

            # Modify memory (more learning)
            for _ in range(10):
                memory.observe(torch.randn(1, 10, 64))

            # Surprise should be different now (continue learning)
            memory.surprise(pattern)

            # Restore
            manager.restore(memory, "test-v1")

            # Surprise should match pre-checkpoint
            surprise_after_restore = memory.surprise(pattern)

            assert abs(surprise_before - surprise_after_restore) < 0.01, (
                f"State should restore exactly: before={surprise_before:.4f}, "
                f"after_restore={surprise_after_restore:.4f}"
            )

    def test_list_checkpoints(self):
        """Should list all created checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            memory = NeuralMemory(dim=64)

            # Create multiple checkpoints
            manager.checkpoint(memory, "v1", "First version")
            manager.checkpoint(memory, "v2", "Second version")
            manager.checkpoint(memory, "v3", "Third version")

            checkpoints = manager.list_checkpoints()
            tags = [cp.tag for cp in checkpoints]

            assert "v1" in tags
            assert "v2" in tags
            assert "v3" in tags
            assert len(checkpoints) == 3

    def test_delete_checkpoint(self):
        """Should be able to delete checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            memory = NeuralMemory(dim=64)

            manager.checkpoint(memory, "to-delete")
            assert len(manager.list_checkpoints()) == 1

            manager.delete("to-delete")
            assert len(manager.list_checkpoints()) == 0


class TestVersioning:
    """Test fork and versioning functionality."""

    def test_fork_creates_copy(self):
        """Forking should create an independent copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)
            version_mgr = VersionManager(checkpoint_mgr)
            memory = NeuralMemory(dim=64)

            # Create initial checkpoint
            checkpoint_mgr.checkpoint(memory, "main")

            # Fork it
            info = version_mgr.fork(memory, "main", "experiment")

            assert info.forked
            assert info.source_tag == "main"
            assert info.new_tag == "experiment"

            # Both should exist
            checkpoints = checkpoint_mgr.list_checkpoints()
            tags = [cp.tag for cp in checkpoints]
            assert "main" in tags
            assert "experiment" in tags

    def test_learning_since_checkpoint(self):
        """Should track how much learning happened since checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)
            version_mgr = VersionManager(checkpoint_mgr)
            memory = NeuralMemory(dim=64)

            # Checkpoint initial state
            checkpoint_mgr.checkpoint(memory, "initial")

            # No learning yet
            learning = version_mgr.learning_since_checkpoint(memory, "initial")
            assert learning["total_learning"] < 0.01

            # Do some learning
            for _ in range(20):
                memory.observe(torch.randn(1, 10, 64))

            # Should show significant learning
            learning = version_mgr.learning_since_checkpoint(memory, "initial")
            assert learning["total_learning"] > 0.1, (
                f"Should show learning: {learning['total_learning']}"
            )
