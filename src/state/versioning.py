"""
Memory versioning - fork and branch operations for neural memory.

Enables experimentation without losing stable state.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .checkpoint import CheckpointManager


@dataclass
class ForkInfo:
    """Information about a memory fork operation."""

    forked: bool
    source_tag: str
    new_tag: str
    source_hash: str
    new_hash: str


class VersionManager:
    """
    Manages versioning operations for neural memory.

    Provides Git-like branching semantics for learned state.
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        """
        Initialize version manager.

        Args:
            checkpoint_manager: Checkpoint manager to use for storage
        """
        self.checkpoint_mgr = checkpoint_manager

    def fork(self, _model: nn.Module, source_tag: str, new_tag: str) -> ForkInfo:
        """
        Fork memory state into a new branch.

        Creates a copy of an existing checkpoint under a new name,
        enabling experimentation without affecting the original.

        Args:
            model: Current model (used to verify state)
            source_tag: Source checkpoint to fork from
            new_tag: Name for the new branch

        Returns:
            ForkInfo with operation details
        """
        source_path = self.checkpoint_mgr.checkpoint_dir / f"{source_tag}.pt"
        new_path = self.checkpoint_mgr.checkpoint_dir / f"{new_tag}.pt"

        if not source_path.exists():
            raise ValueError(f"Source checkpoint '{source_tag}' not found")

        if new_path.exists():
            raise ValueError(f"Checkpoint '{new_tag}' already exists")

        # Copy the checkpoint file
        shutil.copy(source_path, new_path)

        # Copy metadata
        source_meta = self.checkpoint_mgr.metadata["checkpoints"].get(source_tag, {})
        self.checkpoint_mgr.metadata["checkpoints"][new_tag] = {
            **source_meta,
            "forked_from": source_tag,
            "description": f"Forked from {source_tag}",
        }
        self.checkpoint_mgr._save_metadata()

        return ForkInfo(
            forked=True,
            source_tag=source_tag,
            new_tag=new_tag,
            source_hash=source_meta.get("weight_hash", ""),
            new_hash=source_meta.get("weight_hash", ""),
        )

    def get_lineage(self, tag: str) -> list[str]:
        """
        Get the lineage of a checkpoint (all ancestors).

        Args:
            tag: Checkpoint to trace

        Returns:
            List of ancestor tags, oldest first
        """
        lineage = [tag]
        current = tag

        while True:
            meta = self.checkpoint_mgr.metadata["checkpoints"].get(current, {})
            parent = meta.get("forked_from")
            if parent and parent not in lineage:
                lineage.insert(0, parent)
                current = parent
            else:
                break

        return lineage

    def diff_checkpoints(self, _model_class: type, tag1: str, tag2: str) -> dict[str, float]:
        """
        Compare two checkpoints and return weight differences.

        Args:
            model_class: Class to instantiate for loading
            tag1: First checkpoint
            tag2: Second checkpoint

        Returns:
            Dict mapping layer names to L2 distance
        """
        # Load both checkpoints
        state1 = torch.load(self.checkpoint_mgr.checkpoint_dir / f"{tag1}.pt")
        state2 = torch.load(self.checkpoint_mgr.checkpoint_dir / f"{tag2}.pt")

        diffs = {}
        for key in state1:
            if key in state2:
                diff = (state1[key] - state2[key]).pow(2).sum().sqrt().item()
                diffs[key] = diff

        return diffs

    def learning_since_checkpoint(
        self, model: nn.Module, tag: str
    ) -> dict[str, str | float | dict[str, float] | int]:
        """
        Measure how much the model has learned since a checkpoint.

        Args:
            model: Current model state
            tag: Checkpoint to compare against

        Returns:
            Dict with learning metrics
        """
        checkpoint_path = self.checkpoint_mgr.checkpoint_dir / f"{tag}.pt"
        if not checkpoint_path.exists():
            return {"error": f"Checkpoint '{tag}' not found"}

        saved_state = torch.load(checkpoint_path)
        current_state = model.state_dict()

        total_diff = 0.0
        layer_diffs = {}

        for key in saved_state:
            if key in current_state:
                diff = (saved_state[key] - current_state[key]).pow(2).sum().sqrt().item()
                layer_diffs[key] = diff
                total_diff += diff

        return {
            "total_learning": total_diff,
            "layer_diffs": layer_diffs,
            "num_layers_changed": sum(1 for d in layer_diffs.values() if d > 1e-6),
        }
