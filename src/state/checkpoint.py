"""
Checkpoint management for neural memory state.

Like `docker commit` but for neural memory weights.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


@dataclass
class CheckpointInfo:
    """Metadata for a saved checkpoint."""

    tag: str
    created_at: str
    size_mb: float
    weight_hash: str
    description: str = ""


class CheckpointManager:
    """
    Manages saving and restoring neural memory checkpoints.

    Provides Docker-like semantics for memory state management.
    """

    def __init__(self, checkpoint_dir: str = "/app/checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            with self.metadata_file.open() as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"checkpoints": {}}

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        with self.metadata_file.open("w") as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_hash(self, model: nn.Module) -> str:
        """Compute hash of model weights for integrity verification."""
        hasher = hashlib.sha256()
        for param in model.parameters():
            # Use string representation instead of numpy to avoid numpy dependency
            data_str = str(param.data.cpu().flatten().tolist())
            hasher.update(data_str.encode())
        return hasher.hexdigest()[:16]

    def checkpoint(self, model: nn.Module, tag: str, description: str = "") -> CheckpointInfo:
        """
        Save current learned state as a named checkpoint.

        Args:
            model: Neural memory model to checkpoint
            tag: Name for this checkpoint (e.g., "v1.0", "pre-experiment")
            description: Optional description

        Returns:
            CheckpointInfo with metadata
        """
        checkpoint_path = self.checkpoint_dir / f"{tag}.pt"

        # Save model state
        torch.save(model.state_dict(), checkpoint_path)

        # Compute metadata
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        weight_hash = self._compute_hash(model)

        info = CheckpointInfo(
            tag=tag,
            created_at=datetime.now(timezone.utc).isoformat(),
            size_mb=round(size_mb, 2),
            weight_hash=weight_hash,
            description=description,
        )

        # Update metadata
        self.metadata["checkpoints"][tag] = {
            "created_at": info.created_at,
            "size_mb": info.size_mb,
            "weight_hash": info.weight_hash,
            "description": info.description,
        }
        self._save_metadata()

        return info

    def restore(self, model: nn.Module, tag: str) -> CheckpointInfo:
        """
        Restore memory to a previous checkpoint.

        Args:
            model: Neural memory model to restore into
            tag: Checkpoint tag to restore

        Returns:
            CheckpointInfo of restored checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{tag}.pt"

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint '{tag}' not found")

        # Load state (handle CPU/GPU compatibility)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        # Return metadata
        meta = self.metadata["checkpoints"].get(tag, {})
        return CheckpointInfo(
            tag=tag,
            created_at=meta.get("created_at", ""),
            size_mb=meta.get("size_mb", 0),
            weight_hash=self._compute_hash(model),
            description=meta.get("description", ""),
        )

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints."""
        checkpoints = []
        for tag, meta in self.metadata["checkpoints"].items():
            checkpoints.append(
                CheckpointInfo(
                    tag=tag,
                    created_at=meta.get("created_at", ""),
                    size_mb=meta.get("size_mb", 0),
                    weight_hash=meta.get("weight_hash", ""),
                    description=meta.get("description", ""),
                )
            )
        return sorted(checkpoints, key=lambda x: x.created_at, reverse=True)

    def delete(self, tag: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{tag}.pt"

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if tag in self.metadata["checkpoints"]:
            del self.metadata["checkpoints"][tag]
            self._save_metadata()
            return True

        return False
