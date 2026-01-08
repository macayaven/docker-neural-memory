# Docker Patterns Reference

Containerization patterns for Docker Neural Memory.

## Base Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create volume mount points
RUN mkdir -p /app/weights /app/checkpoints
VOLUME ["/app/weights", "/app/checkpoints"]

# Environment configuration
ENV MEMORY_DIM=512
ENV TTT_VARIANT=mlp
ENV LEARNING_RATE=0.01
ENV MCP_PORT=8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${MCP_PORT}/health')" || exit 1

# MCP server entrypoint
EXPOSE ${MCP_PORT}
CMD ["python", "-m", "src.mcp_server"]
```

## requirements.txt

```
torch>=2.0.0
transformers>=4.30.0
mcp>=0.1.0
uvicorn>=0.22.0
pydantic>=2.0.0
numpy>=1.24.0
safetensors>=0.3.0
```

## Docker Compose Configurations

### Single Memory (Development)

```yaml
version: '3.8'

services:
  neural-memory:
    build: .
    image: neuralmemory/base:latest
    container_name: neural-memory
    ports:
      - "8765:8765"
    volumes:
      - memory-weights:/app/weights
      - memory-checkpoints:/app/checkpoints
    environment:
      - MEMORY_DIM=512
      - TTT_VARIANT=mlp
      - LEARNING_RATE=0.01
      - LOG_LEVEL=info
    restart: unless-stopped

volumes:
  memory-weights:
  memory-checkpoints:
```

### Multi-Memory Architecture

For complex applications with different memory concerns:

```yaml
version: '3.8'

services:
  # Long-term project memory (slow, stable)
  project-memory:
    image: neuralmemory/base:latest
    container_name: project-memory
    volumes:
      - project-weights:/app/weights
      - project-checkpoints:/app/checkpoints
    environment:
      - MEMORY_DIM=512
      - LEARNING_RATE=0.005  # Slower, more stable
      - MCP_PORT=8765
    ports:
      - "8765:8765"

  # Fast session memory (ephemeral)
  session-memory:
    image: neuralmemory/base:latest
    container_name: session-memory
    environment:
      - MEMORY_DIM=256
      - LEARNING_RATE=0.05  # Faster adaptation
      - MCP_PORT=8766
    # No volume = ephemeral (resets on restart)
    ports:
      - "8766:8766"

  # Pre-trained domain expert (read-only)
  python-expert:
    image: neuralmemory/code:python
    container_name: python-expert
    volumes:
      - python-weights:/app/weights:ro  # Read-only
    environment:
      - MEMORY_DIM=512
      - LEARNING_RATE=0  # No learning, inference only
      - MCP_PORT=8767
    ports:
      - "8767:8767"

  # MCP Gateway (routes to appropriate memory)
  mcp-gateway:
    image: neuralmemory/gateway:latest
    container_name: mcp-gateway
    ports:
      - "8760:8760"
    environment:
      - PROJECT_MEMORY_URL=http://project-memory:8765
      - SESSION_MEMORY_URL=http://session-memory:8766
      - PYTHON_EXPERT_URL=http://python-expert:8767
    depends_on:
      - project-memory
      - session-memory
      - python-expert

volumes:
  project-weights:
  project-checkpoints:
  python-weights:
```

### GPU-Enabled (For larger models)

```yaml
version: '3.8'

services:
  neural-memory-gpu:
    image: neuralmemory/base:cuda
    container_name: neural-memory-gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - memory-weights:/app/weights
      - memory-checkpoints:/app/checkpoints
    environment:
      - MEMORY_DIM=1024  # Larger with GPU
      - TTT_VARIANT=mlp
      - DEVICE=cuda
    ports:
      - "8765:8765"

volumes:
  memory-weights:
  memory-checkpoints:
```

### Hugging Face Spaces Compatible

```yaml
version: '3.8'

services:
  neural-memory:
    image: neuralmemory/base:hf
    container_name: neural-memory
    ports:
      - "7860:7860"  # HF Spaces default port
    volumes:
      - /data/weights:/app/weights  # HF persistent storage
      - /data/checkpoints:/app/checkpoints
    environment:
      - MEMORY_DIM=512
      - MCP_PORT=7860
      - HF_SPACE=true
```

## State Manager Implementation

```python
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import torch
from safetensors.torch import save_file, load_file


class StateManager:
    """
    Manages neural memory state persistence.
    
    - Saves/restores weight checkpoints
    - Handles versioning and forking
    - Tracks observation history
    """
    
    def __init__(
        self,
        weights_dir: str = "/app/weights",
        checkpoints_dir: str = "/app/checkpoints"
    ):
        self.weights_dir = Path(weights_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        
        # Ensure directories exist
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_path = self.checkpoints_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Observation tracking
        self.observation_count = self.metadata.get("total_observations", 0)
        self._recent_surprises: List[float] = []
    
    def _load_metadata(self) -> Dict:
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {"checkpoints": {}, "total_observations": 0}
    
    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _compute_hash(self, state_dict: Dict) -> str:
        """Compute hash of weights for integrity verification."""
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            hasher.update(key.encode())
            hasher.update(state_dict[key].numpy().tobytes())
        return hasher.hexdigest()[:16]
    
    def save_checkpoint(
        self,
        memory,
        tag: str,
        description: Optional[str] = None
    ) -> Dict:
        """Save current memory state as checkpoint."""
        
        state_dict = memory.memory_net.state_dict()
        weight_hash = self._compute_hash(state_dict)
        
        # Save weights using safetensors (safe, fast)
        checkpoint_path = self.checkpoints_dir / f"{tag}.safetensors"
        save_file(state_dict, checkpoint_path)
        
        # Update metadata
        checkpoint_info = {
            "tag": tag,
            "checkpoint_id": f"{tag}-{weight_hash}",
            "created_at": datetime.utcnow().isoformat(),
            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
            "weight_hash": weight_hash,
            "description": description,
            "observations_at_save": self.observation_count
        }
        
        self.metadata["checkpoints"][tag] = checkpoint_info
        self._save_metadata()
        
        return {
            "checkpoint_id": checkpoint_info["checkpoint_id"],
            "size_mb": checkpoint_info["size_mb"],
            "weight_hash": weight_hash,
            "path": str(checkpoint_path)
        }
    
    def restore_checkpoint(
        self,
        memory,
        tag: str
    ) -> Dict:
        """Restore memory from checkpoint."""
        
        checkpoint_path = self.checkpoints_dir / f"{tag}.safetensors"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint '{tag}' not found")
        
        # Load weights
        state_dict = load_file(checkpoint_path)
        memory.memory_net.load_state_dict(state_dict)
        
        # Reset momentum buffers
        memory.reset_momentum()
        
        checkpoint_info = self.metadata["checkpoints"].get(tag, {})
        observations_since = self.observation_count - checkpoint_info.get("observations_at_save", 0)
        
        return {
            "restored": True,
            "weight_hash": self._compute_hash(state_dict),
            "learning_since_checkpoint": observations_since,
            "warning": f"Rolling back {observations_since} observations" if observations_since > 100 else None
        }
    
    def fork_checkpoint(
        self,
        source_tag: str,
        new_tag: str,
        description: Optional[str] = None
    ) -> Dict:
        """Fork a checkpoint into a new branch."""
        
        source_path = self.checkpoints_dir / f"{source_tag}.safetensors"
        new_path = self.checkpoints_dir / f"{new_tag}.safetensors"
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source checkpoint '{source_tag}' not found")
        
        # Copy checkpoint file
        import shutil
        shutil.copy2(source_path, new_path)
        
        # Load to compute hash
        state_dict = load_file(new_path)
        weight_hash = self._compute_hash(state_dict)
        
        # Update metadata
        source_info = self.metadata["checkpoints"].get(source_tag, {})
        
        self.metadata["checkpoints"][new_tag] = {
            "tag": new_tag,
            "checkpoint_id": f"{new_tag}-{weight_hash}",
            "created_at": datetime.utcnow().isoformat(),
            "size_mb": new_path.stat().st_size / (1024 * 1024),
            "weight_hash": weight_hash,
            "description": description or f"Fork of {source_tag}",
            "forked_from": source_tag,
            "observations_at_save": source_info.get("observations_at_save", 0)
        }
        self._save_metadata()
        
        return {
            "forked": True,
            "source_hash": source_info.get("weight_hash", "unknown"),
            "new_hash": weight_hash,
            "path": str(new_path)
        }
    
    def list_checkpoints(
        self,
        prefix: Optional[str] = None,
        limit: int = 100
    ) -> Dict:
        """List available checkpoints."""
        
        checkpoints = list(self.metadata["checkpoints"].values())
        
        if prefix:
            checkpoints = [c for c in checkpoints if c["tag"].startswith(prefix)]
        
        checkpoints = sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)[:limit]
        
        total_size = sum(c["size_mb"] for c in checkpoints)
        
        return {
            "checkpoints": checkpoints,
            "total_count": len(checkpoints),
            "total_size_mb": total_size
        }
    
    def record_observation(self, surprise: float):
        """Record an observation for tracking."""
        self.observation_count += 1
        self._recent_surprises.append(surprise)
        
        # Keep only recent surprises
        if len(self._recent_surprises) > 100:
            self._recent_surprises = self._recent_surprises[-100:]
        
        # Periodic metadata save
        if self.observation_count % 100 == 0:
            self.metadata["total_observations"] = self.observation_count
            self._save_metadata()
    
    def recent_surprise_avg(self) -> float:
        """Get average recent surprise."""
        if not self._recent_surprises:
            return 0.0
        return sum(self._recent_surprises) / len(self._recent_surprises)
    
    def known_domains(self) -> List[str]:
        """Get list of known domains."""
        return self.metadata.get("domains", ["default"])
    
    def get_recent_observations(self, n: int = 100):
        """Get recent observations for replay during consolidation."""
        # In practice, this would load from a replay buffer
        # For now, return placeholder
        return None


def estimate_capacity(memory) -> float:
    """
    Estimate memory capacity utilization.
    
    Heuristic: High average gradient norm suggests approaching capacity.
    """
    total_grad_norm = 0
    param_count = 0
    
    for param in memory.memory_net.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    avg_grad_norm = total_grad_norm / param_count
    
    # Normalize to 0-1 (heuristic thresholds)
    return min(1.0, avg_grad_norm / 10.0)
```

## Health Check Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class HealthResponse(BaseModel):
    status: str
    memory_loaded: bool
    capacity_used: float
    checkpoints_available: int


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "memory_loaded": memory is not None,
        "capacity_used": estimate_capacity(memory) if memory else 0,
        "checkpoints_available": len(state_manager.list_checkpoints()["checkpoints"])
    }
```

## Image Variants

### Base Image Tags

| Tag | Description | Use Case |
|-----|-------------|----------|
| `latest` | Latest stable release | Production |
| `slim` | Minimal dependencies | Resource-constrained |
| `cuda` | GPU support | Large models |
| `hf` | HF Spaces compatible | Deployment |
| `dev` | Development with debug tools | Development |

### Domain-Specialized Images

| Image | Description |
|-------|-------------|
| `neuralmemory/code:python` | Pre-trained on Python patterns |
| `neuralmemory/code:typescript` | Pre-trained on TypeScript |
| `neuralmemory/domain:legal` | Legal document patterns |
| `neuralmemory/domain:medical` | Medical terminology |

Build specialized images:

```dockerfile
FROM neuralmemory/base:latest

# Pre-trained weights for domain
COPY weights/python-expert.safetensors /app/weights/base.safetensors

# Domain configuration
ENV DOMAIN=python
ENV PRETRAINED=true
ENV LEARNING_RATE=0.005  # Lower LR to preserve training
```

## Publishing to Docker Hub

```bash
# Build and tag
docker build -t neuralmemory/base:latest .
docker build -t neuralmemory/base:$(cat VERSION) .

# Push
docker push neuralmemory/base:latest
docker push neuralmemory/base:$(cat VERSION)

# Multi-architecture (for ARM64/DGX Spark compatibility)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t neuralmemory/base:latest \
  --push .
```
