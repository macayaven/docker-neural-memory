"""
HTTP API wrapper for Neural Memory.

Provides REST endpoints for the comparison demo.
Run alongside or instead of the MCP server.
"""

import json
import logging
import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import MemoryConfig
from .memory.neural_memory import NeuralMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize
app = FastAPI(title="Neural Memory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
config = MemoryConfig(
    dim=int(os.environ.get("MEMORY_DIM", "512")),
    learning_rate=float(os.environ.get("LEARNING_RATE", "0.02")),
)

memory = NeuralMemory(config)
logger.info(f"Neural Memory HTTP API initialized: dim={config.dim}, lr={config.learning_rate}")


# Request/Response models
class ObserveRequest(BaseModel):
    content: str
    learning_rate: float | None = None


class SurpriseRequest(BaseModel):
    content: str


class ObserveResponse(BaseModel):
    surprise: float
    weight_delta: float
    learned: bool
    weight_hash: str


class SurpriseResponse(BaseModel):
    surprise: float
    recommendation: str


class StatsResponse(BaseModel):
    total_observations: int
    weight_parameters: int
    avg_surprise: float
    learning_rate: float
    dimension: int
    weight_hash: str


# Endpoints
@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memory_dim": config.dim,
        "parameters": sum(p.numel() for p in memory.parameters()),
    }


@app.post("/observe", response_model=ObserveResponse)
async def observe(request: ObserveRequest) -> ObserveResponse:
    """Observe content and trigger learning."""
    try:
        hash_before = memory.get_weight_hash()
        result = memory.observe(request.content, learning_rate=request.learning_rate)
        hash_after = memory.get_weight_hash()

        return ObserveResponse(
            surprise=result["surprise"],
            weight_delta=result["weight_delta"],
            learned=hash_before != hash_after,
            weight_hash=hash_after,
        )
    except Exception as e:
        logger.error(f"Observe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/surprise", response_model=SurpriseResponse)
async def surprise(request: SurpriseRequest) -> SurpriseResponse:
    """Check surprise without learning."""
    try:
        score = memory.surprise(request.content)

        if score > 0.7:
            recommendation = "learn"
        elif score < 0.3:
            recommendation = "skip"
        else:
            recommendation = "moderate"

        return SurpriseResponse(surprise=score, recommendation=recommendation)
    except Exception as e:
        logger.error(f"Surprise error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Get memory statistics."""
    try:
        mem_stats = memory.get_stats()
        return StatsResponse(
            total_observations=mem_stats["total_observations"],
            weight_parameters=mem_stats["weight_parameters"],
            avg_surprise=mem_stats["avg_surprise"],
            learning_rate=mem_stats["learning_rate"],
            dimension=mem_stats["dimension"],
            weight_hash=memory.get_weight_hash(),
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset() -> dict[str, str]:
    """Reset memory to initial state."""
    global memory
    memory = NeuralMemory(config)
    return {"status": "reset", "weight_hash": memory.get_weight_hash()}


def main() -> None:
    """Run the HTTP server."""
    import uvicorn

    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
