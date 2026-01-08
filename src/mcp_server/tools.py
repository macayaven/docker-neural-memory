"""
MCP Tool definitions for Docker Neural Memory.

These tools provide a learning-focused interface, different from
traditional memory systems:

Traditional:      Neural Memory:
store(content)    observe(context) - triggers learning
query(prompt)     infer(prompt) - generates from model
-                 surprise(input) - measures novelty
-                 consolidate() - compresses patterns
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ObserveResult:
    """Result from observe operation."""

    surprise: float  # How novel was this? (0-1)
    weight_delta: float  # Magnitude of weight change
    patterns_activated: list[str]  # What patterns fired


@dataclass
class InferResult:
    """Result from infer operation."""

    response: str  # Generated from learned patterns
    confidence: float  # Model certainty
    attention_weights: list[float]  # What memory attended to


@dataclass
class SurpriseResult:
    """Result from surprise measurement."""

    score: float  # Surprise score (0-1)
    nearest_pattern: str  # Most similar learned pattern
    recommendation: str  # "learn", "skip", or "consolidate"


@dataclass
class ConsolidateResult:
    """Result from consolidation."""

    patterns_merged: int
    memory_compressed_by: float  # Percentage
    stability_score: float


@dataclass
class CheckpointResult:
    """Result from checkpoint operation."""

    checkpoint_id: str
    tag: str
    size_mb: float
    weight_hash: str


@dataclass
class StatsResult:
    """Memory statistics."""

    total_observations: int
    weight_parameters: int
    capacity_used: float  # Estimated 0-1
    avg_surprise: float  # Recent learning signal
    domains: list[str]


# Tool schemas for MCP registration
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "observe": {
        "name": "observe",
        "description": "Feed context to the memory. Weights update automatically via test-time training. Unlike store(), this triggers actual learning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Content to learn from",
                },
                "learning_rate": {
                    "type": "number",
                    "description": "Override default learning rate",
                },
                "domain": {
                    "type": "string",
                    "description": "Domain tag for routing",
                },
            },
            "required": ["context"],
        },
    },
    "infer": {
        "name": "infer",
        "description": "Query the memory using learned representations. Unlike query(), this uses the learned model to GENERATE, not retrieve.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What to infer about",
                },
                "temperature": {
                    "type": "number",
                    "description": "Generation temperature",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Response length limit",
                },
            },
            "required": ["prompt"],
        },
    },
    "surprise": {
        "name": "surprise",
        "description": "Measure how surprising/novel an input is. High surprise = worth learning. Low surprise = already known.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input to measure surprise for",
                },
            },
            "required": ["input"],
        },
    },
    "consolidate": {
        "name": "consolidate",
        "description": "Trigger consolidation pass (like sleep for memory). Compresses recent learning into stable long-term patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "checkpoint": {
        "name": "checkpoint",
        "description": "Save current learned state as a named checkpoint. Like `docker commit` but for neural memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Tag name (e.g., 'v1.0', 'pre-experiment')",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description",
                },
            },
            "required": ["tag"],
        },
    },
    "restore": {
        "name": "restore",
        "description": "Restore memory to a previous checkpoint. Like `docker run image:tag` but for learned state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Checkpoint tag to restore",
                },
            },
            "required": ["tag"],
        },
    },
    "fork": {
        "name": "fork",
        "description": "Fork memory state into a new branch. Enables experimentation without losing stable state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_tag": {
                    "type": "string",
                    "description": "Source checkpoint to fork from",
                },
                "new_tag": {
                    "type": "string",
                    "description": "Name for the new branch",
                },
            },
            "required": ["source_tag", "new_tag"],
        },
    },
    "list_checkpoints": {
        "name": "list_checkpoints",
        "description": "List all available checkpoints.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "stats": {
        "name": "stats",
        "description": "Get memory statistics including capacity, observations, and domains.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "attention_map": {
        "name": "attention_map",
        "description": "Visualize what the memory attends to for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to visualize attention for",
                },
            },
            "required": ["query"],
        },
    },
    "explain": {
        "name": "explain",
        "description": "Export learned patterns as interpretable summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of top patterns to explain",
                    "default": 10,
                },
            },
        },
    },
}
