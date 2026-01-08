"""
MCP Server for Docker Neural Memory.

Provides the Model Context Protocol interface for neural memory operations.
"""

import asyncio
import json
import logging
import os
from typing import Any

import torch

from ..memory.consolidation import MemoryConsolidator
from ..memory.neural_memory import NeuralMemory
from ..state.checkpoint import CheckpointManager
from ..state.versioning import VersionManager
from .tools import TOOL_SCHEMAS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralMemoryServer:
    """
    MCP Server for neural memory operations.

    Handles all tool calls and manages the neural memory lifecycle.
    """

    def __init__(self) -> None:
        # Configuration from environment
        self.memory_dim = int(os.environ.get("MEMORY_DIM", "512"))
        self.ttt_variant = os.environ.get("TTT_VARIANT", "mlp")
        self.learning_rate = float(os.environ.get("LEARNING_RATE", "0.01"))

        # Initialize components
        self.memory = NeuralMemory(dim=self.memory_dim)
        self.memory.lr.data = torch.tensor(self.learning_rate)

        self.checkpoint_mgr = CheckpointManager()
        self.version_mgr = VersionManager(self.checkpoint_mgr)
        self.consolidator = MemoryConsolidator()

        # Statistics tracking
        self.total_observations = 0
        self.recent_surprises: list[float] = []
        self.domains: set[str] = set()

        logger.info(
            f"Neural Memory Server initialized: dim={self.memory_dim}, "
            f"variant={self.ttt_variant}, lr={self.learning_rate}"
        )

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor representation."""
        # Simple encoding - in production, use a proper tokenizer/encoder
        encoded = [ord(c) % 256 for c in text]
        # Pad or truncate to fixed size
        target_len = 128
        if len(encoded) < target_len:
            encoded.extend([0] * (target_len - len(encoded)))
        else:
            encoded = encoded[:target_len]

        # Create tensor [1, seq_len, dim]
        tensor = torch.zeros(1, len(encoded), self.memory_dim)
        for i, val in enumerate(encoded):
            tensor[0, i, val % self.memory_dim] = 1.0

        return tensor

    async def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Handle an MCP tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result
        """
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        return await handler(arguments)

    async def _handle_observe(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle observe tool call."""
        context = args["context"]
        domain = args.get("domain")
        lr_override = args.get("learning_rate")

        # Override learning rate if specified
        if lr_override:
            old_lr = self.memory.lr.data.item()
            self.memory.lr.data = torch.tensor(lr_override)

        # Convert to tensor and observe
        tensor = self._text_to_tensor(context)
        result = self.memory.observe(tensor)

        # Restore learning rate
        if lr_override:
            self.memory.lr.data = torch.tensor(old_lr)

        # Update statistics
        self.total_observations += 1
        self.recent_surprises.append(result["surprise"])
        if len(self.recent_surprises) > 100:
            self.recent_surprises.pop(0)
        if domain:
            self.domains.add(domain)

        return {
            "surprise": result["surprise"],
            "weight_delta": result["weight_delta"],
            "patterns_activated": [],  # TODO: implement pattern detection
        }

    async def _handle_infer(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle infer tool call."""
        prompt = args["prompt"]

        tensor = self._text_to_tensor(prompt)
        result = self.memory.infer(tensor)

        # Convert output back to interpretable form
        # In production, use a proper decoder
        confidence = 1.0 - self.memory.surprise(tensor)

        return {
            "response": f"[Neural memory inference for: {prompt[:50]}...]",
            "confidence": max(0.0, min(1.0, confidence)),
            "attention_weights": result["attention_weights"],
        }

    async def _handle_surprise(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle surprise tool call."""
        input_text = args["input"]

        tensor = self._text_to_tensor(input_text)
        surprise = self.memory.surprise(tensor)

        # Determine recommendation based on surprise level
        if surprise > 0.7:
            recommendation = "learn"
        elif surprise < 0.3:
            recommendation = "skip"
        else:
            recommendation = "consolidate"

        return {
            "score": surprise,
            "nearest_pattern": "",  # TODO: implement pattern matching
            "recommendation": recommendation,
        }

    async def _handle_consolidate(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Handle consolidate tool call."""
        # Use recent observations for consolidation
        # In production, would store actual observation tensors
        return self.consolidator.consolidate(
            self.memory.memory_net, [self._text_to_tensor("placeholder")]
        )

    async def _handle_checkpoint(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle checkpoint tool call."""
        tag = args["tag"]
        description = args.get("description", "")

        info = self.checkpoint_mgr.checkpoint(self.memory, tag, description)

        return {
            "checkpoint_id": info.weight_hash,
            "tag": info.tag,
            "size_mb": info.size_mb,
            "weight_hash": info.weight_hash,
        }

    async def _handle_restore(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle restore tool call."""
        tag = args["tag"]

        info = self.checkpoint_mgr.restore(self.memory, tag)
        learning = self.version_mgr.learning_since_checkpoint(self.memory, tag)

        return {
            "restored": True,
            "weight_hash": info.weight_hash,
            "learning_since_checkpoint": learning.get("total_learning", 0),
        }

    async def _handle_fork(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle fork tool call."""
        source_tag = args["source_tag"]
        new_tag = args["new_tag"]

        info = self.version_mgr.fork(self.memory, source_tag, new_tag)

        return {
            "forked": info.forked,
            "source_hash": info.source_hash,
            "new_hash": info.new_hash,
        }

    async def _handle_list_checkpoints(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Handle list_checkpoints tool call."""
        checkpoints = self.checkpoint_mgr.list_checkpoints()

        return {
            "checkpoints": [
                {
                    "tag": cp.tag,
                    "created_at": cp.created_at,
                    "size_mb": cp.size_mb,
                    "description": cp.description,
                }
                for cp in checkpoints
            ]
        }

    async def _handle_stats(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Handle stats tool call."""
        weight_params = sum(p.numel() for p in self.memory.parameters())
        avg_surprise = (
            sum(self.recent_surprises) / len(self.recent_surprises)
            if self.recent_surprises
            else 0.0
        )

        return {
            "total_observations": self.total_observations,
            "weight_parameters": weight_params,
            "capacity_used": min(1.0, self.total_observations / 10000),
            "avg_surprise": avg_surprise,
            "domains": list(self.domains),
        }

    async def _handle_attention_map(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle attention_map tool call."""
        query = args["query"]

        tensor = self._text_to_tensor(query)
        result = self.memory.infer(tensor)

        # Extract attention-like weights from output tensor
        response_tensor = result["response"]
        weights = response_tensor[0, 0, :].softmax(dim=0)

        return {
            "attention_weights": [
                {"pattern": f"pattern_{i}", "weight": w.item()} for i, w in enumerate(weights[:10])
            ],
            "visualization_url": None,
        }

    async def _handle_explain(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle explain tool call."""
        top_k = args.get("top_k", 10)

        # Analyze learned weights to extract patterns
        # This is a simplified version - production would do proper analysis
        patterns = []

        for name, param in self.memory.memory_net.named_parameters():
            if "weight" in name:
                # Find strongest connections
                values, indices = param.abs().flatten().topk(min(top_k, param.numel()))
                for val, idx in zip(values, indices, strict=True):
                    patterns.append(
                        {
                            "description": f"Weight {name}[{idx.item()}]",
                            "strength": val.item(),
                            "examples": [],
                        }
                    )

        # Sort by strength and take top_k
        patterns.sort(key=lambda x: x["strength"], reverse=True)

        return {"patterns": patterns[:top_k]}

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for MCP registration."""
        return list(TOOL_SCHEMAS.values())


async def main() -> None:
    """Run the MCP server."""
    server = NeuralMemoryServer()

    logger.info("Neural Memory MCP Server starting on port 8765")
    logger.info(f"Available tools: {list(TOOL_SCHEMAS.keys())}")

    # Simple stdio-based MCP server loop
    # In production, use proper MCP server implementation
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, input)
            request = json.loads(line)

            response: dict[str, Any] = {}
            if request.get("method") == "tools/list":
                response = {"tools": server.get_tool_schemas()}
            elif request.get("method") == "tools/call":
                params = request.get("params", {})
                result = await server.handle_tool_call(
                    params.get("name"), params.get("arguments", {})
                )
                response = {"result": result}
            else:
                response = {"error": f"Unknown method: {request.get('method')}"}

            print(json.dumps(response), flush=True)

        except EOFError:
            break
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
