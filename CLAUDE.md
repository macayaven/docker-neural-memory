# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Docker Neural Memory** is an implementation of test-time training (TTT) memory, based on Google's Titans architecture. Unlike traditional AI memory solutions that store and retrieve embeddings, this system **learns** during inference—its neural weights update with every interaction.

## Build & Development Commands

```bash
# Local development
pip install -e ".[dev]"          # Dev dependencies

# Run MCP server
python -m src.mcp_server         # MCP server (stdio mode)

# Testing
pytest                           # Run all tests
pytest tests/test_learning.py    # Run specific test file
pytest -k "test_observe"         # Run tests matching pattern
pytest -x                        # Stop on first failure

# Linting (uses ruff, not black)
ruff check src/ tests/           # Lint
ruff check --fix src/ tests/     # Lint with auto-fix
ruff format src/ tests/          # Format
mypy src/                        # Type check
```

## Architecture

### Core Components

```
src/
├── memory/
│   ├── neural_memory.py    # NeuralMemory class (nn.Module) - Titans memory
│   ├── ttt_layer.py        # TTTLayer - test-time training implementation
│   └── consolidation.py    # MemoryConsolidator - pattern compression
├── mcp_server/
│   ├── server.py           # NeuralMemoryServer - MCP protocol handler
│   └── tools.py            # TOOL_SCHEMAS dict - MCP tool definitions
└── config.py               # Pydantic Settings (MemoryConfig, TTTConfig, etc.)
```

### Key Concepts

**Traditional memory (RAG):**
```
store(content) → save embedding
query(prompt)  → similarity search
```

**Neural memory (Titans/TTT):**
```
observe(context) → weights update (learning)
infer(prompt)    → generate from learned model
surprise(input)  → measure novelty
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `observe` | Feed context, trigger learning |
| `infer` | Query using learned representations |
| `surprise` | Measure novelty of input |
| `consolidate` | Compress patterns (like sleep) |
| `stats` | Get memory statistics |
| `attention_map` | Visualize attention weights |
| `explain` | Export patterns as summaries |

## Configuration

Uses Pydantic Settings with environment variable prefixes (see `src/config.py`):

| Prefix | Config Class | Key Variables |
|--------|--------------|---------------|
| `MEMORY_` | `MemoryConfig` | `DIM` (512), `LEARNING_RATE` (0.01), `DEVICE` (cpu) |
| `TTT_` | `TTTConfig` | `VARIANT` (mlp/linear), `NUM_STEPS` (1) |
| `MCP_` | `MCPConfig` | `HOST`, `PORT` (8765), `MODE` (stdio) |

## Testing Verification

When testing learning behavior:
1. **Surprise decreases**: Repeated patterns should show lower surprise scores
2. **Weights change**: `weight_delta` should be non-zero after `observe()`

## Key Papers

- **Titans** (Dec 2024): https://arxiv.org/abs/2501.00663
- **TTT Layers** (Jul 2024): https://arxiv.org/abs/2407.04620
- **TPTT** (Jun 2025): https://arxiv.org/abs/2506.17671
