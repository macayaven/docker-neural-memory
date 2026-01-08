# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**Docker Neural Memory** is a containerized implementation of test-time training (TTT) memory, based on Google's Titans architecture. Unlike traditional AI memory solutions that store and retrieve embeddings, this system **learns** during inference—its neural weights update with every interaction.

See `SPEC.md` for the full technical specification.

## Build & Development Commands

```bash
# Build container
docker compose build

# Run neural memory service
docker compose up -d

# Development (local)
pip install -e ".[dev]"
python -m src.mcp_server

# Run tests
pytest

# Linting
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Architecture

### Core Components

```
src/
├── memory/
│   ├── neural_memory.py    # Titans memory module
│   ├── ttt_layer.py        # TTT-Linear and TTT-MLP
│   └── consolidation.py    # Pattern compression
├── mcp_server/
│   ├── server.py           # MCP server implementation
│   └── tools.py            # Tool definitions
└── state/
    ├── checkpoint.py       # Save/restore weights
    └── versioning.py       # Fork/branch logic
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
| `checkpoint` | Save learned state |
| `restore` | Load previous state |
| `fork` | Branch memory state |
| `list_checkpoints` | List all checkpoints |
| `stats` | Get memory statistics |
| `attention_map` | Visualize attention weights |
| `explain` | Export patterns as summaries |

## Configuration

Environment variables:
- `MEMORY_DIM`: Memory dimension (default: 512)
- `TTT_VARIANT`: "linear" or "mlp" (default: mlp)
- `LEARNING_RATE`: Learning rate for TTT (default: 0.01)

## Testing Checklist

1. **Learning verification**: Surprise decreases on repeated patterns
2. **Persistence**: State survives container restart
3. **Generalization**: Recognizes patterns beyond exact matches

## Key Papers

- **Titans** (Dec 2024): https://arxiv.org/abs/2501.00663
- **TTT Layers** (Jul 2024): https://arxiv.org/abs/2407.04620
- **TPTT** (Jun 2025): https://arxiv.org/abs/2506.17671

## Useful Libraries

- `tptt` - Framework for Titans conversion (PyPI)
- `mcp` - Model Context Protocol server
