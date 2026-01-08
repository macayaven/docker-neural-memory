# Docker Neural Memory

Containerized neural memory that learns at test time using Titans/TTT architecture.

## Overview

Unlike traditional AI memory (RAG) that stores and retrieves embeddings, this system **learns** during inference. Its neural weights update with every interaction, enabling true learning and generalization.

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

## Quick Start

```bash
# Run with Docker
docker compose up -d

# Or development mode
docker compose -f docker-compose.dev.yml run --rm dev

# Run tests
docker compose -f docker-compose.dev.yml run --rm test
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

## Configuration

Environment variables:
- `MEMORY_DIM`: Memory dimension (default: 512)
- `TTT_VARIANT`: "linear" or "mlp" (default: mlp)
- `LEARNING_RATE`: Learning rate for TTT (default: 0.01)

## Key Papers

- [Titans](https://arxiv.org/abs/2501.00663) - Neural long-term memory (Dec 2024)
- [TTT Layers](https://arxiv.org/abs/2407.04620) - Test-time training (Jul 2024)

## License

MIT
