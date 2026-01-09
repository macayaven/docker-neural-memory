# Docker Neural Memory

Containerized neural memory that learns at test time using the Titans/TTT architecture.

Unlike traditional AI memory (RAG) that stores and retrieves embeddings, this system **learns** during inference - its neural weights update with every interaction.

## Live Demo

Try it now: [HuggingFace Space](https://huggingface.co/spaces/macayaven/docker-neural-memory)

## Quick Start

```bash
# Install locally
pip install -e ".[dev]"

# Run MCP server
python -m src.mcp_server
```

## Key Features

- **Test-Time Training**: Memory updates via gradient descent during inference
- **MCP Interface**: Standard Model Context Protocol for tool integration
- **Surprise Scoring**: Quantifies how novel an input is to the learned model

## Architecture

Based on Google's [Titans architecture](https://arxiv.org/abs/2501.00663) and [TTT layers](https://arxiv.org/abs/2407.04620).

```
Traditional Memory:  store(content) -> embedding -> retrieve(query)
Neural Memory:       observe(context) -> weights update -> infer(prompt)
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `observe` | Feed context, trigger learning |
| `infer` | Query using learned representations |
| `surprise` | Measure novelty of input (0-1) |
| `consolidate` | Compress patterns (like sleep) |
| `stats` | Get memory statistics |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## License

MIT
