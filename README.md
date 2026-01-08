# Docker Neural Memory

Containerized neural memory that learns at test time using the Titans/TTT architecture.

Unlike traditional AI memory (RAG) that stores and retrieves embeddings, this system **learns** during inference - its neural weights update with every interaction.

## Quick Start

```bash
# Build and run
docker compose up -d

# Or locally
pip install -e ".[dev]"
python -m src.mcp_server
```

## Key Features

- **Test-Time Training**: Memory updates via gradient descent during inference
- **MCP Interface**: Standard Model Context Protocol for tool integration
- **Checkpoint Management**: Git-like versioning for learned states
- **Observability**: Langfuse integration for training and inference traces

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
| `checkpoint` | Save learned state |
| `restore` | Load previous state |
| `fork` | Branch memory state |

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
