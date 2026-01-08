# Docker Configuration

## Dockerfile

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/neural_memory /app/neural_memory

RUN mkdir -p /app/weights /app/checkpoints
VOLUME ["/app/weights", "/app/checkpoints"]

ENV MEMORY_DIM=512 LEARNING_RATE=0.01
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import neural_memory; print('healthy')"

CMD ["python", "-m", "neural_memory.mcp.server"]
```

## Docker Compose

```yaml
version: "3.8"
services:
  neural-memory:
    build: .
    ports: ["8765:8765"]
    volumes:
      - memory-weights:/app/weights
      - memory-checkpoints:/app/checkpoints

volumes:
  memory-weights:
  memory-checkpoints:
```

## Devcontainer

```json
{
  "name": "Neural Memory Dev",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "charliermarsh.ruff"]
    }
  },
  "postCreateCommand": "pip install -e '.[dev]' && pre-commit install"
}
```
