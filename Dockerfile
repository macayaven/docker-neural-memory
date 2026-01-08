# Docker Neural Memory
# Containerized neural memory that learns at test time

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch pydantic pydantic-settings uvicorn fastapi mcp

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml README.md /app/

# Install package
RUN pip install --no-cache-dir -e .

# Persistent volume mount points for learned state
VOLUME ["/app/weights", "/app/checkpoints"]

# Configuration via environment variables
ENV MEMORY_DIM=512
ENV TTT_VARIANT=mlp
ENV LEARNING_RATE=0.01
ENV SERVER_MODE=http

# Server port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Default: HTTP server (use "python -m src.mcp_server" for MCP mode)
CMD ["python", "-m", "src.http_server"]
