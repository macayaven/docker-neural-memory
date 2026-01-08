# Docker Neural Memory
# Containerized neural memory that learns at test time

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch transformers mcp pydantic uvicorn

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

# MCP server port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()" || exit 1

CMD ["python", "-m", "src.mcp_server"]
