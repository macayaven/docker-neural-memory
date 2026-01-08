# DGX Spark Deployment: Gemma + Neural Memory

Quantitative comparison demo showing the improvement when using Neural Memory
versus standard LLM without memory.

## Quick Start

```bash
# On DGX Spark
cd deploy/dgx-spark

# Start all services
docker compose up -d

# Pull Gemma model (first time only)
docker exec -it dgx-spark-gemma-1 ollama pull gemma2:2b

# Open demo
open http://localhost:7860
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DGX Spark                            │
│                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Gemma     │◄──►│  Neural Memory  │◄──►│   Demo UI   │ │
│  │   (GPU)     │    │  MCP Server     │    │   Gradio    │ │
│  │  :11434     │    │    :8765        │    │   :7860     │ │
│  └─────────────┘    └─────────────────┘    └─────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## What the Demo Shows

### Side-by-Side Comparison

| Metric | With Neural Memory | Without Memory |
|--------|-------------------|----------------|
| Clarifying questions | Decreases over time | Every time |
| Response accuracy | Improves as it learns | Static |
| Token usage | Bounded | Grows with context |
| Surprise score | Drops for familiar patterns | N/A |

### Example Interaction

**Query 1:** "Open my editor"
- Without memory: "Which editor would you like to open?"
- With memory: "Which editor would you like to open?"

**User clarifies:** "VS Code"
- Without memory: (forgets immediately)
- With memory: Learns preference, surprise drops

**Query 2:** "Open my editor"
- Without memory: "Which editor would you like to open?"
- With memory: "Opening VS Code..." (no clarification needed!)

## Metrics Tracked

- **Surprise**: How novel is the input (0-1)
- **Correct predictions**: Responses without clarifying questions
- **Total tokens**: Cumulative token usage
- **Response time**: Latency comparison

## GPU Requirements

- NVIDIA GPU with at least 4GB VRAM for Gemma 2B
- DGX Spark recommended for best performance

## Customization

Environment variables:
- `MEMORY_DIM`: Neural memory dimension (default: 512)
- `LEARNING_RATE`: How fast to learn (default: 0.02)
- `GEMMA_MODEL`: Model to use (default: gemma2:2b)
