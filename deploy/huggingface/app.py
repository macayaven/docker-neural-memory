"""
Docker Neural Memory - Production Demo

REAL neural memory implementation using Titans architecture.
Demonstrates Docker-native AI memory with MCP server integration.

Deploy to: https://huggingface.co/spaces
"""

import sys
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

# Add src to path for real implementation
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MemoryConfig
from src.memory.neural_memory import NeuralMemory

# =============================================================================
# REAL NEURAL MEMORY INSTANCE
# =============================================================================

# Initialize the REAL neural memory - this is actual PyTorch, not a simulation
memory = NeuralMemory(MemoryConfig(dim=256, learning_rate=0.02))

# Track history for visualization
observation_history: list[dict] = []


def reset_memory():
    """Reset to fresh memory state."""
    global memory, observation_history
    memory = NeuralMemory(MemoryConfig(dim=256, learning_rate=0.02))
    observation_history = []
    return "Memory reset. Fresh neural network initialized."


# =============================================================================
# VISUALIZATION
# =============================================================================


def get_weight_sample() -> np.ndarray:
    """Extract 16x16 sample of actual neural weights."""
    with torch.no_grad():
        # Get weights from first linear layer
        weights = memory.memory_net[0].weight.data[:16, :16]
        return weights.cpu().numpy()


def create_weight_visualization() -> plt.Figure:
    """Visualize actual neural network weights."""
    weights = get_weight_sample()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_title(
        f"Neural Memory Weights\n({sum(p.numel() for p in memory.memory_net.parameters()):,} parameters)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("These weights UPDATE during inference (TTT)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Weight Value")
    plt.tight_layout()
    return fig


def create_history_plot() -> plt.Figure:
    """Plot surprise history."""
    fig, ax = plt.subplots(figsize=(8, 3))

    if observation_history:
        surprises = [h["surprise"] for h in observation_history]
        x = range(1, len(surprises) + 1)
        ax.plot(x, surprises, "o-", color="#e74c3c", linewidth=2, markersize=8)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
        ax.set_xlabel("Observation #")
        ax.set_ylabel("Surprise")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No observations yet", ha="center", va="center", fontsize=12, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title("Learning Progress (Surprise Over Time)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================================
# CORE MEMORY OPERATIONS
# =============================================================================


def observe_content(content: str) -> tuple[str, plt.Figure, plt.Figure]:
    """
    Feed content to REAL neural memory - triggers actual gradient updates.
    """
    if not content.strip():
        return "Please enter content to observe.", None, None

    # Get weight hash BEFORE
    hash_before = memory.get_weight_hash()

    # REAL observation with actual gradient descent
    result = memory.observe(content)

    # Get weight hash AFTER
    hash_after = memory.get_weight_hash()

    # Record history
    observation_history.append({
        "content": content[:50],
        "surprise": result["surprise"],
        "weight_delta": result["weight_delta"],
        "learned": result["learned"],
    })

    # Format result
    weights_changed = hash_before != hash_after
    output = f"""## Observation Result

**Content:** "{content[:100]}{'...' if len(content) > 100 else ''}"

### Metrics (REAL - from PyTorch gradient descent)

| Metric | Value |
|--------|-------|
| **Surprise** | {result['surprise']:.4f} |
| **Weight Delta** | {result['weight_delta']:.6f} |
| **Weights Changed** | {'YES' if weights_changed else 'NO'} |
| **Hash Before** | `{hash_before}` |
| **Hash After** | `{hash_after}` |

### What Just Happened

1. Text was encoded to tensor representation
2. Forward pass through neural memory network
3. **Surprise computed** via prediction error (MSE loss)
4. **Gradients calculated** via `torch.autograd.grad()`
5. **Weights updated** via gradient descent: `param -= lr * grad`

This is REAL test-time training. The neural network's weights physically changed.
"""

    return output, create_weight_visualization(), create_history_plot()


def check_surprise(content: str) -> str:
    """Check surprise WITHOUT learning."""
    if not content.strip():
        return "Please enter content to check."

    # REAL surprise computation (no learning)
    surprise = memory.surprise(content)

    return f"""## Surprise Check (No Learning)

**Content:** "{content[:100]}{'...' if len(content) > 100 else ''}"

**Surprise Score:** {surprise:.4f}

Interpretation:
- **< 0.3**: Very familiar - memory has seen similar patterns
- **0.3 - 0.6**: Moderately novel
- **> 0.6**: Highly novel - worth learning

{'This content is FAMILIAR to the memory.' if surprise < 0.3 else 'This content is NOVEL to the memory.' if surprise > 0.6 else 'This content is somewhat familiar.'}
"""


def get_memory_stats() -> str:
    """Get real memory statistics."""
    stats = memory.get_stats()

    return f"""## Memory Statistics

| Metric | Value |
|--------|-------|
| **Total Observations** | {stats['total_observations']} |
| **Parameters** | {stats['weight_parameters']:,} |
| **Dimension** | {stats['dimension']} |
| **Learning Rate** | {stats['learning_rate']:.4f} |
| **Avg Recent Surprise** | {stats['avg_surprise']:.4f} |
| **Current Weight Hash** | `{memory.get_weight_hash()}` |

### This is a Real Neural Network

- **Architecture**: 2-layer MLP with GELU activation and LayerNorm
- **Framework**: PyTorch with autograd
- **Learning**: Test-time training via gradient descent
- **Memory**: ~{stats['weight_parameters'] * 4 / 1024:.1f} KB of weights

Unlike RAG which stores vectors in a database, this IS the memory.
The weights encode everything learned.
"""


# =============================================================================
# DOCKER ECOSYSTEM INTEGRATION
# =============================================================================

DOCKER_INTEGRATION_MD = """
## Docker Ecosystem Integration

This neural memory is designed for **containerized deployment** with full Docker integration.

### MCP Server Interface

The memory exposes tools via Model Context Protocol (MCP):

```python
# MCP Tools Available
@mcp.tool()
def observe(content: str) -> dict:
    '''Feed context, trigger learning.'''
    return memory.observe(content)

@mcp.tool()
def surprise(content: str) -> float:
    '''Measure novelty without learning.'''
    return memory.surprise(content)

@mcp.tool()
def checkpoint(name: str) -> str:
    '''Save learned state to Docker volume.'''
    return save_checkpoint(name)

@mcp.tool()
def restore(name: str) -> str:
    '''Load previous state from Docker volume.'''
    return load_checkpoint(name)
```

### Docker Compose Deployment

```yaml
version: '3.8'
services:
  neural-memory:
    build: .
    ports:
      - "8000:8000"  # MCP server
    volumes:
      - memory-state:/app/checkpoints  # Persistent state
    environment:
      - MEMORY_DIM=512
      - LEARNING_RATE=0.01

volumes:
  memory-state:  # State survives container restarts
```

### Key Docker-Native Features

| Feature | Implementation |
|---------|---------------|
| **State Persistence** | Docker volumes for checkpoints |
| **Horizontal Scaling** | Stateless inference, shared state via volume |
| **CI/CD Integration** | GitHub Actions with Docker build |
| **Resource Control** | Container limits for GPU/memory |
| **Health Checks** | `/health` endpoint with memory stats |

### Why Docker + Neural Memory?

1. **Containerized AI Memory**: Package learned state with your app
2. **Version Control**: Checkpoint states like Git commits
3. **Reproducibility**: Same container = same behavior
4. **Orchestration Ready**: Deploy to Kubernetes, ECS, etc.
5. **MCP Protocol**: Claude Desktop integration via container

---

*This project demonstrates production-grade AI infrastructure with Docker.*
"""

ABOUT_MD = """
## About This Project

### What Makes This Special

This is **NOT a simulation**. The demo runs real PyTorch code:

1. **Real Neural Network**: 2-layer MLP with ~250K parameters
2. **Real Gradient Descent**: `torch.autograd.grad()` computes gradients
3. **Real Weight Updates**: Parameters change during inference
4. **Real Surprise Metric**: MSE loss measures prediction error

### The Titans Architecture

Based on Google's December 2024 paper: [arxiv.org/abs/2501.00663](https://arxiv.org/abs/2501.00663)

**Key Innovation**: The memory IS a neural network. Instead of storing vectors,
it learns patterns by updating weights during inference.

### Docker Integration

- **MCP Server**: Model Context Protocol for Claude Desktop
- **Checkpoints**: Save/restore learned state via Docker volumes
- **Container-Native**: Designed for orchestrated deployment

### Built By

**Carlos Crespo Macaya**
AI Engineer - GenAI Systems & Applied MLOps

- 10+ years production ML experience
- Expert in Docker, Kubernetes, MCP servers
- Currently at HP AICoE building multi-agent systems

This project demonstrates the ability to:
1. Read cutting-edge research (Titans paper)
2. Implement it correctly (PyTorch TTT)
3. Productionize it (Docker, MCP, CI/CD)
4. Make it compelling (this demo)

**Contact:** [macayaven@gmail.com](mailto:macayaven@gmail.com)

**GitHub:** [macayaven/docker-neural-memory](https://github.com/macayaven/docker-neural-memory)
"""


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="Docker Neural Memory", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Docker Neural Memory
    ## Real Test-Time Training - Not a Simulation

    This demo runs **actual PyTorch** code. When you observe content,
    real gradients flow and real weights update.
    """)

    with gr.Tabs():
        # TAB 1: Live Demo
        with gr.TabItem("Live Demo"):
            gr.Markdown("### Watch Real Neural Learning")

            with gr.Row():
                with gr.Column(scale=1):
                    observe_input = gr.Textbox(
                        label="Content to Observe",
                        placeholder="Enter text to trigger real learning...",
                        lines=3,
                    )
                    observe_btn = gr.Button("Observe (Learn)", variant="primary", size="lg")
                    observe_output = gr.Markdown()

                with gr.Column(scale=1):
                    weights_plot = gr.Plot(label="Neural Weights (Real PyTorch)")

            history_plot = gr.Plot(label="Learning History")

            observe_btn.click(
                observe_content,
                inputs=[observe_input],
                outputs=[observe_output, weights_plot, history_plot],
            )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    surprise_input = gr.Textbox(
                        label="Check Surprise (No Learning)",
                        placeholder="Check novelty without updating weights...",
                    )
                    surprise_btn = gr.Button("Check Surprise")
                    surprise_output = gr.Markdown()
                    surprise_btn.click(check_surprise, inputs=[surprise_input], outputs=[surprise_output])

                with gr.Column():
                    stats_btn = gr.Button("Get Memory Stats")
                    stats_output = gr.Markdown()
                    stats_btn.click(get_memory_stats, outputs=[stats_output])

            reset_btn = gr.Button("Reset Memory", variant="secondary")
            reset_output = gr.Markdown()
            reset_btn.click(reset_memory, outputs=[reset_output])

        # TAB 2: Docker Integration
        with gr.TabItem("Docker Integration"):
            gr.Markdown(DOCKER_INTEGRATION_MD)

        # TAB 3: About
        with gr.TabItem("About"):
            gr.Markdown(ABOUT_MD)

    gr.Markdown("""
    ---
    *Docker Neural Memory - Containerized AI memory with real test-time training*

    [GitHub](https://github.com/macayaven/docker-neural-memory) |
    [Contact](mailto:macayaven@gmail.com)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
