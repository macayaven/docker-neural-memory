"""
Docker Neural Memory - Production Demo

REAL neural memory implementation using Titans architecture.
Demonstrates Docker-native AI memory with MCP server integration.

Deploy to: https://huggingface.co/spaces
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import InferenceClient
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")

# =============================================================================
# CUSTOM CSS FOR POLISHED UI
# =============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --neural-cyan: #00d4ff;
    --neural-cyan-glow: rgba(0, 212, 255, 0.3);
    --rag-orange: #ff8c42;
    --purple-accent: #a855f7;
    --bg-deep: #0a0a1a;
    --bg-card: #12122a;
    --bg-card-hover: #1a1a3a;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-subtle: rgba(148, 163, 184, 0.1);
    --success-green: #22c55e;
}

/* Global font settings */
.gradio-container {
    font-family: 'Outfit', system-ui, -apple-system, sans-serif !important;
    background: linear-gradient(180deg, var(--bg-deep) 0%, #0f0f23 100%) !important;
}

/* Headings */
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

/* Code and monospace */
.gradio-container code, .gradio-container pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Tab styling */
.tabs > .tab-nav > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.3s ease !important;
}

.tabs > .tab-nav > button.selected {
    background: linear-gradient(135deg, var(--neural-cyan) 0%, var(--purple-accent) 100%) !important;
    color: white !important;
}

/* Button styling */
.gr-button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--neural-cyan) 0%, var(--purple-accent) 100%) !important;
    border: none !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px var(--neural-cyan-glow) !important;
}

.gr-button-secondary {
    background: transparent !important;
    border: 1px solid var(--text-secondary) !important;
    color: var(--text-secondary) !important;
}

.gr-button-secondary:hover {
    border-color: var(--neural-cyan) !important;
    color: var(--neural-cyan) !important;
}

/* FIX: Labels should NOT look like buttons */
.gr-textbox label, .gr-plot label, .gr-dropdown label, .gr-checkbox label,
label.svelte-1gfkn6j, .label-wrap, span.svelte-1gfkn6j {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: default !important;
}

/* Ensure label containers don't have button styling */
.gr-form > label, .gr-box > label, div[data-testid="textbox"] > label {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}

/* Input styling */
.gr-textbox textarea, .gr-textbox input {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: var(--neural-cyan) !important;
    box-shadow: 0 0 0 3px var(--neural-cyan-glow) !important;
}

/* Card/box styling */
.gr-box, .gr-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}

/* Plot styling */
.gr-plot {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
}

/* Markdown styling */
.prose {
    color: var(--text-primary) !important;
}

.prose h3, .prose h4 {
    color: var(--neural-cyan) !important;
}

/* Smooth animations */
* {
    transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
"""

HEADER_HTML = '''
<div style="
    font-family: 'Outfit', system-ui, sans-serif;
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0a0a1a 100%);
    padding: 40px 30px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(0, 212, 255, 0.2);
    position: relative;
    overflow: hidden;
">
    <!-- Gradient glow effect -->
    <div style="
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 70% 70%, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
        pointer-events: none;
    "></div>

    <div style="position: relative; z-index: 1;">
        <!-- Logo and title -->
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 15px;">
            <div style="
                font-size: 48px;
                background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">🧠</div>
            <div>
                <h1 style="
                    font-size: 2.5em;
                    font-weight: 700;
                    margin: 0;
                    background: linear-gradient(135deg, #00d4ff 0%, #a855f7 50%, #00d4ff 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -0.02em;
                ">Docker Neural Memory</h1>
                <p style="
                    color: #94a3b8;
                    margin: 5px 0 0 0;
                    font-size: 1.1em;
                    font-weight: 300;
                ">Test-Time Training: Evolving LLMs from data hoarders to knowledge creators</p>
            </div>
        </div>

        <!-- Feature badges -->
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 20px;">
            <span style="
                background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
                border: 1px solid rgba(0, 212, 255, 0.3);
                color: #00d4ff;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 500;
            ">⚡ PyTorch TTT</span>
            <span style="
                background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(168, 85, 247, 0.1) 100%);
                border: 1px solid rgba(168, 85, 247, 0.3);
                color: #a855f7;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 500;
            ">🐳 Docker Native</span>
            <span style="
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
                border: 1px solid rgba(34, 197, 94, 0.3);
                color: #22c55e;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 500;
            ">🔌 MCP Server</span>
            <span style="
                background: linear-gradient(135deg, rgba(255, 140, 66, 0.2) 0%, rgba(255, 140, 66, 0.1) 100%);
                border: 1px solid rgba(255, 140, 66, 0.3);
                color: #ff8c42;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 500;
            ">📊 Titans Architecture</span>
        </div>
    </div>
</div>
'''

FOOTER_HTML = '''
<div style="
    font-family: 'Outfit', system-ui, sans-serif;
    background: linear-gradient(135deg, #0a0a1a 0%, #12122a 100%);
    padding: 30px;
    border-radius: 16px;
    margin-top: 30px;
    border: 1px solid rgba(148, 163, 184, 0.1);
">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <!-- Left side: Built by -->
        <div>
            <p style="color: #94a3b8; margin: 0 0 8px 0; font-size: 0.9em;">Built by</p>
            <p style="color: #f8fafc; margin: 0; font-size: 1.2em; font-weight: 600;">Carlos Crespo Macaya</p>
            <p style="color: #64748b; margin: 5px 0 0 0; font-size: 0.85em;">AI Engineer — GenAI Systems & Applied MLOps</p>
        </div>

        <!-- Right side: Social links -->
        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            <a href="https://github.com/macayaven/docker-neural-memory" target="_blank" style="
                display: flex; align-items: center; gap: 8px;
                background: rgba(255,255,255,0.05);
                padding: 10px 16px;
                border-radius: 8px;
                text-decoration: none;
                color: #f8fafc;
                font-size: 0.9em;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            " onmouseover="this.style.borderColor='#f8fafc'; this.style.background='rgba(255,255,255,0.1)';"
               onmouseout="this.style.borderColor='transparent'; this.style.background='rgba(255,255,255,0.05)';">
                <span style="font-size: 1.2em;">🐙</span> GitHub
            </a>
            <a href="https://www.linkedin.com/in/carlos-crespo-macaya/" target="_blank" style="
                display: flex; align-items: center; gap: 8px;
                background: rgba(255,255,255,0.05);
                padding: 10px 16px;
                border-radius: 8px;
                text-decoration: none;
                color: #f8fafc;
                font-size: 0.9em;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            " onmouseover="this.style.borderColor='#0077b5'; this.style.color='#0077b5';"
               onmouseout="this.style.borderColor='transparent'; this.style.color='#f8fafc';">
                <span style="font-size: 1.2em;">💼</span> LinkedIn
            </a>
            <a href="https://www.kaggle.com/macayaven" target="_blank" style="
                display: flex; align-items: center; gap: 8px;
                background: rgba(255,255,255,0.05);
                padding: 10px 16px;
                border-radius: 8px;
                text-decoration: none;
                color: #f8fafc;
                font-size: 0.9em;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            " onmouseover="this.style.borderColor='#20beff'; this.style.color='#20beff';"
               onmouseout="this.style.borderColor='transparent'; this.style.color='#f8fafc';">
                <span style="font-size: 1.2em;">📊</span> Kaggle <span style="background: linear-gradient(135deg, #ffd700, #ffb700); color: #000; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; font-weight: 600;">2×🥇</span>
            </a>
            <a href="https://huggingface.co/macayaven" target="_blank" style="
                display: flex; align-items: center; gap: 8px;
                background: rgba(255,255,255,0.05);
                padding: 10px 16px;
                border-radius: 8px;
                text-decoration: none;
                color: #f8fafc;
                font-size: 0.9em;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            " onmouseover="this.style.borderColor='#ff9d00'; this.style.color='#ff9d00';"
               onmouseout="this.style.borderColor='transparent'; this.style.color='#f8fafc';">
                <span style="font-size: 1.2em;">🤗</span> HuggingFace
            </a>
            <a href="https://scholar.google.com/citations?user=hwvDud0AAAAJ" target="_blank" style="
                display: flex; align-items: center; gap: 8px;
                background: rgba(255,255,255,0.05);
                padding: 10px 16px;
                border-radius: 8px;
                text-decoration: none;
                color: #f8fafc;
                font-size: 0.9em;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            " onmouseover="this.style.borderColor='#4285f4'; this.style.color='#4285f4';"
               onmouseout="this.style.borderColor='transparent'; this.style.color='#f8fafc';">
                <span style="font-size: 1.2em;">🎓</span> Scholar
            </a>
            <a href="https://carlos-crespo.com" target="_blank" style="
                display: flex; align-items: center; gap: 8px;
                background: rgba(255,255,255,0.05);
                padding: 10px 16px;
                border-radius: 8px;
                text-decoration: none;
                color: #f8fafc;
                font-size: 0.9em;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            " onmouseover="this.style.borderColor='#00d4ff'; this.style.color='#00d4ff';"
               onmouseout="this.style.borderColor='transparent'; this.style.color='#f8fafc';">
                <span style="font-size: 1.2em;">🌐</span> Website
            </a>
        </div>
    </div>

    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(148, 163, 184, 0.1); text-align: center;">
        <p style="color: #64748b; margin: 0; font-size: 0.85em;">
            Docker Neural Memory — Containerized AI memory with real test-time training
        </p>
    </div>
</div>
'''

# =============================================================================
# HUGGINGFACE INFERENCE CLIENT
# =============================================================================

# Use a model that is available on HF Serverless Inference free tier
# See: https://huggingface.co/models?inference_provider=hf-inference&pipeline_tag=text-generation
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceTB/SmolLM3-3B")
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Optional - works without for many models

try:
    hf_client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    LLM_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not initialize HF client: {e}")
    hf_client = None
    LLM_AVAILABLE = False

# Add src to path for real implementation
# When deployed to HF Spaces, src/ is copied to the same directory as app.py
sys.path.insert(0, str(Path(__file__).parent))

from src.config import MemoryConfig  # noqa: E402
from src.memory.neural_memory import NeuralMemory  # noqa: E402

# =============================================================================
# REAL NEURAL MEMORY INSTANCE
# =============================================================================

# Initialize the REAL neural memory - this is actual PyTorch, not a simulation
memory = NeuralMemory(MemoryConfig(dim=256, learning_rate=0.02))

# Track history for visualization
observation_history: List[Dict] = []

# =============================================================================
# COMPARISON METRICS & KNOWLEDGE BASE
# =============================================================================


@dataclass
class ComparisonMetrics:
    """Track comparison between vanilla and memory-augmented responses."""

    # With Neural Memory
    nm_queries: int = 0
    nm_correct: int = 0
    nm_hallucinations: int = 0
    nm_response_times: List[float] = field(default_factory=list)

    # Vanilla (no memory)
    vanilla_queries: int = 0
    vanilla_correct: int = 0
    vanilla_hallucinations: int = 0
    vanilla_response_times: List[float] = field(default_factory=list)


metrics = ComparisonMetrics()

# Knowledge base - facts the user teaches
knowledge_base: List[Dict[str, str]] = []

# Store embeddings for t-SNE visualization
embeddings_store: List[Dict] = []


def get_embedding(text: str) -> np.ndarray:
    """Get the neural memory's internal representation of text."""
    with torch.no_grad():
        # Convert text to tensor using memory's encoding
        tensor = memory._encode_text(text)
        # Pass through memory network to get learned representation
        output = memory.memory_net(tensor)
        # Flatten and ensure fixed size (pad or truncate to 256)
        flat = output.cpu().numpy().flatten()
        target_size = 256
        if len(flat) < target_size:
            # Pad with zeros
            flat = np.pad(flat, (0, target_size - len(flat)), mode='constant')
        elif len(flat) > target_size:
            # Truncate
            flat = flat[:target_size]
        return flat


def create_knowledge_base_visualization() -> plt.Figure:
    """Create visualization of the knowledge base (RAG store)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if not knowledge_base:
        ax.text(
            0.5, 0.5,
            "No facts in knowledge base yet.\nAdd facts to see them here.",
            ha="center", va="center", fontsize=14, color="gray"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Knowledge Base (RAG Store)", fontsize=14, fontweight="bold")
        return fig

    # Create a visual list of facts
    n_facts = len(knowledge_base)
    y_positions = np.linspace(0.9, 0.1, min(n_facts, 10))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.set_title(f"Knowledge Base (RAG Store) - {n_facts} Facts", fontsize=14, fontweight="bold")

    # Draw facts as cards
    for i, (y_pos, item) in enumerate(zip(y_positions, knowledge_base[-10:])):
        fact_text = item["fact"]
        if len(fact_text) > 60:
            fact_text = fact_text[:57] + "..."

        # Draw a rounded rectangle
        rect = plt.Rectangle((0.02, y_pos - 0.035), 0.96, 0.07,
                             facecolor="#e8f4f8", edgecolor="#3498db",
                             linewidth=2, alpha=0.8, zorder=1)
        ax.add_patch(rect)

        # Add fact number and text
        ax.text(0.05, y_pos, f"#{len(knowledge_base) - len(knowledge_base[-10:]) + i + 1}",
               fontsize=10, fontweight="bold", color="#2980b9", va="center")
        ax.text(0.12, y_pos, fact_text, fontsize=10, va="center", color="#2c3e50")

    if n_facts > 10:
        ax.text(0.5, 0.02, f"... and {n_facts - 10} more facts",
               ha="center", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    return fig


def create_neural_memory_state_visualization() -> plt.Figure:
    """Create visualization of the neural memory state."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Weight distribution histogram
    ax1 = axes[0]
    with torch.no_grad():
        all_weights = []
        for param in memory.memory_net.parameters():
            all_weights.extend(param.data.cpu().numpy().flatten())
        all_weights = np.array(all_weights)

    ax1.hist(all_weights, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax1.set_title("Weight Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    # 2. Weight heatmap (sample)
    ax2 = axes[1]
    weights = get_weight_sample()
    im = ax2.imshow(weights, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax2.set_title("Weight Matrix Sample (16x16)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, label="Value")

    # 3. Memory stats
    ax3 = axes[2]
    ax3.axis("off")
    stats = memory.get_stats()

    stats_text = f"""
    Neural Memory State
    ───────────────────
    Parameters: {stats['weight_parameters']:,}
    Dimension: {stats['dimension']}
    Learning Rate: {stats['learning_rate']:.4f}

    Observations: {stats['total_observations']}
    Avg Surprise: {stats['avg_surprise']:.4f}

    Weight Stats:
    • Mean: {np.mean(all_weights):.4f}
    • Std: {np.std(all_weights):.4f}
    • Min: {np.min(all_weights):.4f}
    • Max: {np.max(all_weights):.4f}
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
            va="center", transform=ax3.transAxes,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f0f0", "alpha": 0.8})
    ax3.set_title("Memory Statistics", fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def create_tsne_visualization() -> plt.Figure:
    """Create t-SNE visualization of learned representations."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if len(embeddings_store) < 2:
        ax.text(
            0.5, 0.5,
            "Add at least 2 facts to see the embedding space",
            ha="center", va="center", fontsize=14, color="gray"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    # Extract embeddings and labels
    embeddings = np.array([e["embedding"] for e in embeddings_store])
    labels = [e["label"][:30] + "..." if len(e["label"]) > 30 else e["label"]
              for e in embeddings_store]
    surprises = [e["surprise"] for e in embeddings_store]

    # Use PCA if few samples, t-SNE otherwise
    n_samples = len(embeddings)
    if n_samples < 5:
        # PCA for small sample sizes
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        method = "PCA"
    else:
        # t-SNE for larger sample sizes
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        method = "t-SNE"

    # Plot points
    scatter = ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=surprises, cmap="RdYlBu_r",
        s=150, alpha=0.7, edgecolors="white", linewidth=2
    )

    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(
            label, (reduced[i, 0], reduced[i, 1]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, alpha=0.8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7}
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Surprise (Red=Novel, Blue=Familiar)", fontsize=10)

    ax.set_title(f"Neural Memory Embedding Space ({method})\n"
                 f"{n_samples} observations - Similar concepts cluster together",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_embedding_comparison() -> plt.Figure:
    """Create side-by-side: weight heatmap + embedding space."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Weight heatmap
    ax1 = axes[0]
    weights = get_weight_sample()
    im = ax1.imshow(weights, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax1.set_title("Neural Network Weights\n(These update during learning)",
                  fontsize=11, fontweight="bold")
    ax1.axis("off")
    plt.colorbar(im, ax=ax1, label="Weight Value")

    # Right: Embedding space (simplified if few points)
    ax2 = axes[1]
    if len(embeddings_store) < 2:
        ax2.text(0.5, 0.5, "Add facts to see\nembedding space",
                ha="center", va="center", fontsize=12, color="gray")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    else:
        embeddings = np.array([e["embedding"] for e in embeddings_store])
        surprises = [e["surprise"] for e in embeddings_store]

        n_samples = len(embeddings)
        if n_samples < 5:
            reducer = PCA(n_components=2)
        else:
            perplexity = min(30, n_samples - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)

        reduced = reducer.fit_transform(embeddings)

        scatter = ax2.scatter(reduced[:, 0], reduced[:, 1], c=surprises,
                             cmap="RdYlBu_r", s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax2, label="Surprise")
        ax2.grid(True, alpha=0.3)

    ax2.set_title("Learned Representations\n(Similar facts cluster together)",
                  fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def call_llm(prompt: str, context: str = "") -> Tuple[str, float]:
    """Call HuggingFace LLM. Returns (response, time)."""
    if not LLM_AVAILABLE or hf_client is None:
        return "[LLM not available - set HF_TOKEN for comparison demo]", 0.0

    try:
        # Build messages for chat completion
        if context:
            system_msg = f"""You have access to the following knowledge:

{context}

Based ONLY on the knowledge above, answer questions. If the information is not in the knowledge provided, say "I don't have information about that."
"""
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        start = time.time()
        response = hf_client.chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        elapsed = time.time() - start

        # Extract the response content
        answer = response.choices[0].message.content
        return answer.strip() if answer else "", elapsed
    except Exception as e:
        return f"Error: {e!s}", 0.0


def add_to_knowledge_base(fact: str) -> Tuple[str, plt.Figure, plt.Figure, plt.Figure]:
    """Add a fact to the knowledge base and observe it in neural memory."""
    if not fact.strip():
        return (
            "Please enter a fact to add.",
            create_tsne_visualization(),
            create_neural_memory_state_visualization(),
            create_knowledge_base_visualization(),
        )

    # Add to knowledge base
    knowledge_base.append({"fact": fact, "timestamp": time.time()})

    # Observe in neural memory
    result = memory.observe(fact)

    # Store embedding for visualization
    embedding = get_embedding(fact)
    embeddings_store.append({
        "label": fact,
        "embedding": embedding,
        "surprise": result["surprise"],
        "timestamp": time.time(),
    })

    output = f"""### Fact Added

**Fact:** "{fact}"

**Neural Memory Response:**
| Metric | Value |
|--------|-------|
| Surprise | {result['surprise']:.4f} |
| Weight Delta | {result['weight_delta']:.6f} |
| Learned | {'Yes' if result['learned'] else 'No'} |

**Knowledge Base Size:** {len(knowledge_base)} facts
**Embeddings Stored:** {len(embeddings_store)}
"""

    return (
        output,
        create_tsne_visualization(),
        create_neural_memory_state_visualization(),
        create_knowledge_base_visualization(),
    )


def get_knowledge_context() -> str:
    """Get all facts as context string."""
    if not knowledge_base:
        return ""
    return "\n".join([f"- {item['fact']}" for item in knowledge_base])


def call_rag_llm(question: str, knowledge_base: List[Dict]) -> Tuple[str, float, List[str]]:
    """Simulate RAG: retrieve most similar facts by keyword matching."""
    if not LLM_AVAILABLE or hf_client is None:
        return "[LLM not available]", 0.0, []

    # Simple RAG simulation: keyword-based retrieval (top 2 most relevant)
    question_words = set(question.lower().split())
    scored_facts = []
    for item in knowledge_base:
        fact = item["fact"]
        fact_words = set(fact.lower().split())
        # Simple overlap score
        overlap = len(question_words & fact_words)
        scored_facts.append((overlap, fact))

    # Get top 2 most relevant facts
    scored_facts.sort(reverse=True, key=lambda x: x[0])
    retrieved = [f for score, f in scored_facts[:2] if score > 0]

    if retrieved:
        context = "Retrieved facts:\n" + "\n".join([f"- {f}" for f in retrieved])
        system_msg = f"""You are a RAG system. You can ONLY use the retrieved facts below to answer.
If the retrieved facts don't directly answer the question, say "The retrieved information doesn't cover this."

{context}
"""
    else:
        system_msg = "You are a RAG system with no relevant documents retrieved. Say 'No relevant documents found.'"
        retrieved = ["(none retrieved)"]

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]

    try:
        start = time.time()
        response = hf_client.chat_completion(messages=messages, max_tokens=150, temperature=0.7)
        elapsed = time.time() - start
        answer = response.choices[0].message.content
        return answer.strip() if answer else "", elapsed, retrieved
    except Exception as e:
        return f"Error: {e!s}", 0.0, retrieved


def call_neural_memory_llm(question: str, knowledge_base: List[Dict], surprise: float) -> Tuple[str, float]:
    """Neural Memory augmented LLM: uses ALL facts + learned patterns."""
    if not LLM_AVAILABLE or hf_client is None:
        return "[LLM not available]", 0.0

    # Neural memory provides ALL context + pattern awareness
    all_facts = "\n".join([f"- {item['fact']}" for item in knowledge_base])

    # Analyze patterns in the facts
    patterns_hint = ""
    if knowledge_base:
        # Look for approval/rejection patterns
        approvals = [f["fact"] for f in knowledge_base if "approved" in f["fact"].lower() or "liked" in f["fact"].lower()]
        rejections = [f["fact"] for f in knowledge_base if "rejected" in f["fact"].lower() or "disliked" in f["fact"].lower()]
        if approvals or rejections:
            patterns_hint = "\n\nLearned patterns from observations:"
            if approvals:
                patterns_hint += f"\n- Positive signals: {len(approvals)} approvals/likes"
            if rejections:
                patterns_hint += f"\n- Negative signals: {len(rejections)} rejections/dislikes"
            patterns_hint += "\n- Look for common themes in approved vs rejected items"

    system_msg = f"""You are an AI with neural memory that has LEARNED from all observations below.
Unlike simple retrieval, you should:
1. Consider ALL facts holistically
2. Identify PATTERNS across multiple observations
3. Make INFERENCES based on learned patterns
4. Predict based on trends, not just direct matches

Observations (learned knowledge):
{all_facts}
{patterns_hint}

Question novelty (surprise score): {surprise:.2f}
- Low surprise (<0.3): This topic is familiar from your observations
- High surprise (>0.6): This is a novel topic, be cautious
"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]

    try:
        start = time.time()
        response = hf_client.chat_completion(messages=messages, max_tokens=200, temperature=0.7)
        elapsed = time.time() - start
        answer = response.choices[0].message.content
        return answer.strip() if answer else "", elapsed
    except Exception as e:
        return f"Error: {e!s}", 0.0


def compare_responses(question: str) -> Tuple[str, str, str, plt.Figure, plt.Figure]:
    """Compare RAG vs Neural Memory augmented LLM on the same question."""
    global metrics

    if not question.strip():
        return "", "", "", create_neural_memory_state_visualization(), create_knowledge_base_visualization()

    if not LLM_AVAILABLE:
        return (
            "LLM not available. Please set HF_TOKEN environment variable.",
            "LLM not available.",
            "Comparison requires LLM access.",
            create_neural_memory_state_visualization(),
            create_knowledge_base_visualization(),
        )

    # Check surprise (is this question familiar?)
    surprise = memory.surprise(question)

    # Query with NEURAL MEMORY (pattern learning, all context)
    nm_response, nm_time = call_neural_memory_llm(question, knowledge_base, surprise)
    metrics.nm_queries += 1
    metrics.nm_response_times.append(nm_time)

    # Query with RAG (simple retrieval)
    rag_response, rag_time, retrieved_facts = call_rag_llm(question, knowledge_base)
    metrics.vanilla_queries += 1
    metrics.vanilla_response_times.append(rag_time)

    # Simple quality detection
    rag_failed = any(
        phrase in rag_response.lower()
        for phrase in ["doesn't cover", "no relevant", "don't have", "cannot answer"]
    )
    nm_confident = not any(
        phrase in nm_response.lower()
        for phrase in ["i don't know", "i don't have", "cannot"]
    )

    if rag_failed:
        metrics.vanilla_hallucinations += 1
    if nm_confident and knowledge_base:
        metrics.nm_correct += 1

    # Format outputs - Neural Memory
    nm_output = f"""### Neural Memory (Pattern Learning)

**Question:** {question}

**Response:**
> {nm_response}

---
**How it works:**
- Uses **ALL {len(knowledge_base)} facts** holistically
- **Learns patterns** (e.g., approval vs rejection trends)
- **Surprise Score: {surprise:.3f}** - {'familiar topic' if surprise < 0.4 else 'novel topic'}
- Response Time: {nm_time:.2f}s
"""

    # Format outputs - RAG
    retrieved_str = "\n".join([f"  - {f}" for f in retrieved_facts])
    rag_output = f"""### RAG (Retrieval Only)

**Question:** {question}

**Response:**
> {rag_response}

---
**How it works:**
- Retrieved **{len([f for f in retrieved_facts if f != '(none retrieved)'])} facts** by keyword match:
{retrieved_str}
- **No pattern learning** - just similarity search
- Response Time: {rag_time:.2f}s
"""

    # Comparison summary
    comparison = get_comparison_summary()

    return (
        nm_output,
        rag_output,
        comparison,
        create_neural_memory_state_visualization(),
        create_knowledge_base_visualization(),
    )


def get_comparison_summary() -> str:
    """Generate comparison metrics summary."""
    nm_avg_time = (
        sum(metrics.nm_response_times) / len(metrics.nm_response_times)
        if metrics.nm_response_times
        else 0
    )
    rag_avg_time = (
        sum(metrics.vanilla_response_times) / len(metrics.vanilla_response_times)
        if metrics.vanilla_response_times
        else 0
    )

    nm_accuracy = (
        metrics.nm_correct / metrics.nm_queries * 100 if metrics.nm_queries else 0
    )
    rag_fail_rate = (
        metrics.vanilla_hallucinations / metrics.vanilla_queries * 100
        if metrics.vanilla_queries
        else 0
    )

    return f"""## Neural Memory vs RAG Comparison

| Metric | Neural Memory | RAG |
|--------|---------------|-----|
| **Queries** | {metrics.nm_queries} | {metrics.vanilla_queries} |
| **Pattern-Based Answers** | {metrics.nm_correct} ({nm_accuracy:.0f}%) | N/A |
| **Retrieval Failures** | N/A | {metrics.vanilla_hallucinations} ({rag_fail_rate:.0f}%) |
| **Avg Response Time** | {nm_avg_time:.2f}s | {rag_avg_time:.2f}s |

### Knowledge Base: {len(knowledge_base)} facts stored

### Why Neural Memory Wins

| Capability | Neural Memory | RAG |
|------------|---------------|-----|
| **Pattern Learning** | Learns trends across all data | No learning |
| **Inference** | Can predict from patterns | Only retrieves matches |
| **Context Usage** | Uses ALL facts holistically | Uses top-k retrieved |
| **Novelty Detection** | Surprise score | None |
| **Memory Size** | Fixed (neural weights) | Grows with data |

### Key Insight
Neural memory **learns patterns** (e.g., "Carlos rejects bright colors, approves dark themes")
and can **infer preferences** for novel items. RAG just retrieves similar documents.
"""


def reset_comparison() -> Tuple[str, plt.Figure, plt.Figure, plt.Figure]:
    """Reset comparison metrics and knowledge base."""
    global metrics, knowledge_base, embeddings_store
    metrics = ComparisonMetrics()
    knowledge_base = []
    embeddings_store = []
    return (
        "Comparison reset. Knowledge base and embeddings cleared.",
        create_tsne_visualization(),
        create_neural_memory_state_visualization(),
        create_knowledge_base_visualization(),
    )


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
# KEY CONCEPTS (New Educational Tab)
# =============================================================================

KEY_CONCEPTS_HTML = '''
<div style="font-family: 'Outfit', system-ui, sans-serif; padding: 20px; color: #f8fafc;">
    <!-- The Problem -->
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid rgba(252, 129, 129, 0.3);">
        <h3 style="color: #fc8181; margin: 0 0 20px 0; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;">❌</span> The Problem: LLMs Have No Memory
        </h3>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: rgba(0,0,0,0.3); border-radius: 12px; padding: 20px;">
                <p style="color: #a0aec0; margin: 0 0 15px 0; font-size: 0.95em;">Every API call to an LLM starts <strong style="color: #fc8181;">fresh</strong>:</p>
                <div style="background: #0a0a1a; border-radius: 8px; padding: 15px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em;">
                    <div style="color: #64748b;">// Call 1</div>
                    <div style="color: #f8fafc;">User: "My name is Carlos"</div>
                    <div style="color: #22c55e;">AI: "Nice to meet you, Carlos!"</div>
                    <br/>
                    <div style="color: #64748b;">// Call 2 (new session)</div>
                    <div style="color: #f8fafc;">User: "What's my name?"</div>
                    <div style="color: #fc8181;">AI: "I don't know your name."</div>
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: rgba(0,0,0,0.3); border-radius: 12px; padding: 20px;">
                <p style="color: #a0aec0; margin: 0 0 15px 0; font-size: 0.95em;">The model's weights are <strong style="color: #fc8181;">frozen</strong> after training:</p>
                <ul style="color: #a0aec0; margin: 0; padding-left: 20px; line-height: 1.8;">
                    <li>Can't learn new information</li>
                    <li>Can't remember past conversations</li>
                    <li>Can't adapt to user preferences</li>
                    <li>Knowledge is static (training cutoff)</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Two Solutions -->
    <h3 style="color: #f8fafc; margin: 30px 0 20px 0; text-align: center; font-size: 1.3em;">
        Two Solutions to Add Memory
    </h3>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px;">
        <!-- RAG Solution -->
        <div style="background: linear-gradient(135deg, rgba(252, 129, 129, 0.1) 0%, rgba(237, 137, 54, 0.1) 100%); border: 2px solid #fc8181; border-radius: 16px; padding: 25px;">
            <h4 style="color: #fc8181; margin: 0 0 15px 0; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.3em;">📚</span> Solution A: RAG (Retrieval)
            </h4>
            <p style="color: #a0aec0; font-size: 0.9em; margin: 0 0 15px 0;">
                <strong>Store</strong> information externally, <strong>retrieve</strong> relevant pieces when needed.
            </p>
            <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="background: #fc8181; color: #1a1a2e; padding: 4px 10px; border-radius: 4px; font-size: 0.8em; font-weight: 600;">HOW</span>
                </div>
                <ol style="color: #a0aec0; margin: 0; padding-left: 20px; font-size: 0.9em; line-height: 1.7;">
                    <li>Convert facts to vectors (embeddings)</li>
                    <li>Store in vector database</li>
                    <li>On query, find similar vectors</li>
                    <li>Pass retrieved docs to LLM</li>
                </ol>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                <span style="background: rgba(252, 129, 129, 0.2); color: #fc8181; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✓ Simple</span>
                <span style="background: rgba(252, 129, 129, 0.2); color: #fc8181; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✓ Scalable</span>
                <span style="background: rgba(100, 116, 139, 0.3); color: #94a3b8; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✗ No patterns</span>
                <span style="background: rgba(100, 116, 139, 0.3); color: #94a3b8; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✗ Grows</span>
            </div>
        </div>

        <!-- Neural Memory Solution -->
        <div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%); border: 2px solid #00d4ff; border-radius: 16px; padding: 25px;">
            <h4 style="color: #00d4ff; margin: 0 0 15px 0; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.3em;">🧠</span> Solution B: Neural Memory (Learning)
            </h4>
            <p style="color: #a0aec0; font-size: 0.9em; margin: 0 0 15px 0;">
                <strong>Learn</strong> information into neural weights. Memory IS the network.
            </p>
            <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="background: #00d4ff; color: #1a1a2e; padding: 4px 10px; border-radius: 4px; font-size: 0.8em; font-weight: 600;">HOW</span>
                </div>
                <ol style="color: #a0aec0; margin: 0; padding-left: 20px; font-size: 0.9em; line-height: 1.7;">
                    <li>Encode fact as tensor</li>
                    <li>Forward pass through neural net</li>
                    <li>Compute prediction error (surprise)</li>
                    <li><strong style="color: #00d4ff;">Update weights</strong> via gradient descent</li>
                </ol>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                <span style="background: rgba(0, 212, 255, 0.2); color: #00d4ff; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✓ Learns patterns</span>
                <span style="background: rgba(0, 212, 255, 0.2); color: #00d4ff; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✓ Fixed size</span>
                <span style="background: rgba(0, 212, 255, 0.2); color: #00d4ff; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✓ Can infer</span>
                <span style="background: rgba(100, 116, 139, 0.3); color: #94a3b8; padding: 5px 12px; border-radius: 6px; font-size: 0.8em;">✗ Complex</span>
            </div>
        </div>
    </div>

    <!-- Test-Time Training Innovation -->
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; padding: 25px; margin-top: 25px; border: 1px solid rgba(0, 212, 255, 0.3);">
        <h3 style="color: #00d4ff; margin: 0 0 20px 0; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;">⚡</span> The Innovation: Test-Time Training (TTT)
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;">
            <div style="background: rgba(0,0,0,0.3); border-radius: 12px; padding: 20px;">
                <h5 style="color: #a855f7; margin: 0 0 10px 0;">Traditional Training</h5>
                <p style="color: #a0aec0; font-size: 0.9em; margin: 0; line-height: 1.6;">
                    Train once → Freeze weights → Deploy<br/>
                    <span style="color: #64748b;">Model can't learn after deployment</span>
                </p>
            </div>
            <div style="background: rgba(0, 212, 255, 0.1); border-radius: 12px; padding: 20px; border: 1px solid rgba(0, 212, 255, 0.2);">
                <h5 style="color: #00d4ff; margin: 0 0 10px 0;">Test-Time Training (Titans)</h5>
                <p style="color: #a0aec0; font-size: 0.9em; margin: 0; line-height: 1.6;">
                    Deploy → <strong style="color: #00d4ff;">Continue learning</strong> → Weights update<br/>
                    <span style="color: #22c55e;">Model learns from every interaction</span>
                </p>
            </div>
        </div>
        <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 10px;">
            <p style="color: #a0aec0; margin: 0; font-size: 0.9em;">
                <strong style="color: #f8fafc;">This demo implements real TTT:</strong> When you add a fact, actual PyTorch gradients flow and actual neural network weights change. This is not a simulation—it's the Titans architecture from Google's December 2024 paper.
            </p>
        </div>
    </div>
</div>
'''

# =============================================================================
# INCREMENTAL INTEGRATION DIAGRAMS
# =============================================================================

VANILLA_LLM_DIAGRAM_HTML = '''
<div style="font-family: 'Outfit', system-ui, sans-serif; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; color: #fff; margin-bottom: 20px; border: 1px solid rgba(148, 163, 184, 0.2);">
    <h4 style="color: #94a3b8; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
        <span style="background: #374151; color: #f8fafc; padding: 4px 12px; border-radius: 6px; font-size: 0.8em;">Step 1</span>
        Vanilla LLM (The Problem)
    </h4>
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; flex-wrap: wrap;">
        <div style="background: #2d3748; padding: 20px 30px; border-radius: 12px; text-align: center;">
            <div style="font-size: 32px; margin-bottom: 10px;">👤</div>
            <div style="color: #f8fafc; font-weight: 500;">User Query</div>
            <div style="color: #64748b; font-size: 0.85em;">"What's my preference?"</div>
        </div>
        <div style="color: #64748b; font-size: 32px;">→</div>
        <div style="background: linear-gradient(135deg, #805ad5 0%, #553c9a 100%); padding: 20px 30px; border-radius: 12px; text-align: center; border: 2px solid #d6bcfa;">
            <div style="font-size: 32px; margin-bottom: 10px;">🤖</div>
            <div style="color: #f8fafc; font-weight: 600;">LLM</div>
            <div style="color: #e9d8fd; font-size: 0.85em;">Frozen weights</div>
        </div>
        <div style="color: #64748b; font-size: 32px;">→</div>
        <div style="background: rgba(252, 129, 129, 0.2); padding: 20px 30px; border-radius: 12px; text-align: center; border: 2px solid #fc8181;">
            <div style="font-size: 32px; margin-bottom: 10px;">❓</div>
            <div style="color: #fc8181; font-weight: 500;">No Memory</div>
            <div style="color: #a0aec0; font-size: 0.85em;">"I don't know"</div>
        </div>
    </div>
    <div style="margin-top: 15px; padding: 12px; background: rgba(252, 129, 129, 0.1); border-radius: 8px; text-align: center;">
        <span style="color: #fc8181; font-size: 0.9em;">⚠️ LLM has no way to remember user-specific information between sessions</span>
    </div>
</div>
'''

RAG_INTEGRATION_DIAGRAM_HTML = '''
<div style="font-family: 'Outfit', system-ui, sans-serif; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; color: #fff; margin-bottom: 20px; border: 1px solid rgba(255, 140, 66, 0.3);">
    <h4 style="color: #ff8c42; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
        <span style="background: #ff8c42; color: #1a1a2e; padding: 4px 12px; border-radius: 6px; font-size: 0.8em;">Step 2a</span>
        Adding RAG (Retrieval-Augmented Generation)
    </h4>
    <div style="display: flex; align-items: center; justify-content: center; gap: 15px; flex-wrap: wrap;">
        <div style="background: #2d3748; padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">👤</div>
            <div style="color: #f8fafc; font-size: 0.9em;">Query</div>
        </div>
        <div style="color: #ff8c42; font-size: 24px;">→</div>
        <div style="background: rgba(255, 140, 66, 0.2); padding: 15px 20px; border-radius: 10px; text-align: center; border: 1px dashed #ff8c42;">
            <div style="font-size: 24px;">🔍</div>
            <div style="color: #ff8c42; font-size: 0.9em;">Retriever</div>
            <div style="color: #64748b; font-size: 0.75em;">keyword match</div>
        </div>
        <div style="color: #ff8c42; font-size: 24px;">→</div>
        <div style="background: #744210; padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">🗃️</div>
            <div style="color: #faf089; font-size: 0.9em;">Vector DB</div>
            <div style="color: #64748b; font-size: 0.75em;">top-k docs</div>
        </div>
        <div style="color: #ff8c42; font-size: 24px;">→</div>
        <div style="background: #3182ce; padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">📋</div>
            <div style="color: #bee3f8; font-size: 0.9em;">Context</div>
            <div style="color: #64748b; font-size: 0.75em;">prompt injection</div>
        </div>
        <div style="color: #ff8c42; font-size: 24px;">→</div>
        <div style="background: linear-gradient(135deg, #805ad5 0%, #553c9a 100%); padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">🤖</div>
            <div style="color: #f8fafc; font-size: 0.9em;">LLM</div>
        </div>
    </div>
    <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        <div style="padding: 10px; background: rgba(34, 197, 94, 0.1); border-radius: 6px;">
            <span style="color: #22c55e; font-size: 0.85em;">✓ External memory storage</span>
        </div>
        <div style="padding: 10px; background: rgba(252, 129, 129, 0.1); border-radius: 6px;">
            <span style="color: #fc8181; font-size: 0.85em;">✗ No pattern learning</span>
        </div>
    </div>
</div>
'''

NEURAL_MEMORY_INTEGRATION_DIAGRAM_HTML = '''
<div style="font-family: 'Outfit', system-ui, sans-serif; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; color: #fff; margin-bottom: 20px; border: 1px solid rgba(0, 212, 255, 0.3);">
    <h4 style="color: #00d4ff; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
        <span style="background: #00d4ff; color: #1a1a2e; padding: 4px 12px; border-radius: 6px; font-size: 0.8em;">Step 2b</span>
        Adding Neural Memory (Test-Time Training)
    </h4>
    <div style="display: flex; align-items: center; justify-content: center; gap: 15px; flex-wrap: wrap;">
        <div style="background: #2d3748; padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">👤</div>
            <div style="color: #f8fafc; font-size: 0.9em;">Query</div>
        </div>
        <div style="color: #00d4ff; font-size: 24px;">→</div>
        <div style="background: rgba(0, 212, 255, 0.2); padding: 15px 20px; border-radius: 10px; text-align: center; border: 2px solid #00d4ff;">
            <div style="font-size: 24px;">🧠</div>
            <div style="color: #00d4ff; font-size: 0.9em; font-weight: 600;">Neural Memory</div>
            <div style="color: #64748b; font-size: 0.75em;">TTT Module</div>
        </div>
        <div style="color: #00d4ff; font-size: 24px;">→</div>
        <div style="background: #2f855a; padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">📊</div>
            <div style="color: #9ae6b4; font-size: 0.9em;">Patterns</div>
            <div style="color: #64748b; font-size: 0.75em;">+ surprise</div>
        </div>
        <div style="color: #00d4ff; font-size: 24px;">→</div>
        <div style="background: #3182ce; padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">📋</div>
            <div style="color: #bee3f8; font-size: 0.9em;">Rich Context</div>
            <div style="color: #64748b; font-size: 0.75em;">all facts + hints</div>
        </div>
        <div style="color: #00d4ff; font-size: 24px;">→</div>
        <div style="background: linear-gradient(135deg, #805ad5 0%, #553c9a 100%); padding: 15px 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 24px;">🤖</div>
            <div style="color: #f8fafc; font-size: 0.9em;">LLM</div>
        </div>
    </div>
    <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
        <div style="padding: 10px; background: rgba(34, 197, 94, 0.1); border-radius: 6px;">
            <span style="color: #22c55e; font-size: 0.85em;">✓ Learns patterns</span>
        </div>
        <div style="padding: 10px; background: rgba(34, 197, 94, 0.1); border-radius: 6px;">
            <span style="color: #22c55e; font-size: 0.85em;">✓ Fixed memory size</span>
        </div>
        <div style="padding: 10px; background: rgba(34, 197, 94, 0.1); border-radius: 6px;">
            <span style="color: #22c55e; font-size: 0.85em;">✓ Can infer/predict</span>
        </div>
    </div>
</div>
'''

DOCKER_DEPLOYMENT_DIAGRAM_HTML = '''
<div style="font-family: 'Outfit', system-ui, sans-serif; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; color: #fff; border: 1px solid rgba(168, 85, 247, 0.3);">
    <h4 style="color: #a855f7; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
        <span style="background: #a855f7; color: #1a1a2e; padding: 4px 12px; border-radius: 6px; font-size: 0.8em;">Step 3</span>
        Docker Deployment (Production Ready)
    </h4>
    <div style="display: flex; align-items: stretch; justify-content: center; gap: 20px; flex-wrap: wrap;">
        <!-- Docker Container -->
        <div style="background: rgba(168, 85, 247, 0.1); border: 2px solid #a855f7; border-radius: 12px; padding: 20px; min-width: 280px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <span style="font-size: 1.5em;">🐳</span>
                <span style="color: #a855f7; font-weight: 600;">Docker Container</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="background: rgba(0, 212, 255, 0.2); padding: 10px; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.3);">
                    <div style="color: #00d4ff; font-size: 0.85em; font-weight: 500;">🧠 Neural Memory</div>
                    <div style="color: #64748b; font-size: 0.75em;">PyTorch TTT Module</div>
                </div>
                <div style="background: rgba(34, 197, 94, 0.2); padding: 10px; border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.3);">
                    <div style="color: #22c55e; font-size: 0.85em; font-weight: 500;">🔌 MCP Server</div>
                    <div style="color: #64748b; font-size: 0.75em;">Claude Desktop Integration</div>
                </div>
                <div style="background: rgba(255, 140, 66, 0.2); padding: 10px; border-radius: 8px; border: 1px solid rgba(255, 140, 66, 0.3);">
                    <div style="color: #ff8c42; font-size: 0.85em; font-weight: 500;">🌐 HTTP API</div>
                    <div style="color: #64748b; font-size: 0.75em;">REST Endpoints</div>
                </div>
            </div>
        </div>
        <!-- Volume -->
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 10px;">
            <div style="color: #64748b; font-size: 24px;">↔</div>
            <div style="background: #374151; padding: 15px 20px; border-radius: 10px; text-align: center;">
                <div style="font-size: 24px;">💾</div>
                <div style="color: #f8fafc; font-size: 0.9em;">Volume</div>
                <div style="color: #64748b; font-size: 0.75em;">Checkpoints</div>
            </div>
        </div>
    </div>
    <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 10px;">
        <div style="color: #a0aec0; font-size: 0.9em;">
            <strong style="color: #a855f7;">Why Docker?</strong> Learned neural weights persist across container restarts via Docker volumes. Deploy anywhere with identical behavior. Version control your AI's memory state like Git commits.
        </div>
    </div>
</div>
'''

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

# =============================================================================
# ARCHITECTURE DIAGRAMS (How It Works)
# =============================================================================

ARCHITECTURE_INTRO_MD = """
## How It Works: Neural Memory vs RAG Architecture

This section provides a detailed look at how both systems process information and connect to the LLM.
The diagrams below are **faithful representations of our actual implementation**.

---
"""

NEURAL_MEMORY_DIAGRAM_HTML = """
<div style="font-family: system-ui, -apple-system, sans-serif; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; color: #fff;">
    <h3 style="text-align: center; color: #00d4ff; margin-bottom: 30px; font-size: 1.5em;">
        Neural Memory Pipeline (Test-Time Training)
    </h3>

    <!-- Main Flow -->
    <div style="display: flex; flex-direction: column; gap: 20px; max-width: 900px; margin: 0 auto;">

        <!-- Phase 1: Learning Phase -->
        <div style="background: rgba(0, 212, 255, 0.1); border: 2px solid #00d4ff; border-radius: 12px; padding: 20px;">
            <h4 style="color: #00d4ff; margin: 0 0 15px 0;">Phase 1: Learning (When Facts Are Added)</h4>

            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                <!-- Input -->
                <div style="background: #2d3748; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 120px;">
                    <div style="font-size: 24px;">📝</div>
                    <div style="font-weight: bold; color: #fff;">User Fact</div>
                    <div style="font-size: 11px; color: #a0aec0;">"Carlos rejected<br/>bright colors"</div>
                </div>

                <div style="color: #00d4ff; font-size: 24px;">→</div>

                <!-- Encode -->
                <div style="background: #553c9a; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 140px;">
                    <div style="font-size: 24px;">🔢</div>
                    <div style="font-weight: bold; color: #fff;">_encode_text()</div>
                    <div style="font-size: 11px; color: #d6bcfa;">Tensor [1, 64, 256]</div>
                </div>

                <div style="color: #00d4ff; font-size: 24px;">→</div>

                <!-- Forward Pass -->
                <div style="background: #2f855a; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 140px;">
                    <div style="font-size: 24px;">🧠</div>
                    <div style="font-weight: bold; color: #fff;">memory_net(x)</div>
                    <div style="font-size: 11px; color: #9ae6b4;">2-layer MLP<br/>~250K params</div>
                </div>

                <div style="color: #00d4ff; font-size: 24px;">→</div>

                <!-- Compute Loss -->
                <div style="background: #c53030; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 140px;">
                    <div style="font-size: 24px;">📊</div>
                    <div style="font-weight: bold; color: #fff;">MSE Loss</div>
                    <div style="font-size: 11px; color: #feb2b2;">Surprise Score<br/>= Prediction Error</div>
                </div>

                <div style="color: #00d4ff; font-size: 24px;">→</div>

                <!-- Gradient Descent -->
                <div style="background: #d69e2e; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 160px; border: 3px solid #faf089;">
                    <div style="font-size: 24px;">⚡</div>
                    <div style="font-weight: bold; color: #1a202c;">WEIGHT UPDATE</div>
                    <div style="font-size: 11px; color: #744210;">torch.autograd.grad()<br/>param -= lr × grad</div>
                </div>
            </div>

            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; font-size: 12px; color: #a0aec0;">
                <strong style="color: #00d4ff;">Key Point:</strong> The neural network's weights physically change after each fact.
                This is real gradient descent happening at inference time (Test-Time Training / Titans architecture).
            </div>
        </div>

        <!-- Phase 2: Query Phase -->
        <div style="background: rgba(72, 187, 120, 0.1); border: 2px solid #48bb78; border-radius: 12px; padding: 20px;">
            <h4 style="color: #48bb78; margin: 0 0 15px 0;">Phase 2: Query (When Questions Are Asked)</h4>

            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                <!-- Question -->
                <div style="background: #2d3748; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 120px;">
                    <div style="font-size: 24px;">❓</div>
                    <div style="font-weight: bold; color: #fff;">Question</div>
                    <div style="font-size: 11px; color: #a0aec0;">"Will Carlos<br/>like neon?"</div>
                </div>

                <div style="color: #48bb78; font-size: 24px;">→</div>

                <!-- Surprise Check -->
                <div style="background: #553c9a; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 130px;">
                    <div style="font-size: 24px;">🎯</div>
                    <div style="font-weight: bold; color: #fff;">surprise()</div>
                    <div style="font-size: 11px; color: #d6bcfa;">Novelty Score<br/>(No Learning)</div>
                </div>

                <div style="color: #48bb78; font-size: 24px;">→</div>

                <!-- Context Builder -->
                <div style="background: #2f855a; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 160px;">
                    <div style="font-size: 24px;">📦</div>
                    <div style="font-weight: bold; color: #fff;">Build Context</div>
                    <div style="font-size: 11px; color: #9ae6b4;"><strong>ALL facts</strong><br/>+ Pattern hints<br/>+ Surprise score</div>
                </div>

                <div style="color: #48bb78; font-size: 24px;">→</div>

                <!-- System Prompt -->
                <div style="background: #3182ce; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 180px;">
                    <div style="font-size: 24px;">💬</div>
                    <div style="font-weight: bold; color: #fff;">System Prompt</div>
                    <div style="font-size: 10px; color: #bee3f8; text-align: left; margin-top: 5px;">
                        "You have LEARNED from:<br/>
                        • All 4 observations<br/>
                        • Identify PATTERNS<br/>
                        • Make INFERENCES"
                    </div>
                </div>

                <div style="color: #48bb78; font-size: 24px;">→</div>

                <!-- LLM -->
                <div style="background: linear-gradient(135deg, #805ad5 0%, #553c9a 100%); padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 140px; border: 3px solid #d6bcfa;">
                    <div style="font-size: 24px;">🤖</div>
                    <div style="font-weight: bold; color: #fff;">LLM</div>
                    <div style="font-size: 11px; color: #e9d8fd;">SmolLM3-3B<br/>(HuggingFace)</div>
                </div>
            </div>

            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; font-size: 12px; color: #a0aec0;">
                <strong style="color: #48bb78;">Key Point:</strong> The LLM receives ALL facts + learned pattern hints + novelty indicator.
                It's instructed to identify patterns and make inferences, not just retrieve.
            </div>
        </div>

    </div>
</div>
"""

RAG_DIAGRAM_HTML = """
<div style="font-family: system-ui, -apple-system, sans-serif; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; color: #fff; margin-top: 20px;">
    <h3 style="text-align: center; color: #fc8181; margin-bottom: 30px; font-size: 1.5em;">
        RAG Pipeline (Retrieval-Augmented Generation)
    </h3>

    <!-- Main Flow -->
    <div style="display: flex; flex-direction: column; gap: 20px; max-width: 900px; margin: 0 auto;">

        <!-- Phase 1: Storage Phase -->
        <div style="background: rgba(252, 129, 129, 0.1); border: 2px solid #fc8181; border-radius: 12px; padding: 20px;">
            <h4 style="color: #fc8181; margin: 0 0 15px 0;">Phase 1: Storage (When Facts Are Added)</h4>

            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                <!-- Input -->
                <div style="background: #2d3748; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 120px;">
                    <div style="font-size: 24px;">📝</div>
                    <div style="font-weight: bold; color: #fff;">User Fact</div>
                    <div style="font-size: 11px; color: #a0aec0;">"Carlos rejected<br/>bright colors"</div>
                </div>

                <div style="color: #fc8181; font-size: 24px;">→</div>

                <!-- Append to List -->
                <div style="background: #744210; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 180px;">
                    <div style="font-size: 24px;">📋</div>
                    <div style="font-weight: bold; color: #fff;">knowledge_base.append()</div>
                    <div style="font-size: 11px; color: #faf089;">Simple list storage<br/>No transformation</div>
                </div>

                <div style="color: #fc8181; font-size: 24px;">→</div>

                <!-- Storage -->
                <div style="background: #2d3748; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 140px; border: 2px dashed #a0aec0;">
                    <div style="font-size: 24px;">🗃️</div>
                    <div style="font-weight: bold; color: #fff;">Document Store</div>
                    <div style="font-size: 11px; color: #a0aec0;">List of strings<br/>Grows with data</div>
                </div>
            </div>

            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; font-size: 12px; color: #a0aec0;">
                <strong style="color: #fc8181;">Key Point:</strong> Facts are simply stored as-is. <strong>No learning occurs.</strong>
                The system doesn't understand relationships or patterns between facts.
            </div>
        </div>

        <!-- Phase 2: Retrieval Phase -->
        <div style="background: rgba(237, 137, 54, 0.1); border: 2px solid #ed8936; border-radius: 12px; padding: 20px;">
            <h4 style="color: #ed8936; margin: 0 0 15px 0;">Phase 2: Query (When Questions Are Asked)</h4>

            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                <!-- Question -->
                <div style="background: #2d3748; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 120px;">
                    <div style="font-size: 24px;">❓</div>
                    <div style="font-weight: bold; color: #fff;">Question</div>
                    <div style="font-size: 11px; color: #a0aec0;">"Will Carlos<br/>like neon?"</div>
                </div>

                <div style="color: #ed8936; font-size: 24px;">→</div>

                <!-- Tokenize -->
                <div style="background: #553c9a; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 130px;">
                    <div style="font-size: 24px;">✂️</div>
                    <div style="font-weight: bold; color: #fff;">Tokenize</div>
                    <div style="font-size: 11px; color: #d6bcfa;">Split into words<br/>{"will", "carlos",<br/>"like", "neon"}</div>
                </div>

                <div style="color: #ed8936; font-size: 24px;">→</div>

                <!-- Keyword Match -->
                <div style="background: #c53030; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 160px;">
                    <div style="font-size: 24px;">🔍</div>
                    <div style="font-weight: bold; color: #fff;">Keyword Overlap</div>
                    <div style="font-size: 11px; color: #feb2b2;">Count matching words<br/>between Q and each fact</div>
                </div>

                <div style="color: #ed8936; font-size: 24px;">→</div>

                <!-- Top-K -->
                <div style="background: #744210; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 130px; border: 3px solid #faf089;">
                    <div style="font-size: 24px;">🏆</div>
                    <div style="font-weight: bold; color: #fff;">Top-2 Facts</div>
                    <div style="font-size: 11px; color: #faf089;">Only highest<br/>overlap scores</div>
                </div>

                <div style="color: #ed8936; font-size: 24px;">→</div>

                <!-- System Prompt -->
                <div style="background: #3182ce; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 180px;">
                    <div style="font-size: 24px;">💬</div>
                    <div style="font-weight: bold; color: #fff;">System Prompt</div>
                    <div style="font-size: 10px; color: #bee3f8; text-align: left; margin-top: 5px;">
                        "You are a RAG system.<br/>
                        ONLY use retrieved facts.<br/>
                        If not covered, say so."
                    </div>
                </div>

                <div style="color: #ed8936; font-size: 24px;">→</div>

                <!-- LLM -->
                <div style="background: linear-gradient(135deg, #805ad5 0%, #553c9a 100%); padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 140px; border: 3px solid #d6bcfa;">
                    <div style="font-size: 24px;">🤖</div>
                    <div style="font-weight: bold; color: #fff;">LLM</div>
                    <div style="font-size: 11px; color: #e9d8fd;">SmolLM3-3B<br/>(Same model!)</div>
                </div>
            </div>

            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; font-size: 12px; color: #a0aec0;">
                <strong style="color: #ed8936;">Key Point:</strong> The LLM only sees 2 retrieved facts (not all 4).
                "neon" ≠ "bright" keyword-wise, so relevant facts may not be retrieved!
            </div>
        </div>

    </div>
</div>
"""

LLM_INTEGRATION_MD = """
---

## How Each System Connects to the LLM

Both systems use the **exact same LLM** (HuggingFace SmolLM3-3B). The difference is **what context they provide**.

### Neural Memory → LLM Connection

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM PROMPT (Neural Memory)                      │
├─────────────────────────────────────────────────────────────────────┤
│ "You are an AI with neural memory that has LEARNED from all         │
│  observations below. Unlike simple retrieval, you should:           │
│                                                                      │
│  1. Consider ALL facts holistically                                  │
│  2. Identify PATTERNS across multiple observations                   │
│  3. Make INFERENCES based on learned patterns                        │
│  4. Predict based on trends, not just direct matches                 │
│                                                                      │
│  Observations (learned knowledge):                                   │
│  - Carlos rejected the bright colorful design                        │
│  - Carlos rejected the flashy animated homepage                      │
│  - Carlos approved the minimalist dark layout                        │
│  - Carlos approved the clean monochrome interface                    │
│                                                                      │
│  Learned patterns from observations:                                 │
│  - Positive signals: 2 approvals                                     │
│  - Negative signals: 2 rejections                                    │
│  - Look for common themes in approved vs rejected items              │
│                                                                      │
│  Question novelty (surprise score): 0.89                             │
│  - High surprise (>0.6): This is a novel topic, be cautious"         │
├─────────────────────────────────────────────────────────────────────┤
│ USER: "We have a new UI mockup with neon colors - will Carlos       │
│        like it?"                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**What the Neural Memory provides:**
| Component | Purpose |
|-----------|---------|
| **ALL facts** | Complete context for holistic reasoning |
| **Pattern hints** | Extracted approval/rejection counts |
| **Surprise score** | Indicates if question is familiar or novel |
| **Inference instructions** | Tells LLM to identify patterns and predict |

---

### RAG → LLM Connection

```
┌─────────────────────────────────────────────────────────────────────┐
│                       SYSTEM PROMPT (RAG)                            │
├─────────────────────────────────────────────────────────────────────┤
│ "You are a RAG system. You can ONLY use the retrieved facts below   │
│  to answer. If the retrieved facts don't directly answer the        │
│  question, say 'The retrieved information doesn't cover this.'      │
│                                                                      │
│  Retrieved facts:                                                    │
│  - Carlos rejected the bright colorful design                        │
│  (Only 1 fact retrieved - 'neon' didn't match other keywords!)      │
├─────────────────────────────────────────────────────────────────────┤
│ USER: "We have a new UI mockup with neon colors - will Carlos       │
│        like it?"                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**What RAG provides:**
| Component | Purpose |
|-----------|---------|
| **Top-2 facts only** | Limited context based on keyword overlap |
| **No pattern info** | System doesn't analyze relationships |
| **No novelty signal** | No indication of question familiarity |
| **Strict retrieval instructions** | Tells LLM to only use retrieved facts |

---

## The Critical Difference: What Goes Into the LLM

"""

COMPARISON_TABLE_HTML = """
<div style="font-family: system-ui, -apple-system, sans-serif; padding: 20px; background: #1a1a2e; border-radius: 16px; color: #fff; margin: 20px 0;">
    <h3 style="text-align: center; color: #fff; margin-bottom: 20px;">Side-by-Side: What the LLM Receives</h3>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <!-- Neural Memory Column -->
        <div style="background: rgba(0, 212, 255, 0.1); border: 2px solid #00d4ff; border-radius: 12px; padding: 20px;">
            <h4 style="color: #00d4ff; text-align: center; margin: 0 0 15px 0;">🧠 Neural Memory</h4>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="color: #48bb78; font-weight: bold; margin-bottom: 8px;">✅ Facts Provided:</div>
                <div style="font-size: 13px; color: #a0aec0;">ALL 4 facts (complete knowledge)</div>
            </div>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="color: #48bb78; font-weight: bold; margin-bottom: 8px;">✅ Pattern Analysis:</div>
                <div style="font-size: 13px; color: #a0aec0;">
                    • 2 approvals identified<br/>
                    • 2 rejections identified<br/>
                    • "Look for common themes"
                </div>
            </div>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="color: #48bb78; font-weight: bold; margin-bottom: 8px;">✅ Novelty Signal:</div>
                <div style="font-size: 13px; color: #a0aec0;">Surprise score: 0.89 (novel topic)</div>
            </div>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px;">
                <div style="color: #48bb78; font-weight: bold; margin-bottom: 8px;">✅ Instructions:</div>
                <div style="font-size: 13px; color: #a0aec0;">
                    "Identify PATTERNS"<br/>
                    "Make INFERENCES"<br/>
                    "Predict based on trends"
                </div>
            </div>
        </div>

        <!-- RAG Column -->
        <div style="background: rgba(252, 129, 129, 0.1); border: 2px solid #fc8181; border-radius: 12px; padding: 20px;">
            <h4 style="color: #fc8181; text-align: center; margin: 0 0 15px 0;">📚 RAG</h4>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="color: #fc8181; font-weight: bold; margin-bottom: 8px;">⚠️ Facts Provided:</div>
                <div style="font-size: 13px; color: #a0aec0;">Only 1-2 facts (keyword match)<br/>
                <span style="color: #fc8181; font-size: 11px;">"neon" ≠ "bright", "colorful", etc.</span></div>
            </div>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="color: #fc8181; font-weight: bold; margin-bottom: 8px;">❌ Pattern Analysis:</div>
                <div style="font-size: 13px; color: #a0aec0;">None - no relationship detection</div>
            </div>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="color: #fc8181; font-weight: bold; margin-bottom: 8px;">❌ Novelty Signal:</div>
                <div style="font-size: 13px; color: #a0aec0;">None - no familiarity indicator</div>
            </div>

            <div style="background: #2d3748; border-radius: 8px; padding: 15px;">
                <div style="color: #fc8181; font-weight: bold; margin-bottom: 8px;">⚠️ Instructions:</div>
                <div style="font-size: 13px; color: #a0aec0;">
                    "ONLY use retrieved facts"<br/>
                    "If not covered, say so"<br/>
                    <span style="color: #fc8181; font-size: 11px;">No inference allowed</span>
                </div>
            </div>
        </div>
    </div>
</div>
"""

ARCHITECTURE_SUMMARY_MD = """
---

## Why This Architecture Matters

### The Learning Advantage

| Aspect | Neural Memory | RAG |
|--------|---------------|-----|
| **Storage** | Fixed neural weights (~250K params) | Growing document store |
| **Learning** | Yes - weights update per observation | No - just stores text |
| **Retrieval** | Not needed - patterns in weights | Required - keyword matching |
| **Inference** | Can generalize to novel queries | Limited to direct matches |
| **Memory Size** | Constant (doesn't grow) | Linear growth with data |

### When Neural Memory Wins

The architecture shines when:
1. **Pattern Recognition Required** - "Carlos likes X, dislikes Y" → predict for Z
2. **Novel Queries** - Question keywords don't match stored facts
3. **Holistic Reasoning** - Answer requires synthesizing multiple facts
4. **Bounded Memory** - Can't afford growing storage

### When RAG Might Be Better

RAG is simpler when:
1. **Exact Retrieval** - "What did the document say about X?"
2. **Large Corpus** - Millions of documents to search
3. **No Patterns** - Facts are independent, not related
4. **Transparency** - Need to cite exact source documents

---

## Technical Implementation Details

### Neural Memory Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────┐
│  _encode_text(text)                          │
│  ┌─────────────────────────────────────────┐ │
│  │ 1. Convert to ASCII ordinals            │ │
│  │ 2. Pad/truncate to max_seq_len (64)     │ │
│  │ 3. Project to dimension (256)           │ │
│  │ 4. Output: Tensor [1, 64, 256]          │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  memory_net (nn.Sequential)                  │
│  ┌─────────────────────────────────────────┐ │
│  │ Linear(256 → 1024)                      │ │
│  │ GELU activation                         │ │
│  │ LayerNorm(1024)                         │ │
│  │ Linear(1024 → 256)                      │ │
│  └─────────────────────────────────────────┘ │
│  Total: ~262K parameters                     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  _compute_surprise_tensor(input, output)     │
│  ┌─────────────────────────────────────────┐ │
│  │ loss = MSE(output, target)              │ │
│  │ surprise = sigmoid(loss) scaled to 0-1  │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  _update_weights(loss) [IF learn=True]       │
│  ┌─────────────────────────────────────────┐ │
│  │ grads = torch.autograd.grad(loss, θ)    │ │
│  │ for each (param, grad):                 │ │
│  │     param -= learning_rate × grad       │ │
│  └─────────────────────────────────────────┘ │
│  ⚡ This is the key innovation!              │
└─────────────────────────────────────────────┘
```

### RAG Architecture (Simplified for Demo)

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────┐
│  knowledge_base.append({"fact": text, ...})  │
│  Simple list storage - no transformation     │
└─────────────────────────────────────────────┘

Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  Keyword Overlap Scoring                     │
│  ┌─────────────────────────────────────────┐ │
│  │ question_words = set(query.split())     │ │
│  │ for fact in knowledge_base:             │ │
│  │     fact_words = set(fact.split())      │ │
│  │     score = len(question_words ∩ fact_  │ │
│  │              words)                      │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Top-K Selection (K=2 in our demo)           │
│  Return facts with highest overlap scores    │
└─────────────────────────────────────────────┘
```

---

*These diagrams represent the actual implementation in this demo. The code is open source.*
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

with gr.Blocks(title="Docker Neural Memory", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    # Branded header
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # TAB 1: Comparison Demo (NEW - Main Feature)
        with gr.TabItem("LLM Comparison"):
            gr.Markdown("""
            ### Neural Memory vs RAG (Retrieval-Augmented Generation)

            **Step 1:** Teach the system facts about preferences/patterns
            **Step 2:** Ask questions that require **inference**, not just retrieval

            **RAG** retrieves similar documents but can't learn patterns.
            **Neural Memory** learns from ALL observations and can infer from trends.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 1: Teach Facts")
                    fact_input = gr.Textbox(
                        label="Add a Fact",
                        placeholder="e.g., 'Carlos prefers VSCode over Vim'",
                        lines=2,
                    )
                    add_fact_btn = gr.Button("Add to Knowledge Base", variant="secondary")
                    fact_output = gr.Markdown()
                    gr.Markdown("#### Example Facts to Try")
                    gr.Markdown("""
                    **Scenario: Learning User Preferences (Pattern Recognition)**
                    1. "Carlos rejected the bright colorful design"
                    2. "Carlos rejected the flashy animated homepage"
                    3. "Carlos approved the minimalist dark layout"
                    4. "Carlos approved the clean monochrome interface"

                    Then ask: **"We have a new UI mockup with neon colors - will Carlos like it?"**

                    *Neural Memory learns the pattern (Carlos prefers dark/minimal). RAG just retrieves similar facts without inferring the preference pattern.*
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("#### Knowledge Base (RAG Store)")
                    kb_plot = gr.Plot(label="Facts Stored")

            # Visualizations row
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Neural Memory State")
                    neural_state_plot = gr.Plot(label="Neural Network Weights & Stats")
                with gr.Column(scale=1):
                    gr.Markdown("#### Embedding Space")
                    tsne_plot = gr.Plot(label="t-SNE/PCA Visualization")

            add_fact_btn.click(
                add_to_knowledge_base,
                inputs=[fact_input],
                outputs=[fact_output, tsne_plot, neural_state_plot, kb_plot]
            )

            gr.Markdown("---")
            gr.Markdown("#### Step 2: Ask Questions & Compare Responses")

            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Ask a Question",
                        placeholder="e.g., 'We have a new UI mockup with neon colors - will Carlos like it?'",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **Best Questions for Neural Memory:**
                    - Questions requiring **pattern inference**
                    - Questions about **preferences/trends**
                    - Questions needing **generalization**
                    """)

            with gr.Row():
                compare_btn = gr.Button("Compare Responses", variant="primary", size="lg")
                reset_compare_btn = gr.Button("Reset Comparison", variant="secondary")

            # Response display - side by side with clear headers
            with gr.Row():
                with gr.Column():
                    gr.Markdown("##### Neural Memory Response")
                    nm_response = gr.Markdown()
                with gr.Column():
                    gr.Markdown("##### RAG Response")
                    vanilla_response = gr.Markdown()

            comparison_summary = gr.Markdown(label="Comparison Metrics")

            compare_btn.click(
                compare_responses,
                inputs=[question_input],
                outputs=[nm_response, vanilla_response, comparison_summary, neural_state_plot, kb_plot],
            )
            reset_compare_btn.click(
                reset_comparison,
                outputs=[comparison_summary, tsne_plot, neural_state_plot, kb_plot]
            )

        # TAB 2: How It Works (Architecture Diagrams)
        with gr.TabItem("How It Works"):
            gr.Markdown(ARCHITECTURE_INTRO_MD)

            # Neural Memory Diagram
            gr.HTML(NEURAL_MEMORY_DIAGRAM_HTML)

            # RAG Diagram
            gr.HTML(RAG_DIAGRAM_HTML)

            # LLM Integration Explanation
            gr.Markdown(LLM_INTEGRATION_MD)

            # Side-by-side comparison table
            gr.HTML(COMPARISON_TABLE_HTML)

            # Architecture Summary
            gr.Markdown(ARCHITECTURE_SUMMARY_MD)

        # TAB 3: Key Concepts
        with gr.TabItem("Key Concepts"):
            gr.HTML(KEY_CONCEPTS_HTML)

        # TAB 4: Integration & Docker
        with gr.TabItem("Integration & Docker"):
            gr.Markdown("## How Memory Modules Integrate with LLMs")
            gr.Markdown("Follow this incremental explanation to understand how both RAG and Neural Memory attach to a vanilla LLM.")

            # Step 1: Vanilla LLM
            gr.HTML(VANILLA_LLM_DIAGRAM_HTML)

            # Step 2a: RAG Integration
            gr.HTML(RAG_INTEGRATION_DIAGRAM_HTML)

            # Step 2b: Neural Memory Integration
            gr.HTML(NEURAL_MEMORY_INTEGRATION_DIAGRAM_HTML)

            # Step 3: Docker Deployment
            gr.HTML(DOCKER_DEPLOYMENT_DIAGRAM_HTML)

            # Docker details
            gr.Markdown(DOCKER_INTEGRATION_MD)

        # TAB 5: About
        with gr.TabItem("About"):
            gr.Markdown(ABOUT_MD)

    # Polished footer with profile links
    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
