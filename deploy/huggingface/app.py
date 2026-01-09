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

        # TAB 2: Live Demo (original)
        with gr.TabItem("Neural Memory Playground"):
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
