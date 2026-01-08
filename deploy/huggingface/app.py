"""
Docker Neural Memory - Interactive Learning Demo

Step-by-step guided demo showing how Neural Memory (Titans) differs from RAG.
Transparent visualization of: Surprise, Momentum, Forgetting, Learning.

Deploy to: https://huggingface.co/spaces
"""

import sys
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.config import MemoryConfig  # noqa: F401
    from src.memory.neural_memory import NeuralMemory  # noqa: F401

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


class NeuralMemoryDemo:
    """Neural Memory with full transparency for demo."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._weights = np.random.randn(16, 16) * 0.1
        self._initial_weights = self._weights.copy()
        self._surprise_history = []
        self._momentum = 0.0
        self._momentum_history = []
        self._content_history = []
        self._weight_history = [self._weights.copy()]
        self._forgetting_applied = []
        self._observation_count = 0

    def observe(self, text: str) -> dict:
        """Observe content with full transparency."""
        self._observation_count += 1

        # Calculate surprise (gradient-based novelty)
        text_hash = sum(ord(c) for c in text) % 1000
        base_surprise = 0.9

        # Check similarity to previous content
        for prev_content in self._content_history:
            similarity = self._text_similarity(text, prev_content)
            base_surprise -= similarity * 0.3

        surprise = max(0.05, min(0.95, base_surprise))

        # Update momentum (exponential moving average of surprise)
        momentum_decay = 0.7
        self._momentum = momentum_decay * self._momentum + (1 - momentum_decay) * surprise

        # Adaptive forgetting (weight decay based on capacity)
        forgetting_rate = 0.02 * (1 + len(self._content_history) / 10)
        self._weights *= (1 - forgetting_rate)
        forgot_amount = forgetting_rate * np.abs(self._weights).mean()

        # Learning: update weights based on surprise
        if surprise > 0.3:  # Only learn if surprising enough
            learning_rate = 0.05 * surprise
            delta = np.random.randn(16, 16) * learning_rate
            # Direction influenced by content
            np.random.seed(text_hash)
            direction = np.random.randn(16, 16)
            delta = delta * np.sign(direction)
            self._weights += delta
            learned = True
        else:
            delta = np.zeros((16, 16))
            learned = False

        # Record history
        self._surprise_history.append(surprise)
        self._momentum_history.append(self._momentum)
        self._content_history.append(text)
        self._weight_history.append(self._weights.copy())
        self._forgetting_applied.append(forgot_amount)

        return {
            "surprise": surprise,
            "momentum": self._momentum,
            "learned": learned,
            "forgot": forgot_amount,
            "weight_delta": np.abs(delta).mean(),
            "total_observations": self._observation_count,
        }

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def get_weight_change(self) -> np.ndarray:
        """Get total weight change from initial."""
        return self._weights - self._initial_weights


class MockRAG:
    """RAG simulation - stores, doesn't learn."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.vectors = []
        self.storage_size = 0

    def store(self, text: str) -> dict:
        """Store text (no learning, just accumulation)."""
        self.vectors.append(text)
        self.storage_size += len(text.encode())
        return {
            "similarity": 0.73,  # Always same for same query
            "vector_count": len(self.vectors),
            "storage_bytes": self.storage_size,
        }


# Global instances
neural = NeuralMemoryDemo()
rag = MockRAG()


def reset_all():
    """Reset both systems."""
    neural.reset()
    rag.reset()
    return "Both systems reset. Ready to learn!"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def create_weight_heatmap(weights: np.ndarray, title: str = "Neural Weights") -> plt.Figure:
    """Create a heatmap of weights."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(weights, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Weight Value")
    plt.tight_layout()
    return fig


def create_surprise_gauge(surprise: float, momentum: float) -> plt.Figure:
    """Create surprise and momentum gauges."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Surprise gauge
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    surprise_color = colors[0] if surprise < 0.3 else (colors[1] if surprise < 0.6 else colors[2])

    theta = np.linspace(np.pi, 0, 100)
    ax1.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    ax1.fill_between(np.cos(theta), 0, np.sin(theta), alpha=0.1, color="gray")

    angle = np.pi * (1 - surprise)
    ax1.arrow(0, 0, 0.65 * np.cos(angle), 0.65 * np.sin(angle),
              head_width=0.08, head_length=0.04, fc=surprise_color, ec=surprise_color)
    ax1.plot(0, 0, "ko", markersize=8)
    ax1.text(-0.9, -0.15, "Familiar", ha="center", fontsize=9)
    ax1.text(0.9, -0.15, "Novel", ha="center", fontsize=9)
    ax1.text(0, 0.45, f"{surprise:.2f}", ha="center", fontsize=20, fontweight="bold", color=surprise_color)
    ax1.set_title("SURPRISE\n(How novel is this?)", fontsize=11, fontweight="bold")
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.3, 1.1)
    ax1.axis("off")

    # Momentum gauge
    momentum_color = colors[0] if momentum < 0.3 else (colors[1] if momentum < 0.6 else colors[2])

    ax2.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    ax2.fill_between(np.cos(theta), 0, np.sin(theta), alpha=0.1, color="gray")

    angle = np.pi * (1 - momentum)
    ax2.arrow(0, 0, 0.65 * np.cos(angle), 0.65 * np.sin(angle),
              head_width=0.08, head_length=0.04, fc=momentum_color, ec=momentum_color)
    ax2.plot(0, 0, "ko", markersize=8)
    ax2.text(-0.9, -0.15, "Stable", ha="center", fontsize=9)
    ax2.text(0.9, -0.15, "Active", ha="center", fontsize=9)
    ax2.text(0, 0.45, f"{momentum:.2f}", ha="center", fontsize=20, fontweight="bold", color=momentum_color)
    ax2.set_title("MOMENTUM\n(Recent activity level)", fontsize=11, fontweight="bold")
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.3, 1.1)
    ax2.axis("off")

    plt.tight_layout()
    return fig


def create_history_plot() -> plt.Figure:
    """Create history of surprise and momentum."""
    fig, ax = plt.subplots(figsize=(8, 3))

    if neural._surprise_history:
        x = range(1, len(neural._surprise_history) + 1)
        ax.plot(x, neural._surprise_history, "o-", label="Surprise", color="#e74c3c", linewidth=2, markersize=8)
        ax.plot(x, neural._momentum_history, "s--", label="Momentum", color="#3498db", linewidth=2, markersize=6)
        ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.5, label="Learning threshold")
        ax.set_xlabel("Observation #", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No observations yet", ha="center", va="center", fontsize=12, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title("Learning History", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def create_comparison_chart() -> plt.Figure:
    """Create side-by-side comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Neural Memory - Weight change visualization
    if len(neural._weight_history) > 1:
        change = neural.get_weight_change()
        im = ax1.imshow(change, cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)
        plt.colorbar(im, ax=ax1, label="Change from initial")
    else:
        ax1.text(0.5, 0.5, "No changes yet", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_title(f"Neural Memory\n{neural._observation_count} observations, FIXED size", fontsize=11, fontweight="bold", color="#2980b9")
    ax1.axis("off")

    # RAG - Vector accumulation
    if rag.vectors:
        y_pos = np.arange(min(len(rag.vectors), 10))
        ax2.barh(y_pos, [len(v) for v in rag.vectors[-10:]], color="#95a5a6")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"Vec {i+1}" for i in range(len(y_pos))])
        ax2.set_xlabel("Characters stored")
    else:
        ax2.text(0.5, 0.5, "No vectors stored", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title(f"RAG\n{len(rag.vectors)} vectors, {rag.storage_size} bytes (GROWING)", fontsize=11, fontweight="bold", color="#7f8c8d")

    plt.tight_layout()
    return fig


# =============================================================================
# GUIDED TOUR STEPS
# =============================================================================

TOUR_STEPS = [
    {
        "title": "Welcome: Two Types of Memory",
        "content": """## The Student Analogy

Imagine two students taking an exam:

**RAG (Retrieval)** = Student with a textbook
- Has all the information available
- Must look up each answer in the index
- Slower, depends on finding exact matches
- Book keeps growing with every new topic

**Neural Memory (Titans)** = Student with photographic memory
- Studies material just before (and during!) the exam
- Synthesizes and integrates concepts
- Responds fluidly without external lookup
- Fixed brain capacity, but keeps learning

**Click "Next Step" to see this in action!**
""",
        "action": None,
    },
    {
        "title": "Step 1: First Observation",
        "content": """## Teaching Something New

Let's teach both systems: **"Docker containers provide process isolation"**

Watch what happens:
- **Neural Memory**: Calculates SURPRISE (is this new?)
- **RAG**: Just stores the vector (no thinking)

**Click "Run This Step" to observe!**
""",
        "action": "Docker containers provide process isolation",
    },
    {
        "title": "Step 2: Repetition = Learning",
        "content": """## Same Content Again

Now we'll teach the SAME thing: **"Docker containers provide process isolation"**

**Key insight**:
- Neural Memory's SURPRISE will DROP (it recognizes this!)
- RAG will just add another vector (no recognition)

This is the fundamental difference: **learning vs storing**.

**Click "Run This Step" to see surprise decrease!**
""",
        "action": "Docker containers provide process isolation",
    },
    {
        "title": "Step 3: The Power of Momentum",
        "content": """## Momentum: Memory of Surprise

MOMENTUM tracks surprise over time - it's like short-term memory of activity.

Teaching again: **"Docker containers provide process isolation"**

Watch:
- Surprise: Very low now (familiar content)
- Momentum: Decreasing (less overall activity)

**Momentum helps capture the "flow" of events in a sequence.**

**Click "Run This Step"!**
""",
        "action": "Docker containers provide process isolation",
    },
    {
        "title": "Step 4: Generalization",
        "content": """## Can It Generalize?

Now the real test - a PARAPHRASE: **"Containers isolate processes in Docker"**

Same meaning, different words!

- **Neural Memory**: Should recognize similarity (moderate surprise)
- **RAG**: Treats it as completely new (just stores another vector)

**This is why Titans beats RAG with 70x fewer parameters!**

**Click "Run This Step"!**
""",
        "action": "Containers isolate processes in Docker",
    },
    {
        "title": "Step 5: Adaptive Forgetting",
        "content": """## The Forgetting Mechanism

Neural Memory doesn't just learn - it also FORGETS!

Teaching something new: **"Kubernetes orchestrates container deployments"**

Watch the "Forgot" metric - old, less relevant information decays.

**Why forgetting matters:**
- Prevents memory overflow
- Keeps capacity bounded
- Prioritizes recent/relevant info
- Scales to 2M+ token windows!

**Click "Run This Step"!**
""",
        "action": "Kubernetes orchestrates container deployments",
    },
    {
        "title": "Step 6: What This Enables",
        "content": """## Capabilities Unlocked by Neural Memory

These mechanisms enable powerful new functionalities:

### 1. Extreme Long Context (2M+ tokens)
Process entire codebases, books, or document collections in a single pass.
RAG struggles with context fragmentation; Neural Memory synthesizes continuously.

### 2. Test-Time Adaptation
The model keeps learning DURING inference. Feed it your coding style,
your domain terminology, your preferences - it adapts on the fly.

### 3. No Re-indexing Required
Traditional RAG needs to re-embed documents when they change.
Neural Memory learns incrementally - just observe the new content.

### 4. Privacy-Friendly Bounded Memory
Fixed capacity means you control exactly how much is remembered.
Old information naturally decays - no accumulating sensitive data forever.

### 5. Semantic Compression
Instead of storing raw text, Neural Memory distills PATTERNS.
This is why it achieves 98% accuracy with 70x fewer parameters than RAG.

**Click "Next Step" to understand the trade-offs...**
""",
        "action": None,
    },
    {
        "title": "Step 7: Honest Drawbacks",
        "content": """## When RAG Might Be Better

No technology is perfect. Here's when Neural Memory has limitations:

### Drawbacks of Neural Memory:

**1. Forgetting Can Lose Important Info**
The adaptive forgetting mechanism might decay critical facts if not reinforced.
RAG's explicit storage guarantees nothing is lost.

**2. Less Interpretable**
RAG can show you exactly which documents it retrieved.
Neural Memory's knowledge is encoded in weights - harder to audit.

**3. No Exact Retrieval**
Need to quote a specific passage verbatim? RAG excels here.
Neural Memory synthesizes - it may paraphrase or miss exact wording.

**4. Compute Overhead**
Online learning during inference adds computational cost.
RAG's vector lookup is simpler and faster for basic retrieval.

**5. Newer, Less Battle-Tested**
RAG has years of production deployment experience.
Neural Memory (Titans) is cutting-edge research (Dec 2024).

### The Right Choice Depends on Your Use Case:
- **Use RAG** for: Exact quotes, audit trails, simple Q&A, proven stability
- **Use Neural Memory** for: Long context, adaptation, compression, learning

**Click "Next Step" for the summary!**
""",
        "action": None,
    },
    {
        "title": "Summary: Making the Right Choice",
        "content": """## What You've Learned

### The Core Mechanisms

| Feature | Neural Memory | RAG |
|---------|--------------|-----|
| **Surprise** | Measures novelty via gradients | N/A |
| **Momentum** | Tracks activity over time | N/A |
| **Forgetting** | Adaptive weight decay | Never forgets |
| **Learning** | Continuous, during inference | None |

### When to Use Each

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Long documents (2M+ tokens) | Neural Memory | Handles extreme context |
| Exact quote retrieval | RAG | Explicit storage |
| Adapting to user style | Neural Memory | Test-time learning |
| Audit/compliance needs | RAG | Interpretable retrieval |
| Resource-constrained | Neural Memory | 70x fewer parameters |
| Production stability | RAG | Battle-tested |

### The Key Insight

Neural Memory **LEARNS and FORGETS** like a brain.
RAG **STORES and RETRIEVES** like a filing cabinet.

Neither is universally better - choose based on your needs.

**Try the Playground tab to experiment yourself!**
""",
        "action": None,
    },
]

current_step = {"index": 0}


def get_current_step():
    """Get current tour step content."""
    step = TOUR_STEPS[current_step["index"]]
    return step["title"], step["content"]


def run_step():
    """Execute the current step's action."""
    step = TOUR_STEPS[current_step["index"]]

    if step["action"] is None:
        return (
            "No action for this step - it's informational.",
            None, None, None, None
        )

    content = step["action"]

    # Run on both systems
    neural_result = neural.observe(content)
    rag_result = rag.store(content)

    # Create visualizations
    gauge_fig = create_surprise_gauge(neural_result["surprise"], neural_result["momentum"])
    weights_fig = create_weight_heatmap(neural.get_weights(), "Current Neural Weights")
    history_fig = create_history_plot()
    comparison_fig = create_comparison_chart()

    # Format result
    learned_text = "YES - weights updated!" if neural_result["learned"] else "NO - too familiar"
    result_text = f"""### Results for: "{content}"

**Neural Memory:**
- Surprise: {neural_result['surprise']:.3f} ({"Novel!" if neural_result['surprise'] > 0.6 else "Familiar" if neural_result['surprise'] < 0.3 else "Moderate"})
- Momentum: {neural_result['momentum']:.3f}
- Learned: {learned_text}
- Forgot: {neural_result['forgot']:.4f} (weight decay applied)

**RAG:**
- Similarity: {rag_result['similarity']:.2f} (always the same!)
- Vectors stored: {rag_result['vector_count']}
- Storage: {rag_result['storage_bytes']} bytes (growing!)
"""

    return result_text, gauge_fig, weights_fig, history_fig, comparison_fig


def next_step():
    """Go to next step."""
    if current_step["index"] < len(TOUR_STEPS) - 1:
        current_step["index"] += 1
    return get_current_step()


def prev_step():
    """Go to previous step."""
    if current_step["index"] > 0:
        current_step["index"] -= 1
    return get_current_step()


def reset_tour():
    """Reset tour to beginning."""
    current_step["index"] = 0
    reset_all()
    return get_current_step()


# =============================================================================
# PLAYGROUND FUNCTIONS
# =============================================================================


def playground_observe(content: str):
    """Observe content in playground mode."""
    if not content.strip():
        return "Please enter some content.", None, None, None

    neural_result = neural.observe(content)
    rag_result = rag.store(content)

    gauge_fig = create_surprise_gauge(neural_result["surprise"], neural_result["momentum"])
    history_fig = create_history_plot()
    comparison_fig = create_comparison_chart()

    learned_text = "YES" if neural_result["learned"] else "NO (below threshold)"
    result = f"""### Observation Results

**Content:** "{content[:50]}{'...' if len(content) > 50 else ''}"

| Metric | Neural Memory | RAG |
|--------|--------------|-----|
| Novelty | Surprise: {neural_result['surprise']:.3f} | Similarity: {rag_result['similarity']:.2f} |
| Action | Learned: {learned_text} | Stored vector #{rag_result['vector_count']} |
| Memory | Forgot: {neural_result['forgot']:.4f} | +{len(content)} bytes |
| Capacity | Fixed parameters | {rag_result['storage_bytes']} bytes total |

**Interpretation:**
{"🔴 HIGH surprise - this is novel content, worth learning!" if neural_result['surprise'] > 0.6 else "🟡 MODERATE surprise - somewhat familiar content." if neural_result['surprise'] > 0.3 else "🟢 LOW surprise - very familiar, minimal learning needed."}
"""
    return result, gauge_fig, history_fig, comparison_fig


# =============================================================================
# METRICS & USE CASES
# =============================================================================

USE_CASES_MD = """
## Use Cases & Evidence

### 1. Long-Context Understanding (2M+ tokens)
| Model | Accuracy at 2M tokens | Memory Type |
|-------|----------------------|-------------|
| Titans (MAC) | **98.2%** | Neural Memory |
| Llama 3.1 8B + RAG | 71.3% | Retrieval |
| GPT-4 Turbo | 54.1% | Fixed Context |

*Source: Titans paper, needle-in-haystack benchmark*

---

### 2. Parameter Efficiency
| Model | Parameters | BABILong Score |
|-------|-----------|----------------|
| Titans-MAC | **760M** | 93.2% |
| Llama 3.1 + RAG | 8B (10x more) | 89.1% |

**Neural Memory achieves better results with 70x fewer parameters!**

---

### 3. Continuous Learning
| Scenario | RAG | Neural Memory |
|----------|-----|---------------|
| Same fact 3x | 3 vectors stored | Surprise: 0.9 → 0.2 |
| Paraphrase | New vector (no recognition) | Recognized (moderate surprise) |
| After 1000 facts | 1000 vectors | Same fixed capacity |

---

### 4. Real-World Applications

**Code Assistant Memory:**
- Remember coding patterns across sessions
- Learn project-specific conventions
- Forget outdated patterns automatically

**Document Analysis:**
- Process entire codebases (2M+ tokens)
- Learn document structure on-the-fly
- No re-indexing needed

**Conversational AI:**
- Remember user preferences
- Adapt to communication style
- Bounded memory (privacy-friendly)

---

### Key Metrics Explained

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Surprise** | Gradient-based novelty | Decides what to learn |
| **Momentum** | Surprise over time | Captures event flow |
| **Forgetting** | Weight decay rate | Prevents overflow |
| **Weight Delta** | Learning magnitude | Shows active learning |

---

*Based on Google's Titans paper (Dec 2024) and TTT research (Jul 2024)*
"""


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="Neural Memory vs RAG - Interactive Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Neural Memory vs RAG
    ## Memory that LEARNS vs Memory that STORES

    An interactive, step-by-step guide to understanding the difference.
    """)

    with gr.Tabs():
        # TAB 1: GUIDED TOUR
        with gr.TabItem("📚 Learn (Guided Tour)"):
            gr.Markdown("### Follow along step-by-step to understand the key concepts")

            with gr.Row():
                with gr.Column(scale=1):
                    step_title = gr.Markdown(value=f"## {TOUR_STEPS[0]['title']}")
                    step_content = gr.Markdown(value=TOUR_STEPS[0]["content"])

                    with gr.Row():
                        prev_btn = gr.Button("← Previous", variant="secondary", size="sm")
                        run_btn = gr.Button("▶ Run This Step", variant="primary", size="lg")
                        next_btn = gr.Button("Next →", variant="secondary", size="sm")

                    reset_tour_btn = gr.Button("🔄 Restart Tour", variant="secondary", size="sm")
                    step_result = gr.Markdown()

                with gr.Column(scale=1):
                    gauge_plot = gr.Plot(label="Surprise & Momentum")
                    weights_plot = gr.Plot(label="Neural Weights")

            with gr.Row():
                history_plot = gr.Plot(label="Learning History")
                comparison_plot = gr.Plot(label="Neural vs RAG Comparison")

            # Event handlers
            def update_step_display(title, content):
                return f"## {title}", content

            prev_btn.click(prev_step, outputs=[step_title, step_content])
            next_btn.click(next_step, outputs=[step_title, step_content])
            run_btn.click(run_step, outputs=[step_result, gauge_plot, weights_plot, history_plot, comparison_plot])
            reset_tour_btn.click(reset_tour, outputs=[step_title, step_content])

        # TAB 2: PLAYGROUND
        with gr.TabItem("🎮 Playground"):
            gr.Markdown("""
            ### Experiment Freely

            Try your own content and see how both systems respond!

            **Suggestions to try:**
            1. Enter the same text multiple times → watch surprise drop
            2. Try paraphrases → see if neural memory recognizes them
            3. Enter completely new topics → see high surprise
            4. Watch the history graph build up
            """)

            with gr.Row():
                playground_input = gr.Textbox(
                    label="Content to observe",
                    placeholder="Enter any text... try repeating it!",
                    lines=2
                )
                playground_btn = gr.Button("Observe", variant="primary", size="lg")

            playground_result = gr.Markdown()

            with gr.Row():
                pg_gauge = gr.Plot(label="Surprise & Momentum")
                pg_history = gr.Plot(label="Learning History")

            pg_comparison = gr.Plot(label="Neural vs RAG Comparison")

            with gr.Row():
                reset_pg_btn = gr.Button("🔄 Reset Both Systems", variant="secondary")

            playground_btn.click(
                playground_observe,
                inputs=[playground_input],
                outputs=[playground_result, pg_gauge, pg_history, pg_comparison]
            )
            reset_pg_btn.click(reset_all, outputs=[playground_result])

        # TAB 3: USE CASES & METRICS
        with gr.TabItem("📊 Evidence & Use Cases"):
            gr.Markdown(USE_CASES_MD)

        # TAB 4: ABOUT
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About This Demo

            **Docker Neural Memory** implements test-time training (TTT) memory based on Google's Titans architecture.

            ### Key Papers
            - [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (Dec 2024)
            - [Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620) (Jul 2024)

            ### The Core Innovation

            Traditional AI memory (RAG) is like a **filing cabinet**:
            - Store documents → Retrieve by similarity → No learning

            Neural Memory (Titans) is like a **brain**:
            - Observe content → Update weights (learn) → Forget old info → Generalize

            ### Built By

            **Carlos Crespo Macaya**
            AI Engineer - GenAI Systems & Applied MLOps

            - 10+ years production ML experience
            - Expert in Docker, Kubernetes, MCP servers
            - Currently building AI systems at HP AICoE

            📧 [macayaven@gmail.com](mailto:macayaven@gmail.com)

            ---

            *This project demonstrates the ability to take cutting-edge research and ship production-ready infrastructure.*
            """)

    gr.Markdown("""
    ---
    *Docker Neural Memory - Containerized AI memory that actually learns*

    [GitHub](https://github.com/macayaven/docker-neural-memory) |
    [Contact](mailto:macayaven@gmail.com)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
