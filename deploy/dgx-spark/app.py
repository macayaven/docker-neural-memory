"""
Gemma + Neural Memory Comparison Demo

Side-by-side comparison showing quantitative improvement
when using neural memory vs standard context.

Runs on DGX Spark with local Gemma inference.
"""

import json
import os
import time
from dataclasses import dataclass, field

import gradio as gr
import requests

# Configuration
NEURAL_MEMORY_URL = os.getenv("NEURAL_MEMORY_URL", "http://localhost:8765")
GEMMA_URL = os.getenv("GEMMA_URL", "http://localhost:11434")


@dataclass
class Metrics:
    """Track comparison metrics."""

    # With Neural Memory
    nm_queries: int = 0
    nm_correct_predictions: int = 0
    nm_total_tokens: int = 0
    nm_clarifying_questions: int = 0
    nm_response_times: list = field(default_factory=list)
    nm_surprise_history: list = field(default_factory=list)

    # Without Neural Memory (baseline)
    baseline_queries: int = 0
    baseline_correct_predictions: int = 0
    baseline_total_tokens: int = 0
    baseline_clarifying_questions: int = 0
    baseline_response_times: list = field(default_factory=list)


metrics = Metrics()

# User preferences learned
learned_preferences: dict = {}


def call_neural_memory(action: str, content: str = "") -> dict:
    """Call the neural memory MCP server."""
    try:
        if action == "observe":
            resp = requests.post(
                f"{NEURAL_MEMORY_URL}/observe",
                json={"content": content},
                timeout=5,
            )
        elif action == "surprise":
            resp = requests.post(
                f"{NEURAL_MEMORY_URL}/surprise",
                json={"content": content},
                timeout=5,
            )
        elif action == "stats":
            resp = requests.get(f"{NEURAL_MEMORY_URL}/stats", timeout=5)
        else:
            return {"error": f"Unknown action: {action}"}

        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def call_gemma(prompt: str, with_memory_context: str = "") -> tuple[str, float, int]:
    """Call Gemma via Ollama API. Returns (response, time, tokens)."""
    try:
        full_prompt = prompt
        if with_memory_context:
            full_prompt = f"Context from memory:\n{with_memory_context}\n\nUser: {prompt}"

        start = time.time()
        resp = requests.post(
            f"{GEMMA_URL}/api/generate",
            json={
                "model": "gemma2:2b",
                "prompt": full_prompt,
                "stream": False,
            },
            timeout=30,
        )
        elapsed = time.time() - start

        data = resp.json()
        response = data.get("response", "")
        tokens = data.get("eval_count", len(response.split()))

        return response, elapsed, tokens
    except Exception as e:
        return f"Error: {e}", 0.0, 0


def extract_preference(user_input: str, assistant_response: str) -> tuple[str, str] | None:
    """Extract preference from interaction."""
    # Simple heuristic: if user corrects or specifies, it's a preference
    keywords = {
        "editor": ["vscode", "vim", "neovim", "emacs", "sublime", "cursor"],
        "browser": ["chrome", "firefox", "safari", "brave", "arc"],
        "terminal": ["iterm", "warp", "terminal", "alacritty", "kitty"],
        "language": ["python", "javascript", "typescript", "rust", "go"],
    }

    user_lower = user_input.lower()
    for category, options in keywords.items():
        for option in options:
            if option in user_lower:
                return category, option

    return None


def process_with_memory(user_input: str) -> tuple[str, dict]:
    """Process query WITH neural memory."""
    global metrics, learned_preferences

    start_time = time.time()

    # Check surprise first (is this familiar?)
    surprise_result = call_neural_memory("surprise", user_input)
    surprise = surprise_result.get("surprise", 0.5)

    # Build context from learned preferences
    memory_context = ""
    if learned_preferences:
        memory_context = "User preferences:\n"
        for cat, pref in learned_preferences.items():
            memory_context += f"- {cat}: {pref}\n"

    # Query Gemma with memory context
    response, elapsed, tokens = call_gemma(user_input, memory_context)

    # Learn from this interaction
    observe_result = call_neural_memory("observe", f"{user_input} -> {response}")

    # Check if this revealed a preference
    pref = extract_preference(user_input, response)
    if pref:
        category, value = pref
        learned_preferences[category] = value

    # Update metrics
    metrics.nm_queries += 1
    metrics.nm_total_tokens += tokens
    metrics.nm_response_times.append(elapsed)
    metrics.nm_surprise_history.append(surprise)

    # Check if we avoided a clarifying question
    needs_clarification = any(
        q in response.lower() for q in ["which", "what", "please specify", "could you"]
    )
    if needs_clarification:
        metrics.nm_clarifying_questions += 1
    else:
        metrics.nm_correct_predictions += 1

    result_info = {
        "surprise": surprise,
        "tokens": tokens,
        "time": elapsed,
        "learned": observe_result.get("learned", False),
        "preferences": learned_preferences.copy(),
    }

    return response, result_info


def process_without_memory(user_input: str) -> tuple[str, dict]:
    """Process query WITHOUT neural memory (baseline)."""
    global metrics

    start_time = time.time()

    # Query Gemma without any context
    response, elapsed, tokens = call_gemma(user_input)

    # Update metrics
    metrics.baseline_queries += 1
    metrics.baseline_total_tokens += tokens
    metrics.baseline_response_times.append(elapsed)

    # Check if clarifying question was asked
    needs_clarification = any(
        q in response.lower() for q in ["which", "what", "please specify", "could you"]
    )
    if needs_clarification:
        metrics.baseline_clarifying_questions += 1
    else:
        metrics.baseline_correct_predictions += 1

    result_info = {
        "tokens": tokens,
        "time": elapsed,
    }

    return response, result_info


def compare_responses(user_input: str) -> tuple[str, str, str]:
    """Run same query on both systems and compare."""
    if not user_input.strip():
        return "", "", ""

    # Process both
    nm_response, nm_info = process_with_memory(user_input)
    baseline_response, baseline_info = process_without_memory(user_input)

    # Format outputs
    nm_output = f"""### With Neural Memory

{nm_response}

---
**Metrics:**
- Surprise: {nm_info['surprise']:.3f}
- Tokens: {nm_info['tokens']}
- Time: {nm_info['time']:.2f}s
- Learned: {'Yes' if nm_info.get('learned') else 'No'}
"""

    baseline_output = f"""### Without Memory (Baseline)

{baseline_response}

---
**Metrics:**
- Tokens: {baseline_info['tokens']}
- Time: {baseline_info['time']:.2f}s
"""

    # Comparison summary
    comparison = get_metrics_summary()

    return nm_output, baseline_output, comparison


def get_metrics_summary() -> str:
    """Generate metrics comparison summary."""
    nm_avg_time = (
        sum(metrics.nm_response_times) / len(metrics.nm_response_times)
        if metrics.nm_response_times
        else 0
    )
    baseline_avg_time = (
        sum(metrics.baseline_response_times) / len(metrics.baseline_response_times)
        if metrics.baseline_response_times
        else 0
    )

    nm_accuracy = (
        metrics.nm_correct_predictions / metrics.nm_queries * 100
        if metrics.nm_queries
        else 0
    )
    baseline_accuracy = (
        metrics.baseline_correct_predictions / metrics.baseline_queries * 100
        if metrics.baseline_queries
        else 0
    )

    avg_surprise = (
        sum(metrics.nm_surprise_history) / len(metrics.nm_surprise_history)
        if metrics.nm_surprise_history
        else 0.5
    )

    return f"""## Comparison Metrics

| Metric | With Neural Memory | Without Memory |
|--------|-------------------|----------------|
| **Queries** | {metrics.nm_queries} | {metrics.baseline_queries} |
| **Correct (no clarification)** | {metrics.nm_correct_predictions} ({nm_accuracy:.0f}%) | {metrics.baseline_correct_predictions} ({baseline_accuracy:.0f}%) |
| **Clarifying Questions** | {metrics.nm_clarifying_questions} | {metrics.baseline_clarifying_questions} |
| **Total Tokens** | {metrics.nm_total_tokens} | {metrics.baseline_total_tokens} |
| **Avg Response Time** | {nm_avg_time:.2f}s | {baseline_avg_time:.2f}s |
| **Avg Surprise** | {avg_surprise:.3f} | N/A |

### Learned Preferences
{json.dumps(learned_preferences, indent=2) if learned_preferences else "None yet"}

### Key Insight
- Surprise decreases as the system learns your preferences
- Fewer clarifying questions = better UX
- Token count stays bounded (no growing context)
"""


def reset_demo():
    """Reset all metrics and preferences."""
    global metrics, learned_preferences
    metrics = Metrics()
    learned_preferences = {}
    return "Demo reset. All metrics and preferences cleared."


# Gradio Interface
with gr.Blocks(title="Neural Memory Comparison", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Gemma + Neural Memory Comparison
    ## Hard Numbers: With vs Without Memory

    Ask questions that reveal your preferences. Watch the neural memory learn
    while the baseline keeps asking clarifying questions.

    **Try:** "Open my editor", "Search the web", "Run tests"
    """)

    with gr.Row():
        user_input = gr.Textbox(
            label="Your Request",
            placeholder="e.g., 'Open my code editor'",
            lines=2,
        )

    with gr.Row():
        submit_btn = gr.Button("Compare Responses", variant="primary", size="lg")
        reset_btn = gr.Button("Reset Demo", variant="secondary")

    with gr.Row():
        with gr.Column():
            nm_output = gr.Markdown(label="With Neural Memory")
        with gr.Column():
            baseline_output = gr.Markdown(label="Without Memory")

    metrics_output = gr.Markdown(label="Comparison Metrics")

    submit_btn.click(
        compare_responses,
        inputs=[user_input],
        outputs=[nm_output, baseline_output, metrics_output],
    )
    reset_btn.click(reset_demo, outputs=[metrics_output])

    gr.Markdown("""
    ---
    **How it works:**
    1. Both systems receive the same query
    2. Neural Memory checks "surprise" and uses learned preferences
    3. Baseline has no memory of previous interactions
    4. Metrics show the quantitative difference

    *Running on DGX Spark with local Gemma inference*
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
