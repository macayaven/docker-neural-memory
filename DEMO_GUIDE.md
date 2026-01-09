# Docker Neural Memory - Demo Guide

## Live Demo URL
**https://huggingface.co/spaces/macayaven/docker-neural-memory**

---

## Tab 1: LLM Comparison (Main Feature)

This tab demonstrates the key value proposition: **Neural Memory vs RAG**.

### The Key Difference

| Feature | Neural Memory | RAG |
|---------|---------------|-----|
| **How it works** | Learns patterns from ALL facts | Retrieves top-k similar documents |
| **Inference** | Can predict from learned trends | Only returns what matches |
| **Pattern Recognition** | Identifies themes across data | No pattern learning |

### Step 1: Teach Facts (Pattern Learning)

Add these facts one by one to create a **preference pattern**:
```
Carlos rejected the bright colorful design
Carlos rejected the flashy animated homepage
Carlos approved the minimalist dark layout
Carlos approved the clean monochrome interface
```

Watch:
- **Knowledge Base** shows all stored facts
- **Neural Memory State** shows weight updates after each fact
- **Embedding Space** shows how similar facts cluster together

### Step 2: Ask a Pattern-Based Question

In "Ask a Question", type:
```
We have a new UI mockup with neon colors - will Carlos like it?
```

Click **"Compare Responses"** and see:

- **Neural Memory**: Identifies the pattern (Carlos likes dark/minimal, rejects bright/flashy) and predicts "No"
- **RAG**: May fail to retrieve relevant facts (no keyword match for "neon") or give incomplete answer

### What to Look For

| Metric | Neural Memory | RAG |
|--------|---------------|-----|
| **Pattern Inference** | Identifies approval/rejection trends | None |
| **Context Used** | ALL 4 facts | Top 2 by keyword match |
| **Novelty Detection** | Surprise score shows familiarity | None |

### Why This Works

The question "neon colors" doesn't directly match any stored fact keywords:
- **RAG** does keyword matching: "neon" ≠ "bright", "colorful", "dark", etc.
- **Neural Memory** learned the pattern: bright/flashy → rejected, dark/minimal → approved

### Alternative Scenarios

**Scenario 2: Learning Preferences**
1. "Alice prefers Python for data science"
2. "Alice prefers JavaScript for web development"
3. "Alice prefers Go for backend services"
4. Ask: "What should Alice use for a new API server?"

**Scenario 3: Project Knowledge**
1. "Sprint 1 deadline was missed due to scope creep"
2. "Sprint 2 deadline was missed due to scope creep"
3. "Sprint 3 is adding 5 new features"
4. Ask: "Will Sprint 3 meet its deadline?"

---

## Tab 2: Neural Memory Playground

This tab shows the raw neural network learning.

### Observe Content
1. Enter text in "Content to Observe"
2. Click **"Observe (Learn)"**
3. Watch:
   - **Surprise score** decrease on repeated patterns
   - **Weight Delta** shows how much the network changed
   - **Weight visualization** updates in real-time

### Check Surprise Without Learning
1. Enter text in "Check Surprise"
2. Click **"Check Surprise"**
3. See novelty score without updating weights

### Key Insight
- **Surprise < 0.3**: Memory recognizes this pattern
- **Surprise > 0.6**: Novel content, worth learning

---

## Tab 3: Docker Integration

Documentation about how this integrates with Docker and MCP servers.

---

## What Makes This Special

1. **Real PyTorch** - Not a simulation, actual gradient descent
2. **Bounded Memory** - Fixed parameters, doesn't grow like RAG vector DBs
3. **Test-Time Training** - Learns during inference (Titans architecture)
4. **Pattern Inference** - Can predict from learned trends, not just retrieve
5. **Novelty Detection** - Surprise score shows what's familiar vs new
6. **t-SNE Visualization** - See how concepts cluster in embedding space

---

## Troubleshooting

If LLM comparison shows "LLM not available":
- The HF_TOKEN secret may not be set in the Space settings
- The neural memory playground still works without it

If you see errors:
- Try clicking "Reset Memory" or "Reset Comparison"
- Refresh the page to restart the Space
