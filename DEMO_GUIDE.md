# Docker Neural Memory - Demo Guide

## Live Demo URL
**https://huggingface.co/spaces/macayaven/docker-neural-memory**

---

## Tab 1: LLM Comparison (Main Feature)

This tab demonstrates the key value proposition: **memory-augmented LLM vs vanilla LLM**.

### Step 1: Teach Facts to the Memory

1. In the "Add a Fact" textbox, enter facts like:
   ```
   My favorite programming language is Rust
   ```
2. Click **"Add to Knowledge Base"**
3. Watch the **t-SNE visualization** on the right update - each fact becomes a point in embedding space
4. Add more facts:
   ```
   I always use dark mode in my editor
   The project deadline is March 15th
   Our API uses JWT authentication
   The database runs on PostgreSQL 15
   ```

### Step 2: Ask Questions

1. In "Ask a Question", type:
   ```
   What programming language should I use?
   ```
2. Click **"Compare Responses"**
3. See side-by-side results:
   - **With Neural Memory**: Uses your observed facts to give grounded answer
   - **Vanilla LLM**: May hallucinate or give generic answer

### What to Look For

| Metric | With Memory | Vanilla |
|--------|-------------|---------|
| **Grounded Answers** | High % | N/A |
| **Hallucinations** | Low | Higher |
| **Surprise Score** | Decreases over time | N/A |

### Try These Question Sequences

**Sequence 1: Preferences**
1. Teach: "Carlos prefers VSCode over Vim"
2. Teach: "Carlos uses the Dracula theme"
3. Ask: "What editor does Carlos use?"
4. Ask: "What theme should I set up?"

**Sequence 2: Project Context**
1. Teach: "The API endpoint is /api/v2/users"
2. Teach: "Authentication requires Bearer tokens"
3. Ask: "How do I call the users API?"

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
2. **Bounded Memory** - Fixed parameters, doesn't grow like vector DBs
3. **Test-Time Training** - Learns during inference (Titans architecture)
4. **t-SNE Visualization** - See how concepts cluster in embedding space

---

## Troubleshooting

If LLM comparison shows "LLM not available":
- The HF_TOKEN secret may not be set in the Space settings
- The neural memory playground still works without it

If you see errors:
- Try clicking "Reset Memory" or "Reset Comparison"
- Refresh the page to restart the Space
