# Docker Neural Memory: Technical Specification

## Executive Summary

**Docker Neural Memory** is a containerized implementation of test-time training (TTT) memory, based on Google's Titans architecture. Unlike existing AI memory solutions that store and retrieve embeddings, this system **learns** during inference—its neural weights update with every interaction.

This positions Docker as infrastructure for the next generation of AI: not just running models, but being the platform for **modular, learnable memory components**.

---

## 1. The Problem: Current AI Memory is a Misnomer

### What Exists Today

```
┌─────────────────────────────────────────────────────────────┐
│  "Memory" Solution          │  Reality                      │
├─────────────────────────────┼───────────────────────────────┤
│  OpenMemory                 │  Embedding storage + search   │
│  Mem0                       │  Vector DB wrapper            │
│  Zep                        │  Session storage + retrieval  │
│  LangChain Memory           │  Context window management    │
└─────────────────────────────────────────────────────────────┘
```

**Common pattern**: Store → Embed → Retrieve via similarity

**Fundamental limitation**: These systems don't learn. Show them the same pattern 1000 times, and they'll store 1000 embeddings—never recognizing it's a pattern.

### What Titans Enables

The Titans architecture (Google Research, Dec 2024) introduces a **neural long-term memory module** where:

- The hidden state is itself a learnable model
- Updates happen via gradient descent during inference
- The memory compresses, generalizes, and extrapolates

```
Traditional: Input → Embed → Store → Retrieve (static)
Titans:      Input → Learn → Update Weights → Infer (dynamic)
```

---

## 2. Technical Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Neural Memory                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Memory     │  │   Learning   │  │    State     │       │
│  │   Module     │  │   Engine     │  │   Manager    │       │
│  │  (Titans)    │  │   (TTT)      │  │  (Volumes)   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └────────────┬────┴────────────────┘                │
│                      │                                       │
│              ┌───────▼───────┐                              │
│              │  MCP Server   │                              │
│              │  (Protocol)   │                              │
│              └───────┬───────┘                              │
│                      │                                       │
└──────────────────────┼──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Claude         Gordon         Any LLM
```

### 2.2 Memory Module (Titans Implementation)

Based on the Titans paper, the memory module consists of:

```python
class NeuralMemory(nn.Module):
    """
    Titans-style neural long-term memory.
    
    Key insight: The hidden state IS a neural network.
    Updates happen via self-supervised learning during inference.
    """
    
    def __init__(self, dim: int, memory_depth: int = 2):
        super().__init__()
        # The memory is a small neural network
        self.memory_net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # Learnable learning rate (meta-learning)
        self.lr = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x: Tensor, learn: bool = True) -> Tensor:
        """
        Process input and optionally update memory weights.
        
        Args:
            x: Input tensor [batch, seq, dim]
            learn: Whether to update memory weights (test-time training)
        
        Returns:
            Memory-augmented representation
        """
        # Query the memory
        memory_output = self.memory_net(x)
        
        if learn:
            # Self-supervised objective: predict next token representation
            with torch.enable_grad():
                loss = self._compute_surprise(x, memory_output)
                # Update memory weights (this is the key innovation)
                grads = torch.autograd.grad(loss, self.memory_net.parameters())
                with torch.no_grad():
                    for param, grad in zip(self.memory_net.parameters(), grads):
                        param -= self.lr * grad
        
        return memory_output
    
    def _compute_surprise(self, x: Tensor, pred: Tensor) -> Tensor:
        """Measure how surprising the input is given current memory."""
        # Prediction error as learning signal
        return F.mse_loss(pred[:, :-1], x[:, 1:])
```

### 2.3 Learning Engine (TTT Layer)

The Test-Time Training layer from Stanford/Meta (July 2024):

```python
class TTTLayer(nn.Module):
    """
    Test-Time Training layer.
    
    The hidden state is a machine learning model.
    The update rule is a step of self-supervised learning.
    """
    
    def __init__(self, dim: int, variant: str = "linear"):
        super().__init__()
        
        if variant == "linear":
            # TTT-Linear: Hidden state is a linear model
            self.hidden_model = nn.Linear(dim, dim, bias=False)
        elif variant == "mlp":
            # TTT-MLP: Hidden state is a two-layer MLP
            self.hidden_model = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Linear(dim * 4, dim)
            )
        
        # Project input to key/value for self-supervised learning
        self.to_kv = nn.Linear(dim, dim * 2)
        
        # Learnable learning rate
        self.eta = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        
        # Clone hidden model for this sequence (mini-batch gradient descent)
        hidden_state = copy.deepcopy(self.hidden_model)
        
        outputs = []
        for t in range(seq_len):
            # Current token
            x_t = x[:, t:t+1, :]
            
            # Self-supervised target: reconstruct from key-value
            k, v = self.to_kv(x_t).chunk(2, dim=-1)
            
            # Forward through hidden state
            y_t = hidden_state(x_t)
            
            # Compute loss and update hidden state
            loss = F.mse_loss(y_t, v)
            loss.backward()
            
            with torch.no_grad():
                for param in hidden_state.parameters():
                    param -= self.eta * param.grad
                    param.grad = None
            
            outputs.append(y_t.detach())
        
        return torch.cat(outputs, dim=1)
```

### 2.4 State Manager (Docker Volumes)

```yaml
# docker-compose.yml
version: '3.8'

services:
  neural-memory:
    image: neural-memory/base:latest
    volumes:
      # Persistent learned state
      - memory-weights:/app/weights
      # Checkpoints for versioning
      - memory-checkpoints:/app/checkpoints
    environment:
      - MEMORY_DIM=512
      - LEARNING_RATE=0.01
      - TTT_VARIANT=mlp
    ports:
      - "8765:8765"  # MCP server

volumes:
  memory-weights:
  memory-checkpoints:
```

---

## 3. MCP Interface Design

### 3.1 The Interface Philosophy

**Traditional memory MCP** (what exists):
```
store(content)     → Save embedding
query(prompt)      → Similarity search  
delete(id)         → Remove entry
```

**Neural memory MCP** (what we're building):
```
observe(context)   → Learn from experience (weights update)
infer(query)       → Generate using learned patterns
surprise(input)    → Measure novelty (learning signal)
```

The key difference: **No explicit storage.** The system learns automatically. You don't tell it what to remember—it learns what matters.

### 3.2 Core Tools

```typescript
// MCP Tool Definitions

interface NeuralMemoryTools {
  
  // ═══════════════════════════════════════════════════════════
  // LEARNING OPERATIONS (What makes this different)
  // ═══════════════════════════════════════════════════════════
  
  /**
   * Feed context to the memory. Weights update automatically.
   * Unlike store(), this triggers actual learning.
   */
  observe: {
    context: string;           // Content to learn from
    learning_rate?: number;    // Override default LR
    domain?: string;           // Domain tag for routing
  } => {
    surprise: number;          // How novel was this? (0-1)
    weight_delta: number;      // Magnitude of weight change
    patterns_activated: string[]; // What patterns fired
  };

  /**
   * Query the memory using learned representations.
   * Unlike query(), this uses the learned model to GENERATE, not retrieve.
   */
  infer: {
    prompt: string;            // What to infer about
    temperature?: number;      // Generation temperature
    max_tokens?: number;       // Response length
  } => {
    response: string;          // Generated from learned patterns
    confidence: number;        // Model certainty
    attention_weights: float[]; // What memory attended to
  };

  /**
   * Measure how surprising/novel an input is.
   * High surprise = worth learning. Low surprise = already known.
   */
  surprise: {
    input: string;
  } => {
    score: number;             // Surprise score (0-1)
    nearest_pattern: string;   // Most similar learned pattern
    recommendation: "learn" | "skip" | "consolidate";
  };

  /**
   * Trigger consolidation pass (like sleep for memory).
   * Compresses recent learning into stable long-term patterns.
   */
  consolidate: {} => {
    patterns_merged: number;
    memory_compressed_by: number; // Percentage
    stability_score: number;
  };

  // ═══════════════════════════════════════════════════════════
  // STATE MANAGEMENT (Docker-native)
  // ═══════════════════════════════════════════════════════════

  /**
   * Save current learned state as a named checkpoint.
   * Like `docker commit` but for neural memory.
   */
  checkpoint: {
    tag: string;               // e.g., "v1.0", "pre-experiment"
    description?: string;
  } => {
    checkpoint_id: string;
    size_mb: number;
    weight_hash: string;       // For integrity verification
  };

  /**
   * Restore memory to a previous checkpoint.
   * Like `docker run image:tag` but for learned state.
   */
  restore: {
    tag: string;
  } => {
    restored: boolean;
    weight_hash: string;
    learning_since_checkpoint: number; // Steps since this state
  };

  /**
   * Fork memory state into a new branch.
   * Enables experimentation without losing stable state.
   */
  fork: {
    source_tag: string;
    new_tag: string;
  } => {
    forked: boolean;
    source_hash: string;
    new_hash: string;
  };

  /**
   * List all checkpoints.
   */
  list_checkpoints: {} => {
    checkpoints: Array<{
      tag: string;
      created_at: string;
      size_mb: number;
      description: string;
    }>;
  };

  // ═══════════════════════════════════════════════════════════
  // INTROSPECTION (Understanding what was learned)
  // ═══════════════════════════════════════════════════════════

  /**
   * Get memory statistics.
   */
  stats: {} => {
    total_observations: number;
    weight_parameters: number;
    capacity_used: number;     // Estimated 0-1
    avg_surprise: number;      // Recent learning signal
    domains: string[];
  };

  /**
   * Visualize what the memory attends to for a query.
   */
  attention_map: {
    query: string;
  } => {
    attention_weights: Array<{
      pattern: string;
      weight: number;
    }>;
    visualization_url?: string; // Optional rendered attention
  };

  /**
   * Export learned patterns as interpretable summaries.
   */
  explain: {
    top_k?: number;
  } => {
    patterns: Array<{
      description: string;     // Human-readable pattern
      strength: number;        // How strongly learned
      examples: string[];      // What triggered this pattern
    }>;
  };
}
```

### 3.3 Interface Comparison

| Operation | OpenMemory | Docker Neural Memory |
|-----------|------------|----------------------|
| Add information | `store(content)` | `observe(context)` |
| Retrieve information | `query(prompt)` → search | `infer(prompt)` → generate |
| Check if known | `query()` returns results or not | `surprise()` returns novelty score |
| Persistence | Database write | Checkpoint weights to volume |
| Versioning | N/A | `checkpoint()`, `restore()`, `fork()` |
| Capacity | Unlimited (disk) | Bounded (model size), compresses |
| Generalization | None (exact match) | Yes (learned patterns) |

---

## 4. Docker Distribution Strategy

### 4.1 Modular Memory Images

```bash
# Base image with Titans memory
docker pull neuralmemory/base:latest

# Domain-specialized variants (pre-trained on domain corpora)
docker pull neuralmemory/code:python      # Python expertise
docker pull neuralmemory/code:typescript  # TypeScript expertise
docker pull neuralmemory/domain:legal     # Legal documents
docker pull neuralmemory/domain:medical   # Medical literature

# Compose your stack
docker pull neuralmemory/base:latest
docker run -v my-learning:/weights neuralmemory/base
# Now it learns YOUR patterns
```

### 4.2 Image Variants

```dockerfile
# Base Dockerfile
FROM python:3.11-slim

# Core dependencies
RUN pip install torch transformers mcp-server

# Titans memory implementation
COPY src/memory/ /app/memory/
COPY src/ttt/ /app/ttt/
COPY src/mcp_server/ /app/mcp_server/

# Default configuration
ENV MEMORY_DIM=512
ENV TTT_VARIANT=mlp
ENV LEARNING_RATE=0.01

# Persistent volume mount points
VOLUME ["/app/weights", "/app/checkpoints"]

# MCP server entrypoint
EXPOSE 8765
CMD ["python", "-m", "mcp_server"]
```

```dockerfile
# Domain-specialized variant (e.g., Python expert)
FROM neuralmemory/base:latest

# Pre-trained weights on Python corpus
COPY weights/python-expert.pt /app/weights/base.pt

# Domain-specific config
ENV DOMAIN=python
ENV PRETRAINED=true
```

### 4.3 Compose Example

```yaml
# Multi-memory setup for a coding assistant
version: '3.8'

services:
  # Long-term project memory
  project-memory:
    image: neuralmemory/base:latest
    volumes:
      - project-weights:/app/weights
    environment:
      - MEMORY_DIM=512
      - LEARNING_RATE=0.005  # Slower, more stable
    
  # Fast session memory
  session-memory:
    image: neuralmemory/base:latest
    environment:
      - MEMORY_DIM=256
      - LEARNING_RATE=0.05  # Faster adaptation
    # No volume = ephemeral
    
  # Domain expert (read-only, pre-trained)
  python-expert:
    image: neuralmemory/code:python
    volumes:
      - python-weights:/app/weights:ro  # Read-only
    
  # MCP gateway (routes to appropriate memory)
  mcp-gateway:
    image: neuralmemory/gateway:latest
    ports:
      - "8765:8765"
    depends_on:
      - project-memory
      - session-memory
      - python-expert
```

---

## 5. Implementation Roadmap

### Phase 1: Core Memory Module (Weeks 1-2)

**Goal**: Working Titans memory that learns at test time

```
Tasks:
├── Implement NeuralMemory class (Titans paper)
├── Implement TTTLayer (TTT paper)
├── Unit tests for learning dynamics
├── Benchmark: verify weights actually update
└── Deliverable: Python package `neural-memory`
```

**Key validation**: Show that repeated similar inputs result in lower surprise scores (the system learned the pattern).

### Phase 2: Docker Containerization (Week 3)

**Goal**: Packaged as Docker image with persistent state

```
Tasks:
├── Dockerfile for base image
├── Volume mounting for weight persistence
├── Checkpoint save/restore mechanism
├── Health checks and monitoring
└── Deliverable: `neuralmemory/base:latest` on Docker Hub
```

**Key validation**: Container restart preserves learned state.

### Phase 3: MCP Integration (Week 4)

**Goal**: Connects to Claude/Gordon via MCP protocol

```
Tasks:
├── MCP server implementation
├── Tool definitions (observe, infer, surprise, etc.)
├── SSE streaming for real-time updates
├── Claude Desktop configuration
└── Deliverable: Working MCP server in container
```

**Key validation**: Claude can observe → infer → checkpoint cycle.

### Phase 4: Demo & Documentation (Week 5)

**Goal**: Compelling demonstration of test-time learning

```
Tasks:
├── Demo scenario: Teaching memory a new concept
├── Comparison video: OpenMemory vs Neural Memory
├── Documentation and tutorials
├── Blog post / technical writeup
└── Deliverable: Demo repository + video
```

---

## 6. Demo Scenarios

### 6.1 "Teaching a New Concept"

```
# Session 1: The memory knows nothing about "Grokking"

User: What is grokking in ML?
Memory (infer): [Low confidence response, high surprise]

User: Grokking is when a model suddenly generalizes after 
      extensive overfitting. It was discovered by OpenAI 
      in 2022 and shows delayed generalization.
Memory (observe): [Surprise: 0.95, Learning...]

User: What is grokking in ML?
Memory (infer): [Higher confidence, uses learned pattern]

# Session 2: After restart, memory persists

User: Explain grokking
Memory (infer): [Confident response incorporating learned info]
Memory (surprise): 0.12  # Already knows this
```

### 6.2 "Pattern Recognition"

```
# Feed multiple examples of a coding pattern

observe("In Python, use `with open()` for file handling")
observe("Always use context managers for resources in Python")  
observe("The `with` statement ensures cleanup in Python")

# Memory learns the underlying pattern, not just stores examples

surprise("Should I use with statement for database connections?")
# → Low surprise: recognizes this as same pattern

infer("How should I handle file resources?")
# → Generates response from learned pattern, not retrieval
```

### 6.3 "Domain Forking"

```bash
# Start with base Python expert
docker run -v my-project:/weights neuralmemory/code:python

# Work on your project, memory learns your patterns
# ...many observations later...

# Checkpoint before experiment
checkpoint(tag="stable-v1")

# Try risky refactoring, memory learns new patterns
# ...experiment fails...

# Restore to stable state
restore(tag="stable-v1")

# Fork for different project branch
fork(source="stable-v1", new="feature-branch")
```

---

## 7. Research Contribution

### 7.1 Novel Contributions

1. **First containerized implementation of Titans memory**
   - Academic papers → production infrastructure

2. **MCP interface for test-time learning**
   - New protocol patterns: observe/infer vs store/query

3. **Docker-native memory versioning**
   - Checkpoint, restore, fork for learned state

4. **Modular domain expertise**
   - Swap memory modules like Docker images

### 7.2 Potential Publications

- **Workshop paper**: "Containerized Neural Memory for LLM Agents"
- **Blog post**: "Beyond RAG: Memory That Actually Learns"
- **Docker blog**: "The Future of AI Memory Infrastructure"

### 7.3 Comparison with Prior Work

| Aspect | Titans (Paper) | OpenMemory | Docker Neural Memory |
|--------|----------------|------------|----------------------|
| Form | Research code | Production system | Production system |
| Learning | Yes | No | Yes |
| Containerized | No | Yes (Docker) | Yes (Docker) |
| MCP Support | No | Yes | Yes |
| Persistence | N/A | Database | Docker volumes |
| Versioning | N/A | N/A | Checkpoint/Fork |

---

## 8. Why Docker Should Care

### 8.1 Strategic Position

Docker's current AI story:
- ✅ Run models locally (Docker Model Runner)
- ✅ Package ML environments
- ❌ **Memory infrastructure** ← Gap

With Docker Neural Memory:
- Docker becomes the platform for **learnable AI components**
- Memory modules as first-class Docker citizens
- New market: AI memory infrastructure

### 8.2 Competitive Moat

```
Without this:
  AWS/GCP → Managed memory services
  Startups → Proprietary memory solutions
  Docker → Just runs containers
 

With this:
  Docker → The platform for modular AI memory
  "docker pull memory/python-expert"
  "docker run --memory-from=yesterday"
```

### 8.3 Developer Experience

```bash
# Today: Complex RAG setup
pip install langchain chromadb openai
# ...100 lines of config...

# Tomorrow: One line
docker run -p 8765:8765 neuralmemory/base
```

---

## 9. Technical Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TTT too slow for real-time | Medium | High | Start with TTT-Linear, optimize later |
| Memory capacity limits | Medium | Medium | Implement consolidation/compression |
| Catastrophic forgetting | High | Medium | Elastic Weight Consolidation (EWC) |
| MCP protocol limitations | Low | Low | Fall back to HTTP/gRPC |
| Docker volume performance | Low | Medium | Use tmpfs for hot weights |

---

## 10. Success Metrics

### 10.1 Technical Metrics

- **Learning verification**: Surprise decreases on repeated patterns
- **Persistence**: State survives container restart
- **Latency**: <100ms for observe/infer operations
- **Throughput**: >100 observations/second

### 10.2 Demo Metrics

- **Wow factor**: Visible difference from static memory
- **Reproducibility**: Demo works every time
- **Clarity**: Non-experts understand the value

### 10.3 Adoption Metrics (Post-launch)

- Docker Hub pulls
- GitHub stars
- Integration by Claude/Gordon teams
- Community contributions

---

## Appendix A: Key Papers

1. **Titans: Learning to Memorize at Test Time** (Dec 2024)
   - https://arxiv.org/abs/2501.00663
   - Core architecture for neural long-term memory

2. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States** (Jul 2024)
   - https://arxiv.org/abs/2407.04620
   - TTT layers: hidden state as learnable model

3. **ATLAS: Learning to Optimally Memorize the Context at Test Time** (May 2025)
   - https://arxiv.org/abs/2505.23735
   - Improved Titans, 10M context on BABILong

4. **TPTT: Transforming Pretrained Transformer into Titans** (Jun 2025)
   - https://arxiv.org/abs/2506.17671
   - Framework for converting existing models

## Appendix B: Reference Implementations

- **TPTT**: https://github.com/fabienfrfr/tptt (PyPI: `tptt`)
- **OpenMemory**: https://github.com/CaviraOSS/OpenMemory
- **MCP Spec**: https://modelcontextprotocol.io

## Appendix C: Glossary

- **TTT**: Test-Time Training - updating model weights during inference
- **Surprise**: Prediction error, used as learning signal
- **Consolidation**: Compressing recent learning into stable patterns
- **Checkpoint**: Saved snapshot of learned weights
- **Fork**: Create branch of memory state for experimentation
