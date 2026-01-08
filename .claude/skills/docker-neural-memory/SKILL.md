---
name: docker-neural-memory
description: Build containerized neural memory that learns at test time using Titans/TTT architecture. Use when implementing learnable memory modules, test-time training systems, or MCP-connected memory services. Triggers on "neural memory", "test-time training", "Titans memory", "learnable memory", "memory that learns", "TTT layer", or building Docker-based AI memory infrastructure.
---

# Docker Neural Memory

Build containerized neural memory that **learns during inference** using Google's Titans architecture and Stanford's TTT layers.

## Core Concept

```
Traditional Memory (RAG/Vector DB):     Neural Memory (Titans/TTT):
────────────────────────────────────    ────────────────────────────────────
Input → Embed → Store → Retrieve        Input → Learn → Update Weights → Infer
        (static)                                (dynamic)

store(content)  → save embedding        observe(context) → weights update
query(prompt)   → similarity search     infer(prompt)    → generate from model
```

**Key insight**: The memory IS a neural network. Updates happen via gradient descent during inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Neural Memory                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Memory     │  │   Learning   │  │    State     │       │
│  │   Module     │  │   Engine     │  │   Manager    │       │
│  │  (Titans)    │  │   (TTT)      │  │  (Volumes)   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                           │                                  │
│                   ┌───────▼───────┐                         │
│                   │  MCP Server   │                         │
│                   └───────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Workflow

### Phase 1: Core Memory Module

Implement the Titans neural memory. See `references/titans-architecture.md` for detailed implementation.

```python
# Core structure (simplified)
class NeuralMemory(nn.Module):
    def __init__(self, dim: int):
        self.memory_net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.lr = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, x: Tensor, learn: bool = True) -> Tensor:
        output = self.memory_net(x)
        if learn:
            loss = self._compute_surprise(x, output)
            # Update weights during inference (the key innovation)
            self._update_weights(loss)
        return output
```

**Validation**: Verify surprise decreases on repeated patterns.

### Phase 2: TTT Layer Integration

Add Test-Time Training layer for expressive hidden states. See `references/titans-architecture.md`.

Two variants:
- **TTT-Linear**: Hidden state is linear model (faster)
- **TTT-MLP**: Hidden state is 2-layer MLP (more expressive)

### Phase 3: Docker Containerization

Package as Docker image with persistent learned state:

```dockerfile
FROM python:3.11-slim
RUN pip install torch transformers mcp-server

COPY src/ /app/
VOLUME ["/app/weights", "/app/checkpoints"]

ENV MEMORY_DIM=512
ENV TTT_VARIANT=mlp
ENV LEARNING_RATE=0.01

EXPOSE 8765
CMD ["python", "-m", "mcp_server"]
```

See `references/docker-patterns.md` for compose examples and multi-memory setups.

### Phase 4: MCP Interface

Implement MCP server with learning-focused tools. See `references/mcp-interface.md`.

**Core tools** (different from traditional memory):

| Tool | Purpose | Unlike Traditional |
|------|---------|-------------------|
| `observe(context)` | Learn from input | `store()` just saves |
| `infer(prompt)` | Generate from learned model | `query()` just retrieves |
| `surprise(input)` | Measure novelty | No equivalent |
| `consolidate()` | Compress patterns | No equivalent |
| `checkpoint(tag)` | Save learned state | Database backup |
| `restore(tag)` | Load previous state | Database restore |
| `fork(src, new)` | Branch memory | No equivalent |

## File Structure

```
docker-neural-memory/
├── src/
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── neural_memory.py    # Titans memory module
│   │   ├── ttt_layer.py        # TTT-Linear and TTT-MLP
│   │   └── consolidation.py    # Pattern compression
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP server implementation
│   │   └── tools.py            # Tool definitions
│   └── state/
│       ├── __init__.py
│       ├── checkpoint.py       # Save/restore weights
│       └── versioning.py       # Fork/branch logic
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── tests/
    ├── test_learning.py        # Verify weights update
    ├── test_persistence.py     # Verify state survives restart
    └── test_mcp.py             # MCP tool tests
```

## Key Implementation Details

### Surprise Calculation

```python
def compute_surprise(self, x: Tensor, pred: Tensor) -> Tensor:
    """
    Measure how surprising input is given current memory.
    High surprise = worth learning. Low surprise = already known.
    """
    return F.mse_loss(pred[:, :-1], x[:, 1:])
```

### Weight Update During Inference

```python
def _update_weights(self, loss: Tensor):
    """The key innovation: gradient descent during forward pass."""
    grads = torch.autograd.grad(loss, self.memory_net.parameters())
    with torch.no_grad():
        for param, grad in zip(self.memory_net.parameters(), grads):
            param -= self.lr * grad
```

### Catastrophic Forgetting Mitigation

Use Elastic Weight Consolidation (EWC) to protect important weights:

```python
def ewc_loss(self, current_loss: Tensor) -> Tensor:
    """Penalize changes to weights important for past patterns."""
    ewc_penalty = 0
    for name, param in self.memory_net.named_parameters():
        ewc_penalty += (self.fisher[name] * (param - self.optimal[name])**2).sum()
    return current_loss + self.ewc_lambda * ewc_penalty
```

## Testing Checklist

1. **Learning verification**
   ```python
   # Surprise should decrease on repeated patterns
   s1 = memory.observe("Python uses indentation")
   s2 = memory.observe("Python uses indentation")
   assert s2.surprise < s1.surprise
   ```

2. **Persistence verification**
   ```bash
   # Start, learn, stop, restart, verify knowledge retained
   docker run -v weights:/app/weights neuralmemory/base
   # ... observe some patterns ...
   docker stop container
   docker start container
   # ... verify low surprise on same patterns ...
   ```

3. **Generalization verification**
   ```python
   # Should generalize beyond exact matches
   memory.observe("Use with statement for files in Python")
   memory.observe("Use context managers for resources")
   result = memory.infer("How to handle database connections?")
   # Should suggest context managers (generalized pattern)
   ```

## References

- `references/titans-architecture.md` - Detailed Titans/TTT implementation
- `references/mcp-interface.md` - Complete MCP tool specifications
- `references/docker-patterns.md` - Docker compose and deployment patterns

## Key Papers

1. **Titans** (Dec 2024): https://arxiv.org/abs/2501.00663
2. **TTT Layers** (Jul 2024): https://arxiv.org/abs/2407.04620
3. **ATLAS** (May 2025): https://arxiv.org/abs/2505.23735
4. **TPTT** (Jun 2025): https://arxiv.org/abs/2506.17671 - PyPI: `tptt`

## Useful Starting Points

- **TPTT library**: `pip install tptt` - Framework for Titans conversion
- **MCP SDK**: `pip install mcp` - Model Context Protocol server
- **OpenMemory** (reference): https://github.com/CaviraOSS/OpenMemory
