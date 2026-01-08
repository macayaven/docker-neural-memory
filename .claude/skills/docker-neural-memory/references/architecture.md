# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Neural Memory                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │   Memory Module  │  │  Learning Engine │  │ State Manager │  │
│  │    (Titans)      │  │     (TTT)        │  │  (Volumes)    │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           │                     │                    │           │
│           └──────────┬──────────┴────────────────────┘           │
│                      │                                           │
│              ┌───────▼───────┐                                   │
│              │  MCP Server   │                                   │
│              │  (Protocol)   │                                   │
│              └───────┬───────┘                                   │
│                      │                                           │
└──────────────────────┼──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Claude         Gordon         Any LLM
```

## Component Breakdown

### 1. Memory Module (Titans Architecture)

The core innovation from Google's Titans paper.

**Key insight**: The hidden state IS a neural network, not a fixed vector.

```python
class NeuralMemory:
    """
    Hidden state = learnable neural network
    Update rule = gradient descent during inference
    """
    
    memory_net: nn.Sequential  # The actual memory (weights)
    lr: nn.Parameter          # Meta-learned learning rate
```

**Properties**:
- Fixed parameter count (bounded capacity)
- Compresses information into weights
- Generalizes beyond training examples
- Weights update during `observe()`

### 2. Learning Engine (TTT Layer)

From Stanford/Meta's "Learning to Learn at Test Time" paper.

**Two variants**:

| Variant | Hidden State | Expressiveness | Speed |
|---------|--------------|----------------|-------|
| TTT-Linear | Linear layer | Lower | Faster |
| TTT-MLP | 2-layer MLP | Higher | Slower |

**Self-supervised objective**: Predict next token representation.
```python
loss = MSE(predicted, target)  # Surprise signal
grad = autograd(loss)          # Compute gradients
weights -= lr * grad           # Update (TTT!)
```

### 3. State Manager (Docker Volumes)

Maps Docker concepts to neural memory:

| Docker Concept | Neural Memory |
|----------------|---------------|
| `docker commit` | `checkpoint(tag)` |
| `docker run image:tag` | `restore(tag)` |
| `docker tag` | `fork(source, new)` |
| Volume mount | Weight persistence |

**Storage structure**:
```
/app/
├── weights/           # Current learned state
│   └── memory.pt      # PyTorch state dict
└── checkpoints/       # Saved versions
    ├── v1.0.pt
    ├── stable.pt
    └── checkpoints.json  # Metadata
```

### 4. MCP Server

Exposes neural memory to LLMs via Model Context Protocol.

**Tools** (what makes this different from static memory):

| Tool | Purpose | Unlike Static Memory |
|------|---------|---------------------|
| `observe()` | Learn from content | `store()` just saves |
| `infer()` | Generate from learned | `query()` just searches |
| `surprise()` | Measure novelty | N/A in static |
| `consolidate()` | Compress patterns | N/A in static |
| `checkpoint()` | Save learned state | Save database rows |
| `restore()` | Load learned state | Load database rows |
| `fork()` | Branch learned state | N/A in static |

## Data Flow

### Observe (Learning)

```
User: "Python uses indentation for blocks"
        │
        ▼
┌───────────────┐
│ Text Encoder  │ → Embedding
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Memory Net    │ → Prediction
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Compute Loss  │ → Surprise Score
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Backprop +    │ → Gradients
│ Weight Update │
└───────────────┘
        │
        ▼
    Weights Changed!
```

### Infer (Generation)

```
User: "What syntax does Python use?"
        │
        ▼
┌───────────────┐
│ Text Encoder  │ → Query Embedding
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Memory Net    │ → Response Embedding
│  (learned)    │     (from patterns)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Confidence    │ → Certainty Score
│ Estimation    │
└───────────────┘
        │
        ▼
    Response + Confidence
```

## Key Differentiators

### vs. RAG / Vector Search

| Aspect | RAG | Neural Memory |
|--------|-----|---------------|
| Storage | Embeddings in DB | Neural weights |
| Retrieval | Similarity search | Forward pass |
| Capacity | Unlimited | Bounded, compresses |
| Learning | None | Test-time training |
| Generalization | None | Yes |
| Output | Retrieved chunks | Generated response |

### vs. OpenMemory

| Aspect | OpenMemory | Neural Memory |
|--------|------------|---------------|
| Architecture | Graph + embeddings | Titans + TTT |
| Decay model | Salience decay | Weight consolidation |
| Memory type | Episodic/semantic sectors | Learned patterns |
| Learns | No | Yes (weights update) |
| Surprise | N/A | Core metric |

## Scalability Considerations

### Memory Size

```
Parameters = dim × 4 × dim + dim × dim
           = 5 × dim²

For dim=512:  ~1.3M parameters (~5MB)
For dim=1024: ~5.2M parameters (~20MB)
For dim=2048: ~21M parameters (~80MB)
```

### Throughput

- `observe()`: ~10-50ms (includes backward pass)
- `infer()`: ~1-5ms (forward only)
- `surprise()`: ~1-5ms (forward only)
- `checkpoint()`: ~100ms (disk I/O)

### Concurrent Users

With proper batching, single instance can handle:
- 100+ observe/s
- 1000+ infer/s

For higher scale: horizontal scaling with shared checkpoints.

## Security Model

1. **Container isolation**: Memory runs in isolated container
2. **Volume permissions**: Weights only writable by container
3. **API authentication**: Optional API key for MCP
4. **No data exfiltration**: Weights don't contain raw text

## Future Extensions

1. **Multi-head memory**: Separate memory modules for different domains
2. **Hierarchical memory**: Short-term → long-term consolidation
3. **Federated learning**: Learn across instances without sharing data
4. **Memory merging**: Combine knowledge from multiple trained memories
