# Docker Neural Memory - Implementation Plan

> **Derived from**: SPEC.md
> **Approach**: Strict TDD (Test-Driven Development)
> **Status**: Planning

---

## Vision

Build the first production-ready containerized implementation of **test-time training memory** based on Google's Titans architecture. Unlike existing "memory" solutions (RAG, vector DBs) that just store and retrieve, this system **actually learns** during inference.

```
Traditional Memory:  Input → Embed → Store → Retrieve (static)
Neural Memory:       Input → Learn → Update Weights → Infer (dynamic)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Neural Memory                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│   │    Encoder     │   │  Neural Memory │   │     State      │  │
│   │   (Text→Tensor)│   │   (Titans)     │   │   Manager      │  │
│   └───────┬────────┘   └───────┬────────┘   └───────┬────────┘  │
│           │                    │                    │           │
│           └────────────┬───────┴────────────────────┘           │
│                        │                                        │
│                ┌───────▼───────┐                               │
│                │  MCP Server   │                               │
│                │  (Protocol)   │                               │
│                └───────┬───────┘                               │
│                        │                                        │
└────────────────────────┼────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
     Claude          Gordon         Any LLM
```

---

## Implementation Phases

### Phase 1: Core Memory Module
**Goal**: Working Titans memory that learns at test time

| Component | Description | Validation |
|-----------|-------------|------------|
| `NeuralMemory` | Titans memory with gradient-based updates | Weights change on `observe()` |
| `TTTLinear` | Linear hidden state variant | Processes sequences |
| `TTTMLP` | MLP hidden state variant | More expressive |
| `Encoder` | Text → Tensor conversion | Deterministic, reversible |

**Key Test**: Surprise decreases on repeated patterns

### Phase 2: State Management
**Goal**: Persistence and versioning of learned state

| Component | Description | Validation |
|-----------|-------------|------------|
| `CheckpointManager` | Save/restore weight snapshots | State survives restart |
| `VersionManager` | Fork/branch operations | Can rollback experiments |
| `MetadataStore` | Track observations, domains | Stats are accurate |

**Key Test**: Container restart preserves learning

### Phase 3: Consolidation
**Goal**: Memory compression and forgetting prevention

| Component | Description | Validation |
|-----------|-------------|------------|
| `Consolidator` | EWC-based pattern compression | Patterns stabilize |
| `CapacityEstimator` | Track memory utilization | Warns before saturation |
| `PatternExtractor` | Identify learned patterns | Explainability |

**Key Test**: Old knowledge retained after new learning

### Phase 4: MCP Interface
**Goal**: Connect to Claude/Gordon via MCP protocol

| Component | Description | Validation |
|-----------|-------------|------------|
| `MCPServer` | Protocol implementation | Tools register correctly |
| `ToolHandlers` | 11 tool implementations | Each tool works |
| `ErrorHandling` | Graceful failure modes | Errors are informative |

**Key Test**: Claude can observe → infer → checkpoint cycle

### Phase 5: Demo & Polish
**Goal**: Compelling demonstration of value

| Component | Description | Validation |
|-----------|-------------|------------|
| `KillerDemo` | 2-minute wow factor script | Non-experts understand |
| `Benchmarks` | Latency/throughput metrics | <100ms observe/infer |
| `Documentation` | Usage guides, API docs | Complete and accurate |

---

## Component Dependencies

```
Phase 1 (Core)
    │
    ├── Encoder (no deps)
    │
    ├── NeuralMemory (depends on Encoder)
    │
    └── TTTLayer (no deps, alternative to NeuralMemory internals)

Phase 2 (State)
    │
    ├── CheckpointManager (depends on NeuralMemory)
    │
    └── VersionManager (depends on CheckpointManager)

Phase 3 (Consolidation)
    │
    └── Consolidator (depends on NeuralMemory)

Phase 4 (MCP)
    │
    └── MCPServer (depends on all above)

Phase 5 (Demo)
    │
    └── Everything integrated
```

---

## Success Criteria

### Technical Metrics
- [ ] **Learning**: Surprise decreases >50% after 3 observations of same pattern
- [ ] **Persistence**: State identical after container restart
- [ ] **Latency**: <100ms for observe/infer operations
- [ ] **Throughput**: >100 observations/second
- [ ] **Capacity**: Handles 10,000+ observations without degradation

### Demo Metrics
- [ ] **Weights Change**: Visibly different hash before/after observe
- [ ] **Pattern Recognition**: Novel questions get low surprise if related
- [ ] **Bounded Capacity**: Parameter count unchanged after 1000 observations
- [ ] **Persistence**: Knowledge survives Docker restart

---

## Development Environment

```bash
# Build development container
docker compose -f docker-compose.dev.yml build

# Start development shell
docker compose -f docker-compose.dev.yml run --rm dev

# Run tests (TDD workflow)
docker compose -f docker-compose.dev.yml run --rm test

# Run demo
docker compose -f docker-compose.dev.yml run --rm demo
```

---

## File Structure

```
docker-neural-memory/
├── src/
│   ├── __init__.py
│   ├── config.py              # Pydantic settings
│   ├── encoder/
│   │   ├── __init__.py
│   │   └── text_encoder.py    # Text → Tensor
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── neural_memory.py   # Titans memory
│   │   ├── ttt_layer.py       # TTT-Linear/MLP
│   │   └── consolidation.py   # EWC, compression
│   ├── state/
│   │   ├── __init__.py
│   │   ├── checkpoint.py      # Save/restore
│   │   └── versioning.py      # Fork/branch
│   └── mcp_server/
│       ├── __init__.py
│       ├── server.py          # MCP protocol
│       └── tools.py           # Tool definitions
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Fixtures
│   ├── unit/
│   │   ├── test_encoder.py
│   │   ├── test_neural_memory.py
│   │   ├── test_ttt_layer.py
│   │   ├── test_checkpoint.py
│   │   └── test_consolidation.py
│   └── integration/
│       ├── test_persistence.py
│       ├── test_mcp_tools.py
│       └── test_demo_scenarios.py
├── demo/
│   ├── killer_demo.py         # Main demo script
│   └── scenarios/             # Demo scenarios
├── PLAN.md                    # This file
├── TASKS.md                   # TDD task breakdown
├── SPEC.md                    # Technical specification
├── CLAUDE.md                  # Claude Code guidance
├── pyproject.toml
├── Dockerfile
├── Dockerfile.dev
├── docker-compose.yml
└── docker-compose.dev.yml
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TTT too slow | Medium | High | Start with TTT-Linear, profile early |
| Catastrophic forgetting | High | Medium | EWC from Phase 3, test extensively |
| Memory saturation | Medium | Medium | Capacity estimation, consolidation |
| MCP compatibility | Low | Medium | Test with Claude Desktop early |
| Encoding quality | Medium | Medium | Multiple encoder options |

---

## Deployment Strategy

### HuggingFace Spaces (Primary Demo)

**URL**: `https://huggingface.co/spaces/macayaven/docker-neural-memory`

**Purpose**: Shareable demo for recruiters with advocate agent

**Components**:
- `deploy/huggingface/app.py` - Gradio interface with 4 tabs:
  1. **Chat with Advocate** - Recruiter-focused conversational agent
  2. **Live Demo** - Watch weights change, surprise decrease
  3. **Interactive** - Try observe/surprise yourself
  4. **About Carlos** - Background and qualifications
- `deploy/huggingface/Dockerfile` - HF Spaces container
- `deploy/huggingface/requirements.txt` - Minimal dependencies
- `deploy/huggingface/README.md` - Space description

**Deploy**:
```bash
# Create new Space on HuggingFace
# Upload files from deploy/huggingface/
# Space will build and deploy automatically
```

### Docker Hub (Production)

**Image**: `neuralmemory/base:latest`

**Purpose**: Production deployment with MCP interface

```bash
# Build and push
docker build -t neuralmemory/base:latest .
docker push neuralmemory/base:latest

# Run locally
docker run -p 8765:8765 -v memory:/app/weights neuralmemory/base
```

### Demo Executables Location

```
deploy/
├── huggingface/           # HuggingFace Spaces (recruiter demo)
│   ├── app.py             # Gradio app with advocate agent
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
└── docker/                # Docker Hub (production)
    └── (uses root Dockerfile)
```

---

## Recruiter Advocate Agent

The HF Spaces demo includes a conversational agent that:

1. **Explains the project** - What Docker Neural Memory does and why it matters
2. **Pitches Carlos** - Qualifications, experience, and fit for Docker
3. **Demonstrates value** - Shows real learning in action
4. **Handles objections** - Answers questions about experience/skills

**Key talking points**:
- 10+ years production ML with Docker/Kubernetes
- Currently building MCP servers at HP AICoE
- Bridges research (Titans papers) and production (this project)
- Track record: medical AI, RAG systems, multi-agent apps

---

## Next Steps

1. Review TASKS.md for detailed TDD implementation plan
2. Start with Phase 1, Task 1.1 (Encoder tests)
3. Build development container
4. Begin TDD cycle: Red → Green → Refactor
5. Deploy to HuggingFace Spaces for recruiter demos
