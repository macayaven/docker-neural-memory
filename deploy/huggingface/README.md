---
title: Docker Neural Memory
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Docker Neural Memory

**Memory that LEARNS, not just stores**

This demo showcases containerized neural memory using Google's Titans architecture (Dec 2024). Unlike RAG/vector databases that just store and retrieve embeddings, this system's **weights actually update during inference**.

## Key Features

- **Real Learning**: Weights change on every `observe()` call
- **Pattern Recognition**: Surprise decreases as patterns are learned
- **Bounded Capacity**: Fixed parameter count (doesn't grow like vector DBs)
- **Docker-Native**: Designed for containerized deployment with persistent volumes

## How It Works

```
Traditional Memory:  Input → Embed → Store → Retrieve (static)
Neural Memory:       Input → Learn → Update Weights → Infer (dynamic)
```

## Demo Tabs

1. **Chat with Advocate**: Ask about the project and the developer
2. **Live Demo**: Watch weights change and surprise decrease
3. **Interactive**: Try observing your own content
4. **About Carlos**: Meet the developer

## Built By

**Carlos Crespo Macaya** - AI Engineer specializing in GenAI Systems & Applied MLOps

- 10+ years production ML experience
- Expert in Docker, Kubernetes, MCP servers
- Currently at HP AICoE building multi-agent systems

Contact: macayaven@gmail.com

## Links

- [GitHub Repository](https://github.com/macayaven/docker-neural-memory)
- [Technical Specification](https://github.com/macayaven/docker-neural-memory/blob/main/SPEC.md)
- [Titans Paper](https://arxiv.org/abs/2501.00663)
