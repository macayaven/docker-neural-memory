---
title: Docker Neural Memory
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Docker Neural Memory

**Real Test-Time Training - Not a Simulation**

This demo runs **actual PyTorch** code implementing Google's Titans architecture. When you observe content, real gradients flow and real neural network weights update.

## What Makes This Real

- **Real Neural Network**: 2-layer MLP with ~250K parameters
- **Real Gradient Descent**: `torch.autograd.grad()` computes gradients
- **Real Weight Updates**: Parameters physically change during inference
- **Real Surprise Metric**: MSE loss measures prediction error

## Docker-Native Design

This project demonstrates production-grade AI infrastructure:

- **MCP Server**: Model Context Protocol for Claude Desktop integration
- **Docker Volumes**: Persist learned state across container restarts
- **CI/CD Pipeline**: GitHub Actions with Docker build and deploy
- **Kubernetes Ready**: Designed for orchestrated deployment

## Key Features

| Feature | Implementation |
|---------|---------------|
| Test-Time Training | PyTorch autograd during inference |
| State Persistence | Docker volumes for checkpoints |
| MCP Integration | Tools: observe, surprise, checkpoint, restore |
| Bounded Memory | Fixed parameters (doesn't grow like vector DBs) |

## Built By

**Carlos Crespo Macaya** - AI Engineer

- 10+ years production ML experience
- Expert in Docker, Kubernetes, MCP servers
- Currently at HP AICoE building multi-agent systems

Contact: macayaven@gmail.com

## Links

- [GitHub Repository](https://github.com/macayaven/docker-neural-memory)
- [Titans Paper](https://arxiv.org/abs/2501.00663)
