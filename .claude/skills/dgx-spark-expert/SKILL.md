---
name: dgx-spark-expert
description: Comprehensive expert knowledge for NVIDIA DGX Spark workstation. Use when users ask about DGX Spark hardware, software, playbooks, AI Workbench, fine-tuning, inference, troubleshooting, known issues, container workflows, multi-node setup, or any development task on DGX Spark/GB10 Grace Blackwell systems. Triggers on mentions of "DGX Spark", "GB10", "Grace Blackwell desktop", "Spark workstation", or related NVIDIA AI workstation topics.
---

# DGX Spark Expert

Expert guidance for NVIDIA DGX Spark AI workstation development.

## Quick Reference

| Resource | URL |
|----------|-----|
| Playbooks Hub | https://build.nvidia.com/spark |
| User Guide | https://docs.nvidia.com/dgx/dgx-spark/ |
| Support | https://www.nvidia.com/en-us/support/dgx-spark/ |
| Forums | https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10 |
| GitHub Playbooks | https://github.com/NVIDIA/dgx-spark-playbooks |

## Key System Facts

- **Architecture**: ARM64 (not x86) — use ARM64 binaries/containers
- **Memory**: 128GB unified (shared CPU/GPU via UMA)
- **Performance**: 1 PFLOP FP4 with sparsity
- **Max Model Size**: ~200B parameters (single), ~405B (two Sparks stacked)
- **OS**: DGX OS (Ubuntu-based with NVIDIA stack)

## Reference Files

Load these for detailed information:

- `references/hardware-specs.md` — Full specs, UMA details, Spark stacking
- `references/playbooks-index.md` — All 25+ official playbooks with links
- `references/known-issues.md` — Troubleshooting, diagnostics, support
- `references/software-stack.md` — DGX OS, containers, frameworks, tools
- `references/ai-workbench.md` — AI Workbench projects, RAG, agents

## Common Workflows

### Run Inference

**Quick local chat**: Use Ollama + Open WebUI playbook
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.2
```

**Production serving**: Use vLLM or TRT-LLM playbooks
```bash
# vLLM example
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### Fine-tune a Model

1. **Quick experiments**: LLaMA Factory or Unsloth playbooks
2. **Production**: NeMo playbook
3. **Image models**: FLUX Dreambooth playbook

See `references/playbooks-index.md` for all options.

### Create AI Workbench Project

```bash
# Clone official RAG project
nvwb project clone https://github.com/NVIDIA/workbench-example-agentic-rag

# Start project
cd workbench-example-agentic-rag
nvwb start
```

See `references/ai-workbench.md` for detailed workflow.

### Connect Two Sparks

1. Connect via QSFP/CX7 cable
2. Configure netplan on both nodes
3. Exchange SSH keys
4. Install NCCL and run tests
5. See "Connect Two Sparks" and "NCCL" playbooks

### Troubleshoot Issues

1. Check `references/known-issues.md` first
2. Common fixes:
   - Memory issues: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`
   - Check memory: `free -h` (not nvidia-smi for memory)
   - Driver issues: `sudo systemctl status nvidia-persistenced`
3. Forums: https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10

## UMA Memory Management

DGX Spark uses Unified Memory Architecture — CPU and GPU share 128GB.

**Key points**:
- `nvidia-smi` memory display may show "Not Supported" (expected)
- Use `free -h` for actual memory status
- Flush buffer cache if memory pressure: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`
- Models up to ~200B parameters can run locally

## Container Patterns

```bash
# Standard GPU container
docker run --gpus all --runtime nvidia <image>

# With shared memory (required for many ML frameworks)
docker run --gpus all --shm-size=16g <image>

# Mount HuggingFace cache
docker run --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  <image>

# NGC container example
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```

## ARM64 Compatibility

DGX Spark runs ARM64, not x86. When installing software:
- Use `aarch64` or `arm64` package versions
- NGC CLI must be ARM64 Linux version
- Some x86-only tools require alternatives or won't work
- Check NGC for ARM64-compatible containers

## Playbook Selection Guide

| Goal | Recommended Playbook |
|------|---------------------|
| Chat with local LLM | Open WebUI + Ollama |
| Serve LLM API | vLLM or TRT-LLM |
| Fine-tune LLM | LLaMA Factory (quick) or NeMo (production) |
| Build RAG app | RAG in AI Workbench |
| Generate images | Comfy UI |
| AI coding assistant | Vibe Coding |
| Remote access | Tailscale |
| Data science | CUDA-X Data Science |
| Large models (>200B) | Connect Two Sparks |

## Decision Logic

**User asks about inference** → Check model size, recommend vLLM (high throughput) or TRT-LLM (optimized latency), or Ollama for simple use.

**User asks about fine-tuning** → Assess complexity: LLaMA Factory/Unsloth for experiments, NeMo for production, PyTorch for custom needs.

**User asks about AI Workbench** → Load `references/ai-workbench.md`, guide through project creation/cloning.

**User reports error/issue** → Load `references/known-issues.md`, check if known issue, provide diagnostic commands.

**User asks about specs/capabilities** → Load `references/hardware-specs.md`, provide relevant details.

**User wants specific playbook** → Load `references/playbooks-index.md`, provide direct link and summary.
