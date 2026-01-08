# DGX Spark Playbooks Index

Official playbooks at https://build.nvidia.com/spark and https://github.com/NVIDIA/dgx-spark-playbooks

## Quick Start Playbooks

| Playbook | Time | Description |
|----------|------|-------------|
| [VS Code](https://build.nvidia.com/spark/vscode) | 5 min | Install and use VS Code locally or remotely |
| [DGX Dashboard](https://build.nvidia.com/spark/dgx-dashboard) | 30 min | Monitor system and launch JupyterLab |
| [Open WebUI + Ollama](https://build.nvidia.com/spark/open-webui) | 15 min | Chat interface with local models |
| [Set Up Local Network Access](https://build.nvidia.com/spark/connect-to-your-spark) | 5 min | Configure SSH access via NVIDIA Sync |

## Inference Playbooks

| Playbook | Time | Description |
|----------|------|-------------|
| [vLLM](https://build.nvidia.com/spark/vllm) | 30 min | High-throughput inference with PagedAttention |
| [TRT-LLM](https://build.nvidia.com/spark/trt-llm) | 1 hr | TensorRT-LLM for optimized inference |
| [SGLang](https://build.nvidia.com/spark/sglang) | 30 min | SGLang inference server |
| [NIM on Spark](https://build.nvidia.com/spark/nim-llm) | 30 min | Deploy NVIDIA NIM microservices |
| [Speculative Decoding](https://build.nvidia.com/spark/speculative-decoding) | 30 min | Fast inference with draft models |
| [Multi-modal Inference](https://build.nvidia.com/spark/multi-modal-inference) | 1 hr | Vision-language models with TensorRT |
| [NVFP4 Quantization](https://build.nvidia.com/spark/nvfp4-quantization) | 1 hr | Quantize models to NVFP4 for Spark |

## Fine-tuning Playbooks

| Playbook | Time | Description |
|----------|------|-------------|
| [Fine-tune with NeMo](https://build.nvidia.com/spark/nemo-fine-tune) | 1 hr | NVIDIA NeMo framework |
| [Fine-tune with PyTorch](https://build.nvidia.com/spark/pytorch-fine-tune) | 1 hr | Native PyTorch fine-tuning |
| [LLaMA Factory](https://build.nvidia.com/spark/llama-factory) | 1 hr | Easy fine-tuning with LLaMA Factory |
| [Unsloth](https://build.nvidia.com/spark/unsloth) | 1 hr | Optimized fine-tuning with Unsloth |
| [FLUX.1 Dreambooth LoRA](https://build.nvidia.com/spark/flux-finetuning) | 1 hr | Image generation model fine-tuning |

## AI Workbench & Development

| Playbook | Time | Description |
|----------|------|-------------|
| [RAG in AI Workbench](https://build.nvidia.com/spark/rag-ai-workbench) | 30 min | Agentic RAG with AI Workbench |
| [Vibe Coding in VS Code](https://build.nvidia.com/spark/vibe-coding) | 30 min | AI coding assistant with Ollama + Continue |
| [Multi-Agent Chatbot](https://build.nvidia.com/spark/multi-agent-chatbot) | 1 hr | Deploy multi-agent system |

## Multi-Node & Clustering

| Playbook | Time | Description |
|----------|------|-------------|
| [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) | 1 hr | Setup two-node cluster |
| [NCCL for Two Sparks](https://build.nvidia.com/spark/nccl) | 30 min | GPU-to-GPU communication test |

## Data Science & Frameworks

| Playbook | Time | Description |
|----------|------|-------------|
| [CUDA-X Data Science](https://build.nvidia.com/spark/cuda-x-data-science) | 30 min | cuML, cuDF acceleration |
| [Optimized JAX](https://build.nvidia.com/spark/jax) | 2 hrs | JAX optimization for Spark |

## Image Generation

| Playbook | Time | Description |
|----------|------|-------------|
| [Comfy UI](https://build.nvidia.com/spark/comfy-ui) | 45 min | Node-based image generation |

## Utilities & Networking

| Playbook | Time | Description |
|----------|------|-------------|
| [Tailscale](https://build.nvidia.com/spark/tailscale) | 30 min | Remote access anywhere |
| [Text to Knowledge Graph](https://build.nvidia.com/spark/txt2kg) | 30 min | LLM-powered graph extraction |
| [Video Search & Summarization](https://build.nvidia.com/spark/vss) | 1 hr | VSS Blueprint on Spark |

## Playbook Selection Guide

**For inference:**
- Quick chat interface → Open WebUI + Ollama
- Production serving → vLLM or TRT-LLM
- NVIDIA optimized → NIM
- Multi-modal → Multi-modal Inference

**For fine-tuning:**
- Enterprise/production → NeMo
- Quick experiments → LLaMA Factory or Unsloth
- Image models → FLUX Dreambooth

**For development:**
- IDE-based → VS Code
- Notebook-based → DGX Dashboard (JupyterLab)
- AI Workbench projects → RAG in AI Workbench

**For large models (>200B):**
- Connect Two Sparks + NCCL
