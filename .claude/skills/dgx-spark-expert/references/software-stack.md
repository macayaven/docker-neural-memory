# DGX Spark Software Stack

## DGX OS

Ubuntu-based Linux distribution optimized for AI workloads.

**Included**:
- NVIDIA drivers (pre-installed)
- CUDA toolkit
- cuDNN
- TensorRT
- Docker with NVIDIA Container Runtime
- JupyterLab
- NVIDIA AI Workbench

**Update commands**:
```bash
# Check for updates via DGX Dashboard (recommended)
# Or manually:
sudo apt update && sudo apt upgrade

# Check DGX OS version
cat /etc/dgx-release
```

## NVIDIA Sync

Desktop app for remote access setup.

**Features**:
- Configure SSH access
- Setup local network connectivity
- Works from Mac/Windows/Linux client

**Usage**: Download from https://build.nvidia.com/spark/connect-to-your-spark

## DGX Dashboard

Web-based system management at `http://localhost:8888` (or remote IP).

**Features**:
- System monitoring (CPU, GPU, memory, storage)
- JupyterLab launcher
- System updates
- Container management

**Access**:
```bash
# Local
http://localhost:8888

# Remote (replace with Spark's IP)
http://<spark-ip>:8888
```

## Container Runtime

Docker with NVIDIA runtime pre-configured.

**Run GPU containers**:
```bash
# Basic GPU access
docker run --gpus all <image>

# Full GPU + runtime
docker run --gpus all --runtime nvidia <image>

# With shared memory (for PyTorch, etc.)
docker run --gpus all --shm-size=16g <image>

# Mount home directory
docker run --gpus all -v $HOME:/workspace <image>
```

**NGC Container Registry**:
```bash
# Login to NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-ngc-api-key>

# Pull NGC containers
docker pull nvcr.io/nvidia/pytorch:24.01-py3
docker pull nvcr.io/nvidia/tensorflow:24.01-tf2-py3
```

## NGC CLI (ARM64)

Command-line access to NGC resources.

**Installation** (must use ARM64 version):
```bash
# Download ARM64 version
wget https://ngc.nvidia.com/downloads/ngccli_arm64.zip
unzip ngccli_arm64.zip
chmod +x ngc-cli/ngc

# Add to PATH
export PATH=$PATH:$HOME/ngc-cli

# Configure
ngc config set
```

## NVIDIA AI Workbench

Containerized development environment for AI applications.

**Features**:
- Project-based development
- Pre-built templates (RAG, agents, etc.)
- Git integration
- Environment management

**Key workflows**:
- Clone pre-built projects
- Run reproducible AI applications
- Develop RAG systems
- Build multi-agent chatbots

## Pre-installed Frameworks

| Framework | Notes |
|-----------|-------|
| PyTorch | GPU-accelerated, ARM64 optimized |
| TensorFlow | GPU support |
| CUDA | Full toolkit |
| cuDNN | Deep learning primitives |
| TensorRT | Inference optimization |
| RAPIDS | GPU data science (cuDF, cuML) |

## Inference Servers

**vLLM** (recommended for LLMs):
```bash
# Via pip
pip install vllm

# Via Docker
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-chat-hf
```

**TensorRT-LLM** (optimized inference):
```bash
# Use NGC container
docker pull nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3
```

**Ollama** (simple local LLMs):
```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Run model
ollama run llama3.2
```

## Development Tools

**VS Code Remote**:
1. Install VS Code on client machine
2. Install "Remote - SSH" extension
3. Connect: `ssh user@<spark-ip>`

**JupyterLab**:
- Access via DGX Dashboard
- Or run manually: `jupyter lab --ip=0.0.0.0`

## Common Environment Variables

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# HuggingFace cache
export HF_HOME=/home/$USER/.cache/huggingface

# PyTorch memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Software Downloads

Official downloads: https://developer.nvidia.com/topics/ai/dgx-spark

Includes:
- System recovery media
- DGX OS updates
- NVIDIA Sync app
