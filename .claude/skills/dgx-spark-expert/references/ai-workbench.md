# NVIDIA AI Workbench on DGX Spark

## Overview

AI Workbench is NVIDIA's containerized development platform for building and deploying AI applications. It provides reproducible environments for RAG systems, multi-agent workflows, and model development.

## Key Features

- **Containerized**: Full isolation and reproducibility
- **Git-integrated**: Version control for projects
- **GPU-accelerated**: Direct access to Blackwell GPU
- **Pre-built projects**: Clone and run immediately
- **UMA-optimized**: Benefits from unified memory architecture

## Installation

AI Workbench comes pre-installed on DGX Spark. Verify:
```bash
nvwb --version
```

## Creating a New Project

### From Template
```bash
# List available templates
nvwb project list-templates

# Create from template
nvwb project create --template <template-name> --name my-project

# Clone existing project
nvwb project clone <git-url>
```

### Start Project
```bash
# Enter project directory
cd my-project

# Start the workbench environment
nvwb start

# Access web interface (default port 8080)
```

## Available Project Templates

| Template | Description |
|----------|-------------|
| RAG | Retrieval-Augmented Generation with evaluation |
| Hybrid RAG | Multi-source RAG with routing |
| Multi-Agent | Agentic chatbot systems |
| Fine-tuning | Model customization workflows |

## RAG Application Workflow

The official playbook: https://build.nvidia.com/spark/rag-ai-workbench

**Steps**:
1. Clone the RAG project
2. Configure inference endpoint (NVIDIA API or local)
3. Start the application
4. Access Gradio interface
5. Submit queries and evaluate responses

**RAG Features**:
- Query routing
- Response evaluation (relevancy, hallucination)
- Iterative refinement
- Document ingestion and chunking
- Vector embedding generation

## Integration with Inference Engines

AI Workbench projects can use any inference backend:

| Backend | Use Case |
|---------|----------|
| NVIDIA API | Cloud-hosted models, easiest setup |
| vLLM | Local high-throughput serving |
| TRT-LLM | Local optimized inference |
| Ollama | Simple local model serving |
| NIM | NVIDIA optimized microservices |

**Example vLLM integration**:
```python
# In AI Workbench project
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Docker Pattern

AI Workbench uses standard Docker:
```bash
docker run --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/project:/workspace \
  -p 8080:8080 \
  --runtime nvidia \
  <ai-workbench-image>
```

## Access Methods

- **Local**: `http://localhost:8080`
- **Remote**: `http://<spark-ip>:8080`
- **Via Tailscale**: Access from anywhere

## UMA Memory Benefits

AI Workbench on DGX Spark benefits from Unified Memory:
- Dynamic CPU/GPU memory allocation
- Larger model capacity (128GB shared)
- Simplified memory management
- No explicit transfers needed

## Comparison with Other Tools

| Tool | Best For |
|------|----------|
| AI Workbench | Application-oriented, production RAG, agents |
| JupyterLab | Exploratory data science, notebooks |
| VS Code | Traditional IDE development |
| Vibe Coding | AI-assisted coding with Continue |

## Common Commands

```bash
# Project management
nvwb project list
nvwb project start <name>
nvwb project stop <name>

# Environment
nvwb env list
nvwb env activate <env>

# Logs
nvwb logs
```

## Rollback/Cleanup

To remove a project:
```bash
# Stop the project
nvwb project stop <name>

# Delete project files
rm -rf ~/nvwb-projects/<name>
```

No system changes are made outside the AI Workbench environment.

## Resources

- **Playbook**: https://build.nvidia.com/spark/rag-ai-workbench
- **Multi-Agent**: https://build.nvidia.com/spark/multi-agent-chatbot
- **Main docs**: https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/
