# Dockerfile for HuggingFace Spaces
# Deploy: https://huggingface.co/spaces

FROM python:3.11-slim

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first for caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=user src/ ./src/
COPY --chown=user deploy/huggingface/app.py .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Run Gradio app
CMD ["python", "app.py"]
