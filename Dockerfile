# Base image for stable OpenVINO Python bindings
FROM python:3.11-slim-bookworm

# Install required system packages and register official Intel Graphics repository for GPU acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    procps \
    git \
    build-essential \
    && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified' > /etc/apt/sources.list.d/intel-graphics.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js v20 (Strictly standard library, no dependencies)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Download and install Qdrant statically compiled binary
RUN curl -L -o /tmp/qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.10.1/qdrant-x86_64-unknown-linux-gnu.tar.gz \
    && tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin/ qdrant \
    && rm /tmp/qdrant.tar.gz \
    && chmod +x /usr/local/bin/qdrant

# Install Python packages for OpenVINO Inference Server
RUN pip install --no-cache-dir \
    openvino==2026.2.0 \
    openvino-genai==2026.2.0 \
    fastapi \
    uvicorn \
    pydantic \
    numpy

# Set workdir
WORKDIR /app

# Pre-bake local model weights
COPY models/gte-modernbert-ov /app/models/gte-modernbert-ov
COPY models/ettin-17m-ov /app/models/ettin-17m-ov

# Copy zero-dependency source files directly
COPY . .

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables for OpenVINO Inference Server
ENV EMBED_MODEL_PATH=/app/models/gte-modernbert-ov
ENV RERANK_MODEL_PATH=/app/models/ettin-17m-ov
ENV LLM_MODEL_PATH=""

# Expose ports (Qdrant: 6333/6334, OpenVINO: 3002, api-daemon: 6336)
EXPOSE 6333 6334 3002 6336

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
