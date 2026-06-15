# Base image for stable OpenVINO Python bindings
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Layer 1: Core System Utilities and Build Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    procps \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Register Intel Graphics Repository and Install iGPU Runtimes (OpenCL/ICD)
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified' > /etc/apt/sources.list.d/intel-graphics.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        intel-opencl-icd \
        clinfo \
    && rm -rf /var/lib/apt/lists/*

# Layer 3: Install Node.js v20 (Strictly standard library, no dependencies)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Layer 4: Download and Install Qdrant Statically-Compiled Binary
RUN curl -L -o /tmp/qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.10.1/qdrant-x86_64-unknown-linux-gnu.tar.gz \
    && tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin/ qdrant \
    && rm /tmp/qdrant.tar.gz \
    && chmod +x /usr/local/bin/qdrant

# Layer 5: Upgrade Pip to ensure fast installation
RUN pip install --no-cache-dir --upgrade pip

# Layer 6: Install Heavy OpenVINO Engine Dependencies (Maximize Layer Caching)
RUN pip install --no-cache-dir \
    openvino==2026.2.0 \
    openvino-genai==2026.2.0 \
    openvino-tokenizers==2026.2.0.0

# Layer 7: Install Lightweight Server Dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    numpy

# Set working directory
WORKDIR /app

# Layer 8: Explicitly Copy Large Baked-In Model Weights (Avoids rebuilding when code changes)
COPY models/gte-modernbert-ov /app/models/gte-modernbert-ov
COPY models/ettin-17m-ov /app/models/ettin-17m-ov

# Layer 9: Copy Application Source Files and Folders explicitly to avoid copying models or virtual environments
COPY bin /app/bin
COPY benchmarks /app/benchmarks
COPY tests /app/tests
COPY data /app/data
COPY data_hf /app/data_hf
COPY docs /app/docs
COPY server.py /app/server.py
COPY package.json /app/package.json
COPY entrypoint.sh /app/entrypoint.sh

# Ensure entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# Environment Configurations
ENV EMBED_MODEL_PATH=/app/models/gte-modernbert-ov
ENV RERANK_MODEL_PATH=/app/models/ettin-17m-ov
ENV LLM_MODEL_PATH=""
ENV CACHE_DIR=/app/models/model_cache

# Expose required ports:
# - Qdrant: 6333 (HTTP), 6334 (gRPC)
# - OpenVINO Server: 3002
# - Node.js api-daemon: 6336
EXPOSE 6333 6334 3002 6336

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
