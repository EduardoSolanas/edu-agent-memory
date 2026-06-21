FROM python:3.11-slim-bookworm

ARG SYSTEM=intel
ARG HF_TOKEN
ARG HUGGING_FACE_HUB_TOKEN

ENV PYTHONUNBUFFERED=1

# Base system utilities shared by both images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Qdrant is shared by both runtime variants.
RUN curl -L -o /tmp/qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.10.1/qdrant-x86_64-unknown-linux-gnu.tar.gz \
    && tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin/ qdrant \
    && rm /tmp/qdrant.tar.gz \
    && chmod +x /usr/local/bin/qdrant

# Node is used by the api daemon in both variants.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Shared Python runtime dependencies.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic

WORKDIR /app

# Split the image at build time.
RUN set -eux; \
    if [ "$SYSTEM" = "intel" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends gnupg; \
        rm -rf /var/lib/apt/lists/*; \
        curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg; \
        echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified' > /etc/apt/sources.list.d/intel-graphics.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            ocl-icd-libopencl1 \
            intel-opencl-icd \
            clinfo; \
        rm -rf /var/lib/apt/lists/*; \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch; \
        pip install --no-cache-dir \
            openvino==2026.2.0 \
            openvino-genai==2026.2.0 \
            openvino-tokenizers==2026.2.0.0 \
            optimum-intel[openvino]; \
    else \
        apt-get update; \
        apt-get install -y --no-install-recommends libgomp1; \
        rm -rf /var/lib/apt/lists/*; \
        pip install --no-cache-dir \
            onnxruntime-gpu==1.22.0 \
            transformers==4.48.0 \
            numpy \
            safetensors \
            nvidia-cuda-runtime-cu12==12.4.127 \
            nvidia-cuda-nvrtc-cu12==12.4.99 \
            nvidia-cublas-cu12==12.4.5.8 \
            nvidia-curand-cu12==10.3.4.107 \
            nvidia-cufft-cu12==11.0.12.1 \
            nvidia-cusolver-cu12==11.6.4.69 \
            nvidia-cusparse-cu12==12.4.1.24 \
            nvidia-cusparselt-cu12==0.7.1 \
            nvidia-nvjitlink-cu12==12.4.99 \
            nvidia-cudnn-cu12==9.4.0.58; \
    fi

# Intel builds export the OpenVINO embedding and reranker models into the image.
# NVIDIA builds export the ONNX embedder and reranker assets into the image.
COPY bin /app/bin
RUN set -eux; \
    if [ "$SYSTEM" = "intel" ]; then \
        HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" python3 /app/bin/prepare_models.py \
            && test -d /app/models/gte-modernbert-ov \
            && test -d /app/models/ettin-17m-ov; \
    elif [ "$SYSTEM" = "nvidia" ]; then \
        HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" python3 /app/bin/prepare_nvidia_models.py \
            && test -d /app/models/gte-modernbert-base \
            && test -d /app/models/ettin-reranker-17m-v1; \
    fi

# Copy only the runtime application sources.
COPY server.py /app/server.py
COPY server_nvidia.py /app/server_nvidia.py
COPY entrypoint.sh /app/entrypoint.sh

RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh
ENV CACHE_DIR=/app/models/model_cache

EXPOSE 6333 6334 3002 6336

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
