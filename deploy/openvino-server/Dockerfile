FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Install runtime dependencies and register official Intel Graphics repository for GPU acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified' > /etc/apt/sources.list.d/intel-graphics.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        intel-opencl-icd \
        clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install OpenVINO GenAI & FastAPI pipeline packages
RUN pip install --no-cache-dir \
    openvino==2026.2.0 \
    openvino-genai==2026.2.0 \
    fastapi \
    uvicorn \
    pydantic \
    numpy

# Pre-bake the compiled models directly into the image to eliminate volume mount latencies
COPY models/gte-modernbert-ov /app/models/gte-modernbert-ov
COPY models/ettin-17m-ov /app/models/ettin-17m-ov

# Copy the server.py source code
COPY server.py .

# Configure default environment variables to point directly to baked-in model paths
ENV EMBED_MODEL_PATH=/app/models/gte-modernbert-ov
ENV RERANK_MODEL_PATH=/app/models/ettin-17m-ov
ENV LLM_MODEL_PATH=""
ENV CACHE_DIR=/app/models/model_cache

EXPOSE 3002

# Run OpenVINO Inference Server
CMD ["python", "server.py"]
