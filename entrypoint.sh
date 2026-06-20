#!/bin/bash
set -e

# Detect system mode from environment
SYSTEM_MODE="${SYSTEM:-intel}"

# If a custom command is passed to docker run, execute that instead of starting the default stack
if [ "$#" -gt 0 ] && [ "$1" != "bash" ] && [ "$1" != "/bin/bash" ]; then
    echo "[Entrypoint] Running custom command: $@"
    exec "$@"
fi

echo "[Entrypoint] System mode: ${SYSTEM_MODE}"

# Default entrypoint behavior:
echo "[Entrypoint] Creating Qdrant storage directory..."
mkdir -p /app/qdrant_storage

echo "[Entrypoint] Starting Qdrant in the background on port 6333..."
export QDRANT__STORAGE__STORAGE_PATH=/app/qdrant_storage
/usr/local/bin/qdrant &

echo "[Entrypoint] Waiting 3 seconds for Qdrant to boot..."
sleep 3

if [ "${SYSTEM_MODE}" = "intel" ]; then
    echo "[Entrypoint] Intel mode: Starting OpenVINO Inference Server..."
    export EMBED_MODEL_PATH=/app/models/gte-modernbert-ov
    export RERANK_MODEL_PATH=/app/models/ettin-17m-ov
    export LLM_MODEL_PATH=""
    python server.py &
elif [ "${SYSTEM_MODE}" = "nvidia" ]; then
    echo "[Entrypoint] NVIDIA mode: Starting ONNX Runtime Inference Server..."
    export LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export NVIDIA_EMBED_MODEL_PATH="${NVIDIA_EMBED_MODEL_PATH:-/app/models/gte-modernbert-base}"
    export NVIDIA_RERANK_MODEL_PATH="${NVIDIA_RERANK_MODEL_PATH:-/app/models/ettin-reranker-17m-v1}"
    export EMBED_MODEL_PATH="${NVIDIA_EMBED_MODEL_PATH}"
    export RERANK_MODEL_PATH="${NVIDIA_RERANK_MODEL_PATH}"
    python server_nvidia.py &
else
    echo "[Entrypoint] Unknown system mode: ${SYSTEM_MODE}. Exiting."
    exit 1
fi

echo "[Entrypoint] Waiting 5 seconds for inference server to boot..."
sleep 5

echo "[Entrypoint] Checking running processes..."
ps aux

echo "[Entrypoint] Starting Node.js api-daemon in the foreground on port 6336..."
exec node bin/api-daemon.mjs
