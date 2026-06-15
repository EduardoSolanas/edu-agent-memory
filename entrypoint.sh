#!/bin/bash
set -e

# If a custom command is passed to docker run, execute that instead of starting the default stack
if [ "$#" -gt 0 ] && [ "$1" != "bash" ] && [ "$1" != "/bin/bash" ]; then
    echo "[Entrypoint] Running custom command: $@"
    exec "$@"
fi

# Default entrypoint behavior:
echo "[Entrypoint] Creating Qdrant storage directory..."
mkdir -p /app/qdrant_storage

echo "[Entrypoint] Starting Qdrant in the background on port 6333..."
export QDRANT__STORAGE__STORAGE_PATH=/app/qdrant_storage
/usr/local/bin/qdrant &

echo "[Entrypoint] Setting OpenVINO environment variables..."
export EMBED_MODEL_PATH=/app/models/gte-modernbert-ov
export RERANK_MODEL_PATH=/app/models/ettin-17m-ov
export LLM_MODEL_PATH=""

echo "[Entrypoint] Starting OpenVINO Inference Server in the background on port 3002..."
python server.py &

echo "[Entrypoint] Waiting 5 seconds for background services to boot..."
sleep 5

echo "[Entrypoint] Checking running processes..."
ps aux

echo "[Entrypoint] Starting Node.js api-daemon in the foreground on port 6336..."
exec node bin/api-daemon.mjs
