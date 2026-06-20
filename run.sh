#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# edumem Docker build/run script
# ============================================================================
# Usage:
#   ./run.sh build intel      -- build Intel (OpenVINO) image
#   ./run.sh build nvidia     -- build NVIDIA ONNX image
#   ./run.sh run intel        -- run Intel container
#   ./run.sh run nvidia       -- run NVIDIA container
#   ./run.sh stop             -- stop running container
#   ./run.sh clean            -- remove images and containers
# ============================================================================

IMAGE_NAME="edumem"
CONTAINER_NAME="edumem-container"
HF_TOKEN="${HF_TOKEN:-}"
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-}"

if [ -z "$CONTAINER_ENGINE" ]; then
    if command -v docker >/dev/null 2>&1; then
        CONTAINER_ENGINE="docker"
    elif command -v podman >/dev/null 2>&1; then
        CONTAINER_ENGINE="podman"
    else
        echo "ERROR: neither docker nor podman was found in PATH"
        exit 1
    fi
fi

if [ -z "$HF_TOKEN" ] && [ -f ".env" ]; then
    HF_TOKEN="$(grep -m1 '^HF_TOKEN=' .env | sed 's/^HF_TOKEN=//' | tr -d '\r' | sed 's/^["'\'']\(.*\)["'\'']$/\1/')"
fi
if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -f ".env" ]; then
    HUGGING_FACE_HUB_TOKEN="$(grep -m1 '^HUGGING_FACE_HUB_TOKEN=' .env | sed 's/^HUGGING_FACE_HUB_TOKEN=//' | tr -d '\r' | sed 's/^["'\'']\(.*\)["'\'']$/\1/')"
fi
if [ -z "$HF_TOKEN" ] && [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

usage() {
    echo "Usage: $0 {build|run|stop|clean} <intel|nvidia>"
    echo ""
    echo "  build intel  - Build Intel OpenVINO image"
    echo "  build nvidia - Build NVIDIA ONNX image"
    echo "  run intel    - Run Intel container (with /dev/dri mount)"
    echo "  run nvidia   - Run NVIDIA container with host GPU access"
    echo "  stop         - Stop and remove running container"
    echo "  clean        - Remove images and containers"
    exit 1
}

[ $# -lt 2 ] && usage

ACTION="$1"
SYSTEM="$2"

case "$SYSTEM" in
    intel)
        IMAGE="${IMAGE_NAME}:intel"
        ;;
    nvidia)
        IMAGE="${IMAGE_NAME}:nvidia"
        ;;
    *)
        echo "ERROR: SYSTEM must be 'intel' or 'nvidia'"
        exit 1
        ;;
esac

build_image() {
    echo "============================================"
    echo "Building ${SYSTEM} image: ${IMAGE}"
    echo "============================================"

    DOCKER_BUILD_ARGS=(
        --build-arg "SYSTEM=${SYSTEM}"
    )

    if [ -n "$HF_TOKEN" ]; then
        DOCKER_BUILD_ARGS+=(--build-arg "HF_TOKEN=${HF_TOKEN}")
        DOCKER_BUILD_ARGS+=(--build-arg "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
    fi

    case "$SYSTEM" in
        nvidia)
            ;;
    esac

    "$CONTAINER_ENGINE" build "${DOCKER_BUILD_ARGS[@]}" -t "$IMAGE" .
    echo "Image built: ${IMAGE}"
}

run_container() {
    echo "============================================"
    echo "Running ${SYSTEM} container"
    echo "============================================"

    # Stop existing container if running
    "$CONTAINER_ENGINE" rm -f "$CONTAINER_NAME" 2>/dev/null || true

    RUN_ARGS=(
        -d
        --name "$CONTAINER_NAME"
        -p 6333:6333
        -p 6334:6334
        -p 3002:3002
        -p 6336:6336
        -e "SYSTEM=${SYSTEM}"
    )

    case "$SYSTEM" in
        intel)
            RUN_ARGS+=(--device /dev/dri:/dev/dri)
            ;;
        nvidia)
            if [ "$CONTAINER_ENGINE" = "podman" ]; then
                RUN_ARGS+=(--device nvidia.com/gpu=all)
            else
                RUN_ARGS+=(--gpus all)
            fi
            ;;
    esac

    "$CONTAINER_ENGINE" run "${RUN_ARGS[@]}" "$IMAGE"
    echo "Container running: ${CONTAINER_NAME}"
    echo "Ports: 6333 (Qdrant HTTP), 6334 (Qdrant gRPC), 3002 (Inference), 6336 (API daemon)"
}

stop_container() {
    echo "Stopping container..."
    "$CONTAINER_ENGINE" rm -f "$CONTAINER_NAME" 2>/dev/null || echo "Container not found"
}

clean() {
    echo "Cleaning up..."
    "$CONTAINER_ENGINE" rm -f "$CONTAINER_NAME" 2>/dev/null || true
    "$CONTAINER_ENGINE" rmi "$IMAGE" 2>/dev/null || true
    echo "Done"
}

case "$ACTION" in
    build)  build_image ;;
    run)    run_container ;;
    stop)   stop_container ;;
    clean)  clean ;;
    *)      usage ;;
esac
