from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_nvidia_dependencies_are_build_time_isolated():
    dockerfile = _read("Dockerfile")
    intel_branch, nvidia_branch = dockerfile.split('    else \\\n', 1)
    nvidia_branch = nvidia_branch.split('    fi\n', 1)[0]

    assert "openvino==" in intel_branch
    assert "onnxruntime-gpu" not in intel_branch
    assert "transformers==4.48.0" not in intel_branch
    assert "llama-cpp-python" not in nvidia_branch
    assert "onnxruntime-gpu==1.22.0" in nvidia_branch
    assert "transformers==4.48.0" in nvidia_branch
    assert "nvidia-cuda-runtime-cu12==12.4.127" in nvidia_branch
    assert "nvidia-cuda-nvrtc-cu12==12.4.99" in nvidia_branch
    assert "nvidia-cublas-cu12==12.4.5.8" in nvidia_branch
    assert "nvidia-curand-cu12==10.3.4.107" in nvidia_branch
    assert "nvidia-cufft-cu12==11.0.12.1" in nvidia_branch
    assert "nvidia-cusolver-cu12==11.6.4.69" in nvidia_branch
    assert "nvidia-cusparse-cu12==12.4.1.24" in nvidia_branch
    assert "nvidia-cusparselt-cu12==0.7.1" in nvidia_branch
    assert "nvidia-nvjitlink-cu12==12.4.99" in nvidia_branch
    assert "nvidia-cudnn-cu12==9.4.0.58" in nvidia_branch
    assert "openvino==" not in nvidia_branch
    assert "ENV EMBED_MODEL_PATH=" not in dockerfile
    assert "ENV RERANK_MODEL_PATH=" not in dockerfile


def test_runtime_model_and_gpu_settings_are_mode_isolated():
    entrypoint = _read("entrypoint.sh")
    intel_branch, nvidia_and_rest = entrypoint.split(
        'elif [ "${SYSTEM_MODE}" = "nvidia" ]; then', 1
    )
    nvidia_branch = nvidia_and_rest.split("else\n", 1)[0]

    assert "openvino" in intel_branch.lower()
    assert "LD_LIBRARY_PATH" not in intel_branch
    assert "NVIDIA_EMBED_MODEL_PATH" in nvidia_branch
    assert "NVIDIA_RERANK_MODEL_PATH" in nvidia_branch
    assert "LD_LIBRARY_PATH" in nvidia_branch
    assert "/usr/lib/wsl/lib" not in nvidia_branch
    assert "gte-modernbert-base" in nvidia_branch
    assert "ettin-reranker-17m-v1" in nvidia_branch
    assert "gguf" not in nvidia_branch.lower()
    assert "cuda_nvrtc/lib" in nvidia_branch
    assert "curand/lib" in nvidia_branch
    assert "cufft/lib" in nvidia_branch
    assert "cusolver/lib" in nvidia_branch
    assert "cusparse/lib" in nvidia_branch
    assert "cusparselt/lib" in nvidia_branch
    assert "nvjitlink/lib" in nvidia_branch


def test_launcher_keeps_intel_and_nvidia_devices_separate():
    launcher = _read("run.sh")

    assert "RUN_ARGS+=(--device /dev/dri:/dev/dri)" in launcher
    assert "RUN_ARGS+=(--device nvidia.com/gpu=all)" in launcher
    assert "RUN_ARGS+=(--device /dev/dxg)" not in launcher
    assert "RUN_ARGS+=(--gpus all)" in launcher


def test_nvidia_model_prep_targets_exact_repos():
    script = _read("bin/prepare_nvidia_models.py")

    assert "Alibaba-NLP/gte-modernbert-base" in script
    assert "cross-encoder/ettin-reranker-17m-v1" in script
    assert "onnx/model.onnx" in script
    assert "2_Dense/model.safetensors" in script
