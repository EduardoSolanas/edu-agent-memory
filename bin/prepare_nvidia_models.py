#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path("/app")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

EMBED_MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
EMBED_MODEL_DIR = MODELS_DIR / "gte-modernbert-base"
EMBED_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "1_Pooling/config.json",
    "onnx/model.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

RERANK_MODEL_ID = "cross-encoder/ettin-reranker-17m-v1"
RERANK_MODEL_DIR = MODELS_DIR / "ettin-reranker-17m-v1"
RERANK_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_LayerNorm/config.json",
    "3_LayerNorm/model.safetensors",
    "4_Dense/config.json",
    "4_Dense/model.safetensors",
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
]


def load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    for env_path in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key.strip() in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} and value:
                return value
    return None


def _download_files(repo_id: str, target_dir: Path, files: list[str], hf_token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)

    for file_name in files:
        target_path = target_dir / file_name
        if target_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id}:{file_name} -> {target_path}", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )


def _is_populated(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NVIDIA ONNX models into the image.")
    parser.add_argument("--force", action="store_true", help="Re-download models even if they already exist.")
    args = parser.parse_args()

    hf_token = load_hf_token()
    if hf_token:
        print("HF token loaded.")
    else:
        print("HF token not found; public files will be downloaded without auth.")

    need_embed = args.force or not _is_populated(EMBED_MODEL_DIR)
    need_rerank = args.force or not _is_populated(RERANK_MODEL_DIR)

    if not need_embed and not need_rerank:
        print("Both NVIDIA model directories already exist.")
        return

    if args.force:
        shutil.rmtree(EMBED_MODEL_DIR, ignore_errors=True)
        shutil.rmtree(RERANK_MODEL_DIR, ignore_errors=True)

    if need_embed:
        _download_files(EMBED_MODEL_ID, EMBED_MODEL_DIR, EMBED_FILES, hf_token)
    else:
        print(f"Skipping embedder download: {EMBED_MODEL_DIR}")

    if need_rerank:
        _download_files(RERANK_MODEL_ID, RERANK_MODEL_DIR, RERANK_FILES, hf_token)
    else:
        print(f"Skipping reranker download: {RERANK_MODEL_DIR}")

    print("NVIDIA model preparation completed successfully.")


if __name__ == "__main__":
    main()
