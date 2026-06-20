#!/usr/bin/env python3
"""
NVIDIA inference server for ONNX models.

This variant keeps the runtime focused on two shipped models:
- Alibaba-NLP/gte-modernbert-base for embeddings
- cross-encoder/ettin-reranker-17m-v1 for reranking
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NVIDIA Inference Server (ONNX)")

EMBED_MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
RERANK_MODEL_ID = "cross-encoder/ettin-reranker-17m-v1"

MODEL_ROOT = Path(os.getenv("NVIDIA_MODEL_ROOT", "/app/models"))
EMBED_MODEL_PATH = Path(os.getenv("EMBED_MODEL_PATH", str(MODEL_ROOT / "gte-modernbert-base")))
RERANK_MODEL_PATH = Path(os.getenv("RERANK_MODEL_PATH", str(MODEL_ROOT / "ettin-reranker-17m-v1")))
EMBED_MAX_LENGTH = int(os.getenv("NVIDIA_EMBED_MAX_LENGTH", "8192"))
RERANK_MAX_LENGTH = int(os.getenv("NVIDIA_RERANK_MAX_LENGTH", "7999"))

embedder = None
reranker = None
available_providers: list[str] = []
runtime_device = "unknown"


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


class OpenAIEmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = EMBED_MODEL_ID


class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    raw_scores: bool = False


@dataclass(frozen=True)
class RerankerHead:
    dense_weight: np.ndarray
    norm_weight: np.ndarray
    norm_bias: np.ndarray
    score_weight: np.ndarray
    score_bias: np.ndarray


def _load_onnxruntime():
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"[NVIDIA] preload_dlls skipped: {exc}", flush=True)
    return ort


def _load_tokenizer(model_path: Path):
    from transformers import PreTrainedTokenizerFast

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer_file = model_path / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

    config: dict[str, Any] = {}
    tokenizer_config = model_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))

    tokenizer_kwargs: dict[str, Any] = {"tokenizer_file": str(tokenizer_file)}
    for key in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"):
        if isinstance(config.get(key), str):
            tokenizer_kwargs[key] = config[key]
    for key in ("model_max_length", "padding_side", "truncation_side", "clean_up_tokenization_spaces"):
        if key in config:
            tokenizer_kwargs[key] = config[key]

    return PreTrainedTokenizerFast(**tokenizer_kwargs)


def _load_session(ort, model_path: Path):
    onnx_path = model_path / "onnx" / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. "
            f"Available providers: {ort.get_available_providers()}"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def _ort_inputs(session, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    allowed = {item.name for item in session.get_inputs()}
    return {name: np.asarray(value) for name, value in batch.items() if name in allowed}


def _load_reranker_head(model_path: Path) -> RerankerHead:
    from safetensors import safe_open

    dense_path = model_path / "2_Dense" / "model.safetensors"
    norm_path = model_path / "3_LayerNorm" / "model.safetensors"
    score_path = model_path / "4_Dense" / "model.safetensors"

    with safe_open(str(dense_path), framework="np") as handle:
        dense_weight = handle.get_tensor("linear.weight").astype(np.float32)

    with safe_open(str(norm_path), framework="np") as handle:
        norm_weight = handle.get_tensor("norm.weight").astype(np.float32)
        norm_bias = handle.get_tensor("norm.bias").astype(np.float32)

    with safe_open(str(score_path), framework="np") as handle:
        score_weight = handle.get_tensor("linear.weight").astype(np.float32)
        score_bias = handle.get_tensor("linear.bias").astype(np.float32)

    return RerankerHead(
        dense_weight=dense_weight,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        score_weight=score_weight,
        score_bias=score_bias,
    )


def _gelu(values: np.ndarray) -> np.ndarray:
    scaled = values / math.sqrt(2.0)
    erf = np.vectorize(math.erf, otypes=[np.float32])(scaled)
    return 0.5 * values * (1.0 + erf)


def _linear(inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    output = inputs @ weight.T
    if bias is not None:
        output = output + bias
    return output.astype(np.float32)


def _layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    centered = values - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered / np.sqrt(variance + eps)
    return (normalized * weight) + bias


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class OnnxEmbedder:
    def __init__(self, model_path: Path, *, max_length: int = EMBED_MAX_LENGTH):
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = _load_tokenizer(model_path)
        self.session = _load_session(_load_onnxruntime(), model_path)
        self.providers = self.session.get_providers()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        outputs = self.session.run(None, _ort_inputs(self.session, batch))[0]
        embeddings = outputs[:, 0,].astype(np.float32)
        return embeddings.tolist()


class OnnxReranker:
    def __init__(self, model_path: Path, *, max_length: int = RERANK_MAX_LENGTH):
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = _load_tokenizer(model_path)
        self.session = _load_session(_load_onnxruntime(), model_path)
        self.head = _load_reranker_head(model_path)
        self.providers = self.session.get_providers()

    def score(self, query: str, texts: Sequence[str], *, raw_scores: bool = False) -> list[dict[str, float]]:
        if not texts:
            return []

        batch = self.tokenizer(
            [query] * len(texts),
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        hidden = self.session.run(None, _ort_inputs(self.session, batch))[0][:, 0, :].astype(np.float32)
        hidden = _linear(hidden, self.head.dense_weight)
        hidden = _gelu(hidden)
        hidden = _layer_norm(hidden, self.head.norm_weight, self.head.norm_bias)
        hidden = _linear(hidden, self.head.score_weight, self.head.score_bias).reshape(-1)
        scores = hidden if raw_scores else _sigmoid(hidden)

        ranked = [{"index": idx, "score": float(score)} for idx, score in enumerate(scores)]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked


@app.on_event("startup")
async def load_models():
    global embedder, reranker, available_providers, runtime_device

    print("=" * 60, flush=True)
    print("NVIDIA Inference Server (ONNX)", flush=True)
    print(f"  Embedder: {EMBED_MODEL_PATH}", flush=True)
    print(f"  Reranker: {RERANK_MODEL_PATH}", flush=True)
    print("=" * 60, flush=True)

    ort = _load_onnxruntime()
    available_providers = ort.get_available_providers()
    print(f"[NVIDIA] Available providers: {available_providers}", flush=True)
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. "
            f"Available providers: {available_providers}"
        )

    embedder = OnnxEmbedder(EMBED_MODEL_PATH)
    print("Embedding model loaded", flush=True)
    reranker = OnnxReranker(RERANK_MODEL_PATH)
    print("Reranker model loaded", flush=True)
    runtime_device = "cuda" if any(
        "CUDAExecutionProvider" in provider
        for provider in getattr(embedder, "providers", []) + getattr(reranker, "providers", [])
    ) else "cpu"
    if runtime_device != "cuda":
        raise RuntimeError(
            "NVIDIA runtime fell back to CPU. "
            f"Embedder providers={getattr(embedder, 'providers', [])}, "
            f"reranker providers={getattr(reranker, 'providers', [])}"
        )
    print("Server ready!", flush=True)


@app.post("/v1/chat/completions")
async def chat_completions(_: ChatCompletionRequest):
    raise HTTPException(
        status_code=503,
        detail="The NVIDIA image does not ship a local chat model. Use the external judge or LLM service.",
    )


@app.post("/v1/embeddings")
async def embeddings(request: OpenAIEmbeddingRequest):
    try:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        vectors = embedder.encode(inputs)
        data = [
            {
                "object": "embedding",
                "embedding": vector,
                "index": idx,
            }
            for idx, vector in enumerate(vectors)
        ]
        return {
            "object": "list",
            "data": data,
            "model": request.model or EMBED_MODEL_ID,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    except Exception as exc:
        print(f"Embedding error: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/rerank")
async def rerank(request: RerankRequest):
    try:
        return reranker.score(request.query, request.texts, raw_scores=request.raw_scores)
    except Exception as exc:
        print(f"Rerank error: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    loaded = embedder is not None and reranker is not None
    return {
        "status": "ok" if loaded else "unhealthy",
        "device": runtime_device,
        "providers": available_providers,
    }


@app.get("/info")
async def info():
    return {
        "model_id": EMBED_MODEL_ID,
        "model_type": "onnx",
        "device": runtime_device,
        "dimension": 768,
        "reranker_model_id": RERANK_MODEL_ID,
        "providers": available_providers,
    }


if __name__ == "__main__":
    import uvicorn

    print("NVIDIA Inference Server starting on port 3002...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=3002)
