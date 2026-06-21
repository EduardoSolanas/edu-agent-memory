#!/usr/bin/env python3
"""
OpenVINO Inference Server for Hindsight
Serves embeddings, reranker, and LLM models on Intel iGPU or CPU using native openvino_genai pipelines
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openvino_genai
import openvino
import uvicorn
import time
import sys
import math
import threading
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json

from server_text import sanitize_rerank_text

EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/root/openvino-server/models/gte-modernbert-ov")
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "/root/openvino-server/models/ettin-17m-ov")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "512"))
CACHE_DIR = os.getenv("CACHE_DIR", "/root/openvino-server/models/model_cache")
RERANK_CACHE_SIZE = int(os.getenv("RERANK_CACHE_SIZE", "128"))
RERANK_QUERY_MAX_BYTES = int(os.getenv("RERANK_QUERY_MAX_BYTES", "128"))
RERANK_TEXT_MAX_BYTES = int(os.getenv("RERANK_TEXT_MAX_BYTES", "352"))
USE_HF_TOKENIZER = os.getenv("USE_HF_TOKENIZER", "0") == "1"

print("Starting OpenVINO Inference Server (Native GenAI)...", flush=True)

app = FastAPI(title="OpenVINO Inference Server (Native GenAI)")

# Models will be loaded on startup
embed_pipeline = None
rerank_pipeline = None
hf_embedder = None
hf_reranker = None
device = "CPU"

# Single GPU/CPU lock prevents embed/rerank overlap
gpu_lock = threading.Lock()
embed_lock = gpu_lock
rerank_lock = gpu_lock
rerank_cache_lock = threading.Lock()
rerank_cache = OrderedDict()

class EmbedRequest(BaseModel):
    inputs: str | List[str]
    
class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class OpenAIEmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = "sentence-transformers/all-mpnet-base-v2"
    
class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    raw_scores: bool = False
    
class RerankResponse(BaseModel):
    index: int
    score: float

def inverse_sigmoid(p: float) -> float:
    p = max(1e-15, min(1.0 - 1e-15, p))
    return math.log(p / (1.0 - p))


# ============================================================================
# HF Tokenizer + OpenVINO inference path (flag-gated, default OFF)
# ============================================================================

@dataclass(frozen=True)
class RerankerHead:
    """Loads reranker classification head from safetensors files."""
    dense_weight: 'np.ndarray'
    norm_weight: 'np.ndarray'
    norm_bias: 'np.ndarray'
    score_weight: 'np.ndarray'
    score_bias: 'np.ndarray'


def _load_hf_tokenizer(model_path):
    """Load HuggingFace fast tokenizer from model directory."""
    from transformers import PreTrainedTokenizerFast

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer_file = model_path / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

    config: dict = {}
    tokenizer_config = model_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))

    tokenizer_kwargs: dict = {"tokenizer_file": str(tokenizer_file)}
    for key in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"):
        if isinstance(config.get(key), str):
            tokenizer_kwargs[key] = config[key]
    for key in ("model_max_length", "padding_side", "truncation_side", "clean_up_tokenization_spaces"):
        if key in config:
            tokenizer_kwargs[key] = config[key]

    return PreTrainedTokenizerFast(**tokenizer_kwargs)


def _load_reranker_head(model_path):
    """Load reranker classification head from safetensors."""
    from safetensors import safe_open
    import numpy as np

    model_path = Path(model_path)
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


def _linear(inputs, weight, bias=None):
    """Linear layer: y = x @ w.T + b."""
    import numpy as np
    output = inputs @ weight.T
    if bias is not None:
        output = output + bias
    return output.astype(np.float32)


def _layer_norm(values, weight, bias, eps=1e-5):
    """Layer normalization."""
    import numpy as np
    mean = values.mean(axis=-1, keepdims=True)
    centered = values - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered / np.sqrt(variance + eps)
    return (normalized * weight) + bias


def _sigmoid(values):
    """Sigmoid activation."""
    import numpy as np
    return 1.0 / (1.0 + np.exp(-values))


def _gelu(values):
    """GELU activation using erf approximation."""
    import numpy as np
    scaled = values / math.sqrt(2.0)
    erf = np.vectorize(math.erf, otypes=[np.float32])(scaled)
    return 0.5 * values * (1.0 + erf)


def _mean_pooling_with_mask(outputs, attention_mask):
    """
    Mean pooling with attention mask for embeddings.
    outputs: [batch_size, seq_len, hidden_dim]
    attention_mask: [batch_size, seq_len]
    returns: [batch_size, hidden_dim]
    """
    import numpy as np
    # Sum over sequence dimension with masking
    mask = attention_mask.astype(np.float32)[:, :, np.newaxis]  # [batch, seq, 1]
    masked = outputs * mask  # [batch, seq, hidden]
    summed = masked.sum(axis=1)  # [batch, hidden]
    # Avoid division by zero
    lengths = mask.sum(axis=1).clip(min=1.0)  # [batch, 1]
    mean = summed / lengths  # [batch, hidden]
    return mean.astype(np.float32)


class OpenVINOHFEmbedder:
    """Embeddings with HF fast tokenizer + OpenVINO inference."""

    def __init__(self, model_path, device="CPU", max_length=8192):
        import numpy as np

        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length
        self.tokenizer = _load_hf_tokenizer(self.model_path)

        # Load compiled OpenVINO model
        core = openvino.Core()
        ov_model_path = self.model_path / "openvino_model.xml"
        if not ov_model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {ov_model_path}")

        self.compiled_model = core.compile_model(str(ov_model_path), device)
        self.infer_request = self.compiled_model.create_infer_request()

        # Inspect inputs to determine required keys
        input_names = [inp.name for inp in self.compiled_model.inputs]
        self.required_inputs = set(input_names)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Embed texts: tokenize -> infer -> mean pool."""
        import numpy as np

        if not texts:
            return []

        # Tokenize
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        # Prepare input dict: filter to only required input names
        ov_inputs = {}
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in batch and key in self.required_inputs:
                ov_inputs[key] = batch[key].astype(np.int64)

        # Run inference
        outputs = self.infer_request.infer(ov_inputs)

        # Get hidden states (usually the first/only output, shape [batch, seq, hidden])
        hidden_states = list(outputs.values())[0]

        # Mean pooling with attention mask (MUST match embed_config: MEAN, normalize=False)
        embeddings = _mean_pooling_with_mask(hidden_states, batch["attention_mask"])

        return embeddings.tolist()


class OpenVINOHFReranker:
    """Reranking with HF fast tokenizer + OpenVINO inference + external head."""

    def __init__(self, model_path, device="CPU", max_length=7999):
        import numpy as np

        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length
        self.tokenizer = _load_hf_tokenizer(self.model_path)

        # Load compiled OpenVINO model
        core = openvino.Core()
        ov_model_path = self.model_path / "openvino_model.xml"
        if not ov_model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {ov_model_path}")

        self.compiled_model = core.compile_model(str(ov_model_path), device)
        self.infer_request = self.compiled_model.create_infer_request()

        # Inspect inputs
        input_names = [inp.name for inp in self.compiled_model.inputs]
        self.required_inputs = set(input_names)

        # Inspect outputs to determine if we need external head
        output_names = [out.name for out in self.compiled_model.outputs]
        print(f"[HF-Reranker] Output names: {output_names}", flush=True)

        # Load external head (assumption: this model outputs hidden states, not logits)
        self.head = _load_reranker_head(self.model_path)

    def score(self, query: str, texts: List[str], raw_scores: bool = False) -> List[Dict[str, float]]:
        """Rerank texts for query: tokenize -> infer -> head -> sigmoid -> sort."""
        import numpy as np

        if not texts:
            return []

        # Tokenize (query repeated for each text)
        batch = self.tokenizer(
            [query] * len(texts),
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        # Prepare input dict
        ov_inputs = {}
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in batch and key in self.required_inputs:
                ov_inputs[key] = batch[key].astype(np.int64)

        # Run inference
        outputs = self.infer_request.infer(ov_inputs)

        # Get hidden states from [CLS] token (first output, shape [batch, seq, hidden])
        hidden_states = list(outputs.values())[0]
        hidden = hidden_states[:, 0, :].astype(np.float32)  # [batch, hidden]

        # Apply external head
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
    global embed_pipeline, rerank_pipeline, hf_embedder, hf_reranker, device

    try:
        core = openvino.Core()
        available_devices = core.available_devices
        print(f"Detected OpenVINO devices: {available_devices}", flush=True)
        device = "GPU" if "GPU" in available_devices else "CPU"
    except Exception as dev_err:
        print(f"Warning: Failed to query available devices: {dev_err}. Defaulting to CPU.", flush=True)
        device = "CPU"

    print(f"Using inference device: {device}", flush=True)
    print(f"USE_HF_TOKENIZER={USE_HF_TOKENIZER}", flush=True)

    if USE_HF_TOKENIZER:
        # ========== HF Tokenizer + OpenVINO path ==========
        try:
            print(f"Loading HF embedder ({EMBED_MODEL_PATH}) on {device}...", flush=True)
            hf_embedder = OpenVINOHFEmbedder(EMBED_MODEL_PATH, device=device)
            print(f"✓ HF embedder loaded on {device} successfully", flush=True)
        except Exception as e:
            print(f"✗ Failed to load HF embedder: {e}", flush=True)
            raise

        try:
            print(f"Loading HF reranker ({RERANK_MODEL_PATH}) on {device}...", flush=True)
            hf_reranker = OpenVINOHFReranker(RERANK_MODEL_PATH, device=device)
            print(f"✓ HF reranker loaded on {device} successfully", flush=True)
        except Exception as e:
            print(f"✗ Failed to load HF reranker: {e}", flush=True)
            raise
    else:
        # ========== Default: native genai pipelines ==========
        try:
            print(f"Loading embedding model ({EMBED_MODEL_PATH}) on {device}...", flush=True)
            embed_config = openvino_genai.TextEmbeddingPipeline.Config()
            embed_config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
            embed_config.normalize = False

            try:
                embed_pipeline = openvino_genai.TextEmbeddingPipeline(
                    EMBED_MODEL_PATH,
                    device,
                    embed_config,
                    INFERENCE_PRECISION_HINT="f16" if device == "GPU" else "f32",
                    PERFORMANCE_HINT="LATENCY",
                    NUM_STREAMS="1",
                    CACHE_DIR=CACHE_DIR
                )
            except Exception as hints_err:
                print(f"Note: Standard loading due to hints error: {hints_err}", flush=True)
                embed_pipeline = openvino_genai.TextEmbeddingPipeline(
                    EMBED_MODEL_PATH,
                    device,
                    embed_config
                )
            print(f"✓ Embedding model loaded on {device} successfully", flush=True)
        except Exception as e:
            print(f"✗ Failed to load embedding model: {e}", flush=True)
            raise

        try:
            print(f"Loading reranker model ({RERANK_MODEL_PATH}) on {device}...", flush=True)
            try:
                rerank_pipeline = openvino_genai.TextRerankPipeline(
                    RERANK_MODEL_PATH,
                    device,
                    top_n=RERANK_TOP_N,
                    INFERENCE_PRECISION_HINT="f16" if device == "GPU" else "f32",
                    PERFORMANCE_HINT="LATENCY",
                    NUM_STREAMS="1",
                    CACHE_DIR=CACHE_DIR
                )
            except Exception as hints_err:
                print(f"Note: Standard loading due to hints error: {hints_err}", flush=True)
                rerank_pipeline = openvino_genai.TextRerankPipeline(
                    RERANK_MODEL_PATH,
                    device,
                    top_n=RERANK_TOP_N
                )
            print(f"✓ Reranker model loaded on {device} successfully", flush=True)
        except Exception as e:
            print(f"✗ Failed to load reranker model: {e}", flush=True)
            raise

    print("All models loaded successfully!", flush=True)



@app.post("/v1/embeddings")
def openai_embeddings(request: OpenAIEmbeddingRequest):
    try:
        start = time.time()
        inputs = request.input if isinstance(request.input, list) else [request.input]

        if USE_HF_TOKENIZER:
            # HF + OpenVINO path
            with embed_lock:
                embeddings = hf_embedder.encode(inputs)
            path_label = "HF"
        else:
            # Native genai path (default)
            with embed_lock:
                embeddings = embed_pipeline.embed_documents(inputs)
            path_label = "GenAI"

        data = []
        for idx, emb in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "embedding": list(emb),
                "index": idx
            })

        latency = (time.time() - start) * 1000
        print(f"OpenAI Embed ({path_label}): {len(inputs)} texts, {latency:.1f}ms", flush=True)

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        print(f"OpenAI Embed error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


def rerank_in_length_buckets(query: str, texts: List[str]):
    if len(texts) < 64:
        return list(rerank_pipeline.rerank(query, texts))

    word_counts = [len(text.split()) for text in texts]
    max_words = max(word_counts, default=0)
    if max_words <= 96:
        return list(rerank_pipeline.rerank(query, texts))

    bucket_limits = (32, 64, 128, 256, 512, 10**9)
    buckets = [[] for _ in bucket_limits]
    for original_index, (text, words) in enumerate(zip(texts, word_counts)):
        for bucket_index, limit in enumerate(bucket_limits):
            if words <= limit:
                buckets[bucket_index].append((original_index, text))
                break

    non_empty = [b for b in buckets if b]
    if len(non_empty) == 1:
        return list(rerank_pipeline.rerank(query, texts))

    combined = []
    for bucket in non_empty:
        local_texts = [text for _, text in bucket]
        local_results = rerank_pipeline.rerank(query, local_texts)
        for local_index, score in local_results:
            combined.append((bucket[local_index][0], score))

    combined.sort(key=lambda item: item[1], reverse=True)
    return combined

@app.post("/rerank")
def rerank(request: RerankRequest):
    try:
        start = time.time()

        if USE_HF_TOKENIZER:
            # HF + OpenVINO path: HF tokenizer handles pathological cases robustly.
            # Still call sanitize_rerank_text for safety/consistency until validated.
            query = sanitize_rerank_text(
                request.query, max_utf8_bytes=RERANK_QUERY_MAX_BYTES
            )
            texts = [
                sanitize_rerank_text(text, max_utf8_bytes=RERANK_TEXT_MAX_BYTES)
                for text in request.texts
            ]

            with rerank_lock:
                raw_results = hf_reranker.score(query, texts, raw_scores=False)

            results = raw_results  # Already in [{"index": ..., "score": ...}, ...] format

            if request.raw_scores:
                # Convert from sigmoid space back to logits
                results = [
                    {"index": r["index"], "score": inverse_sigmoid(r["score"])}
                    for r in results
                ]

            latency = (time.time() - start) * 1000
            print(f"Rerank (HF): {len(request.texts)} texts, {latency:.1f}ms", flush=True)

        else:
            # Default: native genai pipeline with caching
            # Sanitize query and texts to prevent OpenVINO Tokenizer crash
            query = sanitize_rerank_text(
                request.query, max_utf8_bytes=RERANK_QUERY_MAX_BYTES
            )
            texts = [
                sanitize_rerank_text(text, max_utf8_bytes=RERANK_TEXT_MAX_BYTES)
                for text in request.texts
            ]

            cache_key = (query, tuple(texts))
            with rerank_cache_lock:
                cached = rerank_cache.get(cache_key)
                if cached is not None:
                    rerank_cache.move_to_end(cache_key)

            if cached is None:
                with rerank_lock:
                    raw_results = rerank_in_length_buckets(query, texts)
                if RERANK_CACHE_SIZE > 0:
                    with rerank_cache_lock:
                        rerank_cache[cache_key] = raw_results
                        rerank_cache.move_to_end(cache_key)
                        while len(rerank_cache) > RERANK_CACHE_SIZE:
                            rerank_cache.popitem(last=False)
                cache_status = "miss"
            else:
                raw_results = cached
                cache_status = "hit"

            results = []
            for index, score in raw_results:
                if request.raw_scores:
                    score = inverse_sigmoid(score)
                results.append({"index": index, "score": score})

            latency = (time.time() - start) * 1000
            print(f"Rerank (GenAI {cache_status}): {len(request.texts)} texts, {latency:.1f}ms", flush=True)

        return results

    except Exception as e:
        print(f"Rerank error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "device": device}

@app.get("/info")
async def info():
    return {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "model_type": "embedding",
        "max_input_length": 8192,
        "dimension": 768,
        "device": device
    }

if __name__ == "__main__":
    print("Server starting on port 3002...", flush=True)
    uvicorn_run_placeholder = uvicorn.run
    uvicorn_run_placeholder(app, host="0.0.0.0", port=3002)
