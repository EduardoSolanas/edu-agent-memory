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

from server_text import sanitize_rerank_text

EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/root/openvino-server/models/gte-modernbert-ov")
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "/root/openvino-server/models/ettin-17m-ov")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "512"))
CACHE_DIR = os.getenv("CACHE_DIR", "/root/openvino-server/models/model_cache")
RERANK_CACHE_SIZE = int(os.getenv("RERANK_CACHE_SIZE", "128"))
RERANK_QUERY_MAX_BYTES = int(os.getenv("RERANK_QUERY_MAX_BYTES", "128"))
RERANK_TEXT_MAX_BYTES = int(os.getenv("RERANK_TEXT_MAX_BYTES", "352"))

print("Starting OpenVINO Inference Server (Native GenAI)...", flush=True)

app = FastAPI(title="OpenVINO Inference Server (Native GenAI)")

# Models will be loaded on startup
embed_pipeline = None
rerank_pipeline = None
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

@app.on_event("startup")
async def load_models():
    global embed_pipeline, rerank_pipeline, device
    
    try:
        core = openvino.Core()
        available_devices = core.available_devices
        print(f"Detected OpenVINO devices: {available_devices}", flush=True)
        device = "GPU" if "GPU" in available_devices else "CPU"
    except Exception as dev_err:
        print(f"Warning: Failed to query available devices: {dev_err}. Defaulting to CPU.", flush=True)
        device = "CPU"
        
    print(f"Using inference device: {device}", flush=True)
    
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
        
        with embed_lock:
            embeddings = embed_pipeline.embed_documents(inputs)
            
        data = []
        for idx, emb in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "embedding": list(emb),
                "index": idx
            })
            
        latency = (time.time() - start) * 1000
        print(f"OpenAI Embed: {len(inputs)} texts, {latency:.1f}ms", flush=True)
        
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

        # Sanitize query and texts to prevent OpenVINO Tokenizer crash on empty/whitespace/repeated inputs
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
        print(f"Rerank (Native {cache_status}): {len(request.texts)} texts, {latency:.1f}ms", flush=True)

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
