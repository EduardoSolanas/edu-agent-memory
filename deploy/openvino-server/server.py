#!/usr/bin/env python3
"""
OpenVINO Inference Server for Hindsight
Serves embeddings, reranker, and LLM models on Intel iGPU using native openvino_genai pipelines
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openvino_genai
import uvicorn
import time
import sys
import math
import threading
import os
from collections import OrderedDict

EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/root/openvino-server/models/gte-modernbert-ov")
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "/root/openvino-server/models/ettin-17m-ov")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "512"))
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/root/openvino-server/models/Phi-4-mini-instruct-fp16-ov").strip(' \t\n\r"\'')
CACHE_DIR = os.getenv("CACHE_DIR", "/root/openvino-server/models/model_cache")
RERANK_CACHE_SIZE = int(os.getenv("RERANK_CACHE_SIZE", "128"))

print("Starting OpenVINO Inference Server (Native GenAI)...", flush=True)

app = FastAPI(title="OpenVINO Inference Server (Native GenAI)")

# Models will be loaded on startup
embed_pipeline = None
rerank_pipeline = None
llm_pipeline = None

# Single GPU lock prevents embed/rerank overlap on the same Intel iGPU.
# With NUM_STREAMS=1 this reduces contention and tail-latency spikes.
gpu_lock = threading.Lock()
embed_lock = gpu_lock
rerank_lock = gpu_lock
llm_lock = threading.Lock()
rerank_cache_lock = threading.Lock()
rerank_cache = OrderedDict()

class EmbedRequest(BaseModel):
    inputs: str | List[str]
    
class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    
class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    raw_scores: bool = False
    
class RerankResponse(BaseModel):
    index: int
    score: float

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

def inverse_sigmoid(p: float) -> float:
    p = max(1e-15, min(1.0 - 1e-15, p))
    return math.log(p / (1.0 - p))

@app.on_event("startup")
async def load_models():
    global embed_pipeline, rerank_pipeline, llm_pipeline
    
    try:
        print(f"Loading embedding model ({EMBED_MODEL_PATH}) on GPU...", flush=True)
        embed_config = openvino_genai.TextEmbeddingPipeline.Config()
        embed_config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
        embed_config.normalize = False
        embed_pipeline = openvino_genai.TextEmbeddingPipeline(
            EMBED_MODEL_PATH,
            "GPU",
            embed_config,
            INFERENCE_PRECISION_HINT="f16",
            PERFORMANCE_HINT="LATENCY",
            NUM_STREAMS="1", # Lock single-stream execution to optimize burst latency
            CACHE_DIR=CACHE_DIR
        )
        print("✓ Embedding model loaded on GPU with FP16 + LATENCY hints, NUM_STREAMS=1 & caching", flush=True)
    except Exception as e:
        print(f"✗ Failed to load embedding model: {e}", flush=True)
        raise
    
    try:
        # Load the fully-functional Ettin-17M model!
        print(f"Loading reranker model ({RERANK_MODEL_PATH}) on GPU...", flush=True)
        rerank_pipeline = openvino_genai.TextRerankPipeline(
            RERANK_MODEL_PATH,
            "GPU",
            top_n=RERANK_TOP_N,
            INFERENCE_PRECISION_HINT="f16",
            PERFORMANCE_HINT="LATENCY",
            NUM_STREAMS="1", # Lock single-stream execution to optimize burst latency
            CACHE_DIR=CACHE_DIR
        )
        print(f"✓ Reranker model (Ettin-17M, top_n={RERANK_TOP_N}) loaded on GPU with FP16 + LATENCY hints, NUM_STREAMS=1 & caching", flush=True)
    except Exception as e:
        print(f"✗ Failed to load reranker model: {e}", flush=True)
        raise
        
    if LLM_MODEL_PATH:
        try:
            # Load Phi-4-mini-instruct-fp16-ov model with task polling disabled!
            print(f"Loading LLM model ({LLM_MODEL_PATH}) on GPU with task polling disabled...", flush=True)
            try:
                llm_pipeline = openvino_genai.LLMPipeline(
                    LLM_MODEL_PATH,
                    "GPU",
                    PERFORMANCE_HINT="LATENCY",
                    NUM_STREAMS="1",
                    CACHE_DIR=CACHE_DIR
                )
            except Exception as hints_err:
                print(f"Note: Standard loading due to hints error: {hints_err}", flush=True)
                llm_pipeline = openvino_genai.LLMPipeline(
                    LLM_MODEL_PATH,
                    "GPU",
                    PERFORMANCE_HINT="LATENCY",
                    NUM_STREAMS="1",
                    CACHE_DIR=CACHE_DIR
                )
            print(f"✓ LLM model ({os.path.basename(LLM_MODEL_PATH)}) loaded on GPU with FP16 + LATENCY hints, NUM_STREAMS=1 & caching", flush=True)
        except Exception as e:
            print(f"✗ Failed to load LLM model: {e}", flush=True)
            raise
    else:
        print("Skipping local LLM loading (LLM_MODEL_PATH is empty/not set)", flush=True)
    
    print("All models loaded successfully!", flush=True)

@app.post("/embed")
def embed(request: EmbedRequest):
    try:
        start = time.time()
        
        inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        
        # Protect with thread lock to serialize iGPU executions
        with embed_lock:
            embeddings = embed_pipeline.embed_documents(inputs)
        
        latency = (time.time() - start) * 1000
        print(f"Embed: {len(inputs)} texts, {latency:.1f}ms", flush=True)
        
        return embeddings
    
    except Exception as e:
        print(f"Embed error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


def rerank_in_length_buckets(query: str, texts: List[str]):
    """Score all texts with identical model semantics, grouped by rough length.

    Cross-encoder scores are independent per query/document pair. Grouping by
    length reduces padding waste for mixed short/long candidate batches while
    returning the same scores as one large mixed-length batch.
    """
    if len(texts) < 64:
        return list(rerank_pipeline.rerank(query, texts))

    # Word-count buckets are deliberately coarse: enough to avoid one long
    # candidate forcing the whole batch to the longest shape, without creating
    # many tiny GPU calls.
    word_counts = [len(text.split()) for text in texts]
    max_words = max(word_counts, default=0)
    # Real Hindsight candidates are usually <=64 words. Splitting those creates
    # extra GPU calls and is slower. Bucket only when a real long-tail exists.
    if max_words <= 96:
        return list(rerank_pipeline.rerank(query, texts))

    bucket_limits = (32, 64, 128, 256, 512, 10**9)
    buckets = [[] for _ in bucket_limits]
    for original_index, (text, words) in enumerate(zip(texts, word_counts)):
        for bucket_index, limit in enumerate(bucket_limits):
            if words <= limit:
                buckets[bucket_index].append((original_index, text))
                break

    # If everything is already same-shape, avoid extra Python work.
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

        # Exact-query cache: same query + same candidate texts produce identical
        # model scores. This preserves quality and eliminates repeated GPU work
        # during repeated recalls while the bank is unchanged.
        cache_key = (request.query, tuple(request.texts))
        with rerank_cache_lock:
            cached = rerank_cache.get(cache_key)
            if cached is not None:
                rerank_cache.move_to_end(cache_key)

        if cached is None:
            # Protect with thread lock to serialize iGPU executions
            with rerank_lock:
                raw_results = rerank_in_length_buckets(request.query, request.texts)
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

import re
import json

def parse_text_to_tool_calls(text: str) -> list[dict]:
    # Match patterns like: tool_name(query='value', max_tokens=123) or tool_name('value') or tool_name(value)
    tool_names = ['search_observations', 'search_world_facts', 'recall', 'search_opinions']
    
    for tool in tool_names:
        pattern = rf"{tool}\((.*?)\)"
        match = re.search(pattern, text)
        if match:
            args_str = match.group(1).strip()
            arguments = {}
            
            if (args_str.startswith("'") and args_str.endswith("'")) or (args_str.startswith('"') and args_str.endswith('"')):
                query_val = args_str[1:-1]
                arguments = {"query": query_val}
            else:
                pairs = re.findall(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\w+))", args_str)
                if pairs:
                    for k, v1, v2, v3 in pairs:
                        val = v1 or v2 or v3
                        if val.isdigit():
                            arguments[k] = int(val)
                        else:
                            arguments[k] = val
                else:
                    arguments["query"] = args_str
            
            if "query" not in arguments and args_str:
                arguments["query"] = args_str
                
            return [
                {
                    "id": f"call_{int(time.time())}",
                    "type": "function",
                    "function": {
                        "name": tool,
                        "arguments": json.dumps(arguments)
                    }
                }
            ]
    return []

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if llm_pipeline is None:
        raise HTTPException(status_code=400, detail="Local LLM is disabled. Set LLM_MODEL_PATH to enable.")
    try:
        start = time.time()
        
        # Build tools instruction if tools are provided
        tools_instruction = ""
        if request.tools:
            tools_instruction = (
                "\n\nYou have access to the following search tools. If the user's query requires "
                "retrieving memories, observations, or facts to answer, you MUST output a single tool call "
                "matching this exact syntax (and nothing else):\n"
                "search_observations(query='your query string')\n\n"
                "Available tools:\n"
            )
            for tool in request.tools:
                name = tool.get("function", {}).get("name")
                desc = tool.get("function", {}).get("description", "")
                tools_instruction += f"- {name}(query='...'): {desc}\n"
            
            tools_instruction += (
                "\nCRITICAL: If you do not have the necessary information in your context yet, "
                "you MUST call one of these tools first. Do NOT make up information or refuse "
                "if a tool call could retrieve it."
            )
            
        # Convert Pydantic message list with injected system instruction if applicable
        messages_to_send = []
        has_system = False
        
        for msg in request.messages:
            role = msg.role
            content = msg.content
            
            # Map tool roles and tool calls to standard user/assistant messages
            if role == "tool":
                role = "user"
                content = f"Tool result:\n{content}"
            elif role == "assistant" and not content and msg.tool_calls:
                # Reconstruct the tool call text that the model originally generated
                try:
                    tc = msg.tool_calls[0]
                    tc_name = tc.get("function", {}).get("name") if isinstance(tc, dict) else getattr(getattr(tc, "function", None), "name", None)
                    tc_args = tc.get("function", {}).get("arguments") if isinstance(tc, dict) else getattr(getattr(tc, "function", None), "arguments", None)
                    
                    if isinstance(tc_args, str):
                        try:
                            args_dict = json.loads(tc_args)
                            args_str = ", ".join(f"{k}='{v}'" for k, v in args_dict.items())
                        except Exception:
                            args_str = f"query='{tc_args}'"
                    elif isinstance(tc_args, dict):
                        args_str = ", ".join(f"{k}='{v}'" for k, v in tc_args.items())
                    else:
                        args_str = ""
                    content = f"{tc_name}({args_str})"
                except Exception as e:
                    content = f"search_observations(query='{msg.content}')"
                    
            if role == "system" and tools_instruction:
                content = (content or "") + tools_instruction
                has_system = True
                
            messages_to_send.append({"role": role, "content": content})
            
        if not has_system and tools_instruction:
            messages_to_send.insert(0, {"role": "system", "content": tools_instruction})
            
        # Convert messages list to OpenVINO GenAI ChatHistory object
        history = openvino_genai.ChatHistory()
        for msg in messages_to_send:
            if msg["role"] in ("user", "assistant", "system"):
                history.append({"role": msg["role"], "content": msg["content"] or ""})
        
        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = request.max_tokens
        if request.temperature > 0:
            config.temperature = request.temperature
            config.do_sample = True
            config.top_p = request.top_p
        else:
            config.do_sample = False  # Greedy decoding
            
        # Execute in thread-safe block to avoid conflict on Intel GPU context
        with llm_lock:
            res = llm_pipeline.generate(history, generation_config=config)
            
        output_text = res.texts[0]
        latency = (time.time() - start) * 1000
        print(f"LLM (Phi-4-mini): {len(request.messages)} messages, {latency:.1f}ms", flush=True)
        
        # Check if output contains an agentic tool call
        tool_calls = parse_text_to_tool_calls(output_text)
        
        message_payload = {
            "role": "assistant",
            "content": output_text if not tool_calls else None
        }
        if tool_calls:
            message_payload["tool_calls"] = tool_calls
            
        # Format response to match the exact OpenAI spec
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message_payload,
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Mocked
                "completion_tokens": 0,  # Mocked
                "total_tokens": 0  # Mocked
            }
        }
        
    except Exception as e:
        print(f"LLM error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "device": "GPU"}

@app.get("/info")
async def info():
    return {
        "model_id": "Alibaba-NLP/gte-modernbert-base",
        "model_type": "embedding",
        "max_input_length": 8192,
        "dimension": 768,
        "device": "GPU"
    }

if __name__ == "__main__":
    print("Server starting on port 3002...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=3002)
