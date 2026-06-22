# edumem: Unified Cognitive Memory Subsystem & Performance Harness

This repository contains **edumem** (our Node.js `api-daemon` runtime service), the OpenVINO inference server, and the performance harness used to measure, test, and profile our custom cognitive memory pipeline on **CT 116**.

---

## 🐳 **Unified Single-Image Docker Stack (Qdrant + OpenVINO + api-daemon)**

To standardize execution and eliminate virtualized hardware mapping overhead, the entire memory subsystem is packaged into a **single, unified, self-contained Docker image**. 

### **⚙️ Model Preparation (Crucial First Step)**
The image now builds the OpenVINO embedding and reranker models inside the container. You do not need to pre-run `bin/prepare_models.py` on the host.

Build with your Hugging Face token if you need authenticated downloads. If the token is only in `.env`, load it into PowerShell first:
```powershell
$env:HF_TOKEN = ((Get-Content .env | Where-Object { $_ -match '^HF_TOKEN=' } | Select-Object -First 1) -replace '^HF_TOKEN=', '').Trim('"')
podman build --build-arg HF_TOKEN=$env:HF_TOKEN -t edumem:latest .
```

If you do not need a token, omit the `--build-arg HF_TOKEN=...` flag.

The build exports these OpenVINO models inside the image:
1. `sentence-transformers/all-mpnet-base-v2` -> `models/gte-modernbert-ov` (directory name is legacy; the model exported there is all-mpnet-base-v2)
2. `jhu-clsp/ettin-encoder-17m` -> `models/ettin-17m-ov`

If you still want to export them manually on the host, the script remains available:
```bash
python3 bin/prepare_models.py
python3 bin/prepare_models.py --force
```

### **🔄 Serving Backends**
Two serving backends are available, selected by the `SYSTEM` environment variable (defaults to `intel`):
- **`SYSTEM=intel`** (default): Runs `server.py` — OpenVINO GenAI inference server targeting Intel iGPU (`/dev/dri`) with automatic fallback to CPU. Embeddings: `all-mpnet-base-v2` in `models/gte-modernbert-ov`. Reranker: `ettin-encoder-17m` in `models/ettin-17m-ov`.
- **`SYSTEM=nvidia`** (optional): Runs `server_nvidia.py` — ONNX Runtime inference server on CUDA. Embeddings: `Alibaba-NLP/gte-modernbert-base`. Reranker: `cross-encoder/ettin-reranker-17m-v1`.

Both backends expose identical ports and endpoints: `/v1/embeddings` and `/rerank` on port 3002.

### **🚀 Running the Stack**
The default runtime is CPU-safe. On Linux hosts with an Intel iGPU, you can add the `/dev/dri` mount back in.

```bash
podman run -d \
  --name edumem-app \
  -p 3002:3002 \
  -p 6333:6333 \
  -p 6336:6336 \
  edumem:latest
```

Optional Intel GPU runtime on Linux:
```bash
podman run -d \
  --name edumem-app \
  -p 3002:3002 \
  -p 6333:6333 \
  -p 6336:6336 \
  --privileged \
  -v /dev/dri:/dev/dri \
  edumem:latest
```

### **📦 Packaged Services (Co-located inside a single container):**
1.  **`api-daemon` (Port `6336`)**: Our Node.js REST API memory server. Handles standard `recall`, `retain`, and `reflect` requests.
2.  **`openvino-server` (Port `3002`)**: A custom-built, lightweight FastAPI server running OpenVINO GenAI. Hosts TEI-compatible `/v1/embeddings` (all-mpnet-base-v2, exported to `models/gte-modernbert-ov`) and `/rerank` (ettin-17m-ov) endpoints, automatically falling back from Intel GPU (`/dev/dri`) to optimized CPU execution.
3.  **`qdrant` (Port `6333`)**: High-performance vector database for high-dimensional, high-fidelity memory indexing.

---

## 🏛️ **System Architecture**

The subsystem divides responsibilities across three decoupled layers co-located within a single container boundary:

```text
       [ Calling Agent ] (e.g. Hermes Core on CT 108)
              │
              ▼ (Port 6336 - REST Recall/Retain API)
┌────────────────────────────────────────────────────────┐
│                      api-daemon                        │ (Node.js - Port 6336)
│  - Endpoint routing (/recall, /retain, /reflect)        │
└──────┬───────────────────┬─────────────────────────────┘
       │                   │
       ▼ (Vector queries)  ▼ (Direct /v1/embeddings POSTs)
┌──────────────┐   ┌─────────────────────────────────────┐
│  Qdrant DB   │   │      OpenVINO Inference Server      │ (Python - Port 3002)
│  - Intel iGPU accelerated embedding │
│              │   │  - Native C++ Tokenizers            │
└──────────────┘   └─────────────────────────────────────┘
```

---

## 🔄 **The Ingest Flow: How We Store Stuff**

When a new memory is written (via `/retain`), it flows through our **multi-stage pipeline**:

```text
  [ Raw Chat / Memory String ]
                │
                ▼
┌──────────────────────────────────────────────┐
│  LAYER 1: Ingestion & API Router             │
│  - Receives payload, validates JSON schemas  │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│  LAYER 2: Embedding Generation (OpenVINO)    │
│  - api-daemon POSTs content to port 3002     │
│  - OpenVINO runs ModernBert on Intel iGPU    │
│  - Returns 768-dimensional float array       │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│  LAYER 3: Vector & Payload Indexed           │
│  - Uploads point + payload to Qdrant (:6333) │
└──────────────────────────────────────────────┘
```

---

## 🔍 **Retrieval Execution Flow (RAG)**

When an agent queries the database (via `/recall`), the search leverages our GPU acceleration to maintain sub-100ms REST response latencies:

```text
                  [ User Query ]
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ 1. Compute query vector (iGPU)   │
        │ - POST to OpenVINO on port 3002  │
        └────────────────┬─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ 2. Semantic KNN Search           │
        │ - POST query vector to Qdrant    │
        │ - Retrieve user-scoped payloads  │
        └────────────────┬─────────────────┘
                         │
                         ▼
             [ Sorted Context Results ]
```

*Note: The **Ettin-17M GPU Reranker** is fully active and exposed on `/rerank` (port 3002). It is bypassed in the basic `/recall` REST route to maintain maximum throughput under load, but remains fully available for downstream agent reasoning chains, multi-vector fusion, or custom evaluation pipelines.*

---

## 🧠 **edumem Memory Engine (MEMORIA): Storage & Recall**

The **edumem Python memory engine** implements a structured fact-based recall system that powers `beam.recall` and the BEAM evaluation harness. This is distinct from the Qdrant vector-search path: while Qdrant handles dense semantic retrieval via GPU-accelerated embeddings, MEMORIA stores and retrieves **typed, versioned facts** (metrics, timelines, preferences, knowledge graphs) using language-agnostic fusion techniques.

### **Storage (Write Path)**

When facts are extracted from a message, they flow through **three stages**:

1. **Extraction**: Regex patterns and language-specific templates detect facts across multiple types:
   - **Metrics**: Numbers with units (e.g., "250ms response time" → `response_time_ms: 250ms`)
   - **Timelines**: ISO dates with event context (e.g., "2024-03-15 deployed feature X")
   - **Preferences & Instructions**: Structured rules and user preferences
   - **Knowledge Graph (KG)**: Subject-Predicate-Object triples, including negations (e.g., `user -[negation]-> never used X`)

2. **LLM Canonicalization** (optional): For metric facts, `_llm_canonicalize_facts` uses the LLM client to assign a **canonical snake_case key** so differently-phrased mentions (e.g., "API latency", "response delay") collapse to a single base key. This consolidation happens at write time only; stated targets/goals retain separate keys. Falls back to raw regex keys when no LLM client is present.

3. **Versioned Chaining**: `_insert_fact` stores facts in `memoria_facts` with full **version tracking**. When the same `(session_id, key, fact_type)` receives a new value, the prior row is marked superseded (`valid_to_msg_idx` set) and the previous value is stored in a new `previous_value` field, forming Mem0-style version chains. This enables retrieval of fact histories (e.g., "user previously said X, now says Y").

**Storage Tables** (SQLite, all scoped by `session_id`):
- `memoria_facts`: Versioned facts with columns `version_id`, `previous_value`, `valid_from_msg_idx`, `valid_to_msg_idx`, `message_idx`, `source_memory_id`
- `memoria_timelines`: Dated events and milestones
- `memoria_instructions`: Directives and rules
- `memoria_preferences`: Evolving user preferences
- `memoria_kg`: Knowledge graph triples (includes negation predicates)

### **Recall (RRF Fusion)**

`memoria_retrieve` routes queries through `_memoria_fused_retrieve`, which runs **four parallel specialist retrievers**:

1. **Fact Retriever** (`_memoria_fact_retrieve`): SQL queries on `memoria_facts` keyed by query terms
2. **Timeline Retriever** (`_memoria_timeline_retrieve`): Temporal event matching on `memoria_timelines`
3. **Negation Retriever** (`_memoria_negation_retrieve`): Filters on `memoria_kg` where `predicate='negation'`
4. **Chrono Retriever** (`_memoria_chrono_retrieve`): Time-bounded queries

Results from all four specialists are fused using **Reciprocal Rank Fusion (RRF)** with `k=60`:
```
score(item) = Σ 1/(60 + rank_in_specialist)
```

RRF is **language-agnostic** and requires no intent classifier—it works equally well for English, German, Russian, and Spanish queries. This replaced the prior intent-routing approach (which detected intent keywords to pick a single specialist).

**Measured Impact (BEAM 100K, retrieval level)**:
- **2.6× more answer-bearing facts** surfaced vs. intent-routing (39 nuggets vs. 15 out of 125 total)
- **10 wins, 0 losses** across queries
- **~0.6ms latency** (all-local SQLite, no I/O overhead)

*Note on serving distinction: MEMORIA is the **Python** memory engine used by `beam.recall` and the BEAM evaluation harness. The Node.js `api-daemon` REST `/recall` endpoint currently uses the **Qdrant vector-search path** (embed query → Qdrant KNN; described in Retrieval Execution Flow above) — it does not route through MEMORIA. The two are separate paths.*

---

## ⚡ **Inference Backends & Tokenization**

The embedding and reranking tasks run natively using **OpenVINO** (Intel backend) or **ONNX Runtime** (NVIDIA backend) to eliminate latency spikes and OOM thrashing.

### **A. CPU / iGPU Dynamic Auto-Fallback (Intel Backend)**
The OpenVINO server queries the host execution environments dynamically. If no active OpenCL platform is compiled or bound inside Proxmox unprivileged limits, the pipeline automatically boots in high-performance CPU execution mode with custom parallel thread locks.

### **B. Tokenization**
- **Intel / OpenVINO** (`server.py`): Uses native **C++ tokenizers** (`openvino_tokenizer.xml` and `openvino_detokenizer.xml` compiled into the model directory). Performs token extraction and formatting in raw C++, bypassing Python-to-C++ serialization boundaries. However, the C++ tokenizer has known pathological-input failure modes (very long single tokens, repeated-character runs) that are guarded by input sanitization in `server_text.py`. An optional flag-gated HuggingFace fast-tokenizer path is available via `USE_HF_TOKENIZER=1` for decoupled tokenize-from-inference.
- **NVIDIA / ONNX** (`server_nvidia.py`): Uses the HuggingFace fast (Rust-based) tokenizer (`transformers.PreTrainedTokenizerFast`), decoupled from inference.

### **C. Execution Thread Lock & Length-Bucketing**
*   **Thread Serialization Lock**: The server exposes a single execution lock (`gpu_lock = threading.Lock()`) across both `/embed` and `/rerank` endpoints. Because execution contexts are shared, serializing executions prevents thread thrashing and ensures consistent worst-case performance under load.
*   **Length-Bucketing (Padding Reduction)**: When processing mixed-length batches of memories, the server groups incoming candidate texts into length-buckets (e.g., 32, 64, 128 tokens). This avoids forcing unnecessary padding (which slows down CPU/GPU cycles) onto short memory lines, preserving sub-250ms query times.
