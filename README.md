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
1. `sentence-transformers/all-mpnet-base-v2` -> `models/gte-modernbert-ov`
2. `cross-encoder/ms-marco-MiniLM-L-6-v2` -> `models/ettin-17m-ov`

If you still want to export them manually on the host, the script remains available:
```bash
python3 bin/prepare_models.py
python3 bin/prepare_models.py --force
```

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
2.  **`openvino-server` (Port `3002`)**: A custom-built, lightweight FastAPI server running OpenVINO GenAI. Hosts TEI-compatible `/v1/embeddings` (using `gte-modernbert-base`) and `/rerank` (using `ettin-17m-ov`) endpoints, automatically falling back from Intel GPU (`/dev/dri`) to optimized CPU execution.
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

## ⚡ **OpenVINO C++ Tokenizer & CPU/GPU Pipeline**

The embedding and reranking tasks run natively using **OpenVINO** to completely eliminate latency spikes and OOM thrashing.

### **A. CPU / iGPU Dynamic Auto-Fallback**
The server queries the host execution environments dynamically. If no active OpenCL platform is compiled or bound inside Proxmox unprivileged limits, the pipeline automatically boots in high-performance CPU execution mode with custom parallel thread locks.

### **B. Native C++ Tokenization**
Traditional pipelines use Python's `transformers` tokenizer which runs on a single CPU thread and acts as a major bottleneck. Our OpenVINO server leverages native **C++ tokenizers** (`openvino_tokenizer.xml` and `openvino_detokenizer.xml` compiled into the model directory):
*   Performs token extraction and formatting in raw C++.
*   Bypasses Python-to-C++ serialization boundaries entirely.
*   Moves generated token IDs straight into execution memory buffers, maintaining ultra-low latency.

### **C. Execution Thread Lock & Length-Bucketing**
*   **Thread Serialization Lock**: The server exposes a single execution lock (`gpu_lock = threading.Lock()`) across both `/embed` and `/rerank` endpoints. Because execution contexts are shared, serializing executions prevents thread thrashing and ensures consistent worst-case performance under load.
*   **Length-Bucketing (Padding Reduction)**: When processing mixed-length batches of memories, the server groups incoming candidate texts into length-buckets (e.g., 32, 64, 128 tokens). This avoids forcing unnecessary padding (which slows down CPU/GPU cycles) onto short memory lines, preserving sub-250ms query times.
