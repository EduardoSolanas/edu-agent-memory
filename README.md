# edumem: Unified Cognitive Memory Subsystem & Performance Harness

This repository contains **edumem** (our Node.js `api-daemon` runtime service), the OpenVINO inference server, and the performance harness used to measure, test, and profile our custom cognitive memory pipeline on **CT 116**.

---

## 🐳 **Unified Single-Image Docker Stack (Qdrant + OpenVINO + api-daemon)**

To standardize execution and eliminate virtualized hardware mapping overhead, the entire memory subsystem is packaged into a **single, unified, self-contained Docker image**. 

### **⚙️ Model Preparation (Crucial First Step)**
Before you can build the Docker image or run the cognitive memory pipeline on a fresh clone, you **must** download and prepare the required OpenVINO models. 

An automated, non-interactive Python script `bin/prepare_models.py` is provided to handle this cleanly. It will:
1. Load your Hugging Face credentials (`HF_TOKEN`) from `/opt/edumem/.env` (or standard environment variables).
2. Install `optimum-intel[openvino]` in the virtual environment if not already available.
3. Skip exporting if the models are already prepared (unless `--force` is specified).
4. Download and export the models to FP16 OpenVINO format:
   - **GTE ModernBERT** (`Alibaba-NLP/gte-modernbert-base`) -> `models/gte-modernbert-ov`
   - **Ettin Reranker** (`cross-encoder/ettin-reranker-17m-v1`) -> `models/ettin-17m-ov`

To prepare the models, run:
```bash
python3 bin/prepare_models.py
```

To force re-exporting of already prepared models, use:
```bash
python3 bin/prepare_models.py --force
```

### **🚀 Running the Stack**
The entire cognitive memory platform runs as a single background container, exposing production-grade endpoints:

```bash
# Spin up the unified container with Intel GPU access
docker run -d \
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
