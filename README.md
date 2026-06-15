# edumem: Unified Cognitive Memory Subsystem & Performance Harness

This repository contains **edumem** (consisting of the `api-daemon` runtime service) and the active evaluation/benchmarks used to measure, test, and profile our custom cognitive memory pipeline on **CT 116**.

---

## 🐳 **Single-Command Docker Stack (Qdrant + OpenVINO + api-daemon)**

To completely isolate your system, standardise execution, and eliminate Proxmox LXC setup overhead, the entire memory subsystem is packaged into a **single, unified Docker Compose stack**. 

With a single command, you spin up the entire cognitive memory platform, exposing production-grade endpoints on your host:

```bash
# Navigate to deployment directory
cd /opt/edumem/deploy

# Build and launch all three services in the background
docker compose up --build -d
```

### **📦 Configured Services:**
1.  **`api-daemon` (Port `6336`)**: Our Node.js REST API memory server. Handles standard `recall`/`remember` requests and orchestrates background dreaming and consolidation loops. Uses `network_mode: host` to bypass Docker bridge translation overhead, running at maximum native performance.
2.  **`openvino-server` (Port `3002`)**: A custom-built, lightweight FastAPI server running OpenVINO GenAI. It hosts TEI-compatible `/embed` (using `gte-modernbert-base`) and `/rerank` (using `ettin-17m-ov`) endpoints, automatically falling back from Intel GPU (`/dev/dri`) to optimized CPU execution.
3.  **`edumem-qdrant` (Ports `6333` & `6334`)**: Standard distributed vector database for high-fidelity, high-dimensional memory indexing.

---

## 🏛️ **System Architecture**

The subsystem divides responsibilities across three decoupled layers:

```text
[ Calling Agent ] (e.g. Hermes Core on CT 108)
       │
       ▼ (Port 6336 - REST / JSON Recall/Remember API)
┌────────────────────────────────────────────────────────┐
│                      api-daemon                        │ (Docker Container: Port 6336)
│  - Scheduled Dreaming & Consolidation (Every 12h)      │
│  - Standard API-compatible Recall & Reflection         │
└──────┬───────────────────┬─────────────────────────────┘
       │                   │
       ▼ (Vector queries)  ▼ (Direct /embed & /rerank HTTP POSTs)
┌──────────────┐   ┌─────────────────────────────────────┐
│  Qdrant DB   │   │      OpenVINO Inference Server      │ (Docker Container: Port 3002)
│  (Port 6333) │   │  - Local CPU / GPU model loading    │
│              │   │  - Native C++ Tokenizers            │
└──────────────┘   └─────────────────────────────────────┘
```

---

## 💾 **1. Relational + Vector Storage Engine**

While the production runtime uses Qdrant for distributed vector queries, our evaluation and sandbox engine uses **`sqlite-vec`**:

### **What is `sqlite-vec`?**
It is a native C-extension for SQLite that adds high-performance vector search capabilities directly into standard relational databases. 

### **Why we migrated to it for benchmarks:**
1.  **Zero-Network Latency**: By running vector indexing inside the same SQLite file as relational data, we avoid HTTP/gRPC network overhead, keeping memory reads sub-millisecond.
2.  **Atomic Consistency**: Relational metadata (timestamps, user IDs, sequence numbers) and raw vectors are written/committed together in a single atomic transaction.
3.  **Instant Caching**: It allows us to dump and restore precompiled databases (`.db` files) in `<0.05` seconds, completely bypassing slow message-by-message vector ingestion.

---

## 🗺️ **2. Storage Database Schema (Relational + Vector)**

Our database schema manages both semantic (vector) search and chronological (timeline) reasoning across four main tables:

```text
               ┌──────────────────────────────────────────────────┐
               │              sqlite-vec Database                 │
               └──────────────────────┬───────────────────────────┘
                                      │
         ┌────────────────────────────┼───────────────────────────┐
         ▼                            ▼                           ▼
┌──────────────────┐         ┌──────────────────┐        ┌──────────────────┐
│  memoria_facts   │         │    vec_facts     │        │memoria_timelines │
├──────────────────┤         ├──────────────────┤        ├──────────────────┤
│ id (TEXT, PK)    │◄───────>│ rowid (INT, PK)  │        │ id (TEXT, PK)    │
│ text (TEXT)      │         │ fact_id (TEXT,FK)│        │ fact_id (TEXT,FK)│
│ user_id (TEXT)   │         │ embedding(F32[768│        │ date (TEXT)      │
│ created_at (INT) │         └──────────────────┘        │ epoch (INT)      │
└────────┬─────────┘         (sqlite-vec virtual table)  └──────────────────┘
         │                                               (Chronological index)
         ▼ (FK)
┌──────────────────┐
│memoria_sequences │
├──────────────────┤
│ id (TEXT, PK)    │
│ order_idx (INT)  │
│ milestone (TEXT) │
└──────────────────┘
(Episodic step ordering)
```

---

## 🔄 **3. The Ingest Flow: How We Store Stuff**

When a new message or memory is written, it flows through a **3-Layer Pipeline**:

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
│  LAYER 2: Synthesis & Extraction (LLM)       │
│  - Parses semantic constraints               │
│  - Extracts atomic facts                     │
│  - Identifies chronological dates            │
│  - Maps sequential milestones                │
└───────────────┬──────────────────────────────┘
                │
                ├──────────────────────────────┐
                ▼ (Relational Pipeline)        ▼ (Vector Pipeline)
┌──────────────────────────────────────────────┐┌──────────────────────────────────────────────┐
│  LAYER 3A: Relational Write                  ││  LAYER 3B: Vector Write                      │
│  - Writes to SQL:                            ││  - Request /embed from OpenVINO Server      │
│    * `memoria_facts`                         ││  - OpenVINO tokenizes in C++ & runs iGPU   │
│    * `memoria_timelines` (chronology)        ││  - Returns FP16 vector (768 dimensions)     │
│    * `memoria_sequences` (milestones)        ││  - Writes vector to `vec_facts` virtual tbl │
└───────────────────────┬──────────────────────┘└──────────────────────┬───────────────────────┘
                        │                                              │
                        └──────────────────────┬───────────────────────┘
                                               ▼
                                  [ Atomic sqlite-vec COMMIT ]
```

---

## 🔍 **4. Retrieval Execution Flow (RAG)**

When an agent queries the database (e.g. during a recall step), the search combines **semantic vectors** and **structured indexes**:

```text
                  [ User Query ]
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ Get Embedding vector from GPU    │
        └────────────────┬─────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼ (Standard Path)               ▼ (Temporal Query Path)
┌────────────────────────────────┐ ┌────────────────────────────────┐
│ Semantic KNN Search            │ │ Timeline Query                 │
│ - Search `vec_facts` virtual   │ │ - Join `memoria_facts` with    │
│   table using Cosine Distance  │ │   `memoria_timelines`          │
│ - Joins matching `rowid` to    │ │ - Sort by `epoch` ascending    │
│   `memoria_facts` metadata     │ │ - Extract evolution history    │
└────────────────┬───────────────┘ └────────────────┬───────────────┘
                 │                                  │
                 └───────────────┬──────────────────┘
                                 ▼
                    [ Top-K Candidate Slices ]
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Ettin-17M GPU Reranker   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                     [ Clean, Ranked Context ]
```

---

## ⚡ **5. OpenVINO C++ Tokenizer & CPU/GPU Pipeline**

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
