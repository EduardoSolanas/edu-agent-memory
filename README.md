# api-daemon Architecture & Performance Harness

This repository contains the **api-daemon** memory subsystem service and the active evaluation/benchmarks used to measure, test, and profile our custom cognitive memory pipeline.

📖 **Documentation & Visuals:**
*   [Setup & Reproduction Guide](setup.md) — Standing step-by-step reproduction and environment configurations.
*   [Interactive Architecture Diagram](docs/architecture.html) — High-resolution, dark-themed SVG visualization of the 3-layer system pipeline, database tables, and iGPU local model server execution path.

---

## 🏛️ **System Architecture**

The subsystem divides responsibilities across three decoupled layers on the host (**CT 116**):

```text
[ Calling Agent ] (e.g. Hermes on CT 108)
       │
       ▼ (Port 6336 - REST / JSON)
┌────────────────────────────────────────────────────────┐
│                      api-daemon                        │ (Node.js Workspace: /opt/edumem)
│  - Scheduled Dreaming & Consolidation (Every 12h)      │
│  - Standard API-compatible Recall & Reflection         │
└──────┬───────────────────┬─────────────────────────────┘
       │                   │
       ▼ (Vector queries)  ▼ (Direct /embed & /rerank HTTP POSTs)
┌──────────────┐   ┌─────────────────────────────────────┐
│  Qdrant DB   │   │      OpenVINO Inference Server      │ (Docker / host port 3002)
│  (Port 6333) │   │  - Intel iGPU execution (/dev/dri)  │
│              │   │  - Native C++ Tokenizers            │
└──────────────┘   └─────────────────────────────────────┘
```

---

## 💾 **1. The Storage Engine: SQLite + `sqlite-vec`**

While the production runtime uses Qdrant for distributed vector queries, our evaluation and sandbox engine uses **`sqlite-vec`**:

### **What is `sqlite-vec`?**
It is a native C-extension for SQLite that adds high-performance vector search capabilities directly into standard relational databases. 

### **Why we migrated to it:**
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

## 🔄 **3. The Ingestion Pipeline: How We Store Stuff**

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

## ⚡ **5. OpenVINO C++ Tokenizer & iGPU Pipeline**

The embedding and reranking tasks run natively on your **Intel iGPU** rather than CPU to completely eliminate latency spikes and OOM thrashing.

### **A. iGPU Device Mapping**
The OpenVINO Docker container accesses the Intel integrated GPU through host device mapping:
*   **LXC Config**: Ensure `/dev/dri` is mapped into your LXC container.
*   **Docker Volumes**: Bound as `devices: ["/dev/dri:/dev/dri"]` (or using Host network mode with direct path binds as implemented on CT 116).

### **B. Native C++ Tokenization**
Traditional pipelines use Python's `transformers` tokenizer which runs on a single CPU thread and acts as a major bottleneck. Your OpenVINO server leverages native **C++ tokenizers** (`openvino_tokenizer.xml` and `openvino_detokenizer.xml` compiled into the model directory):
*   Performs token extraction and formatting in raw C++.
*   Bypasses Python-to-C++ serialization boundaries entirely.
*   Moves generated token IDs straight into GPU memory buffers, maintaining low latency.

### **C. iGPU Thread Lock & Length-Bucketing**
*   **GPU Thread Serialization Lock**: The server exposes a single GPU lock (`gpu_lock = threading.Lock()`) across both `/embed` and `/rerank` endpoints. Because the Intel iGPU context is shared, serializing executions prevents multi-stream GPU thrashing and ensures consistent worst-case performance under load.
*   **Length-Bucketing (Padding Reduction)**: When processing mixed-length batches of memories, the server groups incoming candidate texts into length-buckets (e.g., 32, 64, 128 tokens). This avoids forcing unnecessary padding (which slows down GPU cycles) onto short memory lines, preserving sub-250ms query times.


---

## 📦 **5. Unified Docker Stack (Qdrant + OpenVINO)**

To completely isolate your system from legacy configurations, the workspace includes a unified **Docker Compose stack** under `deploy/`. This single-command setup starts both the **Qdrant Vector Database** and your local **OpenVINO Inference Server** with native Intel iGPU/GPU acceleration:

```bash
# Navigate to deployment directory
cd /opt/edumem/deploy

# Build and launch both containers in background
docker compose up --build -d
```

### **Services & Ports Configured:**
*   **`openvino-server` (Port `3002`)**: Hosts the high-performance TEI-like `/embed`, `/rerank`, and `/v1/chat/completions` endpoints. Uses `network_mode: host` to eliminate Docker routing overhead, maps `/dev/dri` directly to your Intel iGPU, and mounts your compiled models folder.
*   **`edumem-qdrant` (Port `6333` & `6334`)**: Standard distributed vector store with persistent storage bound to `qdrant_data`.
