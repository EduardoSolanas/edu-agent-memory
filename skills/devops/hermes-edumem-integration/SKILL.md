---
name: hermes-edumem-integration
description: "Use when configuring, setting up, or querying the edumem cognitive memory server on CT 116 from Hermes."
version: 1.0.0
author: Eduardo Solanas
license: MIT
metadata:
  hermes:
    tags: [cognitive-memory, edumem, qdrant, openvino, integration]
    related_skills: [hermes-agent, hermes-config-troubleshooting]
---

# edumem Cognitive Memory Integration Guide

## Overview
This skill contains the recipes, endpoints, and diagnostic workflows for integrating and operating the **edumem** cognitive memory engine on **CT 116** (`192.168.1.160`) from Hermes Agent.

The `edumem` stack runs as a single, self-contained multi-process Docker container (`edumem-app` from image `edumem:latest`) co-locating the Node.js `api-daemon` REST server, a Qdrant vector database, and an OpenVINO inference server utilizing Intel Graphics (iGPU) hardware acceleration.

---

## When to Use
- When configuring or troubleshooting Hermes memory providers to target `edumem`.
- When writing custom memory retrieval adapters or client plugins.
- When verifying the performance or health of the GPU-accelerated embedding engine.

---

## Core Configuration & Endpoints

All services are hosted on **CT 116 (`192.168.1.160`)** and exposed via mapped host ports:

| Service | Host Endpoint | Purpose / Function |
| :--- | :--- | :--- |
| **`api-daemon`** | `http://192.168.1.160:6336` | Primary memory REST entry point (`/recall`, `/retain`, `/reflect`) |
| **`OpenVINO`** | `http://192.168.1.160:3002` | Hardware-accelerated embeddings (`gte-modernbert`) & reranking (`ettin-17m`) |
| **`Qdrant`** | `http://192.168.1.160:6333` | Dense vector index and JSON payload storage |

---

## Linked Files & Templates
- `templates/config.yaml`: Starter snippet to add to your Hermes `~/.hermes/config.yaml` to register and activate the `edumem` provider.

---

## 💤 Dreaming & Consolidation (The Sleep Cycle)

To prevent memory retrieval degradation and perform semantic deduplication, the `edumem` provider leverages the local Python-native `BeamMemory.sleep_all_sessions` consolidation pipeline on **CT 116**.

In your **Hermes `edumem` Memory Provider**, this is integrated directly into the **`on_session_end` lifecycle hook**:
1.  When a Hermes agent conversation or session finishes (e.g. on CLI exit), the `on_session_end` event fires automatically.
2.  The provider spins up a non-blocking background thread on the client.
3.  The thread securely triggers the `sleep_all_sessions(force=True)` routine on **CT 116** over SSH:
    ```bash
    ssh root@192.168.1.160 "cd /opt/edumem && .venv/bin/python3 -c 'from edumem.core.beam import BeamMemory; BeamMemory().sleep_all_sessions(force=True)'"
    ```
4.  This consolidates raw working memory into episodic summaries, updates the `consolidated_facts` table, and resolves conflicts without blocking the live agent conversation loop.

---

## One-Shot API Recipes

### 1. Ingestion (`RETAIN`)
Writes a new semantic memory/fact to the user-scoped database.
```bash
curl -s -X POST http://192.168.1.160:6336/v1/default/banks/locomo/memories/retain \
     -H 'Content-Type: application/json' \
     -d '{
       "content": "Tim has been planning to travel to Tokyo, Japan since 2024 to see the cherry blossoms.",
       "tags": ["user:tim_user"]
     }'
```

### 2. Semantic Search (`RECALL`)
Queries the vector database to retrieve the top matching context rows.
```bash
curl -s -X POST http://192.168.1.160:6336/v1/default/banks/locomo/memories/recall \
     -H 'Content-Type: application/json' \
     -d '{
       "query": "Where does Tim want to travel?",
       "tags": ["user:tim_user"]
     }'
```

### 3. Native Model Inference (`EMBED`)
Directly generates high-dimensional vectors on the Intel GPU.
```bash
curl -s -X POST http://192.168.1.160:3002/v1/embeddings \
     -H 'Content-Type: application/json' \
     -d '{
       "input": "This is a verification test of the GPU routing performance."
     }'
```

---

## Common Pitfalls & Troubleshooting

### 1. Model compilation takes 8+ seconds on start
- **Cause:** On cold startup, the Intel OpenCL driver compiles model execution shaders from scratch.
- **Fix:** Ensure the container volume is persistent or caching is enabled (`CACHE_DIR` environment variable mapped). Once the precompiled GPU binary executable buffer (`cl_cache`) is built, subsequent loads complete in **under 1 second**.

### 2. High peak DRAM consumption causing container OOM crashes
- **Cause:** Duplicating constant weight buffers in CPU memory before uploading to GPU.
- **Fix:** Verify you are running OpenVINO version `2026.2.0` or higher, which enables optimized zero-copy IR read mode (memory-mapping `.bin` weight structures directly via `mmap` to prevent duplication).

---

## Verification Checklist

- [ ] Container `edumem-app` is running and healthy: `docker ps`
- [ ] Hardware acceleration is engaged: `docker logs edumem-app | grep GPU`
- [ ] Health check endpoints return success: `curl http://192.168.1.160:6336/health`
- [ ] Vector search returns relevant payloads with Cosine similarity score > 0.6
