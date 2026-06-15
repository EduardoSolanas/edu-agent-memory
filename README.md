# api-daemon Workspace

This repository contains the **api-daemon** service and the active evaluation/benchmarks used to measure, test, and profile our custom cognitive memory pipeline.

---

## 🚀 Overview of api-daemon

The `api-daemon` runs on CT 116 (port **`6336`**) and is the single, authoritative entrypoint for calling agents (e.g., Hermes on CT 108) to query and interact with memory.

It hosts two primary production endpoints compatible with the Standard Memory API protocol:
1.  **RAG Recall (`POST /v1/default/banks/:bank/memories/recall`)**: For fast vector-based retrieval.
2.  **RAG Reflection (`POST /v1/default/banks/:bank/reflect`)**: For generating chronological, formatted participant context blocks.

In the background, it runs a scheduled 12-hour dreaming and consolidation job using local embedding (`gte-modernbert-base`) and reranking (`Ettin-17M`) servers to optimize, decay, and deduplicate agent memories.

---

## 🛠️ Step-by-Step Setup & Reproduction

### 1. System Requirements & Packages
Configure your target container or machine (e.g. Debian/Ubuntu container):
```bash
apt-get update
apt-get install -y curl git sqlite3 libsqlite3-dev python3 python3-pip python3-venv build-essential
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
```

### 2. Qdrant Vector DB
Start Qdrant inside a Docker container:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v /opt/qdrant_storage:/qdrant/storage \
    --name mnemosyne-qdrant \
    --restart always \
    qdrant/qdrant:latest
```

### 3. Clone & Initialize Workspace
Clone the repository to `/opt/edumem` and install standard node requirements:
```bash
git clone https://github.com/EduardoSolanas/edu-agent-memory.git /opt/edumem
cd /opt/edumem
npm install
```

### 4. Create Python Virtual Environment
Initialize the Python environment used by the evaluation harness:
```bash
cd /opt/edumem/personamemv2
python3 -m venv .venv
source .venv/bin/activate
pip install datasets requests tqdm numpy scipy pydantic sqlite-vec
```

### 5. Start and Register api-daemon Service
Register `api-daemon` as a persistent systemd service at `/etc/systemd/system/api-daemon.service`:
```ini
[Unit]
Description=api-daemon
After=network.target 

[Service]
Type=simple
User=root
WorkingDirectory=/opt/edumem
EnvironmentFile=-/etc/default/api-daemon
EnvironmentFile=-/etc/environment
ExecStart=/usr/bin/node /opt/edumem/bin/api-daemon.mjs
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Reload and start:
```bash
systemctl daemon-reload
systemctl enable api-daemon.service
systemctl start api-daemon.service
```

### 6. Verify with Smoke Tests
```bash
# Node.js backend check (recalled count should be 1)
node /opt/edumem/benchmarks/smoke.mjs

# Dry-run of Python BEAM evaluation
python3 /opt/edumem/benchmarks/run_beam_official.py --provider gemini --model gemini-2.5-flash --dry-run
```
