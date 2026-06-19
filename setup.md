# api-daemon Setup & Reproduction Guide

This guide walks you through setting up and reproducing the **`api-daemon`** cognitive memory pipeline on a fresh Linux container or machine.

---

## 🛠️ Step-by-Step Setup

### **1. System Requirements & Packages**
Configure your target container or machine (e.g. Debian/Ubuntu container):
```bash
# Update and install build & SQLite dependencies
apt-get update
apt-get install -y curl git sqlite3 libsqlite3-dev python3 python3-pip python3-venv build-essential

# Install Node.js LTS (v20+)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
```

### **2. Unified Docker Compose Stack (Highly Recommended)**
To launch both the **Qdrant Vector Database** and the **OpenVINO Inference Server** (hosting the benchmark-supported `/v1/embeddings` and `/rerank` endpoints) in a unified, single-command environment with native Intel iGPU acceleration:

```bash
# Navigate to the deploy folder
cd /opt/edumem/deploy

# Build and start the services in the background
docker compose up --build -d
```

This will cleanly configure and spin up:
1. **`edumem-qdrant`** (Port `6333` and `6334` for database storage).
2. **`openvino-server`** (Exposing `/v1/embeddings`, `/rerank`, and `/v1/chat/completions` on host port `3002` with integrated GPU devices mapped).

*Alternatively, if you prefer to run Qdrant as a standalone container without the inference server:*
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v /opt/qdrant_storage:/qdrant/storage \
    --name edumem-qdrant \
    --restart always \
    qdrant/qdrant:latest
```

### **3. Clone & Initialize Workspace**
Clone the repository to `/opt/edumem` and install standard node requirements:
```bash
git clone https://github.com/EduardoSolanas/edu-agent-memory.git /opt/edumem
cd /opt/edumem
npm install
```

### **4. Create Python Virtual Environment**
Initialize the isolated Python runtime used by the evaluation and analysis suite:
```bash
cd /opt/edumem/personamemv2
python3 -m venv .venv
source .venv/bin/activate

# Install requirements and sqlite-vec binary for local sandbox testing
pip install --upgrade pip
pip install datasets requests tqdm numpy scipy pydantic sqlite-vec
```

### **5. OpenVINO Models**
The Dockerfile now runs the model export step during `podman build` or `docker build`, so you do not need to pre-export the models on the host.

Build with an authenticated Hugging Face token if needed. If the token is only in `.env`, load it into PowerShell first:
```powershell
$env:HF_TOKEN = ((Get-Content .env | Where-Object { $_ -match '^HF_TOKEN=' } | Select-Object -First 1) -replace '^HF_TOKEN=', '').Trim('"')
podman build --build-arg HF_TOKEN=$env:HF_TOKEN -t edumem:latest .
```

If you prefer to export manually on the host, the script is still available:
```bash
python3 /opt/edumem/bin/prepare_models.py
python3 /opt/edumem/bin/prepare_models.py --force
```

The build and the script both prepare these FP16 OpenVINO models:
1. **Embedding model** (`sentence-transformers/all-mpnet-base-v2`, 768 dims) -> `models/gte-modernbert-ov`
2. **MiniLM Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) -> `models/ettin-17m-ov`

For the official BEAM runner, the only supported dense embedding path is:
`http://localhost:3002/v1/embeddings`
with `EDUMEM_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2` and
`EDUMEM_EMBEDDING_DIM=768`.

### **6. Start and Register `api-daemon` Service**
1. Generate the service defaults environment file at **`/etc/default/api-daemon`** containing your active API keys:
   ```bash
   GEMINI_API_KEY=your_g...here
   OPENAI_API_KEY=your_o...here
   ```

2. Register the daemon as a systemd service file at **`/etc/systemd/system/api-daemon.service`**:
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

3. Reload and start:
   ```bash
   systemctl daemon-reload
   systemctl enable api-daemon.service
   systemctl start api-daemon.service
   ```

---

## 🧪 **Verification & Smoke Testing**

### **1. Check the active `api-daemon` Service**
```bash
systemctl status api-daemon.service
```
Expected output: `Active: active (running) ... api-daemon Server running on http://0.0.0.0:6336`

### **2. Execute the Node.js Memory Check**
```bash
node /opt/edumem/benchmarks/smoke.mjs
```
Expected output:
```text
STORE "fb79bfe4-b76b-4c86-bd16-335385214ee9"
RECALL_COUNT 1
TOP_TEXT edumem smoke test memory: agent-memory keeps Hindsight on 8888 and edumem Qdrant on 6333.
```

### **3. Run the Python BEAM Benchmark Suite**
Confirm your Python environment, HuggingFace dataset caching, and model configurations are correct by running a quick dry-run:
```bash
python3 /opt/edumem/benchmarks/run_beam_official.py --provider gemini --model gemini-2.5-flash --dry-run
```
Expected output: `Dry run complete. Exiting.`
