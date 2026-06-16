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
To launch both the **Qdrant Vector Database** and the **OpenVINO Inference Server** (hosting the TEI-like `/embed` and `/rerank` endpoints) in a unified, single-command environment with native Intel iGPU acceleration:

```bash
# Navigate to the deploy folder
cd /opt/edumem/deploy

# Build and start the services in the background
docker compose up --build -d
```

This will cleanly configure and spin up:
1. **`edumem-qdrant`** (Port `6333` and `6334` for database storage).
2. **`openvino-server`** (Exposing `/embed`, `/rerank`, and `/v1/chat/completions` on host port `3002` with integrated GPU devices mapped).

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

### **5. Prepare & Export OpenVINO Models**
Before building the Docker image or running the OpenVINO Inference Server, you need to download the raw Hugging Face models and export them to OpenVINO IR (FP16) format.

A convenient, non-interactive script is provided for this purpose:
```bash
python3 /opt/edumem/bin/prepare_models.py
```

This script will:
1. Parse `/opt/edumem/.env` to read `HF_TOKEN` if present (falling back to standard system environment variables like `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`).
2. Automatically check if `optimum-intel[openvino]` is installed in the active virtual environment `/opt/edumem/.venv`, and install it if missing.
3. Check if `models/gte-modernbert-ov` and `models/ettin-17m-ov` already exist. If they do, it will skip downloading/exporting to prevent redundant disk and network operations.
4. Execute `optimum-cli export openvino` under the hood to export:
   - **GTE ModernBERT** (`Alibaba-NLP/gte-modernbert-base`) with `--task feature-extraction` and `--weight-format fp16` to `models/gte-modernbert-ov`.
   - **Ettin Reranker** (`cross-encoder/ettin-reranker-17m-v1`) with `--task text-classification` and `--weight-format fp16` to `models/ettin-17m-ov`.

*Note: If you need to force re-exporting of existing models, you can run the script with the `--force` flag:*
```bash
python3 /opt/edumem/bin/prepare_models.py --force
```

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
