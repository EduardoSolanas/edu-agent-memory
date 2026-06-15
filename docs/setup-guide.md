# edumem Setup & Reproduction Guide

This guide walks you through setting up and reproducing the **edumem** cognitive memory pipeline on a fresh Linux container or machine.

---

## 1. System Requirements & Prereqs

Ensure the target system (e.g. Debian/Ubuntu container) has the following packages installed:

```bash
# Update package list and install system dependencies
apt-get update
apt-get install -y curl git sqlite3 libsqlite3-dev python3 python3-pip python3-venv build-essential
```

### Node.js (v18+)
```bash
# Install Node.js LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
```

---

## 2. Qdrant Vector Database Setup

`edumem` uses Qdrant for fast vector searches. Run Qdrant inside a Docker container:

```bash
# Start Qdrant with local persistence on standard ports
docker run -d -p 6333:6333 -p 6334:6334 \
    -v /opt/qdrant_storage:/qdrant/storage \
    --name mnemosyne-qdrant \
    --restart always \
    qdrant/qdrant:latest
```

---

## 3. Clone & Initialize edumem

Clone your repository to `/opt/mnemosyne` and install the Node.js daemon dependencies:

```bash
# Clone the repository
git clone https://github.com/EduardoSolanas/edu-agent-memory.git /opt/mnemosyne

# Install mnemosy-ai dependency
cd /opt/mnemosyne
npm install
```

---

## 4. Python Environment & Benchmarks Setup

`edumem` relies on a Python virtual environment to execute academic evaluations (like BEAM) and handle local vector operations using `sqlite-vec`.

```bash
# Create the python virtual environment
cd /opt/mnemosyne/personamemv2
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install core evaluation packages
pip install --upgrade pip
pip install datasets requests tqdm numpy scipy pydantic

# Install sqlite-vec for local vector sandbox capabilities
pip install sqlite-vec
```

---

## 5. Configure the Cognitive Daemon Service

`edumem` runs a Node.js daemon that schedules dreaming (decay, deduplication) and memory consolidation. It also exposes a Hindsight-compatible RAG API on port `6336`.

Create the systemd service file: `/etc/systemd/system/mnemosyne-cognitive-daemon.service`

```ini
[Unit]
Description=Mnemosyne Cognitive Daemon
After=network.target 

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mnemosyne
# Expose Gemini / OpenAI API keys through local configuration files
EnvironmentFile=-/etc/default/hindsight
EnvironmentFile=-/etc/environment
ExecStart=/usr/bin/node /opt/mnemosyne/bin/mnemosyne-daemon.mjs
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the daemon:
```bash
systemctl daemon-reload
systemctl enable mnemosyne-cognitive-daemon.service
systemctl start mnemosyne-cognitive-daemon.service
```

---

## 6. Integrating with OpenVINO GenAI

The embedding and reranking tasks run on our local OpenVINO servers to eliminate latency spikes.

Ensure your `mnemosyne-daemon.mjs` config maps to your active OpenVINO endpoints:
*   **Embeddings**: `http://127.0.0.1:3002/embed` (running `gte-modernbert-base`)
*   **Reranking**: `http://127.0.0.1:3002/rerank` (running `Ettin-17M`)

---

## 7. Fast Verification

Confirm the setup is running correctly using the offline smoke tests:

```bash
# Run the Node.js local DB connection check
node /opt/mnemosyne/benchmarks/smoke.mjs

# Run the official BEAM fast-cached evaluation (using precompiled DB)
python3 /opt/mnemosyne/benchmarks/run_beam_official.py --provider gemini --model gemini-2.5-flash
```
