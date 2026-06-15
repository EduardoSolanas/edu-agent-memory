# AGENTS.md — edumem Memory System Benchmarking and Testing Guide

This document defines the architecture, standards, and execution steps for PersonaMem-v2 and LongMemEval memory benchmarks on the **agent-memory** container (CT 116).

---

## How to Run the Tests

Always run tests using the designated virtual environment on CT 116: `/opt/edumem/personamemv2/.venv/bin/python3`.
Set `PYTHONPATH=/opt/edumem/personamemv2` before executing to ensure internal packages resolve correctly.

### 1. PersonaMem-v2 32k Benchmark (Full 70-User Run)
Exercises the real `BeamMemory` database pipeline under a 32k-token context footprint.
```bash
env PYTHONPATH=/opt/edumem/personamemv2 LIMIT=70 /opt/edumem/personamemv2/.venv/bin/python3 /opt/edumem/personamemv2/benchmarks/personamem_v2_32k.py
```
*(For quick smoke checks, run with a smaller subset limit: `LIMIT=5`)*

### 2. LongMemEval Chronological Timeline Benchmark
Exercises chronological-retrieval and multi-session historical timeline profiling.
```bash
env LIMIT=500 /opt/edumem/personamemv2/.venv/bin/python3 /opt/LongMemEval/run_longmemeval_full.py
```
*(For quick smoke checks, run with `LIMIT=5`)*

### 3. E2E Suite and Gate Tests
Validates the system against specialized persona scenarios and privacy preservation constraints.
```bash
# E2E 3-Persona Gate
env PYTHONPATH=/opt/edumem/personamemv2 /opt/edumem/personamemv2/.venv/bin/python3 /opt/edumem/personamemv2/benchmarks/e2e/three_personas.py

# E2E Single Persona (ID 521)
env PYTHONPATH=/opt/edumem/personamemv2 /opt/edumem/personamemv2/.venv/bin/python3 /opt/edumem/personamemv2/benchmarks/e2e/single_persona.py

# E2E Profile Smoke Test
env PYTHONPATH=/opt/edumem/personamemv2 /opt/edumem/personamemv2/.venv/bin/python3 /opt/edumem/personamemv2/benchmarks/e2e/profile_smoke.py
```

---

## What NOT to Do (Anti-Patterns and Pitfalls)

To preserve scientific rigor and system integrity, adhere strictly to these rules:

1. **Do NOT Use Offline In-Memory Simulations or TF-IDF Mocks**
   - *Pitfall:* Using keyword searchers (such as `best_evidence` or raw python word overlap) masks the real-world performance of the system.
   - *Requirement:* All tests must exercise the real production path via `from edumem.core.beam import BeamMemory`, forcing actual writes, indexing, and queries on disk.

2. **Do NOT Use Raw OS Copying (`shutil.copy2`) on Active SQLite Files**
   - *Pitfall:* Copying an active database using raw filesystem operations can break or corrupt SQLite virtual tables, particularly `fts5` and `sqlite-vec` (resulting in `Error opening vector blob`).
   - *Requirement:* Always let the SQLite engine initialize files cleanly, or use appropriate database-level dump/restore commands.

3. **Do NOT Use Prompt-Hacking or Hardcoded Synonym Lists**
   - *Pitfall:* Hardcoding specialized word lists to artificially boost dataset-specific terms (e.g., boosting "cereal" or "stroller" to match specific questions) constitutes target leakage and overfits the benchmarks.
   - *Requirement:* Any improvements must come from generalized architectural enhancements, such as query-level synonym expansion, FTS5/vector weight tuning, or chronological context engineering.

4. **Do NOT Run Parallel Tests with Unlimited Concurrency**
   - *Pitfall:* Issuing unrestricted parallel requests to background LLM APIs (e.g. NAN API or OpenAI) will trigger strict `429 Rate Limit Exceeded` exceptions or crash local inference servers.
   - *Requirement:* Set safe semaphores or run within a ThreadPoolExecutor limited to a maximum of **4 workers**.

5. **Do NOT Forget to Isolate Database Files Between Test Iterations**
   - *Pitfall:* Accumulating stale memory files across multiple test runs causes database bloat and cross-contamination, leading to false-positive or false-negative results.
   - *Requirement:* Explicitly clean, wipe, or `.unlink()` the target database file (such as `results/lme/databases/banks/{qid}.db`) before starting a fresh run for a user.
