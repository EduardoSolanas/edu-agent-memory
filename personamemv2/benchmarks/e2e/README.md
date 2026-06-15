# PersonaMem-v2 E2E benchmark suite

Purpose: lightweight end-to-end checks for the new compact-profile personalization pipeline.

## Commands

```bash
cd /opt/edumem/personamemv2
.venv/bin/python3 benchmarks/e2e/single_persona.py
.venv/bin/python3 benchmarks/e2e/three_personas.py
.venv/bin/python3 benchmarks/e2e/profile_smoke.py
.venv/bin/python3 benchmarks/e2e/run_all.py
```

## Fixtures

Fixtures live under `benchmarks/e2e/fixtures/`. They are small deterministic slices used for smoke/regression checks only.

## Full benchmarks

Use the full-scale runners one level up:

```bash
LIMIT=70 .venv/bin/python3 benchmarks/personamem_v2_32k.py
LIMIT=70 .venv/bin/python3 benchmarks/personamem_v2_128k.py
.venv/bin/python3 benchmarks/grounded_audit_32k.py
.venv/bin/python3 benchmarks/grounded_audit_128k.py
```
