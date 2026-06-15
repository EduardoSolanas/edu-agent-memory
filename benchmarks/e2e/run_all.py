#!/usr/bin/env python3
"""Run lightweight PersonaMem-v2 end-to-end benchmark checks."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python3"
CASES = [
    "single_persona.py",
    "three_personas.py",
    "profile_smoke.py",
    "test_fts_sanitization.py",
]


def main() -> int:
    suite = Path(__file__).resolve().parent
    for case in CASES:
        print(f"\n=== {case} ===", flush=True)
        completed = subprocess.run([str(PYTHON), str(suite / case)], cwd=str(ROOT))
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
