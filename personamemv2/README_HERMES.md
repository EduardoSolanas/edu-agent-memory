# PersonaMem-v2 new-system experiments

Purpose: isolated workspace for PersonaMem-v2, which is a different benchmark from v1.

Key difference from v1:
- v2 is implicit-persona personalization, not simple explicit preference recall.
- Raw turn-level recall scored poorly on first smoke: `2/10`.
- Serious v2 tuning builds compact inferred per-user memory profiles, then answers from those profiles.

Current layout:
- `benchmarks/e2e/` — lightweight end-to-end benchmark checks and fixtures.
- `benchmarks/personamem_v2_32k.py` — full 70-scenario 32k context benchmark.
- `benchmarks/personamem_v2_128k.py` — full 70-scenario 128k context benchmark.
- `benchmarks/grounded_audit_32k.py` — grounded validation audit for 32k outputs.
- `benchmarks/grounded_audit_128k.py` — grounded validation audit for 128k outputs.
- `data_hf/` — local Hugging Face dataset cache.

Rules:
- Use the new compact-profile pipeline for PersonaMem-v2 validation.
- Do not use the old Qdrant daemon path.
- Do not use `related_conversation_snippet` as memory input; it is only for debugging/oracle analysis.
- Keep benchmark fixtures under `benchmarks/e2e/fixtures/`, not in the project root.
