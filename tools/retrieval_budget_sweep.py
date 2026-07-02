#!/usr/bin/env python3
"""
Retrieval Budget Sweep
======================
Measures nugget recall at several EDUMEM_MAX_CONTEXT_CHARS budgets over one
ingested BEAM 100K conversation, with NO answer LLM.

Usage:
  EDUMEM_RETRIEVAL_E2E=1 python tools/retrieval_budget_sweep.py

Prints a table: budget -> overall recall + per-ability recall + mean ctx chars.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ============================================================
#  Env setup (before importing edumem modules)
# ============================================================

os.environ.setdefault("EDUMEM_EMBEDDING_API_URL", "http://127.0.0.1:3002")
os.environ.setdefault("EDUMEM_EMBEDDING_MODEL", "Alibaba-NLP/gte-modernbert-base")
os.environ.setdefault("EDUMEM_EMBEDDINGS_VIA_API", "1")
os.environ.setdefault("EDUMEM_BENCHMARK_PURE_RECALL", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_beam_retrieval_recall import build_nuggets, nugget_recall, ABILITY_MAP, get_cached_beams
from tools.evaluate_beam_end_to_end import (
    answer_with_memory,
)
from edumem.core.beam import BeamMemory

BUDGETS = [2000, 4000, 8000, 16000, 48000]


def _measure_recall_at_budget(items: list[dict], budget: int) -> tuple[float, dict, float]:
    """Return (overall_recall, per_ability_recall, mean_ctx_chars) at given budget.

    items: prebuilt list of {beam, question, ability, nuggets} across all sessions.
    """
    os.environ["EDUMEM_MAX_CONTEXT_CHARS"] = str(budget)

    # Need to re-import to pick up new env value for MAX_MEMORY_CONTEXT_CHARS
    import tools.evaluate_beam_end_to_end as _mod
    _mod.MAX_MEMORY_CONTEXT_CHARS = budget

    ability_recalls: dict[str, list[float]] = {}
    all_chars: list[int] = []

    for it in items:
        ctx = answer_with_memory(None, it["beam"], it["question"],
                                 ability=it["ability"], context_only=True)
        recall = nugget_recall(it["nuggets"], ctx)
        all_chars.append(len(ctx))
        ability_recalls.setdefault(it["ability"], []).append(recall)

    all_recalls = [r for v in ability_recalls.values() for r in v]
    overall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    per_ability = {ab: sum(v) / len(v) for ab, v in ability_recalls.items()}
    mean_chars = sum(all_chars) / len(all_chars) if all_chars else 0.0
    return overall, per_ability, mean_chars


def main():
    db_path, convs_meta = get_cached_beams()
    print("Cache ready.\n", flush=True)

    # Build a flat list of evaluable items across all isolated sessions, each
    # carrying its own per-session BeamMemory (mirrors the recall test).
    beam_cache: dict[str, BeamMemory] = {}
    items: list[dict] = []
    # Bound to the first conversation (5 budgets x all convs is ~90 min); one
    # conv is representative for the budget curve. Override with SWEEP_ALL_CONVS=1.
    if os.environ.get("SWEEP_ALL_CONVS") != "1":
        convs_meta = convs_meta[:1]
    for sid, questions in convs_meta:
        for q in questions:
            rubric = q.get("rubric", [])
            if not rubric:
                continue
            nuggets = build_nuggets(rubric)
            if not nuggets:
                continue
            if sid not in beam_cache:
                beam_cache[sid] = BeamMemory(db_path=db_path, session_id=sid)
            items.append({
                "beam": beam_cache[sid],
                "question": q["question"],
                "ability": ABILITY_MAP.get(q.get("ability", ""), q.get("ability", "")),
                "nuggets": nuggets,
            })

    all_abilities = sorted({it["ability"] for it in items})

    # Header
    ab_cols = "  ".join(f"{ab:>7}" for ab in all_abilities)
    print(f"{'Budget':>8}  {'Overall':>8}  {ab_cols}  {'MeanChars':>10}")
    print("-" * (8 + 2 + 8 + 2 + len(all_abilities) * 9 + 2 + 10))

    for budget in BUDGETS:
        overall, per_ability, mean_chars = _measure_recall_at_budget(items, budget)
        ab_vals = "  ".join(f"{per_ability.get(ab, 0.0):>7.3f}" for ab in all_abilities)
        print(f"{budget:>8}  {overall:>8.3f}  {ab_vals}  {mean_chars:>10.0f}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
