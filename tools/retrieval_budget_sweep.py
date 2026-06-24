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

os.environ.setdefault("EDUMEM_EMBEDDING_API_URL", "http://localhost:3002")
os.environ.setdefault("EDUMEM_EMBEDDING_MODEL", "Alibaba-NLP/gte-modernbert-base")
os.environ.setdefault("EDUMEM_EMBEDDINGS_VIA_API", "1")
os.environ.setdefault("EDUMEM_BENCHMARK_PURE_RECALL", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_beam_retrieval_recall import build_nuggets, nugget_recall, ABILITY_MAP, get_cached_beam_and_conv
from tools.evaluate_beam_end_to_end import (
    answer_with_memory,
)

BUDGETS = [1500, 3000, 6000, 12000, 16000]


def _measure_recall_at_budget(beam, questions: list[dict], budget: int) -> tuple[float, dict, float]:
    """Return (overall_recall, per_ability_recall, mean_ctx_chars) at given budget."""
    os.environ["EDUMEM_MAX_CONTEXT_CHARS"] = str(budget)

    # Need to re-import to pick up new env value for MAX_MEMORY_CONTEXT_CHARS
    import tools.evaluate_beam_end_to_end as _mod
    _mod.MAX_MEMORY_CONTEXT_CHARS = budget

    ability_recalls: dict[str, list[float]] = {}
    all_chars: list[int] = []

    for q in questions:
        rubric = q.get("rubric", [])
        if not rubric:
            continue
        ability_raw = q.get("ability", "")
        ability = ABILITY_MAP.get(ability_raw, ability_raw)
        nuggets = build_nuggets(rubric)
        if not nuggets:
            continue

        ctx = answer_with_memory(None, beam, q["question"], ability=ability, context_only=True)
        recall = nugget_recall(nuggets, ctx)
        all_chars.append(len(ctx))
        ability_recalls.setdefault(ability, []).append(recall)

    all_recalls = [r for v in ability_recalls.values() for r in v]
    overall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    per_ability = {ab: sum(v) / len(v) for ab, v in ability_recalls.items()}
    mean_chars = sum(all_chars) / len(all_chars) if all_chars else 0.0
    return overall, per_ability, mean_chars


def main():
    beam, conv = get_cached_beam_and_conv()
    print("Cache ready.\n", flush=True)

    questions = conv.get("questions", [])
    # Collect all abilities for header
    all_abilities = sorted({
        ABILITY_MAP.get(q.get("ability", ""), q.get("ability", ""))
        for q in questions if q.get("rubric")
    })

    # Header
    ab_cols = "  ".join(f"{ab:>7}" for ab in all_abilities)
    print(f"{'Budget':>8}  {'Overall':>8}  {ab_cols}  {'MeanChars':>10}")
    print("-" * (8 + 2 + 8 + 2 + len(all_abilities) * 9 + 2 + 10))

    for budget in BUDGETS:
        overall, per_ability, mean_chars = _measure_recall_at_budget(beam, questions, budget)
        ab_vals = "  ".join(f"{per_ability.get(ab, 0.0):>7.3f}" for ab in all_abilities)
        print(f"{budget:>8}  {overall:>8.3f}  {ab_vals}  {mean_chars:>10.0f}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
