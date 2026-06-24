#!/usr/bin/env python3
"""TEMPORARY graph A/B -- not committed.

Isolates the KG (graph) recall source. Both arms use the SAME ingested DBs and
the SAME RRF fusion. They differ ONLY in whether the `_memoria_kg_retrieve`
specialist participates:

  GRAPH-ON : KG triples (entities/relations) fuse alongside fact/timeline/
             negation/chrono.
  GRAPH-OFF: `_memoria_kg_retrieve` is stubbed to empty -> graph excluded,
             every other specialist unchanged.

We measure nugget recall in the retrieved context (fast, no judge). NO
re-ingestion: we reuse the ON-arm DBs already on disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Real recall path: dense gte-modernbert via the container.
os.environ["EDUMEM_EMBEDDING_API_URL"] = "http://localhost:3002"
os.environ["EDUMEM_EMBEDDING_MODEL"] = "Alibaba-NLP/gte-modernbert-base"
os.environ["EDUMEM_EMBEDDINGS_VIA_API"] = "1"
os.environ["EDUMEM_LLM_EXTRACTION"] = "0"  # retrieval only; no write path here

from tools.evaluate_beam_end_to_end import load_beam_dataset, ABILITY_MAP
from edumem.core.beam import BeamMemory

# conv index -> persisted ON-arm DB (session ab_1_<ci>)
TMP = Path("C:/Users/eduar/AppData/Local/Temp")
DBS = {
    0: TMP / "abx_1_0_ztp95pkl" / "beam.db",   # full graph: facts+entities+relations
    1: TMP / "abx_1_1_chjeck7i" / "beam.db",   # relations-only graph
}

_EMPTY = {"context": "", "facts": [], "source": "fallback"}

# Fact abilities the graph is meant to lift (KU/MR/IE/TR/SUM). The others
# (ABS/CR/EO/IF/PF) have meta-judgment rubrics that are not literally present
# in retrieved context, so they floor nugget recall at 0 and are excluded.
FACT_ABILITIES = {
    "knowledge_update", "multi_session_reasoning", "information_extraction",
    "temporal_reasoning", "summarization",
}


def build_nuggets(rubric) -> list:
    prefixes = (
        "llm response should state:", "llm response should contain:",
        "llm response should mention:", "response should state:",
        "response should contain:", "response should mention:",
        "the response should state:", "the response should mention:",
        "the response should contain:",
    )
    out = []
    for item in rubric or []:
        text = (item.get("criterion") or item.get("text") or item.get("description") or ""
                ) if isinstance(item, dict) else str(item)
        low = text.strip().lower()
        for p in prefixes:
            if low.startswith(p):
                low = low[len(p):].strip()
                break
        for marker in ("state:", "contain:", "mention:", "states:", "contains:", "mentions:"):
            idx = low.find(marker)
            if idx != -1 and idx < 60:
                low = low[idx + len(marker):].strip()
                break
        low = low.strip().strip(".").strip()
        if low:
            out.append(low)
    return out


def _norm(s: str) -> str:
    return "".join((s or "").lower().split())


def nugget_in_context(nugget: str, context: str) -> bool:
    if not nugget:
        return False
    n = " ".join(nugget.lower().split())
    c = " ".join((context or "").lower().split())
    return (n in c) or (_norm(nugget) in _norm(context))


def retrieve_recall(beam, question, ability, nuggets) -> int:
    try:
        res = beam.memoria_retrieve(question, ability=ability, top_k=10)
        ctx = res.get("context", "") if isinstance(res, dict) else ""
    except Exception as e:
        print(f"  retrieve error: {e}")
        ctx = ""
    return sum(1 for n in nuggets if nugget_in_context(n, ctx))


def main():
    ds = load_beam_dataset(["100K"], max_conversations=2)
    conversations = ds.get("100K", [])
    print(f"Loaded {len(conversations)} conversations; "
          f"questions: {[len(c['questions']) for c in conversations]}")

    on_pairs = []   # (recalled_on, total)
    off_pairs = []  # (recalled_off, total)

    for ci, conv in enumerate(conversations):
        db = DBS.get(ci)
        if not db or not db.exists():
            print(f"conv {ci}: no persisted DB ({db}) -- skipping")
            continue
        beam = BeamMemory(session_id=f"ab_1_{ci}", db_path=db, llm_client=None)
        kg_rows = beam.conn.execute("SELECT COUNT(*) FROM memoria_kg").fetchone()[0]
        print(f"\nconv {ci}: {db.parent.name}  (kg_rows={kg_rows})")

        for q in conv["questions"]:
            if q.get("ability") not in FACT_ABILITIES:
                continue
            question = q.get("question", "")
            nuggets = build_nuggets(q.get("rubric", []))
            if not question or not nuggets:
                continue
            ability = ABILITY_MAP.get(q.get("ability", ""), None)

            # GRAPH-ON
            r_on = retrieve_recall(beam, question, ability, nuggets)
            # GRAPH-OFF: stub the KG specialist to empty, restore after
            orig = beam._memoria_kg_retrieve
            beam._memoria_kg_retrieve = lambda *a, **k: dict(_EMPTY)
            try:
                r_off = retrieve_recall(beam, question, ability, nuggets)
            finally:
                beam._memoria_kg_retrieve = orig

            on_pairs.append((r_on, len(nuggets)))
            off_pairs.append((r_off, len(nuggets)))

    def tot(p):
        return sum(r for r, _ in p)

    on_more = off_more = same = 0
    for (ron, _), (roff, _) in zip(on_pairs, off_pairs):
        if ron > roff:
            on_more += 1
        elif roff > ron:
            off_more += 1
        else:
            same += 1

    print("\n" + "=" * 60)
    print("GRAPH A/B RESULTS  (same DBs, same fusion, KG source toggled)")
    print("=" * 60)
    n = len(on_pairs)
    total = sum(t for _, t in on_pairs)
    print(f"questions scored : {n}")
    print(f"nuggets (total)  : {total}")
    print(f"  GRAPH-ON  recalled : {tot(on_pairs)}")
    print(f"  GRAPH-OFF recalled : {tot(off_pairs)}")
    print(f"\nPaired per-question:")
    print(f"  GRAPH helped (on>off) : {on_more}")
    print(f"  GRAPH hurt   (off>on) : {off_more}")
    print(f"  tie                   : {same}")
    delta = tot(on_pairs) - tot(off_pairs)
    print(f"\nVERDICT: graph fusion changed nugget recall by {delta:+d} "
          f"({tot(on_pairs)} vs {tot(off_pairs)}).")


if __name__ == "__main__":
    main()
