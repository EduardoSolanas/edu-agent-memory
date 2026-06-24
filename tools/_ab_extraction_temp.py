#!/usr/bin/env python3
"""TEMPORARY A/B -- not committed.

Retrieval-level A/B: does write-time LLM extraction (EDUMEM_LLM_EXTRACTION=1)
surface more answer-bearing nuggets in the retrieved context than the current
default (regex + LLM metric-consolidation, flag=0)?

Both arms use identical recall (RRF fusion). They differ ONLY in the
write/extraction path. We measure NUGGET RECALL in the retrieved context (fast,
no judge), like the prior RRF A/B. NO_EMBEDDINGS=1 for speed (RRF specialists
are local SQL).
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Dense embeddings via the container (the REAL recall path — gte-modernbert).
os.environ["EDUMEM_EMBEDDING_API_URL"] = "http://localhost:3002"
os.environ["EDUMEM_EMBEDDING_MODEL"] = "Alibaba-NLP/gte-modernbert-base"

# Load NAN key from .env without printing it.
def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

NAN_KEY = os.environ.get("NAN_APY_KEY", "")
NAN_BASE = "https://api.nan.builders/v1"
if not NAN_KEY:
    print("ERROR: NAN_APY_KEY not found in .env")
    sys.exit(1)

from tools.evaluate_beam_end_to_end import (
    load_beam_dataset, ingest_conversation, LLMClient, ABILITY_MAP,
)
from edumem.core.beam import BeamMemory


def make_llm():
    return LLMClient(model="qwen3.6", api_key=NAN_KEY, base_url=NAN_BASE)


def build_nuggets(rubric) -> list[str]:
    """Strip rubric prefixes, lowercase -> expected answer-nuggets."""
    prefixes = (
        "llm response should state:",
        "llm response should contain:",
        "llm response should mention:",
        "llm response should state that",
        "llm response should contain",
        "llm response should mention",
        "response should state:",
        "response should contain:",
        "response should mention:",
        "the response should state:",
        "the response should mention:",
        "the response should contain:",
    )
    out = []
    for item in rubric or []:
        if isinstance(item, dict):
            text = item.get("criterion") or item.get("text") or item.get("description") or ""
        else:
            text = str(item)
        low = text.strip().lower()
        for p in prefixes:
            if low.startswith(p):
                low = low[len(p):].strip()
                break
        # generic "... state/contain/mention: X" fallback
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
    """Whitespace-normalized + space-stripped containment (250ms == 250 ms)."""
    if not nugget:
        return False
    n_norm = " ".join(nugget.lower().split())
    c_norm = " ".join((context or "").lower().split())
    if n_norm in c_norm:
        return True
    # space-stripped variant for units like "250 ms"
    return _norm(nugget) in _norm(context)


def storage_stats(beam: BeamMemory) -> dict:
    cur = beam.conn.cursor()
    try:
        total = cur.execute("SELECT COUNT(*) FROM memoria_facts").fetchone()[0]
        distinct = cur.execute("SELECT COUNT(DISTINCT key) FROM memoria_facts").fetchone()[0]
        chains = cur.execute(
            "SELECT COUNT(*) FROM memoria_facts WHERE previous_value IS NOT NULL"
        ).fetchone()[0]
    except Exception as e:
        return {"error": str(e)}
    return {"facts": total, "distinct_keys": distinct, "version_chains": chains}


def run_arm(arm_flag: str, conversations: list[dict]) -> dict:
    os.environ["EDUMEM_LLM_EXTRACTION"] = arm_flag
    arm_name = "ON(LLM)" if arm_flag == "1" else "OFF(regex)"
    print(f"\n=== ARM {arm_name} (EDUMEM_LLM_EXTRACTION={arm_flag}) ===")

    per_q = []  # list of (n_recalled, n_total)
    storage = {"facts": 0, "distinct_keys": 0, "version_chains": 0}
    ingest_secs = 0.0

    for ci, conv in enumerate(conversations):
        llm = make_llm()
        tmpdir = tempfile.mkdtemp(prefix=f"abx_{arm_flag}_{ci}_")
        db_path = Path(tmpdir) / "beam.db"
        beam = BeamMemory(session_id=f"ab_{arm_flag}_{ci}",
                          db_path=db_path, llm_client=llm)

        t0 = time.perf_counter()
        ingest_conversation(beam, conv["messages"], llm=llm)
        dt = time.perf_counter() - t0
        ingest_secs += dt
        print(f"  conv {ci}: ingested {len(conv['messages'])} msgs in {dt:.1f}s")

        st = storage_stats(beam)
        if "error" not in st:
            for k in storage:
                storage[k] += st.get(k, 0)
        else:
            print(f"  conv {ci}: storage_stats error: {st['error']}")

        for q in conv["questions"]:
            question = q.get("question", "")
            rubric = q.get("rubric", [])
            nuggets = build_nuggets(rubric)
            if not question or not nuggets:
                continue
            ability = ABILITY_MAP.get(q.get("ability", ""), None)
            try:
                res = beam.memoria_retrieve(question, ability=ability, top_k=10)
                context = res.get("context", "") if isinstance(res, dict) else ""
            except Exception as e:
                print(f"  retrieve error: {e}")
                context = ""
            recalled = sum(1 for n in nuggets if nugget_in_context(n, context))
            per_q.append((recalled, len(nuggets)))

    return {"arm": arm_name, "per_q": per_q, "storage": storage,
            "ingest_secs": ingest_secs, "llm_calls": None}


def main():
    print("Loading BEAM 100K, first 2 conversations...")
    ds = load_beam_dataset(["100K"], max_conversations=2)
    conversations = ds.get("100K", [])
    if not conversations:
        print("ERROR: no conversations loaded")
        sys.exit(1)
    print(f"Loaded {len(conversations)} conversations; "
          f"questions: {[len(c['questions']) for c in conversations]}")

    off = run_arm("0", conversations)
    on = run_arm("1", conversations)

    def total_recalled(arm):
        return sum(r for r, _ in arm["per_q"])

    def total_nuggets(arm):
        return sum(t for _, t in arm["per_q"])

    # paired per-question comparison (same question order across arms)
    on_more = off_more = same = 0
    for (ro, _), (rn, _) in zip(off["per_q"], on["per_q"]):
        if rn > ro:
            on_more += 1
        elif ro > rn:
            off_more += 1
        else:
            same += 1

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for arm in (off, on):
        print(f"\nArm {arm['arm']}:")
        print(f"  questions scored : {len(arm['per_q'])}")
        print(f"  nuggets recalled : {total_recalled(arm)} / {total_nuggets(arm)}")
        print(f"  ingest time      : {arm['ingest_secs']:.1f}s")
        print(f"  storage          : facts={arm['storage']['facts']} "
              f"distinct_keys={arm['storage']['distinct_keys']} "
              f"version_chains={arm['storage']['version_chains']}")

    print(f"\nPaired per-question (n={len(off['per_q'])}):")
    print(f"  ON surfaced MORE  : {on_more}")
    print(f"  OFF surfaced MORE : {off_more}")
    print(f"  tie               : {same}")

    delta = total_recalled(on) - total_recalled(off)
    print(f"\nVERDICT: LLM extraction recalled {total_recalled(on)} vs regex "
          f"{total_recalled(off)} answer-nuggets (delta {delta:+d}).")


if __name__ == "__main__":
    main()
