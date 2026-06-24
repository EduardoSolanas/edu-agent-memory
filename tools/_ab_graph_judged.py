#!/usr/bin/env python3
"""TEMPORARY judged graph A/B -- not committed.

Reuses the official answer_with_memory + judge_with_rubrics, but over an
ALREADY-INGESTED case-0 DB (no re-ingestion) and serially (no worker lock).
Toggles ONLY EDUMEM_KG_FUSION between arms, so storage + everything else is
identical; the single difference is whether the KG specialist fuses at recall.

Per-question: retrieve -> qwen3.6 answer (enable_thinking=false, 300s timeout)
-> deepseek judge (reasoning_effort=low). ~11s/question; 20 q x 2 arms ~= 7 min.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
NAN_KEY = os.environ.get("NAN_APY_KEY", "")
# Harness LLMClient reads these at import time -- set BEFORE importing it.
os.environ["OPENROUTER_API_KEY"] = NAN_KEY
os.environ["OPENROUTER_BASE_URL"] = "https://api.nan.builders/v1"
os.environ["EDUMEM_EMBEDDING_API_URL"] = "http://localhost:3002"
os.environ["EDUMEM_EMBEDDING_MODEL"] = "Alibaba-NLP/gte-modernbert-base"
os.environ["EDUMEM_EMBEDDINGS_VIA_API"] = "1"
os.environ["EDUMEM_LLM_EXTRACTION"] = "0"        # retrieval/answer only
os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = "1"  # no oracle bypasses
os.environ["EDUMEM_MAX_CONTEXT_CHARS"] = "16000"
os.environ["BEAM_LLM_TIMEOUT"] = "300"

from tools.evaluate_beam_end_to_end import (
    load_beam_dataset, ABILITY_MAP, LLMClient,
    answer_with_memory, judge_with_rubrics, _summarize_judge_result,
)
from edumem.core.beam import BeamMemory

DB = Path("C:/Users/eduar/AppData/Local/Temp/abx_1_0_ztp95pkl/beam.db")
ARMS = [("graph", "1"), ("nograph", "0")]
_TAG = os.environ.get("GRAPH_AB_TAG", "")
CKPT = PROJECT_ROOT / (f"graph_judged_scores{('_'+_TAG) if _TAG else ''}.jsonl")  # per (qi,arm)


def _load_ckpt() -> dict:
    """Return {(qi, arm): score} from prior (possibly killed) runs."""
    done = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[(r["qi"], r["arm"])] = r
            except Exception:
                continue
    return done


def _append_ckpt(rec: dict):
    with open(CKPT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def main():
    if not NAN_KEY:
        print("ERROR: NAN_APY_KEY missing"); sys.exit(1)
    if not DB.exists():
        print(f"ERROR: pre-ingested DB not found: {DB}"); sys.exit(1)

    ds = load_beam_dataset(["100K"], max_conversations=1)
    conv = ds["100K"][0]
    questions = conv["questions"]
    print(f"conv 0: {len(questions)} questions; DB={DB.parent.name}")

    llm = LLMClient(model="qwen3.6", api_key=NAN_KEY, base_url="https://api.nan.builders/v1")
    judge = LLMClient(model="deepseek-v4-flash", api_key=NAN_KEY, base_url="https://api.nan.builders/v1")
    beam = BeamMemory(session_id="ab_1_0", db_path=DB, llm_client=None)

    done = _load_ckpt()
    print(f"resume: {len(done)} (qi,arm) cells already scored")

    # scores[arm][ability_code] = [official_score, ...]
    scores = {arm: defaultdict(list) for arm, _ in ARMS}

    for qi, q in enumerate(questions):
        question = q.get("question", "")
        rubric = q.get("rubric", [])
        ab_raw = q.get("ability", "")
        ability = ABILITY_MAP.get(ab_raw)
        if not question or not rubric or not ability:
            continue
        for arm, flag in ARMS:
            if (qi, arm) in done:
                sc = done[(qi, arm)]["score"]
                scores[arm][ability].append(sc)
                continue
            os.environ["EDUMEM_KG_FUSION"] = flag
            try:
                ans = answer_with_memory(llm, beam, question, ability=ability)
                if isinstance(ans, tuple):
                    ans = ans[0]
                judgment = judge_with_rubrics(judge, question, rubric, ans, ability)
                sc = _summarize_judge_result(judgment)["official_score"]
            except Exception as e:
                print(f"  q{qi} {arm} ERROR: {type(e).__name__}: {str(e)[:80]}")
                sc = 0.0
            scores[arm][ability].append(sc)
            _append_ckpt({"qi": qi, "arm": arm, "ability": ability, "score": sc})
            print(f"  q{qi:2d} [{ability:3}] {arm:7} score={sc:.2f}", flush=True)

    # --- Report ---
    abilities = sorted({a for arm, _ in ARMS for a in scores[arm]})
    print("\n" + "=" * 64)
    print("JUDGED GRAPH A/B (case 0, KG fusion ON vs OFF; same DB, same answer/judge)")
    print("=" * 64)
    print(f"{'ability':12} {'graph':>8} {'nograph':>8} {'delta':>8}")
    g_tot = n_tot = 0.0
    g_cnt = 0
    for ab in abilities:
        g = scores["graph"].get(ab, [])
        n = scores["nograph"].get(ab, [])
        gm = sum(g) / len(g) if g else 0.0
        nm = sum(n) / len(n) if n else 0.0
        print(f"{ab:12} {gm:8.3f} {nm:8.3f} {gm - nm:+8.3f}")
        g_tot += sum(g); n_tot += sum(n); g_cnt += len(g)
    go = g_tot / g_cnt if g_cnt else 0.0
    no = n_tot / g_cnt if g_cnt else 0.0
    print("-" * 40)
    print(f"{'OVERALL':12} {go:8.3f} {no:8.3f} {go - no:+8.3f}")
    print(f"\nVERDICT: graph fusion moved overall official score by {go - no:+.3f} "
          f"({go:.3f} vs {no:.3f}) across {g_cnt} questions.")


if __name__ == "__main__":
    import faulthandler, traceback
    faulthandler.enable()
    try:
        main()
    except BaseException:
        print("\n!!! TOP-LEVEL CRASH !!!", flush=True)
        traceback.print_exc()
        raise
