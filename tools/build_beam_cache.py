#!/usr/bin/env python3
"""Build a fresh BEAM recall cache with all LLMs + docker inference active.

Creates a NEW timestamped DB — never touches the existing beam_100K_x3.db.
After ingest + sleep consolidation, runs the nugget-recall gate and prints
per-ability results identical to test_beam_retrieval_recall.py.

Usage:
    python tools/build_beam_cache.py [--out PATH] [--n-convs N]

Requirements:
    - Docker container up at 127.0.0.1:3002 (embed + rerank)
    - NAN_APY_KEY or EDUMEM_LLM_API_KEY in .env

What's active vs the baseline build:
    - use_cloud=True          — ExtractionClient: SPO facts + conclusions (qwen3.6)
    - llm_client              — mem0-style consolidation gate per fact (default on)
                                also drives the always-on rolling-summary track
    - EDUMEM_RERANKER_URL     — cross-encoder rerank in retrieval (docker :3002/rerank)
    - EDUMEM_EMBEDDINGS_VIA_API=1 — dense embeddings via docker :3002
"""

from __future__ import annotations

import argparse
import os
import re
import string
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── env setup BEFORE any edumem import ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_file():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        for prefix in ("NAN_APY_KEY=", "EDUMEM_LLM_API_KEY="):
            if line.startswith(prefix):
                key = line[len(prefix):].strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault("EDUMEM_LLM_API_KEY", key)
                break


_load_env_file()

os.environ.setdefault("EDUMEM_LLM_BASE_URL",       "https://api.nan.builders/v1")
os.environ.setdefault("EDUMEM_LLM_MODEL",           "qwen3.6")
os.environ.setdefault("EDUMEM_EXTRACTION_TIMEOUT",  "300")
os.environ.setdefault("EDUMEM_EMBEDDING_API_URL",   "http://127.0.0.1:3002")
os.environ.setdefault("EDUMEM_EMBEDDING_MODEL",     "Alibaba-NLP/gte-modernbert-base")
os.environ.setdefault("EDUMEM_EMBEDDINGS_VIA_API",  "1")
os.environ.setdefault("EDUMEM_RERANKER_URL",        "http://127.0.0.1:3002/rerank")
os.environ.setdefault("EDUMEM_LLM_FACT_CONSOLIDATION", "1")  # mem0-style gate
os.environ.setdefault("EDUMEM_BENCHMARK_PURE_RECALL",  "1")

# ── imports (after env) ────────────────────────────────────────────────────────

from tools.evaluate_beam_end_to_end import (   # noqa: E402
    ABILITY_MAP,
    LLMClient,
    answer_with_memory,
    ingest_conversation,
    load_beam_dataset,
)
from edumem.core.beam import BeamMemory         # noqa: E402

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Build a fresh BEAM recall cache with all LLMs active."
)
parser.add_argument(
    "--out", type=Path, default=None,
    help="Output DB path (default: tests/.beam_recall_cache/beam_100K_x3_TIMESTAMP.db)",
)
parser.add_argument(
    "--n-convs", type=int, default=3,
    help="Number of BEAM 100K conversations to ingest (default: 3)",
)
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_db: Path = args.out or (
    PROJECT_ROOT / "tests" / ".beam_recall_cache" / f"beam_100K_x3_{timestamp}.db"
)
out_db.parent.mkdir(parents=True, exist_ok=True)

if out_db.exists():
    print(f"[build] ERROR: output DB already exists: {out_db}")
    print("[build] Choose a different --out path or remove the file first.")
    sys.exit(1)

print(f"[build] ── NEW CACHE BUILD ──────────────────────────────────────────", flush=True)
print(f"[build] Output DB  : {out_db}", flush=True)
print(f"[build] n_convs    : {args.n_convs}", flush=True)
print(f"[build] LLM model  : {os.environ['EDUMEM_LLM_MODEL']}", flush=True)
print(f"[build] LLM URL    : {os.environ['EDUMEM_LLM_BASE_URL']}", flush=True)
print(f"[build] Embed URL  : {os.environ['EDUMEM_EMBEDDING_API_URL']}", flush=True)
print(f"[build] Reranker   : {os.environ['EDUMEM_RERANKER_URL']}", flush=True)
print(f"[build] Summaries  : always-on (driven by llm_client)", flush=True)
print(f"[build] Key set    : {'yes' if os.environ.get('EDUMEM_LLM_API_KEY') else 'NO — will fail'}", flush=True)
print(flush=True)

# ── load dataset ──────────────────────────────────────────────────────────────

print(f"[build] Loading BEAM 100K ({args.n_convs} conversations)...", flush=True)
data = load_beam_dataset(["100K"], max_conversations=args.n_convs)
convs = data.get("100K", [])
assert convs, "No 100K conversations loaded from dataset"
print(f"[build] Loaded {len(convs)} conversations.", flush=True)

# ── ingest ────────────────────────────────────────────────────────────────────

llm = LLMClient(model=os.environ["EDUMEM_LLM_MODEL"])
convs_meta: list[tuple[str, list]] = []

for ci, conv in enumerate(convs):
    sid = f"retrieval-recall-cache-conv{ci}"
    n_msg = len(conv.get("messages", []))
    print(f"\n[build] ── Conv {ci}/{len(convs)-1}: {conv['id']} ({n_msg} messages) -> '{sid}' ──", flush=True)

    beam = BeamMemory(db_path=out_db, session_id=sid, use_cloud=True, llm_client=llm)
    ingest_conversation(beam, conv["messages"], llm=llm)

    print(f"[build] Running sleep consolidation for '{sid}'...", flush=True)
    sleep_rounds = 0
    try:
        while True:
            result = beam.sleep(force=True)
            status = result.get("status", "")
            items = result.get("items_consolidated", 0)
            print(f"[build]   sleep round {sleep_rounds}: status={status} items={items}", flush=True)
            sleep_rounds += 1
            if status in ("no_op", "error"):
                break
    except Exception as exc:
        print(
            f"[build] WARNING: sleep consolidation failed for '{sid}': "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()

    beam.conn.close()
    convs_meta.append((sid, conv.get("questions", [])))

print(f"\n[build] All {len(convs_meta)} conversations ingested into {out_db}", flush=True)

# ── nugget-recall gate (same logic as test_beam_retrieval_recall.py) ──────────

_STOPWORDS = frozenset({
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "need","dare","ought","used","to","of","in","on","at","by","for","with","about",
    "against","between","through","during","before","after","above","below","from",
    "up","down","out","off","over","under","again","and","but","or","nor","not","so",
    "yet","both","either","neither","each","few","more","most","other","some","such",
    "than","too","very","just","this","that","these","those","it","its","they","them",
    "their","we","our","you","your","he","she","his","her","i","my","me","what",
    "which","who","when","where","why","how","all","any","both","no","into",
})
_RUBRIC_PREFIX_RE = re.compile(
    r"^(?:llm\s+response\s+should\s+(?:state|contain|mention|include|say)\s*:?\s*"
    r"|the\s+(?:response|answer|llm)\s+should\s+(?:state|contain|mention|include|say)\s*:?\s*"
    r"|response\s+should\s+(?:state|contain|mention|include|say)\s*:?\s*"
    r"|should\s+(?:state|contain|mention|include|say)\s*:?\s*"
    r"|(?:state|contain|mention|include|say)\s+that\s+)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _sig_tokens(text: str) -> list[str]:
    return [t for t in text.split() if len(t) >= 3 and t not in _STOPWORDS]


def _strip_rubric(raw: str) -> str:
    s = _RUBRIC_PREFIX_RE.sub("", raw.strip()).strip()
    return s or raw.strip()


def nugget_recall(nuggets: list[str], context: str) -> float:
    if not nuggets:
        return 1.0
    ctx_norm = _normalize(context)
    ctx_tokens = set(ctx_norm.split())
    present = 0
    for n in nuggets:
        nug = _normalize(_strip_rubric(n))
        if not nug:
            present += 1
            continue
        if nug in ctx_norm:
            present += 1
            continue
        sig = _sig_tokens(nug)
        if not sig:
            continue
        if sum(1 for t in sig if t in ctx_tokens) / len(sig) >= 0.70:
            present += 1
    return present / len(nuggets)


print("\n[recall] Running nugget-recall gate...", flush=True)

beam_cache: dict[str, BeamMemory] = {}
ability_recalls: dict[str, list[float]] = {}
ability_ctx_chars: dict[str, list[int]] = {}
skipped = 0

for sid, qs in convs_meta:
    for q in qs:
        rubric = q.get("rubric", [])
        if not rubric:
            skipped += 1
            continue

        nuggets: list[str] = []
        for item in rubric:
            raw = (
                item.get("criterion") or item.get("text") or item.get("content") or ""
                if isinstance(item, dict) else str(item)
            )
            n = _strip_rubric(raw).strip()
            if n:
                nuggets.append(n)
        if not nuggets:
            skipped += 1
            continue

        ability = ABILITY_MAP.get(q.get("ability", ""), q.get("ability", ""))

        if sid not in beam_cache:
            beam_cache[sid] = BeamMemory(db_path=out_db, session_id=sid)
        beam = beam_cache[sid]

        ctx = answer_with_memory(None, beam, q["question"], ability=ability, context_only=True)
        assert isinstance(ctx, str), f"context_only should return str, got {type(ctx)}"

        r = nugget_recall(nuggets, ctx)
        ability_recalls.setdefault(ability, []).append(r)
        ability_ctx_chars.setdefault(ability, []).append(len(ctx))

assert ability_recalls, "No questions with rubrics were evaluated"

print(f"\n{'Ability':<8} {'Mean Recall':>12} {'Mean Ctx Chars':>16} {'N':>5}")
print("-" * 45)
all_recalls: list[float] = []
for ab in sorted(ability_recalls):
    recalls = ability_recalls[ab]
    chars = ability_ctx_chars.get(ab, [])
    mean_r = sum(recalls) / len(recalls)
    mean_c = sum(chars) / len(chars) if chars else 0
    all_recalls.extend(recalls)
    print(f"{ab:<8} {mean_r:>12.3f} {mean_c:>16.0f} {len(recalls):>5}")
print("-" * 45)
overall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
print(f"{'OVERALL':<8} {overall:>12.3f} {'':>16} {len(all_recalls):>5}")
print(f"\nSkipped (no rubric/nuggets): {skipped}")
print(f"Gate (>=0.30): {'PASS' if overall >= 0.30 else 'FAIL'} ({overall:.3f})")
print(f"\n[done] DB written to: {out_db}")
