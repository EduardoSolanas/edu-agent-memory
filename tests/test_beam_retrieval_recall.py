"""
BEAM Retrieval-Recall Integration Test
=======================================
Pre-LLM recall gate: measures whether answer-bearing rubric nuggets appear
in the context string that the answer-LLM WOULD see, WITHOUT calling any
answer or judge LLM.

When EDUMEM_RETRIEVAL_E2E=1, reads the prebuilt recall cache
(tests/.beam_recall_cache/beam_100K_x3.db) and runs recall against it. Skips by
default so the offline fast suite does not silently include a benchmark-scale
retrieval run just because a local cache file exists. Needs the embeddings
container at 127.0.0.1:3002.

Run:
  EDUMEM_RETRIEVAL_E2E=1 python -m pytest tests/test_beam_retrieval_recall.py -q -s
"""

from __future__ import annotations

import os
import re
import string
import sys
from pathlib import Path

import pytest

# Gate: this is a benchmark-style recall test, not part of the default fast
# suite. The cache may exist locally for development, but its presence alone
# must not make `python -m pytest tests/ -q` take minutes.
_CACHE_DB = Path(__file__).resolve().parent / ".beam_recall_cache" / "beam_100K_x3.db"
pytestmark = pytest.mark.skipif(
    os.environ.get("EDUMEM_RETRIEVAL_E2E") != "1" or not _CACHE_DB.exists(),
    reason=(
        "retrieval recall benchmark is opt-in; set EDUMEM_RETRIEVAL_E2E=1 "
        f"and ensure cache exists at {_CACHE_DB}"
    ),
)

# ============================================================
#  Env setup (must happen before importing edumem modules)
# ============================================================

def _setup_retrieval_env():
    """Set env vars for retrieval-only mode (no answer/judge LLM)."""
    # 127.0.0.1, not localhost: on Windows/WSL localhost resolves to IPv6 ::1
    # first and stalls ~2s per request before falling back to the IPv4 port.
    os.environ.setdefault("EDUMEM_EMBEDDING_API_URL", "http://127.0.0.1:3002")
    os.environ.setdefault("EDUMEM_EMBEDDING_MODEL", "Alibaba-NLP/gte-modernbert-base")
    os.environ.setdefault("EDUMEM_EMBEDDINGS_VIA_API", "1")
    os.environ.setdefault("EDUMEM_BENCHMARK_PURE_RECALL", "1")

_setup_retrieval_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _setup_write_llm_env():
    """Cache BUILD uses the LLM (use_cloud extraction: facts + conclusions via
    qwen3.6). Point the canonical EDUMEM_LLM_* vars at NAN/qwen3.6, loading the
    key from .env if not already in the environment. Only the one-time build
    needs this; the recall test reads the cached output with NO LLM.

    Also sets EDUMEM_EXTRACTION_TIMEOUT=300 (large prompts need >60s for qwen3.6)"""
    if not os.environ.get("EDUMEM_LLM_API_KEY"):
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("NAN_APY_KEY=") and "=" in line:
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        os.environ["EDUMEM_LLM_API_KEY"] = key
                    break
    os.environ.setdefault("EDUMEM_LLM_BASE_URL", "https://api.nan.builders/v1")
    os.environ.setdefault("EDUMEM_LLM_MODEL", "qwen3.6")
    os.environ.setdefault("EDUMEM_EXTRACTION_TIMEOUT", "300")


from tools.evaluate_beam_end_to_end import (
    ABILITY_MAP,
    LLMClient,
    answer_with_memory,
    ingest_conversation,
    load_beam_dataset,
)
from edumem.core.beam import BeamMemory

# NOTE: connection pooling for the embedding API now lives in production
# (`edumem.core.embeddings._EMBED_API_SESSION`, a module-level requests.Session).
# This module no longer shadows `_embed_api` — the import-time monkey-patch that
# did so was process-wide pollution (it replaced _embed_api for every test in
# the suite, not just this one). The live benchmark now uses the real pooled
# path directly.

# NOTE: the cross-encoder reranker now lives in the real pipeline
# (`edumem.core.beam._fusion_rerank`, called from `_memoria_fused_retrieve`),
# so this module no longer shadows `_memoria_fused_retrieve`. The reranker
# fires automatically when the endpoint at EDUMEM_RERANKER_URL is up; if it
# is down, fusion degrades silently to RRF order.

# ============================================================
#  Nugget helpers
# ============================================================

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "i", "my", "me", "what", "which", "who",
    "when", "where", "why", "how", "all", "any", "both", "no", "into",
})

_RUBRIC_PREFIX_RE = re.compile(
    r'^(?:llm\s+response\s+should\s+(?:state|contain|mention|include|say)\s*:?\s*'
    r'|the\s+(?:response|answer|llm)\s+should\s+(?:state|contain|mention|include|say)\s*:?\s*'
    r'|response\s+should\s+(?:state|contain|mention|include|say)\s*:?\s*'
    r'|should\s+(?:state|contain|mention|include|say)\s*:?\s*'
    r'|(?:state|contain|mention|include|say)\s+that\s+)',
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _significant_tokens(text: str) -> list[str]:
    """Return tokens that are significant (len>=3, not a stopword)."""
    return [t for t in text.split() if len(t) >= 3 and t not in _STOPWORDS]


def _strip_rubric_prefix(raw: str) -> str:
    """Strip common rubric instruction prefixes to get the actual nugget."""
    stripped = _RUBRIC_PREFIX_RE.sub("", raw.strip())
    return stripped.strip() or raw.strip()


def build_nuggets(rubric: list) -> list[str]:
    """Extract nugget strings from a rubric list (dicts or plain strings)."""
    nuggets = []
    for item in rubric:
        if isinstance(item, dict):
            raw = item.get("criterion") or item.get("text") or item.get("content") or ""
        else:
            raw = str(item)
        nugget = _strip_rubric_prefix(raw).strip()
        if nugget:
            nuggets.append(nugget)
    return nuggets


def nugget_recall(nuggets: list[str], context: str) -> float:
    """Return fraction of nuggets present in context.

    A nugget is 'present' if:
      1. Its normalized form is a substring of the normalized context, OR
      2. >= 70% of its significant tokens (len>=3, non-stopword) appear in
         the normalized context tokens.
    """
    if not nuggets:
        return 1.0
    ctx_norm = _normalize(context)
    ctx_tokens = set(ctx_norm.split())

    present = 0
    for nugget in nuggets:
        nug_norm = _normalize(nugget)
        if not nug_norm:
            present += 1
            continue
        # Check 1: substring match
        if nug_norm in ctx_norm:
            present += 1
            continue
        # Check 2: token overlap (>=70% of significant tokens)
        sig_tokens = _significant_tokens(nug_norm)
        if not sig_tokens:
            # No significant tokens: fall back to plain substring check (already failed)
            continue
        overlap = sum(1 for t in sig_tokens if t in ctx_tokens)
        if overlap / len(sig_tokens) >= 0.70:
            present += 1

    return present / len(nuggets)


# ============================================================
#  Integration test
# ============================================================

RECALL_GATE = 0.30  # Minimum acceptable overall recall
CACHE_DB = PROJECT_ROOT / "tests" / ".beam_recall_cache" / "beam_100K_x3.db"


def get_cached_beams():
    """Return (db_path, convs_meta) for BEAM 100K convs 0-2.

    convs_meta is [(session_id, [questions]), ...] — one entry per conversation
    with its own isolated session_id. Builds the DB once if missing, reuses
    otherwise. Uses separate BeamMemory per conversation during build and
    returns the DB path plus metadata so the test can create per-session
    BeamMemory instances at query time."""
    N_CONVS = 3
    print(f"[cache] Loading BEAM 100K (max_conversations={N_CONVS})...", flush=True)
    data = load_beam_dataset(["100K"], max_conversations=N_CONVS)
    convs = data.get("100K", [])
    assert convs, "No 100K conversations loaded"

    convs_meta = []

    if CACHE_DB.exists():
        print(f"[cache] reusing existing DB at {CACHE_DB}", flush=True)
        # Build metadata from conversation data (questions are cheap, no LLM)
        for ci, conv in enumerate(convs):
            sid = f"retrieval-recall-cache-conv{ci}"
            convs_meta.append((sid, conv.get("questions", [])))
    else:
        print(f"[cache] building with LLM extraction (use_cloud=True + llm_client)...", flush=True)
        _setup_write_llm_env()
        llm = LLMClient(model=os.environ.get("EDUMEM_LLM_MODEL", "qwen3.6"))
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        for ci, conv in enumerate(convs):
            sid = f"retrieval-recall-cache-conv{ci}"
            print(f"[cache] Ingesting conversation {ci} ({conv['id']}) into session '{sid}' with {len(conv['messages'])} messages...", flush=True)
            beam = BeamMemory(db_path=CACHE_DB, session_id=sid,
                              use_cloud=True, llm_client=llm)
            ingest_conversation(beam, conv["messages"], llm=llm)
            try:
                while beam.sleep(force=True).get("status") not in ("no_op", "error"):
                    pass
            except Exception as sleep_err:
                # A broken consolidation cycle silently degrades conclusion/
                # episodic coverage; surface it instead of hiding it so a bad
                # build is visible rather than producing a quietly-worse cache.
                import traceback
                print(f"[cache] WARNING: sleep consolidation failed for '{sid}': "
                      f"{type(sleep_err).__name__}: {sleep_err}", flush=True)
                traceback.print_exc()
            beam.conn.close()
            convs_meta.append((sid, conv.get("questions", [])))
        print(f"[cache] All conversations ingested.", flush=True)

    return str(CACHE_DB), convs_meta


@pytest.fixture(scope="module")
def beam_with_convs(tmp_path_factory):
    """Return (db_path, convs_meta) for BEAM 100K convs 0-2 (module-scoped)."""
    return get_cached_beams()


def _assert_retrieval_backends_live():
    """Hard precondition: embeddings + reranker endpoints must actually respond.

    `embeddings.available()` only checks that the API URL is SET, not that the
    endpoint is reachable. When the container is down, `embed_query()` returns
    None and `beam.recall()` silently falls back to keyword-only — producing
    degraded recall numbers that masquerade as a regression. This benchmark
    measures dense+rerank recall, so a dead endpoint is a hard error, not a
    silent downgrade. Fail with an actionable message instead.
    """
    from edumem.core import embeddings as _emb
    from edumem.core.beam import _fusion_rerank

    emb_url = os.environ.get("EDUMEM_EMBEDDING_API_URL", "<unset>")
    rer_url = os.environ.get("EDUMEM_RERANKER_URL", "http://127.0.0.1:3002/rerank")

    # 1. Embedding endpoint must return a real vector (not None).
    probe = _emb.embed_query("database migration postgresql")
    assert probe is not None and getattr(probe, "size", 0) > 0, (
        f"Embedding endpoint at {emb_url} is unreachable or returned no vector. "
        f"Dense recall would silently degrade to keyword-only. "
        f"Start the container: `docker start edumem-server` and confirm "
        f"`curl {emb_url}/health` returns 200."
    )

    # 2. Reranker endpoint must return real scores (not None = degraded to RRF).
    scores = _fusion_rerank("which database", ["We use PostgreSQL", "Redis cache"])
    assert scores is not None and isinstance(scores, list) and len(scores) == 2, (
        f"Reranker endpoint at {rer_url} is unreachable or returned no scores. "
        f"Fusion would silently degrade to RRF order. "
        f"Start the container: `docker start edumem-server`."
    )


def test_retrieval_recall_per_ability(beam_with_convs):
    """Check that rubric nuggets appear in the retrieved context for each question."""
    # Fail loudly if the dense/rerank backends are down — never measure
    # keyword-only recall and report it as if the full pipeline ran.
    _assert_retrieval_backends_live()

    db_path, convs_meta = beam_with_convs

    # Collect all questions with their session_id
    questions_with_sid = []
    for sid, qs in convs_meta:
        for q in qs:
            q["_session_id"] = sid
            questions_with_sid.append(q)

    assert questions_with_sid, "No questions loaded"

    # Cache one BeamMemory per session_id
    beam_cache: dict[str, BeamMemory] = {}

    ability_recalls: dict[str, list[float]] = {}
    ability_ctx_chars: dict[str, list[int]] = {}

    skipped = 0
    for q in questions_with_sid:
        rubric = q.get("rubric", [])
        if not rubric:
            skipped += 1
            continue

        ability_raw = q.get("ability", "")
        ability = ABILITY_MAP.get(ability_raw, ability_raw)
        nuggets = build_nuggets(rubric)
        if not nuggets:
            skipped += 1
            continue

        sid = q["_session_id"]
        if sid not in beam_cache:
            beam_cache[sid] = BeamMemory(db_path=db_path, session_id=sid)
        beam = beam_cache[sid]

        ctx = answer_with_memory(
            None,
            beam,
            q["question"],
            ability=ability,
            context_only=True,
        )
        assert isinstance(ctx, str), f"context_only should return str, got {type(ctx)}"

        recall = nugget_recall(nuggets, ctx)
        ctx_chars = len(ctx)

        ability_recalls.setdefault(ability, []).append(recall)
        ability_ctx_chars.setdefault(ability, []).append(ctx_chars)

    assert ability_recalls, "No questions with rubrics were evaluated"

    # Print per-ability table
    print(f"\n{'Ability':<8} {'Mean Recall':>12} {'Mean Ctx Chars':>16} {'N':>5}")
    print("-" * 45)
    all_recalls = []
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
    print(f"Gate: overall recall >= {RECALL_GATE}")

    assert overall >= RECALL_GATE, (
        f"Overall nugget recall {overall:.3f} is below gate {RECALL_GATE}. "
        f"Per-ability: {{{', '.join(f'{k}: {sum(v)/len(v):.3f}' for k, v in sorted(ability_recalls.items()))}}}"
    )
