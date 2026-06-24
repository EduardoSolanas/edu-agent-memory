"""
BEAM Retrieval-Recall Integration Test
=======================================
Pre-LLM recall gate: measures whether answer-bearing rubric nuggets appear
in the context string that the answer-LLM WOULD see, WITHOUT calling any
answer or judge LLM.

Gated by EDUMEM_RETRIEVAL_E2E=1 (skip by default).

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
import requests

# Gate: skip entire module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("EDUMEM_RETRIEVAL_E2E") != "1",
    reason="retrieval e2e; set EDUMEM_RETRIEVAL_E2E=1 and have embeddings container up at localhost:3002"
)

# ============================================================
#  Env setup (must happen before importing edumem modules)
# ============================================================

def _setup_retrieval_env():
    """Set env vars for retrieval-only mode (no answer/judge LLM)."""
    os.environ.setdefault("EDUMEM_EMBEDDING_API_URL", "http://localhost:3002")
    os.environ.setdefault("EDUMEM_EMBEDDING_MODEL", "Alibaba-NLP/gte-modernbert-base")
    os.environ.setdefault("EDUMEM_EMBEDDINGS_VIA_API", "1")
    os.environ.setdefault("EDUMEM_BENCHMARK_PURE_RECALL", "1")

_setup_retrieval_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_beam_end_to_end import (
    ABILITY_MAP,
    answer_with_memory,
    ingest_conversation,
    load_beam_dataset,
)
from edumem.core.beam import BeamMemory

# ============================================================
#  Connection-pooled embedding: monkey-patch _embed_api to use
#  a requests.Session (HTTP keep-alive, TCP connection reuse)
# ============================================================

_EMBED_SESSION = requests.Session()

from edumem.core import embeddings as _emb_mod

_orig_embed_api = _emb_mod._embed_api


def _pooled_embed_api(texts):
    """Replacement for edumem.core.embeddings._embed_api with connection pooling."""
    import numpy as _np
    import os as _os

    base_url = _os.environ.get("EDUMEM_EMBEDDING_API_URL", "https://openrouter.ai/api/v1")
    is_custom = "openrouter.ai" not in base_url

    if is_custom and not base_url.endswith("/v1"):
        url = f"{base_url.rstrip('/')}/v1/embeddings"
    else:
        url = f"{base_url.rstrip('/')}/embeddings"

    payload = {
        "model": _emb_mod._DEFAULT_MODEL,
        "input": texts,
    }
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://edumem.site",
        "X-Title": "edumem Embedding",
    }
    api_key = _emb_mod._OPENAI_API_KEY
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with _emb_mod._EMBED_API_LOCK:
        for attempt in range(3):
            try:
                resp = _EMBED_SESSION.post(url, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                _emb_mod._API_CALL_COUNT += 1
                return _np.array(embeddings, dtype=_np.float32)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    import time as _time
                    _time.sleep(2 ** attempt)
                    continue
                return None

    return None


_emb_mod._embed_api = _pooled_embed_api

# ============================================================
#  Cross-encoder reranker: monkey-patch _memoria_fused_retrieve
#  to add a reranker step after RRF fusion (Hindsight pattern:
#  RRF → cross-encoder reranker → token-limit trim)
# ============================================================

_RERANK_SESSION = requests.Session()
_RERANK_AVAILABLE = False
try:
    _RERANK_SESSION.get("http://localhost:3002/health", timeout=2)
    _RERANK_AVAILABLE = True
except Exception:
    pass

from edumem.core.beam import BeamMemory as _BeamMemory

_orig_memoria_fused = _BeamMemory._memoria_fused_retrieve


def _patched_memoria_fused(self, query: str, top_k: int = 10) -> dict:
    """Original RRF fusion + cross-encoder reranker after fusion."""
    result = _orig_memoria_fused(self, query, top_k=top_k)
    facts = result.get("facts", [])
    if not facts:
        return result

    # Convert facts to text for reranker scoring
    texts = []
    for f in facts:
        if isinstance(f, dict):
            parts = []
            for k, v in f.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (str, int, float)):
                    parts.append(str(v))
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    parts.extend(v)
            texts.append(" | ".join(parts))
        else:
            texts.append(str(f))

    if not _RERANK_AVAILABLE:
        return result

    try:
        rerank_url = "http://localhost:3002/rerank"
        resp = _RERANK_SESSION.post(
            rerank_url,
            json={"query": query, "texts": texts},
            timeout=30,
        )
        resp.raise_for_status()
        rerank_data = resp.json()
        # rerank returns sorted results: [{"index": idx, "score": score}, ...]
        # Reorder facts by reranker score descending
        score_map = {item["index"]: item["score"] for item in rerank_data}
        scored = [(score_map.get(i, 0.0), i, f) for i, f in enumerate(facts)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        reranked_facts = [item[2] for item in scored[:top_k]]

        # Rebuild context string matching _memoria_fused_retrieve's format
        from edumem.core import query_mode as _qmode
        def _item_msg_idx(item):
            if not isinstance(item, dict):
                return None
            return (item.get("message_idx") or item.get("msg_idx")
                    or item.get("updated_msg_idx") or item.get("valid_from_msg_idx"))

        if _qmode.is_ordering_query(query):
            reranked_facts.sort(key=lambda f: (_item_msg_idx(f) if _item_msg_idx(f) is not None else 1 << 30))

        context_lines = []
        for f in reranked_facts:
            if not isinstance(f, dict):
                context_lines.append(str(f))
                continue
            idx = _item_msg_idx(f)
            tag = f"[MSGIDX:{idx}] " if idx is not None else ""
            if "sequence" in f:
                body = str(f.get("sequence", ""))
            elif "date" in f:
                body = f"[{f.get('date', '')}] {str(f.get('description', ''))[:200]}"
            elif "predicate" in f:
                body = f"{f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')}"
            elif "object" in f and f.get("source") == "negation":
                body = f"user said never/not: {f.get('object', '')}"
            elif "key" in f:
                body = f"{f.get('key', '')}: {f.get('value', '')}"
            else:
                body = ", ".join(f"{k}: {v}" for k, v in f.items() if not k.startswith("_"))
            line = f"{tag}{body}".strip()
            if line:
                context_lines.append(line)

        result["context"] = "\n".join(context_lines)
        result["facts"] = reranked_facts
        result["reranked"] = True
    except Exception as e:
        # reranker unavailable or failed; fall through to RRF-only result
        pass

    return result


_BeamMemory._memoria_fused_retrieve = _patched_memoria_fused


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
CACHE_DB = PROJECT_ROOT / "tests" / ".beam_recall_cache" / "beam_100K_conv0.db"


def get_cached_beam_and_conv():
    """Return (beam, conv) for BEAM 100K conv 0, reusing a cached ingested DB.

    Builds the DB once (ingest with llm=None) if missing; otherwise just opens it.
    - Always loads conversation metadata (cheap, needed for questions).
    - If CACHE_DB exists: open BeamMemory and return (beam, conv) WITHOUT re-ingesting.
    - If it does NOT exist: create parent dir, ingest conversation, then return (beam, conv).
    - Uses same session_id whether building or reusing, so retrieval queries ingested rows.
    """
    # Always load conversation metadata (cheap)
    print("[cache] Loading BEAM 100K (max_conversations=1)...", flush=True)
    data = load_beam_dataset(["100K"], max_conversations=1)
    convs = data.get("100K", [])
    assert convs, "No 100K conversations loaded"
    conv = convs[0]

    session_id = "retrieval-recall-cache"

    if CACHE_DB.exists():
        # Reuse existing cache
        print(f"[cache] reusing existing DB at {CACHE_DB}", flush=True)
        beam = BeamMemory(db_path=CACHE_DB, session_id=session_id)
    else:
        # Build cache
        print(f"[cache] building... (this will take ~11 minutes)", flush=True)
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        beam = BeamMemory(db_path=CACHE_DB, session_id=session_id)
        print(f"[cache] Ingesting conversation {conv['id']} ({len(conv['messages'])} messages)...", flush=True)
        ingest_conversation(beam, conv["messages"], llm=None)
        print(f"[cache] Ingestion complete.", flush=True)

    return beam, conv


@pytest.fixture(scope="module")
def beam_with_conv(tmp_path_factory):
    """Return cached BEAM 100K conversation (module-scoped)."""
    return get_cached_beam_and_conv()


def test_retrieval_recall_per_ability(beam_with_conv):
    """Check that rubric nuggets appear in the retrieved context for each question."""
    beam, conv = beam_with_conv

    questions = conv.get("questions", [])
    assert questions, "No questions in conversation"

    ability_recalls: dict[str, list[float]] = {}
    ability_ctx_chars: dict[str, list[int]] = {}

    skipped = 0
    for q in questions:
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
