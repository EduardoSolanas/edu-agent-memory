# Cross-encoder reranker wired into MEMORIA fusion

**Date:** 2026-06-24
**Status:** Approved (design sections 1–4)
**Scope:** Move the cross-encoder reranker from a test monkey-patch into the
real recall pipeline, as a single, well-bounded unit. Remove the now-redundant
harness rerank.

## Problem

The BEAM retrieval-recall benchmark sits at **0.347 overall recall** (passes the
0.30 gate, but ~65% of rubric nuggets are not surfaced). Root cause: the 30 facts
returned by `_memoria_fused_retrieve` are dominated by generic timeline/milestone
scaffolding regardless of question topic. RRF fusion over 6 specialists cannot
distinguish content-relevant facts from temporal noise — it only fuses *rank*
positions, not query relevance.

A cross-encoder reranker (Hindsight pattern: RRF → cross-encoder → trim) already
solves this in a **test monkey-patch** (`tests/test_beam_retrieval_recall.py`,
lines ~114–219) and in the **harness** (`tools/evaluate_beam_end_to_end.py`,
`_rerank` at line 2825). Neither is correct:

- The test monkey-patch targets the *right* object (the fused facts inside
  `_memoria_fused_retrieve`) but is trapped in the test — production recall never
  benefits.
- The harness `_rerank` (called at line 3047) targets the *wrong* object: by the
  time it runs, the MEMORIA facts are bundled into a single `memories[0]` blob
  (line 3032). It reranks the bundle, not the facts inside it, so it cannot
  surface the right facts.

There is exactly **one** correct insertion point: inside
`_memoria_fused_retrieve` in `edumem/core/beam.py`, after RRF fusion reorders the
fused facts by query relevance.

## Reranker contract (confirmed from existing code)

- **Request:** `POST <EDUMEM_RERANKER_URL>` (default
  `http://localhost:3002/rerank`) with JSON `{"query": str, "texts": [str]}`.
- **Response:** `[{"index": int, "score": float}, ...]` — `index` refers back to
  the position in the submitted `texts` array; higher `score` = more relevant.
- **Must no-op gracefully when down** — the offline fast suite (~229 tests) runs
  with no network and must stay green.

## Design

### Section 1 — The rerank helper (the single seam)

A module-level function plus a shared `requests.Session` in
`edumem/core/beam.py`, placed alongside `_rrf_fuse`:

```python
_FUSION_RERANK_SESSION = requests.Session()   # keep-alive, TCP reuse

def _fusion_rerank(query: str, fact_texts: list) -> list | None:
    """Cross-encoder rerank of fused facts. Returns [{"index","score"}] or None.

    POSTs {"query": ..., "texts": [...]} to EDUMEM_RERANKER_URL (default
    http://localhost:3002/rerank). Returns None on ANY failure or empty input,
    so callers fall through to RRF order. Gated by EDUMEM_FUSION_RERANK
    (default "1"); set "0" to disable. Single seam a test stubs.
    """
```

- Uses `requests` for simplicity (the repo already depends on it; the test
  monkey-patch uses `requests.Session` for the same keep-alive benefit).
- **Failure mode:** `None` on timeout / connection error / malformed JSON / empty
  input / feature disabled. Caller treats `None` as "keep RRF order." This is
  what keeps the offline fast suite green.
- **No persistent health-check:** a failed POST returns `None` fast (5s timeout).
  The reranker is optional, not a dependency.

### Section 2 — Wiring into `_memoria_fused_retrieve`

The rerank reorders fused facts between the RRF fuse and the EO ordering sort:

```
fused_keys = _rrf_fuse(ranked_lists, k=60)     # existing
... reassemble final_items + final_facts ...   # existing
# ---- NEW: cross-encoder rerank ----
... EO msg_idx sort ...                        # existing, runs AFTER rerank
... render final_context_lines ...             # existing, outputs new order
```

New block (after `final_items` is assembled, before the EO sort):

```python
reranked = False
_fact_texts = [_fact_render_text(f) for _, f in final_items]
_scores = _fusion_rerank(query, _fact_texts)   # None when disabled/down
if _scores is not None:
    _score_map = {item["index"]: item["score"] for item in _scores}
    # Stable index (enumerate) — avoids O(n^2) and dict-unhashable-fact pitfalls.
    indexed = list(enumerate(final_items))
    indexed.sort(key=lambda pair: -_score_map.get(pair[0], 0.0))
    final_items = [item for _, item in indexed]
    final_facts = [f for _, f in final_items]
    reranked = True
```

The `EDUMEM_FUSION_RERANK` gate lives **inside** `_fusion_rerank` (Section 1) — it
returns `None` when disabled, so the wiring block only checks `_scores is not None`.
No double-gating.

**Three deliberate decisions:**

1. **`_fact_render_text(fact)`** — a small module-level helper that extracts the
   same text used for rendering (sequence / date / predicate / key / fallback).
   The reranker scores exactly what the prompt will show. This refactor pulls the
   body-extraction switch out of the inlined render loop into a reusable function
   (improves the code I'm working in, per working-in-existing-codebases guidance).

2. **EO ordering still wins.** The rerank block runs first, then the existing
   `if is_ordering_query(query): final_items.sort(by msg_idx)` re-sorts by
   first-appearance order. For EO, message order is the graded signal (tau-b over
   positions), so it must override rerank. For all other abilities, rerank order
   stands. **No behavior change for EO.**

3. **Graceful fallback chain:** reranker down or disabled → `_scores is None` →
   skip the sort → facts stay in RRF order → identical to today. No `if`-guards
   that make tests vacuous.

A `result["reranked"]` boolean is added to the return dict for diagnostics —
`True` only when the rerank actually fired.

### Section 3 — Removing the harness rerank

With the reranker in fusion, the harness `_rerank` is both redundant and aimed at
the wrong object. It is removed:

- **Remove the call** at `evaluate_beam_end_to_end.py` lines 3042–3047 (the
  "Phase 5.5 local cross-encoder" block + `_rerank_top_n` computation).
- **Remove the orphaned helpers** `_rerank` (line 2825) and
  `_apply_rerank_scores` (line 2792) — each has exactly one caller, both removed
  by this change.
- **Keep `_probe_reranker` (line 2813)** — still used by the CLI preflight (line
  4705) to gate runs: "refuse to start if the reranker endpoint is down." With
  the reranker now in the pipeline, this preflight becomes more meaningful.
- **Keep the diagnostics plumbing** (`_summarize_reranker_run`,
  `_finalize_reranker_run_health`, `rerank_diag`) — verified to have no dangling
  references after the call removal; removed only if orphaned.
- **Keep `_intent_from_reranker_scores` (line 101)** — that is the intent
  *classifier* use of the reranker (4-way scoring), a different feature.

**Invariant:** after this change there is exactly one rerank of recall content,
in `_memoria_fused_retrieve`, and zero rerank logic in the harness answer path.

### Section 4 — Test changes

**New offline test — `tests/test_fusion_rerank.py`** (AGENTS.md mandates tests
for retrieval changes):

```python
def test_fusion_rerank_reorders_facts(monkeypatch):
    # Stub the seam — no network. Scores deliberately INVERT natural order.
    monkeypatch.setattr(edumem.core.beam, "_fusion_rerank",
        lambda q, texts: [{"index": 2, "score": 0.9},
                          {"index": 1, "score": 0.5},
                          {"index": 0, "score": 0.1}])
    beam = <ingest tiny synthetic conv: 3 distinct facts>
    result = beam.memoria_retrieve("query", top_k=3)
    # Assert facts come back in RERANK order (C, B, A), NOT RRF order.
    # FAILS if the wiring is reverted — not vacuous.
    assert <fact_C> before <fact_B> before <fact_A>

def test_fusion_rerank_offline_fallback(monkeypatch):
    # Endpoint-down simulation: seam returns None.
    monkeypatch.setattr(edumem.core.beam, "_fusion_rerank", lambda q, t: None)
    result = beam.memoria_retrieve("query", top_k=3)
    assert result["reranked"] is False
    # order equals RRF order (the offline-suite reality)
```

**Trim `tests/test_beam_retrieval_recall.py`:**

- **Remove** the cross-encoder monkey-patch block (lines ~114–219):
  `_patched_memoria_fused`, `_orig_memoria_fused`, `_RERANK_SESSION`,
  `_RERANK_AVAILABLE`, and the
  `_BeamMemory._memoria_fused_retrieve = _patched_memoria_fused` assignment.
- **Keep** the connection-pooled `_embed_api` monkey-patch (lines ~60–112) —
  ingest-speed infra (HTTP keep-alive), unrelated to rerank.
- The live test now exercises the **real** fusion-rerank path. If the endpoint is
  down, `_fusion_rerank` returns `None` and the test still runs (RRF-only) — it
  just won't show the rerank improvement.

No changes to `test_beam_evaluator.py` or `test_memoria_regressions.py` — those
assert on rendering *format* (`[Fact CURRENT]`, `MSGIDX:`), unchanged by
reordering. Confirmed by running the full suite.

## Out of scope

AGENTS.md "next steps" not touched here (separate efforts):
- Specialist deduplication (content hash before RRF).
- Polyphonic recall enablement (`EDUMEM_POLYPHONIC_RECALL=1`).
- Gap #1: ability-aware intent routing for versioned-fact surfacing.

The reranker only reorders existing fused facts — it cannot retrieve NEW ones,
so it is an upper-bound-only improvement signal on the live recall benchmark.

## Verification

- `python -m pytest tests/ -q` stays green offline (reranker no-ops when down).
- New `test_fusion_rerank.py` passes (reorder asserted via stubbed seam).
- Live: `EDUMEM_RETRIEVAL_E2E=1 python -m pytest tests/test_beam_retrieval_recall.py
  -q -s` — recall ≥ 0.347 baseline.
