# Fusion Cross-encoder Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the cross-encoder reranker out of a test monkey-patch into the real recall pipeline (`_memoria_fused_retrieve` in `edumem/core/beam.py`), so production recall benefits — and remove the redundant/wrong-object harness rerank.

**Architecture:** Approach B (approved): a module-level `_fusion_rerank` helper + shared `requests.Session` is the single testable seam. `_memoria_fused_retrieve` calls it after RRF fusion to reorder fused facts by query relevance; the EO ordering sort still runs after to preserve first-appearance order for ordering queries. The harness `_rerank` call and its orphaned helpers are removed. A new offline test stubs the seam.

**Tech Stack:** Python 3, `requests` (HTTP keep-alive), `pytest`, SQLite (BEAM storage). The reranker service is an HTTP endpoint at `EDUMEM_RERANKER_URL` (default `http://localhost:3002/rerank`) returning `[{"index": int, "score": float}]`.

---

## Current state (already done — context only, do not redo)

Two edits are **already committed/working** from earlier in this session:
1. `import requests` added to `edumem/core/beam.py` (top imports).
2. Module-level `_FUSION_RERANK_SESSION = requests.Session()`, `_fact_render_text(fact)`, and `_fusion_rerank(query, fact_texts)` added after `_rrf_fuse` (beam.py ~line 2376–2438).

These are correct and match the spec. **Do not re-add them.** The tasks below assume they exist. Verify with a grep at the start of Task 1.

---

## Task 1: Refactor the fusion render loop to use `_fact_render_text`

The inlined body-extraction switch in `_memoria_fused_retrieve`'s render loop (beam.py lines ~5376–5392) duplicates `_fact_render_text`. This refactor de-duplicates so rerank and render use the same text. It also makes the reranker-scored text identical to prompt text (spec Section 2, decision 1).

**Files:**
- Modify: `edumem/core/beam.py` lines ~5369–5392 (the render loop)
- Test: existing suite — this is a pure refactor, covered by `tests/test_beam_evaluator.py` format assertions

- [ ] **Step 1: Verify the helper exists (do not re-add)**

Run:
```bash
grep -n "_fact_render_text\|_fusion_rerank" edumem/core/beam.py
```
Expected: 3+ matches (the def + the rerank call site inside `_fusion_rerank`). If absent, STOP — the pre-work was not applied; re-apply from the spec Section 1.

- [ ] **Step 2: Replace the inlined body switch with a call to `_fact_render_text`**

In `_memoria_fused_retrieve`, replace the render loop body (the block from `for specialist_name, fact in final_items:` through the `if line:` append). Keep the `[MSGIDX:N]` tagging logic — that stays in the loop because it depends on `_item_msg_idx`, which is not part of `_fact_render_text`.

Change this (beam.py ~5369–5392):
```python
        final_context_lines = []
        for specialist_name, fact in final_items:
            if not isinstance(fact, dict):
                final_context_lines.append(str(fact))
                continue
            idx = _item_msg_idx(fact)
            tag = f"[MSGIDX:{idx}] " if idx is not None else ""
            if "sequence" in fact:  # chrono / sequence specialist
                body = f"{fact.get('sequence', '')}"
            elif "date" in fact:  # timeline specialist
                body = f"[{fact.get('date', '')}] {str(fact.get('description', ''))[:200]}"
            elif "predicate" in fact:  # entity KG specialist
                body = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
            elif "object" in fact:  # negation specialist
                body = f"user said never/not: {fact.get('object', '')}"
            elif "key" in fact:  # fact specialist
                body = f"{fact.get('key', '')}: {fact.get('value', '')}"
            else:
                body = ", ".join(
                    f"{k}: {v}" for k, v in fact.items() if not k.startswith("_")
                )
            line = f"{tag}{body}".strip()
            if line:
                final_context_lines.append(line)
```
to:
```python
        final_context_lines = []
        for specialist_name, fact in final_items:
            idx = _item_msg_idx(fact) if isinstance(fact, dict) else None
            tag = f"[MSGIDX:{idx}] " if idx is not None else ""
            line = f"{tag}{_fact_render_text(fact)}".strip()
            if line:
                final_context_lines.append(line)
```

- [ ] **Step 3: Run the static rendering tests**

Run: `python -m pytest tests/test_beam_evaluator.py -q`
Expected: PASS (these assert on `[Fact CURRENT]`, `[Fact CHANGED]`, `MSGIDX:` formats — unchanged by this refactor).

- [ ] **Step 4: Run the full fast suite to confirm no regression**

Run: `python -m pytest tests/ -q`
Expected: ~229 passed (offline, no network — `_fusion_rerank` returns `None` when the endpoint is down).

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py
git commit -m "refactor(beam): de-duplicate fusion render via _fact_render_text"
```

---

## Task 2: Write the failing test for fusion rerank wiring (TDD)

Write the test that proves the wiring reorders fused facts by rerank score, *before* adding the wiring. It must fail.

**Files:**
- Create: `tests/test_fusion_rerank.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion_rerank.py`:
```python
"""Offline test for cross-encoder rerank wiring in _memoria_fused_retrieve.

Stubs the `_fusion_rerank` seam (no network) and asserts the fused facts come
back in rerank-score order, not RRF order. Fails if the wiring is reverted.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import edumem.core.beam as beam_mod
from edumem.core.beam import BeamMemory


def _build_beam_with_facts(tmp_path: Path) -> BeamMemory:
    """Build a BeamMemory and inject 3 distinct versioned facts directly."""
    db_path = tmp_path / "fusion_rerank.db"
    beam = BeamMemory(session_id="rerank-test", db_path=db_path)
    conn = beam.conn
    # Three distinct facts. RRF over the fact specialist returns them in
    # insertion/key order (A, B, C). The rerank stub will invert that.
    rows = [
        {"key": "alpha_setting", "value": "value_A", "source_memory_id": "A"},
        {"key": "bravo_setting", "value": "value_B", "source_memory_id": "B"},
        {"key": "charlie_setting", "value": "value_C", "source_memory_id": "C"},
    ]
    for r in rows:
        conn.execute(
            "INSERT OR REPLACE INTO memoria_facts "
            "(session_id, key, value, fact_type, source_memory_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("rerank-test", r["key"], r["value"], "preference", r["source_memory_id"]),
        )
    conn.commit()
    return beam


def test_fusion_rerank_reorders_facts(monkeypatch, tmp_path):
    """Rerank scores must override RRF order; the stub inverts natural order."""
    beam = _build_beam_with_facts(tmp_path)

    # Stub the seam: NO network. Scores deliberately invert insertion order
    # so index 2 (charlie) ranks highest, index 0 (alpha) lowest.
    def _stub(query, fact_texts):
        return [{"index": 2, "score": 0.9}, {"index": 1, "score": 0.5}, {"index": 0, "score": 0.1}]

    monkeypatch.setattr(beam_mod, "_fusion_rerank", _stub)

    result = beam.memoria_retrieve("query about settings", top_k=3)
    context = result["context"]
    # charlie must appear BEFORE alpha in the rendered context.
    assert context.index("charlie_setting") < context.index("alpha_setting"), (
        f"rerank did not reorder: {context!r}"
    )
    assert result.get("reranked") is True


def test_fusion_rerank_offline_fallback(monkeypatch, tmp_path):
    """Endpoint-down (None) must keep RRF order and mark reranked=False."""
    beam = _build_beam_with_facts(tmp_path)
    monkeypatch.setattr(beam_mod, "_fusion_rerank", lambda q, t: None)

    result = beam.memoria_retrieve("query about settings", top_k=3)
    assert result.get("reranked") is False
    # Context still populated (RRF order, whatever it is).
    assert result["context"], "context should still render without rerank"
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run: `python -m pytest tests/test_fusion_rerank.py -q`
Expected: FAIL on `test_fusion_rerank_reorders_facts` — the wiring is not yet added so `charlie` stays after `alpha` in RRF order, and `result["reranked"]` does not exist (`KeyError` or `False`).
`test_fusion_rerank_offline_fallback` may pass trivially (reranked key absent → `is False` holds for missing key only if the test uses `.get`; it does).

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_fusion_rerank.py
git commit -m "test(fusion-rerank): failing test for rerank wiring (red)"
```

---

## Task 3: Wire the rerank into `_memoria_fused_retrieve`

Add the reorder block between RRF reassembly and the EO ordering sort. This makes Task 2's test pass.

**Files:**
- Modify: `edumem/core/beam.py` — `_memoria_fused_retrieve`, after `final_items` assembly (~line 5338) and before the `from . import query_mode` import (~line 5339)

- [ ] **Step 1: Insert the rerank block**

In `_memoria_fused_retrieve`, immediately after the reassembly loop ends (after the `for key in fused_keys[:top_k]:` block, before `from . import query_mode as _query_mode`), insert:

```python
        # ---- Cross-encoder rerank: reorder fused facts by query relevance ----
        # Runs after RRF fusion; the EO msg_idx sort below still wins for
        # ordering queries. None (endpoint down / disabled) keeps RRF order.
        reranked = False
        _fact_texts = [_fact_render_text(f) for _, f in final_items]
        _scores = _fusion_rerank(query, _fact_texts)
        if _scores is not None:
            _score_map = {item["index"]: item["score"] for item in _scores}
            # Stable enumerate index (avoids O(n^2) list.index + unhashable dict pitfalls).
            _indexed = list(enumerate(final_items))
            _indexed.sort(key=lambda pair: -_score_map.get(pair[0], 0.0))
            final_items = [item for _, item in _indexed]
            final_facts = [f for _, f in final_items]
            reranked = True
```

- [ ] **Step 2: Add `reranked` to the result dict**

In the same method, the `result = {...}` dict (~line 5396) gets the new key:
```python
        result = {
            "context": "\n".join(final_context_lines),
            "facts": final_facts,
            "source": "rrf_fused",
            "source_memory_ids": list(final_ids),
            "reranked": reranked,
            "rrf_timing": timing,
        }
```

- [ ] **Step 3: Run the Task 2 tests to verify they PASS**

Run: `python -m pytest tests/test_fusion_rerank.py -q`
Expected: PASS (both tests — reorder asserts succeed, fallback marks `reranked=False`).

- [ ] **Step 4: Run the full fast suite**

Run: `python -m pytest tests/ -q`
Expected: ~229 + 2 passed. No regression.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py
git commit -m "feat(recall): cross-encoder rerank of fused facts in _memoria_fused_retrieve"
```

---

## Task 4: Remove the harness `_rerank` call + orphaned helpers

The harness rerank is redundant (now in fusion) and targets the wrong object (the bundled blob). Remove the call and the helpers that become orphaned.

**Files:**
- Modify: `tools/evaluate_beam_end_to_end.py`

- [ ] **Step 1: Verify orphan status before removing**

Run:
```bash
grep -n "_rerank(\|_apply_rerank_scores(" tools/evaluate_beam_end_to_end.py
```
Expected matches:
- `_apply_rerank_scores` def (~2792) + one call inside `_rerank` (~2850).
- `_rerank` def (~2825) + one call at ~3047.
Both have exactly one external caller each → safe to remove both helpers after removing the 3047 call. If `_rerank` has additional callers, STOP and only remove the 3047 call.

- [ ] **Step 2: Remove the call site (the "Phase 5.5" block)**

Delete lines ~3042–3047:
```python
    # ---- Reranking (Phase 5.5: local cross-encoder) ----
    # Ordering (EO) is graded by tau-b over ALL items, so keep a wider reranked set
    # for ordering queries -- the char-budgeted context builder trims later. A flat
    # top_k cap here would drop topic mentions and make the ordering incomplete.
    _rerank_top_n = top_k * 3 if is_ordering_query(question) else top_k
    memories = _rerank(question, memories, top_n=_rerank_top_n, diag=diag)
```
The EO message-index sort at the next line (`if is_ordering_query(question): memories.sort(...)`) stays.

- [ ] **Step 3: Remove the orphaned `_rerank` and `_apply_rerank_scores` helpers**

Delete the two function definitions `_apply_rerank_scores` (~2792–2810) and `_rerank` (~2825–2863). **Do NOT delete** `_probe_reranker` (~2813) — it is still used by the CLI preflight.

- [ ] **Step 4: Check for dangling references**

Run:
```bash
grep -n "_rerank\b\|_apply_rerank_scores\|_rerank_top_n" tools/evaluate_beam_end_to_end.py
```
Expected: no matches. (The diagnostics plumbing `_summarize_reranker_run` / `_finalize_reranker_run_health` / `rerank_diag` reference the *diagnostics dict keys*, not the `_rerank` function — those stay and are fed by fusion's `result["reranked"]` if surfaced. Confirm no NameError.)

- [ ] **Step 5: Verify the harness still imports and the offline suite passes**

Run:
```bash
python -c "import tools.evaluate_beam_end_to_end"
python -m pytest tests/ -q
```
Expected: import succeeds; full suite green.

- [ ] **Step 6: Commit**

```bash
git add tools/evaluate_beam_end_to_end.py
git commit -m "refactor(harness): remove redundant/wrong-object _rerank; rerank now in fusion"
```

---

## Task 5: Remove the redundant reranker monkey-patch from the live test

The live recall test's monkey-patch shadowed `_memoria_fused_retrieve` to add the reranker. Now that it's real, the patch is dead weight that hides the real path. Remove it; keep the embedding connection-pooling patch.

**Files:**
- Modify: `tests/test_beam_retrieval_recall.py` lines ~114–219

- [ ] **Step 1: Delete the cross-encoder monkey-patch block**

Remove from the comment header `# Cross-encoder reranker: monkey-patch ...` (line ~114) through the assignment `_BeamMemory._memoria_fused_retrieve = _patched_memoria_fused` (line ~219). This includes `_RERANK_SESSION`, `_RERANK_AVAILABLE`, `_orig_memoria_fused`, and `_patched_memoria_fused`.

- [ ] **Step 2: Confirm the embedding pool patch is intact**

The block `# Connection-pooled embedding: monkey-patch _embed_api ...` (lines ~55–112) and its `_emb_mod._embed_api = _pooled_embed_api` assignment MUST remain. Verify:
```bash
grep -n "_pooled_embed_api\|_EMBED_SESSION" tests/test_beam_retrieval_recall.py
```
Expected: matches present (the embedding patch kept).

- [ ] **Step 3: Verify the test file still parses and imports**

Run:
```bash
python -c "import ast; ast.parse(open('tests/test_beam_retrieval_recall.py').read()); print('OK')"
```
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_beam_retrieval_recall.py
git commit -m "test(recall): drop redundant fusion-rerank monkey-patch (now real in pipeline)"
```

---

## Task 6: Update AGENTS.md to remove the reranker contradiction

The doc both says "added a reranker" (in the test) and lists the pipeline reranker as "None". Now that it's real, fix the architecture table and the "What was done" note.

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the architecture table**

In the "Architecture comparison" table, change the `edumem MEMORIA` reranker cell from:
`**None** (reranker exists at container but not used in pipeline)`
to:
`Cross-encoder (RRF → rerank → EO-sort; gated `EDUMEM_FUSION_RERANK`, default ON)`

- [ ] **Step 2: Update the "What was done" note and the "Key gap" line**

In "### What was done", change "added a Hindsight-style cross-encoder reranker after RRF fusion (gated on container availability)" to "wired a Hindsight-style cross-encoder reranker into `_memoria_fused_retrieve` after RRF fusion (default ON; gated `EDUMEM_FUSION_RERANK`)".

Replace the "Key gap vs Hindsight" paragraph: the gap (no reranker after RRF) is now closed. Change it to note the reranker is wired in fusion and that it only reorders existing fused facts (cannot retrieve new ones).

- [ ] **Step 3: Add a one-line note to "What to try next"**

Strike item 2 ("Add cross-encoder reranker after RRF") — it's done. Renumber.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): reranker now wired into fusion; fix the table contradiction"
```

---

## Self-Review (run before handoff)

- [ ] **Spec coverage:** Sections 1–4 all have tasks (1=helpers [done pre-work] + render refactor Task 1; 2=wiring Task 3; 3=harness removal Task 4; 4=tests Task 2 + Task 5).
- [ ] **Placeholder scan:** No TBD/TODO; every code step shows full code.
- [ ] **Type consistency:** `_fusion_rerank(query, fact_texts) -> list | None`; result key is `"reranked"` (bool); helper is `_fact_render_text(fact) -> str`. Consistent across Tasks 1–3 and the test in Task 2.
- [ ] **Verify the offline suite is green at the end:** `python -m pytest tests/ -q`.
