# Semantic recall over versioned facts (`memoria_facts`)

**Date:** 2026-06-24
**Status:** Revised (3rd revision, post two validation rounds)
**Scope:** Add embedding-based recall over the `memoria_facts` table so paraphrased questions (ABS/SUM, currently at 0.000 recall) can find facts whose `key`/`value` share no literal tokens with the query. Write-time-only backfill (no migration of existing rows).

## Problem (evidence-backed)

Overall recall is 0.347; **ABS and PF sit at 0.000** (AGENTS.md). Root cause: `_memoria_fact_retrieve` (beam.py:5500) matches via four pure-SQL lexical passes — numbers, capitalized terms, a synonym map, and `context_snippet LIKE '%term%'`. An ABS question like "How did user feedback influence UI/UX?" retrieves nothing because no fact literally contains those words. The facts exist (63 version chains stored at ingest) but are unreachable by paraphrase. Semantic recall over the facts is the targeted fix.

## Why not "just reuse the episodic vec helpers"

Two validation rounds established that the obvious approach is wrong:

- `_vec_insert` (beam.py:1923) and `_vec_search` (beam.py:1966) are **hardcoded to `vec_episodes`** — no table parameter. They cannot be called verbatim for a facts table.
- `_vec_available` (beam.py:1358) and `_effective_vec_type` (beam.py:1906) probe **`vec_episodes` specifically**. A facts-table guard built on them checks the wrong table.
- `_memoria_fact_retrieve`'s post-Pass-4 block (beam.py:5617–5728) deduplicates by key, keeps the latest version, collapses direct-metric questions, sorts by `version_id` desc, then truncates to `top_k`. **There is no valid RRF-fuse insertion point inside this method** — fusing before it gets overwritten; fusing after fuses two already-arbitrary, already-truncated lists. Also `_rrf_fuse` takes string keys, not dicts.

This spec resolves all three.

## Design

### Part A — Generalize the vec helpers with a `table` parameter

Add an optional `table: str = "vec_episodes"` parameter to all four vec helpers so existing callers are byte-for-byte unchanged, and facts-specific helpers can target `vec_facts`:

- `_vec_available(conn, table="vec_episodes")` — change the probe from `vec_episodes` to `table`.
- `_effective_vec_type(conn, table="vec_episodes")` — change the `sqlite_master` lookup from `vec_episodes` to `table`.
- `_vec_insert(conn, rowid, embedding, table="vec_episodes")` — change the three INSERT statements (`vec_episodes` → `table`).
- `_vec_search(conn, embedding, k=20, table="vec_episodes")` — change the three SELECT statements (`vec_episodes` → `table`).

The unit-normalization (the sqlite-vec 0.1.9 1024-dim workaround), the inlined-`k`-literal requirement, and the per-vec-type quantize pairing all live inside these helpers and carry over unchanged. **No new SQL is hand-written for the facts table** — the same proven, vec-type-aware statements are reused with a parameterized name. This is the concrete resolution to validator Defect 1.

**Critical threading requirement (validator round 3, item 1):** the internal calls must also carry the table name, or vec-type detection silently reads the wrong table's schema:
- `_effective_vec_type` internally calls `_vec_available(conn)` at beam.py:1908 — must become `_vec_available(conn, table=table)`.
- `_vec_insert` internally calls `_effective_vec_type(conn)` at 1932 — must become `_effective_vec_type(conn, table=table)`.
- `_vec_search` internally calls `_effective_vec_type(conn)` at 1973 — must become `_effective_vec_type(conn, table=table)`.

Harmless today (both tables created with the same type), but a correctness landmine if the two tables ever diverge. The plan must enforce threading at all three internal call sites.

### Part B — Repurpose the existing (dead) `vec_facts` table

`vec_facts` already exists (beam.py:1059) but is orphaned — nothing inserts into or queries it (confirmed by repo-wide grep). Repurpose it for versioned-fact embeddings rather than creating a third table (`vec_memoria_facts`). Removes the naming-collision risk (validator NEW-C) and the "wrong table checked" risk, since Part A's helpers now probe the table they're given.

**Schema change:** none required. `vec_facts` is already `vec0(embedding <type>[EMBEDDING_DIM])`, same shape as `vec_episodes`. Its rowids will be `memoria_facts.id` (the AUTOINCREMENT PK, beam.py:820).

### Part C — Write path: embed live facts at insert

A new method `_embed_fact(conn, rowid, ctx, key, value)`:

```python
def _embed_fact(self, conn, rowid: int, ctx: str, key: str, value: str) -> None:
    """Best-effort embedding of a LIVE fact into vec_facts. Swallows all failures."""
    text = (ctx.strip() or f"{key}: {value}").strip()  # context_snippet carries the semantic content; fall back to key:value
    if not text:
        return
    try:
        from . import embeddings as _e
        if not _e.available():
            return
        embs = _e.embed([text])
        if not embs or embs[0] is None:
            return
        _vec_insert(conn, rowid, embs[0].tolist(), table="vec_facts")
    except Exception:
        pass
```

**Embed-text choice (validator Defect 5):** `context_snippet` when present, else `"{key}: {value}"`. Dates/versions have generic keys (`named_date`) but meaningful surrounding utterances in `context_snippet`; embedding the snippet carries real semantic content (it's what lexical Pass 4 already searches). When `ctx` is empty, fall back to `key: value` (unchanged from baseline for that row).

**Call sites — only LIVE rows (validator Defect 3):**
- `_insert_fact` plain branch (beam.py:5052) — embed after insert.
- `_insert_fact` new-version branch (beam.py:5044) — embed after insert (the new live value).
- `_insert_change_fact` new-value branch (beam.py:5106) — embed after insert.
- **Skip:** `_insert_fact` date branch (5017 — produces a *live* row, but skipped by content-quality choice: dates have generic keys like `named_date` and embed poorly), `_insert_change_fact` superseded-old branch (5096 — **dead row**, `valid_to_msg_idx` set: embedding it pollutes the index with stale values).

**lastrowid capture (validator Defect 3):** each indexed INSERT is restructured to retain its cursor: `cur = self.conn.execute(...); rid = cur.lastrowid`, then `_embed_fact(self.conn, rid, ctx, key, value)`. `Connection.execute` returns a cursor exposing `lastrowid` in sqlite3.

**Commit semantics (validator NEW-3b):** `_vec_insert` calls `_real_commit()`/`commit()` (1960-1963), which force-commits per call. Doing that per-fact inside `_insert_fact` means N commits per ingest. Resolution: **the write path does NOT call `_vec_insert` directly inside `_insert_fact`.** Instead it enqueues `(rid, text)` onto a per-beam pending list, and a new `_flush_fact_embeddings()` method batch-embeds (a single `embed([...])` call over all pending texts — cheaper than N `embed([text])` calls) and inserts after the ingest loop completes, called from the batch ingest boundary.

Note (validator round 3, item 4): this flush pattern is **novel**, not a mirror of how episodes are embedded — episodes embed per-row inside `_consolidate_episode` (beam.py:4049-4065), not at a boundary. The justification for deferring is (a) avoiding N per-fact force-commits and (b) enabling one batched `embed([...])` call — not protecting a live `_deferred_commits` batching path (validator confirmed `with _deferred_commits` is never used anywhere, so there is no active batching to protect). If `_flush_fact_embeddings` is never called (e.g. a direct `_insert_fact` test call), facts simply have no vector — lexical retrieval still works.

### Part D — Read path: a 7th fusion specialist

Rather than fuse inside `_memoria_fact_retrieve` (impossible per validator Defect 2), the semantic pass is a **7th specialist** in `_memoria_fused_retrieve` (beam.py:5150), after the existing 6. This is structurally sound: the fusion loop (5310-5317) keys on `source_memory_id`; semantic facts get a synthetic `source_memory_id: f"semantic:fact:{rid}"`, producing unique keys that RRF-combine with the lexical specialists without collision.

New method `_memoria_semantic_retrieve(self, query, top_k=10) -> dict`:

```python
def _memoria_semantic_retrieve(self, query, top_k=10):
    from . import embeddings as _e
    if not _e.available() or not _vec_available(self.conn, table="vec_facts"):
        return {"context": "", "facts": [], "source": "fallback"}
    try:
        q_emb = _e.embed_query(query)   # @lru_cache'd — 1 net/new embed per question
        if q_emb is None:
            return {"context": "", "facts": [], "source": "fallback"}
        hits = _vec_search(self.conn, q_emb, k=top_k * 2, table="vec_facts")
        if not hits:
            return {"context": "", "facts": [], "source": "fallback"}
        rids = [h["rowid"] for h in hits]
        qmarks = ",".join("?" * len(rids))
        # validator Defect NEW-3: filter to LIVE facts only (valid_to_msg_idx IS NULL)
        rows = self.conn.execute(
            f"SELECT fact_type, key, value, context_snippet, previous_value, "
            f"updated_msg_idx, version_id, source_memory_id, message_idx, "
            f"valid_from_msg_idx, id "
            f"FROM memoria_facts WHERE id IN ({qmarks}) "
            f"AND session_id = ? AND valid_to_msg_idx IS NULL",
            (*rids, self.session_id),
        ).fetchall()
        # reorder rows to match _vec_search distance ranking
        order = {r[10]: i for i, r in enumerate(rows)}  # r[10] == id
        rows.sort(key=lambda r: order.get(r[10], 1 << 30))
        facts = []
        for r in rows:
            d = dict(zip(['type','key','value','context','previous_value',
                          'updated_msg_idx','version_id','source_memory_id',
                          'message_idx','valid_from_msg_idx','id'], r))
            d['source_memory_id'] = f"semantic:fact:{r[10]}"  # synthetic fusion key
            facts.append(d)
        ctx_lines = [self._format_versioned_fact(f, intent='').split('] ',1)[-1] for f in facts]  # readable body
        return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_semantic"}
    except Exception:
        return {"context": "", "facts": [], "source": "fallback"}
```

**Superseded-fact filter (validator NEW-3):** the JOIN includes `valid_to_msg_idx IS NULL`, so only live facts surface. Combined with the Part C cleanup hook below, the semantic index cannot return stale values.

**Cleanup hook (validator NEW-3, completeness):** when a fact is superseded (the `UPDATE memoria_facts SET valid_to_msg_idx` at beam.py:5033 and 5090), also delete its vec row: `self.conn.execute("DELETE FROM vec_facts WHERE rowid = ?", (superseded_id,))`. This keeps the index from accumulating dead vectors even though the `IS NULL` filter already protects correctness. (Both the filter and the hook are belt-and-suspenders; the filter is the correctness guarantee, the hook is hygiene.)

**Per-question cost (validator NEW-CHECK A):** this specialist adds **1 new query-embed per question** to a previously pure-SQL fusion path (the other 6 specialists are pure SQL). `embed_query` is `@lru_cache`'d (embeddings.py:214), so repeated identical queries within a run are free; the cost is ~tens of ms (local fastembed) or one network round-trip (API) per distinct question. Acceptable and explicitly acknowledged.

### Part E — Offline test safety (validator Defect 4)

`test_memoria_regressions.py` and `test_ku_update_framing.py` call `_insert_fact` directly with **no** `EDUMEM_NO_EMBEDDINGS` and no mock, and there is no `conftest.py`/`pytest.ini` setting a global default. With Part C, a clean CI sandbox (no cached ONNX, no network) could newly attempt a lazy fastembed download.

Resolution (part of this change, not optional):
- These two test files gain a module-level `monkeypatch`/`autouse` fixture (or `os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"` in setup) so `_embed_fact`'s `available()` check returns False and no embed is attempted. This is documented in the spec because it's a required test change, not an afterthought.
- The `try/except → pass` in `_embed_fact` covers exception failures; the `available()` gate covers the `EDUMEM_NO_EMBEDDINGS` path. The residual lazy-download hang is a **pre-existing latent fastembed risk** (it affects existing `vec_episodes` equally) — documented as known, out of scope to fix here.

## No new feature flag

Validator Defect 6 was initially thought to need a flag, then re-examined: `EDUMEM_NO_EMBEDDINGS` (master switch) + `_vec_available(table="vec_facts")` + `embeddings.available()` cover every case where the specialist must skip. No per-feature flag is added (YAGNI). If benchmarking later needs to isolate this one pass for A/B, a flag is a one-line addition then.

## Verification

- Fast suite: `python -m pytest tests/ -q` stays green (the two affected tests get the env fix; offline → Part E gates → specialist returns fallback → no behavior change).
- New offline test: stub `_vec_search`/`_e.embed_query` (no network), ingest facts, assert semantic specialist surfaces a fact whose key/value share no tokens with the query (the ABS-style case that's 0.000 today). Must FAIL if Part D is reverted.
- Live benchmark (`EDUMEM_RETRIEVAL_E2E=1`): re-ingest (write-time-only, so the cached DB must be rebuilt), measure ABS/PF recall lift off 0.000.

## Out of scope

- The lazy-fastembed-download hang (pre-existing, affects episodes too).
- Richer Hindsight-style "conclusions" extraction (separate effort; only worthwhile if semantic recall proves lexical-reach was the bottleneck).
- Backfill of pre-existing DBs (write-time-only decision: rebuild the cached DB to measure).
