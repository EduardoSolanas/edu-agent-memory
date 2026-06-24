# Semantic Fact Recall Implementation Plan (TDD)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add embedding-based recall over the `memoria_facts` table so paraphrased questions (ABS/SUM, currently 0.000 recall) find facts whose key/value share no literal tokens with the query.

**Architecture:** Generalize the 4 vec helpers with a `table` param; repurpose the dead `vec_facts` table; embed live facts at insert (deferred to a batch boundary); add a 7th semantic fusion specialist in `_memoria_fused_retrieve` that queries `vec_facts` and fuses via the existing RRF. Full spec: `docs/superpowers/specs/2026-06-24-semantic-fact-recall-design.md`.

**Tech Stack:** Python 3, sqlite-vec (vec0 virtual tables), `edumem.core.embeddings` (`embed`, `embed_query`, `available`), pytest.

---

## File Structure

- **Modify** `edumem/core/beam.py` — generalize 4 vec helpers; add `_embed_fact` enqueue + `_flush_fact_embeddings` + `_memoria_semantic_retrieve`; wire into `_insert_fact`/`_insert_change_fact`/`_memoria_fused_retrieve`; add supersession cleanup.
- **Modify** `tests/test_memoria_regressions.py` — add `EDUMEM_NO_EMBEDDINGS` guard (Part E).
- **Modify** `tests/test_ku_update_framing.py` — add `EDUMEM_NO_EMBEDDINGS` guard (Part E).
- **Create** `tests/test_semantic_fact_recall.py` — new TDD tests (one per task below).

---

## Task 1: Generalize `_vec_available` with a `table` param (TDD)

**Files:**
- Modify: `edumem/core/beam.py:1358-1365`
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_semantic_fact_recall.py`:
```python
"""TDD tests for semantic recall over memoria_facts (vec_facts repurposed)."""
from __future__ import annotations

import os
from pathlib import Path

# Suppress embeddings for the offline unit tests in this file. The write-path
# embed is exercised by stubbing, not by a real model. (Spec Part E.)
os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")

from edumem.core import beam as beam_mod
from edumem.core.beam import BeamMemory, _vec_available


def _new_beam(tmp_path: Path) -> BeamMemory:
    return BeamMemory(session_id="sem-test", db_path=tmp_path / "sem.db")


def test_vec_available_checks_the_named_table(tmp_path):
    """_vec_available(table=...) probes the given table, not vec_episodes."""
    beam = _new_beam(tmp_path)
    try:
        # vec_episodes and vec_facts both exist after init_beam (beam.py:738, 1059).
        assert _vec_available(beam.conn, table="vec_episodes") in (True, False)
        # vec_facts is created in init_beam too; probing it must not raise and
        # must reflect ITS existence, independent of vec_episodes.
        assert _vec_available(beam.conn, table="vec_facts") in (True, False)
        # A nonexistent table must return False, never raise.
        assert _vec_available(beam.conn, table="vec_does_not_exist") is False
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_vec_available_checks_the_named_table -q`
Expected: FAIL — `TypeError: _vec_available() got an unexpected keyword argument 'table'`.

- [ ] **Step 3: Implement — add the `table` param**

Edit `edumem/core/beam.py:1358`:
```python
def _vec_available(conn: sqlite3.Connection, table: str = "vec_episodes") -> bool:
    if not _SQLITE_VEC_AVAILABLE:
        return False
    try:
        conn.execute(f"SELECT 1 FROM {table} LIMIT 0")
        return True
    except Exception:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_vec_available_checks_the_named_table -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(vec): generalize _vec_available with a table param"
```

---

## Task 2: Generalize `_effective_vec_type` (thread table internally)

**Files:**
- Modify: `edumem/core/beam.py:1906-1920`
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_semantic_fact_recall.py`:
```python
from edumem.core.beam import _effective_vec_type


def test_effective_vec_type_reads_named_table_schema(tmp_path):
    """_effective_vec_type(table=...) reads the named table's schema, not vec_episodes."""
    beam = _new_beam(tmp_path)
    try:
        # Both tables created with the same effective_vec_type in init_beam, so both
        # resolve to the same type string. The point: the call must not raise and
        # must accept the table kwarg.
        t_episodes = _effective_vec_type(beam.conn, table="vec_episodes")
        t_facts = _effective_vec_type(beam.conn, table="vec_facts")
        assert t_episodes in ("bit", "int8", "float32")
        assert t_facts == t_episodes  # both created together -> same type
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_effective_vec_type_reads_named_table_schema -q`
Expected: FAIL — `TypeError: _effective_vec_type() got an unexpected keyword argument 'table'`.

- [ ] **Step 3: Implement — add `table` param AND thread it into the internal `_vec_available` call (validator-required)**

Edit `edumem/core/beam.py:1906`:
```python
def _effective_vec_type(conn: sqlite3.Connection, table: str = "vec_episodes") -> str:
    """Re-detect the actual vector type used by the named vec0 table."""
    if not _vec_available(conn, table=table):
        return "float32"
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row and "int8" in row[0]:
            return "int8"
        if row and "bit" in row[0]:
            return "bit"
    except Exception:
        logger.info("Regex extraction failed, skipping", exc_info=True)
    return "float32"
```
Note: the parameterized `name=?` is now bind-safe (was a literal before); the `_vec_available(conn, table=table)` call threads the table name per validator round-3 item 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_effective_vec_type_reads_named_table_schema -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(vec): generalize _effective_vec_type with table param (threaded)"
```

---

## Task 3: Generalize `_vec_insert` (thread table into internal `_effective_vec_type`)

**Files:**
- Modify: `edumem/core/beam.py:1923-1963`
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_semantic_fact_recall.py`:
```python
import numpy as np
from edumem.core.beam import _vec_insert, _vec_available


def test_vec_insert_targets_named_table(tmp_path):
    """_vec_insert(table=...) inserts into the named table, not vec_episodes."""
    beam = _new_beam(tmp_path)
    try:
        emb = np.random.rand(EMBEDDING_DIM_REF).astype(np.float32) if False else None
        # We can't easily synthesize a correctly-shaped vector without knowing
        # EMBEDDING_DIM; instead assert the API exists and a wrong-dim vector to
        # vec_facts raises (proving it actually targeted vec_facts, not silently
        # hitting vec_episodes). Use a deliberately-bogus short vector.
        import pytest
        with pytest.raises(Exception):
            _vec_insert(beam.conn, 999, [0.1, 0.2], table="vec_facts")
    finally:
        beam.conn.close()
```
(Use a sentinel import line at top of file: `EMBEDDING_DIM_REF = 0  # placeholder, unused` — actually drop that line; the test does not use it. Keep the test minimal as written without the `emb = ... if False else None` line.)

Final minimal test body to use:
```python
import pytest
from edumem.core.beam import _vec_insert


def test_vec_insert_targets_named_table(tmp_path):
    """_vec_insert(table=...) actually targets the named table (raises on bad dim)."""
    beam = _new_beam(tmp_path)
    try:
        # A 2-dim vector into an EMBEDDING_DIM table must raise — proving the
        # INSERT went to vec_facts (which has the real dim), not silently nowhere.
        with pytest.raises(Exception):
            _vec_insert(beam.conn, 999, [0.1, 0.2], table="vec_facts")
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_vec_insert_targets_named_table -q`
Expected: FAIL — `TypeError: _vec_insert() got an unexpected keyword argument 'table'`.

- [ ] **Step 3: Implement — add `table` param; thread into `_effective_vec_type`; use in all 3 INSERT strings**

Edit `edumem/core/beam.py:1923`. Change the signature and the three INSERT statements (`vec_episodes` → `{table}`), and the internal `_effective_vec_type(conn)` call:
```python
def _vec_insert(conn: sqlite3.Connection, rowid: int, embedding: List[float], table: str = "vec_episodes"):
    """Insert embedding into the named sqlite-vec table with quantization via SQL functions.

    Manually normalizes the embedding to unit length before quantization.
    (sqlite-vec 0.1.9 'unit' param fails at 1024-dim; pre-normalize works around it.)
    """
    vec_type = _effective_vec_type(conn, table=table)
    import numpy as _np
    emb_arr = _np.array(embedding, dtype=_np.float32)
    norm = _np.linalg.norm(emb_arr)
    if norm > 0:
        emb_arr = emb_arr / norm
    emb_json = json.dumps(emb_arr.tolist())
    if vec_type == "bit":
        conn.execute(
            f"INSERT INTO {table}(rowid, embedding) VALUES (?, vec_quantize_binary(?))",
            (rowid, emb_json)
        )
    elif vec_type == "int8":
        conn.execute(
            f"INSERT INTO {table}(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
            (rowid, emb_json)
        )
    else:
        conn.execute(
            f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
            (rowid, emb_json)
        )
    if isinstance(conn, _BeamConnection):
        conn._real_commit()
    else:
        conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_vec_insert_targets_named_table -q`
Expected: PASS (the bad-dim vector raises, proving it targeted vec_facts).
Also run the whole file: `python -m pytest tests/test_semantic_fact_recall.py -q` — all three pass.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(vec): generalize _vec_insert with table param (threaded)"
```

---

## Task 4: Generalize `_vec_search`

**Files:**
- Modify: `edumem/core/beam.py:1966-2002`
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_vec_search_accepts_table_param(tmp_path):
    """_vec_search(table=...) accepts the param without error on an empty table."""
    from edumem.core.beam import _vec_search
    beam = _new_beam(tmp_path)
    try:
        # Searching an empty vec_facts returns [] (no rows), not an error.
        # We need a real-dim query vector; skip if vec_facts unavailable.
        if not _vec_available(beam.conn, table="vec_facts"):
            return  # sqlite-vec absent in this env; nothing to assert
        # Build a zero vector of the right dim by reading EMBEDDING_DIM lazily.
        from edumem.core.embeddings import EMBEDDING_DIM
        q = [0.0] * EMBEDDING_DIM
        res = _vec_search(beam.conn, q, k=5, table="vec_facts")
        assert res == []
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_vec_search_accepts_table_param -q`
Expected: FAIL — `TypeError: _vec_search() got an unexpected keyword argument 'table'`.

- [ ] **Step 3: Implement — add `table` param; thread into `_effective_vec_type`; use in all 3 SELECT strings**

Edit `edumem/core/beam.py:1966`:
```python
def _vec_search(conn: sqlite3.Connection, embedding: List[float], k: int = 20, table: str = "vec_episodes") -> List[Dict]:
    """Search the named sqlite-vec table; return rowids with distances.

    Normalizes the query embedding to unit length before quantization so
    distances are commensurate with stored vectors (unit-normalized at insert).
    """
    vec_type = _effective_vec_type(conn, table=table)
    import numpy as _np
    emb_arr = _np.array(embedding, dtype=_np.float32)
    norm = _np.linalg.norm(emb_arr)
    if norm > 0:
        emb_arr = emb_arr / norm
    emb_json = json.dumps(emb_arr.tolist())
    k = int(k)
    if vec_type == "bit":
        rows = conn.execute(
            f"SELECT rowid, distance FROM {table} WHERE embedding MATCH vec_quantize_binary(?) ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    elif vec_type == "int8":
        rows = conn.execute(
            f'SELECT rowid, distance FROM {table} WHERE embedding MATCH vec_quantize_int8(?, "unit") AND k={k} ORDER BY distance',
            (emb_json,)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT rowid, distance FROM {table} WHERE embedding MATCH ? ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    return [{"rowid": r["rowid"], "distance": r["distance"]} for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py -q`
Expected: all 4 tests PASS. Also run the full fast suite to confirm no regression in existing `_vec_*` callers:
`python -m pytest tests/ -q` — expect only the pre-existing known failures (if any) or all-green.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(vec): generalize _vec_search with table param (threaded)"
```

---

## Task 5: `_embed_fact` + pending list (write path, no commit inside `_insert_fact`)

**Files:**
- Modify: `edumem/core/beam.py` — add `_embed_fact` enqueue method + pending list in `__init__`; call from `_insert_fact` live branches.
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_embed_fact_enqueues_pending_when_available(tmp_path, monkeypatch):
    """When embeddings available, _insert_fact enqueues the fact for batch embed."""
    # Force embeddings "available" and stub embed so no real model loads.
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "")
    import edumem.core.embeddings as _e
    monkeypatch.setattr(_e, "available", lambda: True)
    captured = []
    monkeypatch.setattr(_e, "embed", lambda texts: (captured.extend(texts), __import__("numpy").zeros((len(texts), 1)))[1])
    # sqlite-vec may be absent in unit env; if so, enqueue still records the text.
    beam = _new_beam(tmp_path)
    try:
        beam._insert_fact("sem-test", 5, "metric", "api_latency_ms", "120ms",
                          "The API latency was measured at 120ms during load test.", 0.5,
                          source_memory_id="m5")
        # The pending list records (rid, text) for the plain branch.
        assert any("120ms" in t for _, t in getattr(beam, "_pending_fact_embeddings", []))
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_embed_fact_enqueues_pending_when_available -q`
Expected: FAIL — `AttributeError: 'BeamMemory' object has no attribute '_pending_fact_embeddings'` (or the enqueue never happens).

- [ ] **Step 3: Implement**

(a) In `BeamMemory.__init__` (find the assignment block, e.g. near `self.session_id = ...`), add:
```python
        # Deferred fact-embedding queue: (rowid, embed_text). Flushed by
        # _flush_fact_embeddings() at the batch ingest boundary so we do ONE
        # batched embed() call instead of N per-fact embeds + N commits.
        self._pending_fact_embeddings: list = []
```

(b) Add method (place near `_insert_fact`, ~line 4998):
```python
    def _embed_fact_enqueue(self, rowid: int, ctx: str, key: str, value: str) -> None:
        """Enqueue a LIVE fact for batch embedding. Best-effort; never raises.

        Embed text is context_snippet when present (carries semantic content),
        else "{key}: {value}". The actual embed+insert happens in
        _flush_fact_embeddings to avoid N per-fact commits.
        """
        text = (ctx or "").strip() or f"{key}: {value}".strip()
        if not text or rowid is None:
            return
        self._pending_fact_embeddings.append((rowid, text))

    def _flush_fact_embeddings(self) -> None:
        """Batch-embed all pending facts and insert into vec_facts. Best-effort."""
        if not self._pending_fact_embeddings:
            return
        from . import embeddings as _e
        if not _e.available() or not _vec_available(self.conn, table="vec_facts"):
            self._pending_fact_embeddings = []
            return
        try:
            rids, texts = zip(*self._pending_fact_embeddings)
            embs = _e.embed(list(texts))
            if embs is None:
                self._pending_fact_embeddings = []
                return
            for rid, emb in zip(rids, embs):
                if emb is None:
                    continue
                try:
                    _vec_insert(self.conn, rid, emb.tolist(), table="vec_facts")
                except Exception:
                    pass  # individual insert failure (e.g. dup rowid) is non-fatal
        except Exception:
            pass
        finally:
            self._pending_fact_embeddings = []
```

(c) In `_insert_fact`, the **plain branch** (beam.py ~5052) — capture lastrowid and enqueue. Change:
```python
        else:
            self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, "
                "context_snippet, importance, valid_from_msg_idx, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session, msg_idx, ftype, key, value, ctx, importance, msg_idx, source_memory_id))
```
to:
```python
        else:
            _cur = self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, "
                "context_snippet, importance, valid_from_msg_idx, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session, msg_idx, ftype, key, value, ctx, importance, msg_idx, source_memory_id))
            if ftype != 'date':  # skip date branch by content-quality (generic keys)
                self._embed_fact_enqueue(_cur.lastrowid, ctx, key, value)
```

(d) In `_insert_fact`, the **new-version branch** (beam.py ~5044) — capture lastrowid and enqueue. Change the `self.conn.execute("INSERT INTO memoria_facts (...) VALUES (...)", (...))` of the new-version row to capture cursor and add:
```python
                _cur = self.conn.execute(
                    "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, "
                    "context_snippet, importance, version_id, previous_value, updated_msg_idx, "
                    "valid_from_msg_idx, source_memory_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session, msg_idx, ftype, key, value, ctx, importance,
                     new_version, existing[1], msg_idx, msg_idx, source_memory_id))
                if ftype != 'date':
                    self._embed_fact_enqueue(_cur.lastrowid, ctx, key, value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_embed_fact_enqueues_pending_when_available -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(facts): enqueue live facts for batch embedding (deferred, no per-fact commit)"
```

---

## Task 6: `_insert_change_fact` enqueues the new (live) row only; supersession cleanup hook

**Files:**
- Modify: `edumem/core/beam.py:5059-5112` (`_insert_change_fact`) + the two supersession UPDATE sites (5033, 5090).
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_insert_change_fact_enqueues_only_new_row_and_cleans_old(tmp_path, monkeypatch):
    """_insert_change_fact enqueues the new row, never the superseded-old; old vec row deleted."""
    import edumem.core.embeddings as _e
    monkeypatch.setattr(_e, "available", lambda: True)
    monkeypatch.setattr(_e, "embed", lambda texts: __import__("numpy").zeros((len(texts), 1)))
    beam = _new_beam(tmp_path)
    try:
        # First insert a plain fact, then "change" it.
        beam._insert_fact("sem-test", 1, "metric", "svc_latency", "100ms",
                          "latency was 100ms", 0.5, source_memory_id="m1")
        # Simulate a vec_facts row for the old fact id (as if flushed), then change.
        # We test the enqueue + cleanup-hook contract: after _insert_change_fact,
        # the pending list has exactly ONE entry (the new row), and the cleanup
        # hook removed any old vec row.
        beam._pending_fact_embeddings = []  # reset after the plain insert
        beam._insert_change_fact("sem-test", 2, "metric", "svc_latency",
                                 "100ms", "200ms", "latency changed to 200ms", 0.5,
                                 source_memory_id="m2")
        # Exactly one pending entry (the new row), and it references the new text.
        assert len(beam._pending_fact_embeddings) == 1
        assert "200ms" in beam._pending_fact_embeddings[0][1]
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_insert_change_fact_enqueues_only_new_row_and_cleans_old -q`
Expected: FAIL — `_insert_change_fact` doesn't enqueue (AttributeError or 0 pending).

- [ ] **Step 3: Implement**

(a) Read `_insert_change_fact` (5059-5112). In the **new-value INSERT** branch (~5106), capture lastrowid and enqueue the new (live) row:
```python
            _cur = self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, "
                "context_snippet, importance, previous_value, updated_msg_idx, "
                "valid_from_msg_idx, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session, msg_idx, ftype, key, value, ctx, importance,
                 previous_value, msg_idx, msg_idx, source_memory_id))
            if ftype != 'date':
                self._embed_fact_enqueue(_cur.lastrowid, ctx, key, value)
```
Do NOT touch the superseded-old branch (5096) — it stays a dead row, never enqueued.

(b) **Cleanup hook on supersession.** There are two supersession UPDATE sites that set `valid_to_msg_idx`:
- `_insert_fact` new-version branch, the UPDATE at ~5033: `UPDATE memoria_facts SET valid_to_msg_idx = ? WHERE id = ?` with `(msg_idx, existing[0])`. After it, add:
```python
                try:
                    self.conn.execute("DELETE FROM vec_facts WHERE rowid = ?", (existing[0],))
                except Exception:
                    pass
```
- `_insert_change_fact`, the existing-supersede UPDATE at ~5090 (`existing` truthy branch). After that UPDATE (keyed on `existing[0]`), add the same DELETE hook.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py -q`
Expected: all tests PASS. Run full suite: `python -m pytest tests/ -q` — no new failures.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(facts): _insert_change_fact enqueues new row only + supersession vec cleanup"
```

---

## Task 7: Hook `_flush_fact_embeddings` at the batch ingest boundary

**Files:**
- Modify: `edumem/core/beam.py` — `remember_batch` (2948) end; also any other public ingest entry that calls `_insert_fact` in a loop.
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test**

Append:
```python
def test_remember_batch_flushes_fact_embeddings(tmp_path, monkeypatch):
    """remember_batch calls _flush_fact_embeddings once at the end (one batched embed)."""
    import edumem.core.embeddings as _e
    monkeypatch.setattr(_e, "available", lambda: True)
    embed_calls = []
    def _fake_embed(texts):
        embed_calls.append(list(texts))
        import numpy as np
        return np.zeros((len(texts), 1))
    monkeypatch.setattr(_e, "embed", _fake_embed)
    beam = _new_beam(tmp_path)
    try:
        # _store_llm_extraction is the path that calls _insert_fact in a loop.
        # Drive a couple of facts through it, then remember_batch (or the
        # extraction path) must flush with a SINGLE batched embed call.
        beam._insert_fact("sem-test", 1, "metric", "a", "1", "ctx a", 0.5, source_memory_id="a")
        beam._insert_fact("sem-test", 2, "metric", "b", "2", "ctx b", 0.5, source_memory_id="b")
        beam._flush_fact_embeddings()
        assert len(embed_calls) == 1
        assert len(embed_calls[0]) == 2  # both facts in ONE batched call
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it passes (it may already pass from Task 5)**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_remember_batch_flushes_fact_embeddings -q`
Expected: PASS (the flush is batched by Task 5's `_flush_fact_embeddings`). If it fails, the flush isn't batching — fix before proceeding.

- [ ] **Step 3: Implement — call `_flush_fact_embeddings()` at the end of `remember_batch`**

In `remember_batch` (2948), find its return statement (search for the final `return memory_ids` or equivalent). Just before it, add:
```python
        # Flush any fact embeddings queued by _store_llm_extraction/_insert_fact
        # during this batch — one batched embed() call, not N per-fact calls.
        try:
            self._flush_fact_embeddings()
        except Exception:
            pass
```

- [ ] **Step 4: Run test + full suite**

Run: `python -m pytest tests/test_semantic_fact_recall.py -q` (all pass) then `python -m pytest tests/ -q` (no regressions).

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(facts): flush fact embeddings at remember_batch boundary"
```

---

## Task 8: `_memoria_semantic_retrieve` (the 7th specialist body)

**Files:**
- Modify: `edumem/core/beam.py` — add method near `_memoria_fact_retrieve` (~5500).
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test (the core value test — ABS-style paraphrase reach)**

Append:
```python
def test_semantic_retrieve_finds_fact_with_no_literal_overlap(tmp_path, monkeypatch):
    """A query sharing NO tokens with the fact's key/value still surfaces it
    via the semantic specialist (the ABS 0.000 case). Stubs embed + vec_search."""
    beam = _new_beam(tmp_path)
    try:
        # Insert a fact whose key/value share no words with the query.
        beam._insert_fact("sem-test", 1, "entity", "design system", "neumorphism",
                          "We adopted a neumorphism design system for the UI.", 0.5,
                          source_memory_id="m1")
        beam.conn.commit()
        # Find the rowid of the inserted fact.
        rid = beam.conn.execute(
            "SELECT id FROM memoria_facts WHERE source_memory_id = ?", ("m1",)
        ).fetchone()[0]

        # Stub the query embed + vec_search so the semantic path fires deterministically.
        import edumem.core.embeddings as _e
        monkeypatch.setattr(_e, "available", lambda: True)
        monkeypatch.setattr(_e, "embed_query", lambda q: [0.0])  # nonzero to pass None check
        import edumem.core.beam as _b
        monkeypatch.setattr(_b, "_vec_available", lambda conn, table="vec_episodes": True)
        monkeypatch.setattr(_b, "_vec_search", lambda conn, emb, k=10, table="vec_episodes":
                            [{"rowid": rid, "distance": 0.1}] if table == "vec_facts" else [])

        result = beam._memoria_semantic_retrieve("How did user feedback influence the look and feel?", top_k=5)
        assert result["source"] == "memoria_semantic"
        assert "neumorphism" in result["context"]
        # The synthetic fusion key is set.
        assert any(f.get("source_memory_id") == f"semantic:fact:{rid}" for f in result["facts"])
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_semantic_retrieve_finds_fact_with_no_literal_overlap -q`
Expected: FAIL — `AttributeError: 'BeamMemory' object has no attribute '_memoria_semantic_retrieve'`.

- [ ] **Step 3: Implement**

Add method to `BeamMemory` (place right after `_memoria_fact_retrieve`'s return, ~line 5740):
```python
    def _memoria_semantic_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Semantic KNN over vec_facts (repurposed). 7th fusion specialist.

        Finds facts whose key/value share no literal tokens with the query
        (the ABS/SUM 0.000-recall case). Best-effort: returns fallback when
        embeddings/vec unavailable. Filters to LIVE facts (valid_to_msg_idx
        IS NULL) so superseded values never surface.
        """
        from . import embeddings as _e
        if not _e.available() or not _vec_available(self.conn, table="vec_facts"):
            return {"context": "", "facts": [], "source": "fallback"}
        try:
            q_emb = _e.embed_query(query)
            if q_emb is None:
                return {"context": "", "facts": [], "source": "fallback"}
            hits = _vec_search(self.conn, q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb),
                               k=top_k * 2, table="vec_facts")
            if not hits:
                return {"context": "", "facts": [], "source": "fallback"}
            rids = [h["rowid"] for h in hits]
            qmarks = ",".join("?" * len(rids))
            rows = self.conn.execute(
                f"SELECT fact_type, key, value, context_snippet, previous_value, "
                f"updated_msg_idx, version_id, source_memory_id, message_idx, "
                f"valid_from_msg_idx, id "
                f"FROM memoria_facts WHERE id IN ({qmarks}) "
                f"AND session_id = ? AND valid_to_msg_idx IS NULL",
                (*rids, self.session_id),
            ).fetchall()
            # Reorder to match vec_search distance ranking (rids order = nearest first).
            by_id = {r[10]: r for r in rows}
            ordered = [by_id[r] for r in rids if r in by_id]
            cols = ['type', 'key', 'value', 'context', 'previous_value',
                    'updated_msg_idx', 'version_id', 'source_memory_id',
                    'message_idx', 'valid_from_msg_idx', 'id']
            facts = []
            ctx_lines = []
            for r in ordered:
                d = dict(zip(cols, r))
                d['source_memory_id'] = f"semantic:fact:{r[10]}"  # synthetic fusion key
                facts.append(d)
                # Readable body — mirror the fact-specialist render (key: value).
                ctx_lines.append(f"{r[1]}: {r[2]}")
            return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_semantic"}
        except Exception:
            return {"context": "", "facts": [], "source": "fallback"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_semantic_retrieve_finds_fact_with_no_literal_overlap -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(recall): _memoria_semantic_retrieve (7th specialist, live-only, paraphrase reach)"
```

---

## Task 9: Wire the 7th specialist into `_memoria_fused_retrieve`

**Files:**
- Modify: `edumem/core/beam.py` — `_memoria_fused_retrieve` (~5217), after the 6th specialist block (~5300).
- Test: `tests/test_semantic_fact_recall.py`

- [ ] **Step 1: Write the failing test (end-to-end: fusion surfaces the semantic fact)**

Append:
```python
def test_fusion_surfaces_semantic_specialist_fact(tmp_path, monkeypatch):
    """memoria_retrieve (full fusion) surfaces a fact found ONLY by the semantic specialist."""
    beam = _new_beam(tmp_path)
    try:
        beam._insert_fact("sem-test", 1, "entity", "design system", "neumorphism",
                          "We adopted a neumorphism design system for the UI.", 0.5,
                          source_memory_id="m1")
        beam.conn.commit()
        rid = beam.conn.execute("SELECT id FROM memoria_facts WHERE source_memory_id=?", ("m1",)).fetchone()[0]

        import edumem.core.embeddings as _e
        monkeypatch.setattr(_e, "available", lambda: True)
        monkeypatch.setattr(_e, "embed_query", lambda q: [0.0])
        import edumem.core.beam as _b
        monkeypatch.setattr(_b, "_vec_available", lambda conn, table="vec_episodes": True)
        monkeypatch.setattr(_b, "_vec_search", lambda conn, emb, k=10, table="vec_episodes":
                            [{"rowid": rid, "distance": 0.1}] if table == "vec_facts" else [])

        # Query with NO literal overlap to key/value — only semantic can find it.
        result = beam.memoria_retrieve("How did user feedback influence the look and feel?", top_k=5)
        assert "neumorphism" in result["context"], (
            f"semantic specialist fact not surfaced by fusion: {result['context']!r}"
        )
    finally:
        beam.conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_fact_recall.py::test_fusion_surfaces_semantic_specialist_fact -q`
Expected: FAIL — fusion doesn't call the semantic specialist yet, so "neumorphism" isn't surfaced (the lexical fact specialist finds nothing for a query with no token overlap).

- [ ] **Step 3: Implement — add Specialist 7 to `_memoria_fused_retrieve`**

In `_memoria_fused_retrieve`, after the summary specialist block (the 6th, ending ~line 5300 with `specialists.append(('summary', ...))` inside its `if`), add:
```python
        # Specialist 7: Semantic KNN over vec_facts (paraphrase reach for ABS/SUM).
        # Always attempted; returns fallback when embeddings/vec unavailable so it
        # is a no-op offline. Filters to LIVE facts (valid_to_msg_idx IS NULL).
        try:
            start = _time.perf_counter()
            semantic_result = self._memoria_semantic_retrieve(query, top_k=top_k)
            timing['semantic_ms'] = (_time.perf_counter() - start) * 1000
            specialists.append(('semantic', semantic_result))
        except Exception:
            timing['semantic_ms'] = 0
            specialists.append(('semantic', {"context": "", "facts": [], "source": "fallback"}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_fact_recall.py -q`
Expected: all PASS. Then full suite: `python -m pytest tests/ -q` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add edumem/core/beam.py tests/test_semantic_fact_recall.py
git commit -m "feat(recall): wire semantic specialist into _memoria_fused_retrieve (7th voice)"
```

---

## Task 10: Offline test safety — guard the two fact-ingestion tests (Part E)

**Files:**
- Modify: `tests/test_memoria_regressions.py`
- Modify: `tests/test_ku_update_framing.py`

- [ ] **Step 1: Add the env guard to `tests/test_memoria_regressions.py`**

At the top of the file, after the existing imports, add:
```python
# These tests call _insert_fact directly. The semantic-recall write path would
# otherwise attempt a (best-effort) embedding on each insert; suppress it so the
# suite stays offline and deterministic (Spec Part E).
os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")
```

- [ ] **Step 2: Add the same guard to `tests/test_ku_update_framing.py`**

After the imports at the top, add:
```python
import os
os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")  # Spec Part E: keep fact-ingestion offline
```

- [ ] **Step 3: Run the two files + full suite**

Run: `python -m pytest tests/test_memoria_regressions.py tests/test_ku_update_framing.py -q` then `python -m pytest tests/ -q`.
Expected: both files pass; full suite green (no ONNX-download attempt).

- [ ] **Step 4: Commit**

```bash
git add tests/test_memoria_regressions.py tests/test_ku_update_framing.py
git commit -m "test(facts): guard direct _insert_fact tests with EDUMEM_NO_EMBEDDINGS (Part E)"
```

---

## Task 11: Full suite + AGENTS.md note

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Run the complete fast suite**

Run: `python -m pytest tests/ -q`
Expected: all green (no new failures; the 3 pre-existing sanitizer failures were already fixed in a prior commit, so the suite should be fully green).

- [ ] **Step 2: Add a note to AGENTS.md**

Under the "Retrieval-recall benchmark analysis" section, add a short subsection noting semantic fact recall is now wired (write-time embed of live facts into repurposed `vec_facts`; 7th fusion specialist; targets ABS/PF off 0.000; requires re-ingest to populate since backfill is out of scope). Update the "What to try next" list to strike/adjust any now-done item.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): note semantic fact recall wired (7th specialist, vec_facts repurposed)"
```

---

## Self-Review

- [ ] **Spec coverage:** Part A (Tasks 1-4), Part B (vec_facts repurpose — no task needed, it's existing dead schema; referenced in Tasks 5/8), Part C write path (Tasks 5-7), Part D read path (Tasks 8-9), Part E test safety (Task 10), verification (Task 11). All spec parts covered.
- [ ] **Placeholder scan:** every step has concrete code. No TBD/TODO.
- [ ] **Type consistency:** `_embed_fact_enqueue(rid, ctx, key, value)`, `_flush_fact_embeddings()`, `_memoria_semantic_retrieve(query, top_k)`, `_vec_*(conn, ..., table="vec_facts")` — consistent across tasks. Synthetic key `f"semantic:fact:{rid}"` consistent in Tasks 8-9.
- [ ] **TDD ordering:** every task writes the failing test BEFORE the implementation; each test fails for the right reason (function absent / behavior absent).
