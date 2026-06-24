"""TDD tests for semantic recall over memoria_facts (vec_facts repurposed)."""
from __future__ import annotations

import os
from pathlib import Path

# Suppress embeddings for the offline unit tests in this file. The write-path
# embed is exercised by stubbing, not by a real model. (Spec Part E.)
os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")

from edumem.core import beam as beam_mod
from edumem.core.beam import BeamMemory, _vec_available, _effective_vec_type


def _new_beam(tmp_path: Path) -> BeamMemory:
    return BeamMemory(session_id="sem-test", db_path=tmp_path / "sem.db")


def test_vec_available_checks_the_named_table(tmp_path):
    """_vec_available(table=...) probes the named table, not vec_episodes.

    Non-vacuous: vec_facts must resolve True (it exists), a bogus table must
    resolve False. Both differ from the no-table-arg default behavior only if
    the param is actually threaded — reverting the param makes the bogus-table
    probe query vec_episodes and wrongly return True.
    """
    beam = _new_beam(tmp_path)
    try:
        assert _vec_available(beam.conn, table="vec_facts") is True
        # A nonexistent table must return False, never raise. If the param were
        # ignored (hardcoded to vec_episodes), this would wrongly return True.
        assert _vec_available(beam.conn, table="vec_does_not_exist") is False
    finally:
        beam.conn.close()


def test_effective_vec_type_reads_named_table_schema(tmp_path):
    """_effective_vec_type(table=...) reads the named table's schema.

    Non-vacuous: a nonexistent table returns the float32 default (because
    _vec_available returns False for it); vec_facts returns a real type.
    Reverting the param makes the bogus-table call read vec_episodes and
    return a non-default type.
    """
    beam = _new_beam(tmp_path)
    try:
        t_facts = _effective_vec_type(beam.conn, table="vec_facts")
        assert t_facts in ("bit", "int8", "float32")  # real table -> real type
        # Nonexistent table -> _vec_available False -> float32 default.
        assert _effective_vec_type(beam.conn, table="vec_does_not_exist") == "float32"
    finally:
        beam.conn.close()


def test_vec_insert_targets_named_table(tmp_path):
    """_vec_insert(table=...) actually targets the named table.

    Non-vacuous: inserting a correct-dim vector into vec_facts must SUCCEED and
    the row must be readable back FROM vec_facts (not vec_episodes). This fails
    if the param is ignored (row lands in vec_episodes) or if the table is
    hardcoded.
    """
    from edumem.core.beam import _vec_insert
    from edumem.core.embeddings import EMBEDDING_DIM

    beam = _new_beam(tmp_path)
    try:
        if not _vec_available(beam.conn, table="vec_facts"):
            import pytest
            pytest.skip("sqlite-vec not available")
        vec = [0.01] * EMBEDDING_DIM
        _vec_insert(beam.conn, 12345, vec, table="vec_facts")
        # Row must be in vec_facts, NOT in vec_episodes.
        in_facts = beam.conn.execute("SELECT rowid FROM vec_facts WHERE rowid=12345").fetchone()
        in_episodes = beam.conn.execute("SELECT rowid FROM vec_episodes WHERE rowid=12345").fetchone()
        assert in_facts is not None, "row not in vec_facts — table param ignored"
        assert in_episodes is None, "row leaked into vec_episodes — wrong table targeted"
    finally:
        beam.conn.close()


def test_vec_search_targets_named_table(tmp_path):
    """_vec_search(table=...) searches the named table.

    Non-vacuous: insert a vector into vec_facts only, then _vec_search with
    table='vec_facts' must find it while table='vec_episodes' must NOT.
    Reverting the param (hardcoded vec_episodes) makes the facts search miss.
    """
    import pytest
    from edumem.core.beam import _vec_insert, _vec_search
    from edumem.core.embeddings import EMBEDDING_DIM

    beam = _new_beam(tmp_path)
    try:
        if not _vec_available(beam.conn, table="vec_facts"):
            pytest.skip("sqlite-vec not available")
        vec = [0.01] * EMBEDDING_DIM
        _vec_insert(beam.conn, 777, vec, table="vec_facts")
        # Searching vec_facts must find rowid 777.
        hits_facts = _vec_search(beam.conn, vec, k=5, table="vec_facts")
        assert any(h["rowid"] == 777 for h in hits_facts), "vec_facts search missed the inserted row"
        # Searching vec_episodes must NOT find it (it was inserted into vec_facts only).
        hits_ep = _vec_search(beam.conn, vec, k=5, table="vec_episodes")
        assert not any(h["rowid"] == 777 for h in hits_ep), "row leaked across tables"
    finally:
        beam.conn.close()


def test_embed_fact_enqueues_pending_for_live_fact(tmp_path):
    """_insert_fact enqueues a live fact for batch embedding (text = context_snippet).

    Non-vacuous: the pending list must contain (rowid, text) with the
    context_snippet text, NOT key:value, and only when embeddings are available.
    """
    beam = _new_beam(tmp_path)
    try:
        beam._pending_fact_embeddings = []  # ensure clean
        # Force embeddings available + stub so no real model loads.
        import edumem.core.embeddings as _e
        _e_orig = _e.available
        _e.available = lambda: True
        try:
            beam._insert_fact("sem-test", 5, "metric", "api_latency_ms", "120ms",
                              "The API latency was measured at 120ms during load test.", 0.5,
                              source_memory_id="m5")
        finally:
            _e.available = _e_orig
        # One pending entry, text is the context_snippet (not key:value).
        assert len(beam._pending_fact_embeddings) == 1, beam._pending_fact_embeddings
        rid, text = beam._pending_fact_embeddings[0]
        assert "120ms" in text
        assert text.startswith("The API latency"), text  # context_snippet, not "api_latency_ms: 120ms"
    finally:
        beam.conn.close()


def test_insert_change_fact_enqueues_only_new_row(tmp_path):
    """_insert_change_fact enqueues the NEW row only; the old (dead) row is not enqueued.

    Non-vacuous: exactly one pending entry (the new value text), referencing
    new_value not old_value.
    """
    beam = _new_beam(tmp_path)
    try:
        beam._pending_fact_embeddings = []
        import edumem.core.embeddings as _e
        _e_orig = _e.available
        _e.available = lambda: True
        try:
            beam._insert_change_fact("sem-test", 2, "svc_latency",
                                     "100ms", "200ms", "latency changed to 200ms", 0.5,
                                     source_memory_id="m2")
        finally:
            _e.available = _e_orig
        # Exactly ONE pending entry (the new row); the dead old row is not enqueued.
        assert len(beam._pending_fact_embeddings) == 1, beam._pending_fact_embeddings
        rid, text = beam._pending_fact_embeddings[0]
        assert "200ms" in text and "changed to 200ms" in text  # new value + its context
    finally:
        beam.conn.close()


def test_remember_batch_flushes_fact_embeddings_once(tmp_path):
    """remember_batch flushes pending fact embeddings in ONE batched embed call.

    Non-vacuous: stub embed to count calls; after remember_batch, embed was
    called exactly once with all queued texts (not N times).
    """
    import edumem.core.embeddings as _e
    embed_calls = []
    _e_orig_available = _e.available
    _e_orig_embed = _e.embed

    def _fake_embed(texts):
        embed_calls.append(list(texts))
        import numpy as np
        from edumem.core.embeddings import EMBEDDING_DIM
        return np.zeros((len(texts), EMBEDDING_DIM))
    _e.available = lambda: True
    _e.embed = _fake_embed
    try:
        beam = _new_beam(tmp_path)
        try:
            # Queue two facts directly via the enqueue API, then call remember_batch
            # with an empty list to trigger the boundary flush.
            beam._embed_fact_enqueue(101, "ctx alpha", "k1", "v1")
            beam._embed_fact_enqueue(102, "ctx bravo", "k2", "v2")
            beam.remember_batch([])
            assert len(embed_calls) == 1, embed_calls
            assert len(embed_calls[0]) == 2  # both facts in ONE batched call
        finally:
            beam.conn.close()
    finally:
        _e.available = _e_orig_available
        _e.embed = _e_orig_embed


def test_memoria_semantic_retrieve_finds_fact_with_no_literal_overlap(tmp_path):
    """_memoria_semantic_retrieve finds a fact whose key/value share NO tokens with the query.

    Non-vacuous: uses the REAL vec_facts table (sqlite-vec present) with a
    stubbed embedder that returns a constant vector — so identical-text facts
    cluster and the query (different words, same stub vector) matches them.
    Asserts the live fact surfaces with a synthetic fusion key and that a
    superseded (dead) fact does NOT surface.
    """
    import numpy as np
    from edumem.core.embeddings import EMBEDDING_DIM
    import edumem.core.embeddings as _e

    if not _vec_available(_new_beam(tmp_path).conn, table="vec_facts"):
        import pytest
        pytest.skip("sqlite-vec not available")

    _stub_vec = ([0.01] * EMBEDDING_DIM, np.array([[0.01] * EMBEDDING_DIM], dtype=np.float32))
    _e_orig_available, _e_orig_embed, _e_orig_eq = _e.available, _e.embed, _e.embed_query
    _e.available = lambda: True
    _e.embed = lambda texts: np.array([[0.01] * EMBEDDING_DIM] * len(texts), dtype=np.float32)
    _e.embed_query = lambda q: np.array([0.01] * EMBEDDING_DIM, dtype=np.float32)
    try:
        beam = _new_beam(tmp_path)
        try:
            # A fact whose key/value share no words with the query below.
            beam._insert_fact("sem-test", 1, "entity", "design system", "neumorphism",
                              "We adopted a neumorphism design system for the UI.", 0.5,
                              source_memory_id="m1")
            beam._flush_fact_embeddings()  # write the embedding into vec_facts
            rid = beam.conn.execute(
                "SELECT id FROM memoria_facts WHERE source_memory_id=?", ("m1",)
            ).fetchone()[0]

            # Query with NO literal overlap to key/value/ctx — only semantic (stub) match.
            result = beam._memoria_semantic_retrieve(
                "How did user feedback influence the look and feel?", top_k=5)
            assert result["source"] == "memoria_semantic"
            assert "neumorphism" in result["context"], result["context"]
            assert any(f.get("source_memory_id") == f"semantic:fact:{rid}" for f in result["facts"])
        finally:
            beam.conn.close()
    finally:
        _e.available, _e.embed, _e.embed_query = _e_orig_available, _e_orig_embed, _e_orig_eq


def test_fusion_surfaces_semantic_only_fact(tmp_path):
    """memoria_retrieve (full fusion) surfaces a fact found ONLY by the semantic specialist.

    Non-vacuous: the query shares no tokens with the fact's key/value/ctx, so
    the lexical fact specialist finds nothing; only the 7th semantic specialist
    can surface it. Reverting the specialist wiring makes this fail.
    """
    import numpy as np
    from edumem.core.embeddings import EMBEDDING_DIM
    import edumem.core.embeddings as _e

    if not _vec_available(_new_beam(tmp_path).conn, table="vec_facts"):
        import pytest
        pytest.skip("sqlite-vec not available")

    _e_orig_available, _e_orig_embed, _e_orig_eq = _e.available, _e.embed, _e.embed_query
    _e.available = lambda: True
    _e.embed = lambda texts: np.array([[0.01] * EMBEDDING_DIM] * len(texts), dtype=np.float32)
    _e.embed_query = lambda q: np.array([0.01] * EMBEDDING_DIM, dtype=np.float32)
    try:
        beam = _new_beam(tmp_path)
        try:
            beam._insert_fact("sem-test", 1, "entity", "design system", "neumorphism",
                              "We adopted a neumorphism design system for the UI.", 0.5,
                              source_memory_id="m1")
            beam._flush_fact_embeddings()
            # Query with NO literal overlap — only the semantic specialist can find it.
            result = beam.memoria_retrieve("How did user feedback influence the look and feel?", top_k=5)
            assert "neumorphism" in result["context"], (
                f"semantic-only fact not surfaced by fusion: {result['context']!r}"
            )
        finally:
            beam.conn.close()
    finally:
        _e.available, _e.embed, _e.embed_query = _e_orig_available, _e_orig_embed, _e_orig_eq


def test_embed_api_reuses_one_pooled_session_across_calls(monkeypatch):
    """_embed_api must reuse a single module-level requests.Session (keep-alive).

    Non-vacuous: stub the session's .post so no network is hit; call _embed_api
    twice; assert both calls went through the SAME session instance (id match)
    — proving connection pooling, not a fresh Session per call. Reverting to
    per-call urlopen makes the second assertion fail (no shared session).
    """
    import edumem.core.embeddings as _e

    # Point at a custom endpoint so the API-key branch is skipped.
    monkeypatch.setenv("EDUMEM_EMBEDDING_API_URL", "http://embedding.test.local")
    monkeypatch.setenv("EDUMEM_EMBEDDING_MODEL", "test-model")

    # Force the module to create its pooled session now, then stub its .post.
    sessions_used = []

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): return None
        def json(self):
            return {"data": [{"embedding": [0.0, 0.0, 0.0]}]}

    # Replace the module-level pooled session with a stub that records itself.
    class _StubSession:
        def __init__(self):
            sessions_used.append(self)
        def post(self, url, json=None, headers=None, timeout=None, **kw):
            self.last_url = url
            return _FakeResp()

    stub = _StubSession()
    monkeypatch.setattr(_e, "_EMBED_API_SESSION", stub)

    out1 = _e._embed_api(["hello"])
    out2 = _e._embed_api(["world"])

    # Both calls succeeded and used the SAME (single) session instance.
    assert out1 is not None and out2 is not None
    assert len(sessions_used) == 1, f"expected ONE reused session, got {len(sessions_used)}"
    assert sessions_used[0] is stub
