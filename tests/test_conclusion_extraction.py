"""TDD tests for SUM conclusion extraction (Hindsight-style synthesis).

Conclusions are stored as memoria_facts with fact_type='conclusion', flowing
through the same embed -> vec_facts -> semantic-specialist path. These tests
verify the extraction prompt wiring + storage + retrieval, stubbing the LLM.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")

import edumem.core.embeddings as _e
from edumem.core.beam import BeamMemory, _vec_available


def _new_beam(tmp_path: Path) -> BeamMemory:
    return BeamMemory(session_id="concl-test", db_path=tmp_path / "concl.db")


def test_extract_conclusions_returns_synthesized_insights(monkeypatch):
    """extract_conclusions calls the LLM with the conclusion prompt and parses JSON."""
    from edumem.extraction import ExtractionClient
    from edumem.extraction.prompts import CONCLUSION_SYSTEM_PROMPT

    captured = {}

    def _fake_chat(self, messages, temperature=0.0, max_tokens=4096):
        captured["system"] = messages[0]["content"] if messages else ""
        # Return a valid conclusion array
        return ('[{"text": "Security progressed from password hashing (Werkzeug) '
                'through token auth to a Redis-backed account lockout, driven by '
                'repeated brute-force concerns.", "theme": "security", '
                '"source": [40, 120], "confidence": 0.85}]')

    monkeypatch.setattr(ExtractionClient, "chat", _fake_chat)
    client = ExtractionClient()
    msgs = [{"role": "user", "content": "I added password hashing then token auth."}]
    conclusions = client.extract_conclusions(msgs)

    assert CONCLUSION_SYSTEM_PROMPT in captured["system"], "must use the conclusion prompt"
    assert len(conclusions) == 1
    assert "password hashing" in conclusions[0]["text"]
    assert conclusions[0]["theme"] == "security"


def test_conclusion_stored_and_retrievable_as_semantic_fact(monkeypatch, tmp_path):
    """A stored conclusion is findable by semantic recall (the SUM win condition).

    Stubs embed so no network; verifies the conclusion lands in memoria_facts with
    fact_type='conclusion' AND in vec_facts, and the semantic specialist surfaces it
    for a paraphrased narrative query with no literal overlap.
    """
    import numpy as np
    from edumem.core.embeddings import EMBEDDING_DIM

    if not _vec_available(_new_beam(tmp_path).conn, table="vec_facts"):
        import pytest
        pytest.skip("sqlite-vec not available")

    # Stub embedders: constant vector so any text matches any query.
    _orig_av, _orig_em, _orig_eq = _e.available, _e.embed, _e.embed_query
    _e.available = lambda: True
    _e.embed = lambda texts: np.array([[0.01] * EMBEDDING_DIM] * len(texts), dtype=np.float32)
    _e.embed_query = lambda q: np.array([0.01] * EMBEDDING_DIM, dtype=np.float32)
    try:
        beam = _new_beam(tmp_path)
        try:
            # Store a conclusion directly via the same path the extractor will use.
            beam._insert_fact("concl-test", 1, "conclusion", "security",
                              "Security progressed from password hashing through token "
                              "auth to a Redis-backed account lockout mechanism.",
                              "Security work covered hashing, tokens, and lockout.", 0.85,
                              source_memory_id="c1")
            beam._flush_fact_embeddings()

            # Confirm storage shape.
            row = beam.conn.execute(
                "SELECT fact_type, key, value FROM memoria_facts WHERE source_memory_id=?",
                ("c1",)
            ).fetchone()
            assert row and row[0] == "conclusion", f"expected conclusion, got {row}"

            # Semantic specialist must surface it for a narrative query with NO literal overlap.
            res = beam._memoria_semantic_retrieve(
                "Give me a comprehensive summary of how I handled security", top_k=5)
            assert res["source"] == "memoria_semantic"
            assert "lockout" in res["context"] or "token auth" in res["context"], res["context"]
        finally:
            beam.conn.close()
    finally:
        _e.available, _e.embed, _e.embed_query = _orig_av, _orig_em, _orig_eq
