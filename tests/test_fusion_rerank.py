"""Offline test for cross-encoder rerank wiring in _memoria_fused_retrieve.

Stubs the `_fusion_rerank` seam (no network) and asserts the fused facts come
back in rerank-score order, not RRF order. Fails if the wiring is reverted.
"""
from __future__ import annotations

from pathlib import Path

import edumem.core.beam as beam_mod
from edumem.core.beam import BeamMemory


def _build_beam_with_facts(tmp_path: Path) -> BeamMemory:
    """Build a BeamMemory and inject 3 distinct facts directly into memoria_facts.

    The fact specialist (`_memoria_fact_retrieve`) is pure-SQL, no embeddings:
    Pass 4 matches `context_snippet LIKE '%<query word>%'`. We embed a shared
    distinctive word ("gadget") in every fact's snippet so a query containing
    "gadget" surfaces all three, regardless of embedding availability.
    """
    db_path = tmp_path / "fusion_rerank.db"
    beam = BeamMemory(session_id="rerank-test", db_path=db_path)
    conn = beam.conn
    rows = [
        ("Alpha_Cfg", "value_alpha", "A"),
        ("Bravo_Cfg", "value_bravo", "B"),
        ("Charlie_Cfg", "value_charlie", "C"),
    ]
    for key, value, smid in rows:
        conn.execute(
            "INSERT OR REPLACE INTO memoria_facts "
            "(session_id, key, value, fact_type, context_snippet, source_memory_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("rerank-test", key, value, "preference", "gadget detail", smid),
        )
    conn.commit()
    return beam


def test_fusion_rerank_reorders_facts(monkeypatch, tmp_path):
    """Rerank scores must override RRF order; the stub inverts natural order."""
    beam = _build_beam_with_facts(tmp_path)

    # Stub the seam: NO network. Return one score per submitted text, inverting
    # natural insertion order so the last (Charlie) ranks highest, first (Alpha)
    # lowest. `index` refers to position in the fact_texts list.
    def _stub(query, fact_texts):
        n = len(fact_texts)
        # highest score for the LAST text, lowest for the FIRST -> reversed.
        return [{"index": n - 1 - i, "score": float(n - i)} for i in range(n)]

    monkeypatch.setattr(beam_mod, "_fusion_rerank", _stub)

    result = beam.memoria_retrieve("query about gadget", top_k=3)
    context = result["context"]

    # Charlie must render BEFORE Alpha (rerank reversed the order).
    assert "Charlie_Cfg" in context and "Alpha_Cfg" in context, (
        f"expected facts not surfaced: {context!r}"
    )
    assert context.index("Charlie_Cfg") < context.index("Alpha_Cfg"), (
        f"rerank did not reorder: {context!r}"
    )
    assert result.get("reranked") is True


def test_fusion_rerank_offline_fallback(monkeypatch, tmp_path):
    """Endpoint-down (None) must keep RRF order and mark reranked=False."""
    beam = _build_beam_with_facts(tmp_path)
    monkeypatch.setattr(beam_mod, "_fusion_rerank", lambda q, t: None)

    result = beam.memoria_retrieve("query about gadget", top_k=3)
    assert result.get("reranked") is False
    assert result["context"], "context should still render without rerank"
