import os

# Spec Part E: these tests call _insert_fact directly; keep fact-ingestion offline
# so the semantic-recall write path doesn't attempt an embedding.
os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")

from edumem.core.beam import BeamMemory, init_beam
from edumem.core.query_mode import build_system_prompt


def test_current_fact_context_is_explicitly_not_a_conflict(tmp_path):
    """A CURRENT fact is one update chain, not two competing assertions."""
    db_path = tmp_path / "current-update.db"
    init_beam(str(db_path))
    beam = BeamMemory(db_path=str(db_path), session_id="current-update")
    try:
        beam._insert_fact(
            "current-update", 10, "metric", "response_time_ms", "300ms",
            "Response time was 300ms", 0.9, source_memory_id="msg10",
        )
        beam._insert_fact(
            "current-update", 20, "metric", "response_time_ms", "250ms",
            "Response time is now 250ms", 0.9, source_memory_id="msg20",
        )
        beam.conn.commit()

        context = beam._memoria_fact_retrieve(
            "What is the average response time?", top_k=10, intent="current"
        )["context"]
        prompt = build_system_prompt("What is the average response time?")

        assert "[Fact CURRENT" in context
        assert "250ms" in context and "was: 300ms" in context
        assert "never treat the current and was values as contradictory" in prompt.lower()
    finally:
        beam.conn.close()


def test_explicit_contradiction_question_still_requires_conflict_resolution():
    prompt = build_system_prompt("Is there a contradiction about the response time?")

    assert "CONTRADICTION RESOLUTION" in prompt
    assert "Present BOTH sides" in prompt
