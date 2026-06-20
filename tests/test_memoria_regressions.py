from __future__ import annotations

from pathlib import Path

from edumem.core.beam import BeamMemory


def test_memoria_fact_retrieve_keeps_only_current_value_in_context(tmp_path):
    db_path = Path(tmp_path) / "beam.db"
    beam = BeamMemory(session_id="regression-session", db_path=db_path)
    try:
        beam._insert_fact(
            "regression-session",
            1,
            "version",
            "database version",
            "PostgreSQL 14",
            "The database version was PostgreSQL 14.",
            0.5,
            source_memory_id="mem-1",
        )
        beam._insert_fact(
            "regression-session",
            2,
            "version",
            "database version",
            "PostgreSQL 15",
            "The database version was updated to PostgreSQL 15.",
            0.5,
            source_memory_id="mem-2",
        )
        beam._insert_fact(
            "regression-session",
            3,
            "version",
            "database version",
            "PostgreSQL 16",
            "The database version is PostgreSQL 16 now.",
            0.5,
            source_memory_id="mem-3",
        )
        beam.conn.commit()

        result = beam.memoria_retrieve("What is the database version?", ability="IE", top_k=5)

        assert result["source"] == "memoria_facts"
        assert result["context"] == "[Fact version] database version: PostgreSQL 16"
        assert "evolved:" not in result["context"]
        assert "->" not in result["context"]

        assert len(result["facts"]) == 1
        fact = result["facts"][0]
        assert fact["value"] == "PostgreSQL 16"
        assert fact["previous_value"] == "PostgreSQL 15"
        assert [entry["value"] for entry in fact["history"]] == [
            "PostgreSQL 14",
            "PostgreSQL 15",
        ]
        assert [entry["version_id"] for entry in fact["history"]] == [0, 1]
        assert fact["history"][0]["previous_value"] == "PostgreSQL 14"
        assert fact["history"][1]["previous_value"] == "PostgreSQL 15"
        assert fact["history"][1]["updated_msg_idx"] == 2
    finally:
        beam.conn.close()
