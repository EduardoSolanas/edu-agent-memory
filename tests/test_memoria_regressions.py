from __future__ import annotations

import os
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

        # RRF fusion is now the only path; verify it retrieves the facts correctly
        assert result["source"] == "rrf_fused"
        # Fusion format: no markup, just key: value pairs
        assert "database version" in result["context"]
        assert "PostgreSQL 16" in result["context"]

        assert len(result["facts"]) >= 1
        # Find the database version fact in results
        db_fact = next((f for f in result["facts"] if f.get("key") == "database version"), None)
        assert db_fact is not None, f"Expected database version fact in {result['facts']}"
        assert db_fact["value"] == "PostgreSQL 16"
        assert db_fact["previous_value"] == "PostgreSQL 15"
        assert [entry["value"] for entry in db_fact["history"]] == [
            "PostgreSQL 14",
            "PostgreSQL 15",
        ]
    finally:
        beam.conn.close()
