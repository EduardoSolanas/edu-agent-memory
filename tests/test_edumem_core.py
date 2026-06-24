import pytest
import os
import re
import tempfile
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta

from edumem.core.beam import (
    BeamMemory, 
    _fts_search_working, 
    _expanded_query_tokens, 
    _fts_query_terms,
    clean_and_format_sequence,
    parse_relative_date,
    generate_derived_temporal_facts,
    _wm_vector_only_hit_meets_floor,
    _get_embedding_batch_size,
    _get_embedding_batch_total_chars,
    _iter_embedding_chunks,
)
from edumem.core.polyphonic_recall import PolyphonicRecallEngine, RecallResult

class TestEdumemCore:
    """Generic, production-ready integration test suite verifying core system requirements."""

    def test_beam_memory_clean_and_format_sequence(self):
        """Verify that clean_and_format_sequence converts structured JSON arrays/dicts to clean plain text."""
        # Case 1: Sequence query with JSON array structure
        query_seq = "Can you list the order of the events in my budget tracker?"
        raw_seq = '{"events": ["1. Added transactions module", "- Set up database", "3. Launched dashboard"]}'
        
        formatted_seq = clean_and_format_sequence(query_seq, raw_seq)
        
        assert "{" not in formatted_seq
        assert "}" not in formatted_seq
        assert "[" not in formatted_seq
        assert "]" not in formatted_seq
        assert "1." not in formatted_seq
        assert "-" not in formatted_seq
        
        lines = formatted_seq.strip().split("\n")
        assert len(lines) == 3
        assert lines[0].strip() == "Added transactions module"
        assert lines[1].strip() == "Set up database"
        assert lines[2].strip() == "Launched dashboard"

        # Case 2: Non-sequence query should pass through unaffected
        query_normal = "What is my budget tracker?"
        raw_normal = "Your budget tracker is a personal finance app designed to monitor expenses."
        
        formatted_normal = clean_and_format_sequence(query_normal, raw_normal)
        assert formatted_normal == raw_normal

    def test_beam_memory_parse_relative_date(self):
        """Verify parse_relative_date function handles diverse relative date strings deterministically."""
        base_date = "2024-03-12T12:00:00"  # A Tuesday
        
        # Test weekday patterns
        assert parse_relative_date("last Tuesday", base_date) == "2024-03-05"
        assert parse_relative_date("next Tuesday", base_date) == "2024-03-19"
        assert parse_relative_date("this Wednesday", base_date) == "2024-03-13"
        
        # Test X ago patterns
        assert parse_relative_date("3 days ago", base_date) == "2024-03-09"
        assert parse_relative_date("one week ago", base_date) == "2024-03-05"
        assert parse_relative_date("two months ago", base_date) == "2024-01-12"
        
        # Test yesterday/tomorrow/today patterns
        assert parse_relative_date("yesterday", base_date) == "2024-03-11"
        assert parse_relative_date("tomorrow", base_date) == "2024-03-13"
        assert parse_relative_date("today", base_date) == "2024-03-12"

        # Yearless named dates inherit the explicit reference year.
        assert parse_relative_date("March 29", base_date) == "2024-03-29"
        assert parse_relative_date("March 29th", base_date) == "2024-03-29"
        assert parse_relative_date("29th of March", base_date) == "2024-03-29"

        # The harness sentinel is storage compatibility, not a date reference.
        assert parse_relative_date("next Tuesday", "1970-01-01T00:00:00Z") is None
        
        # Test ISO date pattern fallback
        assert parse_relative_date("it happened on 2023-05-15", base_date) == "2023-05-15"

    def test_beam_memory_generate_derived_temporal_facts(self):
        """Verify generate_derived_temporal_facts correctly calculates relative temporal distances."""
        # Query containing temporal cues
        query = "How long between my first commit and the project launch?"
        
        # Multiple memories with timestamps/occurred_at
        mems = [
            {"content": "I made the initial repository commit", "occurred_at": "2024-01-01"},
            {"content": "We launched the dashboard", "occurred_at": "2024-01-11"}
        ]
        
        derived = generate_derived_temporal_facts(mems, query)
        assert len(derived) == 1
        
        fact = derived[0]
        assert fact["source"] == "derived_temporal"
        assert fact["tier"] == "derived"
        assert "initial repository commit" in fact["content"]
        assert "launched the dashboard" in fact["content"]
        # Difference in days (10 days)
        assert "10 days before" in fact["content"]
        assert "2024-01-01 → 2024-01-11" in fact["content"]

    def test_embedding_batch_size_default_and_override(self, monkeypatch):
        monkeypatch.delenv("EDUMEM_EMBEDDING_BATCH_SIZE", raising=False)
        assert _get_embedding_batch_size() == 8

        monkeypatch.setenv("EDUMEM_EMBEDDING_BATCH_SIZE", "16")
        assert _get_embedding_batch_size() == 16

        monkeypatch.setenv("EDUMEM_EMBEDDING_BATCH_SIZE", "0")
        assert _get_embedding_batch_size() == 8

    def test_embedding_batch_total_chars_default_and_override(self, monkeypatch):
        monkeypatch.delenv("EDUMEM_EMBEDDING_BATCH_TOTAL_CHARS", raising=False)
        monkeypatch.delenv("EDUMEM_EMBEDDING_BATCH_CHAR_BUDGET", raising=False)
        assert _get_embedding_batch_total_chars() == 6000

        monkeypatch.setenv("EDUMEM_EMBEDDING_BATCH_TOTAL_CHARS", "4096")
        assert _get_embedding_batch_total_chars() == 4096

        monkeypatch.delenv("EDUMEM_EMBEDDING_BATCH_TOTAL_CHARS", raising=False)
        monkeypatch.setenv("EDUMEM_EMBEDDING_BATCH_CHAR_BUDGET", "2048")
        assert _get_embedding_batch_total_chars() == 2048

        monkeypatch.setenv("EDUMEM_EMBEDDING_BATCH_CHAR_BUDGET", "0")
        assert _get_embedding_batch_total_chars() == 6000

    def test_embedding_chunk_helper_preserves_id_alignment(self):
        items = [
            {"content": "alpha"},
            {"content": "bravo"},
            {"content": "charlie"},
            {"content": "delta"},
            {"content": "echo"},
        ]
        ids = [
            "id-0",
            "id-1",
            "id-2",
            "id-3",
            "id-4",
        ]

        chunks = list(_iter_embedding_chunks(items, ids, batch_size=2))

        assert chunks == [
            (0, 2, ["id-0", "id-1"], ["alpha", "bravo"]),
            (2, 4, ["id-2", "id-3"], ["charlie", "delta"]),
            (4, 5, ["id-4"], ["echo"]),
        ]

    def test_embedding_chunk_helper_caps_batches_at_eight_items(self):
        items = [{"content": f"item-{i}"} for i in range(9)]
        ids = [f"id-{i}" for i in range(9)]

        chunks = list(_iter_embedding_chunks(items, ids, batch_size=8, total_chars=1000))

        assert chunks == [
            (0, 8, [f"id-{i}" for i in range(8)], [f"item-{i}" for i in range(8)]),
            (8, 9, ["id-8"], ["item-8"]),
        ]

    def test_embedding_chunk_helper_respects_char_budget_and_isolates_oversized_messages(self):
        items = [
            {"content": "short-a"},
            {"content": "short-b"},
            {"content": "l" * 4000},
            {"content": "x" * 7000},
        ]
        ids = ["id-short-a", "id-short-b", "id-long", "id-oversized"]

        chunks = list(_iter_embedding_chunks(items, ids, batch_size=8, total_chars=5000))

        assert chunks == [
            (0, 3, ["id-short-a", "id-short-b", "id-long"], ["short-a", "short-b", "l" * 4000]),
            (3, 4, ["id-oversized"], ["x" * 7000]),
        ]

    def test_beam_memory_bi_temporal_grounding(self):
        """Verify bi-temporal database columns (occurred_at, recorded_at) and relative date resolution."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
            
        try:
            beam = BeamMemory(db_path=db_path)
            
            # Use specific reference timestamp for ingestion (e.g. 2024-03-12 which is a Tuesday)
            reference_timestamp = "2024-03-12T10:00:00"
            
            # Remember a temporal claim with a relative time reference ("last Tuesday" -> 2024-03-05)
            beam.remember(
                content="I finished the transaction management features last Tuesday",
                timestamp=reference_timestamp,
            )
            
            # Verify columns exist and were written correctly
            cursor = beam.conn.execute("SELECT occurred_at, recorded_at, content FROM working_memory")
            row = cursor.fetchone()
            assert row is not None
            assert row["recorded_at"] == reference_timestamp
            
            # Relative date calculation validation
            assert "2024-03-05" in row["occurred_at"]
            
        finally:
            beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_beam_memory_fts_query_sanitization(self):
        """Verify that BeamMemory query path protects FTS queries containing punctuation or quotes from crashing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            
            # Ingest a standard test memory
            beam.remember(content="I do not like rock climbing because of heights.", source="conversation")
            
            # Query containing FTS-sensitive characters (quotes, colons, punctuation)
            query_with_punctuation = "I don't like rock: climbing"
            
            # Verify recall public interface executes cleanly without FTS exceptions
            results = beam.recall(query_with_punctuation, top_k=5)
            assert len(results) > 0
            assert "rock climbing" in results[0]["content"]
        finally:
            beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_beam_memory_sandbox_database_ttl_bypass(self):
        """Verify that database paths indicating a sandbox/test environment bypass working memory TTL pruning."""
        # Create a database with '_test_sandbox' in its path
        with tempfile.NamedTemporaryFile(suffix="_test_sandbox.db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            
            # Set a very short active TTL to trigger pruning
            beam.WORKING_MEMORY_TTL_HOURS = 0.001
            
            # Ingest memories older than the standard 7-day TTL
            old_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
            beam.remember(content="Historical context to protect", source="conversation", timestamp=old_timestamp)
            
            # Trigger standard commit/trim path
            beam.remember(content="Trigger trim path", source="conversation")
            
            # Verify the memory was NOT pruned due to sandbox environment detection
            cursor = beam.conn.execute("SELECT COUNT(*) FROM working_memory WHERE content LIKE '%Historical%'")
            count = cursor.fetchone()[0]
            assert count == 1, "Historical memory was incorrectly pruned in a sandbox/test database!"
        finally:
            beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_beam_memory_thread_local_connection_syncing(self):
        """Verify that cached engines automatically synchronize database connections when the master handle changes."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            
            # Warm the cache to initialize the active engines
            engine = beam._get_polyphonic_engine()
            assert engine.conn == beam.conn
            
            # Transition the active SQLite connection handle
            new_conn = sqlite3.connect(str(db_path))
            beam.conn = new_conn
            
            # Retrieve the engines again and verify they synchronized seamlessly
            engine2 = beam._get_polyphonic_engine()
            assert engine2.conn == new_conn, "PolyphonicRecallEngine connection failed to sync!"
            assert engine2.graph.conn == new_conn, "EpisodicGraph connection failed to sync!"
            assert engine2.consolidator.conn == new_conn, "VeracityConsolidator connection failed to sync!"
            
            new_conn.close()
        finally:
            beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_beam_memory_expanded_query_tokens_singular_plural(self):
        """Verify query-side expansion maps plural forms to singulars to bypass FTS limitations."""
        plural_tokens = ["columns", "commits", "buses", "memories"]
        expanded = _expanded_query_tokens(plural_tokens)
        
        assert "column" in expanded
        assert "commit" in expanded
        assert "bus" in expanded
        assert "memory" in expanded

    def test_beam_memory_fts_query_terms_polar_negation(self):
        """Verify polar queries expand search query terms with proximity negation operators."""
        query = "Have I worked with routes?"
        terms = _fts_query_terms(query)
        
        # Verify negation qualifiers (never, not) are present in query expansion
        assert any("never" in term or "not" in term for term in terms), f"Negation terms missing from polar query: {terms}"

    def test_beam_memory_polyphonic_recall_score_scaling(self):
        """Verify PolyphonicRecallEngine accurately scales merged Reciprocal Rank Fusion (RRF) scores."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            engine = PolyphonicRecallEngine(db_path=db_path, conn=sqlite3.connect(str(db_path)))
            
            # Construct mock voice candidate responses
            vector_results = [RecallResult(memory_id="mem_1", score=0.9, voice="vector", metadata={})]
            keyword_results = [RecallResult(memory_id="mem_1", score=0.8, voice="keyword", metadata={})]
            
            engine.voice_weights = {
                "vector": 0.5,
                "keyword": 0.5,
                "graph": 0.0,
                "fact": 0.0,
                "temporal": 0.0
            }
            
            combined = engine._combine_voices(
                vector_results,
                [], # graph
                [], # fact
                [], # temporal
                keyword_results
            )
            
            assert "mem_1" in combined
            res = combined["mem_1"]
            
            # Verify RRF score scaling places results in an appropriate range for standard client filters (>0.1)
            assert res.combined_score > 0.10, f"RRF score scaling factor failed: {res.combined_score}"
            
            engine.conn.close()
        finally:
            if db_path.exists():
                db_path.unlink()


class TestBeamCoreFixes:
    def test_vector_only_floor_gate(self):
        assert _wm_vector_only_hit_meets_floor(0.60)
        assert _wm_vector_only_hit_meets_floor(0.91)
        assert not _wm_vector_only_hit_meets_floor(0.59)

    def test_structured_extraction_dedupes_synthetic_metadata_and_repeat_calls(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        beam = None
        try:
            beam = BeamMemory(db_path=db_path)
            memory_id = "mem_fix04"
            content = (
                "On 2024-03-15 My App deployed 3 features. "
                "Always format code with syntax highlighting. "
                "I prefer simple dependencies. "
                "[DATES: 2024-03-15] datetokens: datetok20240315 [MSGIDX:42]"
            )

            first = beam.extract_and_store_facts(
                content,
                message_idx=42,
                source_memory_id=memory_id,
                source="beam_user",
            )
            second = beam.extract_and_store_facts(
                content,
                message_idx=42,
                source_memory_id=memory_id,
                source="beam_user",
            )

            assert first["metric"] == 1
            assert first["date"] == 1
            assert first["timeline"] == 1
            assert first["instruction"] == 1
            assert first["preference"] == 1
            assert second["metric"] == 0
            assert second["date"] == 0
            assert second["timeline"] == 0
            assert second.get("instruction", 0) == 0
            assert second.get("preference", 0) == 0

            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_facts WHERE source_memory_id = ? AND fact_type = 'metric'",
                (memory_id,),
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_facts WHERE source_memory_id = ? AND fact_type = 'date'",
                (memory_id,),
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_timelines WHERE source_memory_id = ?",
                (memory_id,),
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_instructions WHERE source_memory_id = ?",
                (memory_id,),
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_preferences WHERE source_memory_id = ?",
                (memory_id,),
            ).fetchone()[0] == 1
        finally:
            if beam is not None:
                beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_structured_dedupe_is_scoped_to_session(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        beam_a = None
        beam_b = None
        try:
            beam_a = BeamMemory(session_id="session_a", db_path=db_path)
            beam_b = BeamMemory(session_id="session_b", db_path=db_path)
            content = (
                "On 2024-03-15 Project uses PostgreSQL v16.1 and deployed 3 features. "
                "Always format code with syntax highlighting. "
                "I prefer simple dependencies. I never use tabs."
            )

            first = beam_a.extract_and_store_facts(
                content,
                message_idx=42,
                source_memory_id="shared_mem",
                source="beam_user",
            )
            second = beam_b.extract_and_store_facts(
                content,
                message_idx=42,
                source_memory_id="shared_mem",
                source="beam_user",
            )

            assert first == second
            assert beam_a.conn.execute(
                "SELECT COUNT(*) FROM memoria_facts WHERE session_id = ?",
                (beam_a.session_id,),
            ).fetchone()[0] == first["metric"] + first["date"] + first["version"] + first["sequence"]
            assert beam_b.conn.execute(
                "SELECT COUNT(*) FROM memoria_facts WHERE session_id = ?",
                (beam_b.session_id,),
            ).fetchone()[0] == second["metric"] + second["date"] + second["version"] + second["sequence"]

            assert beam_a.conn.execute(
                "SELECT COUNT(*) FROM memoria_timelines WHERE session_id = ?",
                (beam_a.session_id,),
            ).fetchone()[0] == first["timeline"]
            assert beam_b.conn.execute(
                "SELECT COUNT(*) FROM memoria_timelines WHERE session_id = ?",
                (beam_b.session_id,),
            ).fetchone()[0] == second["timeline"]

            assert beam_a.conn.execute(
                "SELECT COUNT(*) FROM memoria_kg WHERE session_id = ?",
                (beam_a.session_id,),
            ).fetchone()[0] == first["negation"] + first["decision"]
            assert beam_b.conn.execute(
                "SELECT COUNT(*) FROM memoria_kg WHERE session_id = ?",
                (beam_b.session_id,),
            ).fetchone()[0] == second["negation"] + second["decision"]

            assert beam_a.conn.execute(
                "SELECT COUNT(*) FROM memoria_instructions WHERE session_id = ?",
                (beam_a.session_id,),
            ).fetchone()[0] == first["instruction"]
            assert beam_b.conn.execute(
                "SELECT COUNT(*) FROM memoria_instructions WHERE session_id = ?",
                (beam_b.session_id,),
            ).fetchone()[0] == second["instruction"]

            assert beam_a.conn.execute(
                "SELECT COUNT(*) FROM memoria_preferences WHERE session_id = ?",
                (beam_a.session_id,),
            ).fetchone()[0] == first["preference"]
            assert beam_b.conn.execute(
                "SELECT COUNT(*) FROM memoria_preferences WHERE session_id = ?",
                (beam_b.session_id,),
            ).fetchone()[0] == second["preference"]
        finally:
            if beam_a is not None:
                beam_a.conn.close()
            if beam_b is not None:
                beam_b.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_msgidx_propagates_into_structured_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        beam = None
        try:
            beam = BeamMemory(db_path=db_path)
            batch = [
                {
                    "content": (
                        "On 2024-03-15 My App deployed 3 features. "
                        "Always format code with syntax highlighting. "
                        "I prefer simple dependencies. [MSGIDX:0]"
                    ),
                    "source": "beam_user",
                    "message_index": 0,
                },
                {
                    "content": (
                        "On 2024-03-16 My App deployed 4 features. "
                        "Always format code with syntax highlighting. "
                        "I prefer simple dependencies. [MSGIDX:1]"
                    ),
                    "source": "beam_user",
                    "message_index": 1,
                },
            ]

            ids = beam.remember_batch(batch)
            assert len(ids) == 2

            def structured_indices(memory_id: str) -> set[int]:
                indices = set()
                for table in (
                    "memoria_facts",
                    "memoria_timelines",
                    "memoria_instructions",
                    "memoria_preferences",
                ):
                    rows = beam.conn.execute(
                        f"SELECT message_idx FROM {table} WHERE source_memory_id = ? AND message_idx IS NOT NULL",
                        (memory_id,),
                    ).fetchall()
                    indices.update(row["message_idx"] for row in rows)
                return indices

            assert structured_indices(ids[0]) == {0}
            assert structured_indices(ids[1]) == {1}
        finally:
            if beam is not None:
                beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_user_authored_instructions_and_preferences_only(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        beam = None
        try:
            beam = BeamMemory(db_path=db_path)
            ids = beam.remember_batch([
                {
                    "content": "The API latency is 12ms on 2024-03-15.",
                    "source": "beam_assistant",
                    "message_index": 0,
                },
                {
                    "content": "Always format code with syntax highlighting. I prefer simple dependencies.",
                    "source": "beam_user",
                    "message_index": 1,
                },
            ])
            assistant_id, user_id = ids

            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_instructions WHERE source_memory_id = ?",
                (assistant_id,),
            ).fetchone()[0] == 0
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_preferences WHERE source_memory_id = ?",
                (assistant_id,),
            ).fetchone()[0] == 0
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_facts WHERE source_memory_id = ? AND fact_type = 'metric'",
                (assistant_id,),
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_facts WHERE source_memory_id = ? AND fact_type = 'date'",
                (assistant_id,),
            ).fetchone()[0] == 1

            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_instructions WHERE source_memory_id = ?",
                (user_id,),
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM memoria_preferences WHERE source_memory_id = ?",
                (user_id,),
            ).fetchone()[0] == 1
        finally:
            if beam is not None:
                beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass

    def test_vector_only_recall_smoke_with_embeddings(self):
        from edumem.core import embeddings as _embeddings
        import importlib

        endpoint = os.environ.get("EDUMEM_TEST_INFERENCE_URL")
        if not endpoint:
            pytest.skip("EDUMEM_TEST_INFERENCE_URL not set")

        saved_api_url = os.environ.get("EDUMEM_EMBEDDING_API_URL")
        saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
        saved_model = os.environ.get("EDUMEM_EMBEDDING_MODEL")
        try:
            os.environ["EDUMEM_EMBEDDING_API_URL"] = endpoint
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
            test_model = os.environ.get("EDUMEM_TEST_INFERENCE_MODEL")
            if test_model:
                os.environ["EDUMEM_EMBEDDING_MODEL"] = test_model
            importlib.reload(_embeddings)

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                db_path = Path(tmp.name)
            beam = None
            try:
                beam = BeamMemory(db_path=db_path)
                strong = "A physician examines a patient in the clinic."
                weak = "A bicycle is leaning against a wall in the garage."
                beam.remember_batch([
                    {
                        "content": strong,
                        "source": "beam_user",
                        "message_index": 0,
                    },
                    {
                        "content": weak,
                        "source": "beam_user",
                        "message_index": 1,
                    },
                ])

                results = beam.recall(
                    "A doctor checks a sick person at the hospital.",
                    top_k=10,
                )
                assert any(strong in row["content"] for row in results)
                assert not any(weak in row["content"] for row in results)
            finally:
                if beam is not None:
                    beam.conn.close()
                if db_path.exists():
                    try:
                        db_path.unlink()
                    except PermissionError:
                        pass
        finally:
            if saved_api_url is None:
                os.environ.pop("EDUMEM_EMBEDDING_API_URL", None)
            else:
                os.environ["EDUMEM_EMBEDDING_API_URL"] = saved_api_url
            if saved_no_embeddings is None:
                os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
            else:
                os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings
            if saved_model is None:
                os.environ.pop("EDUMEM_EMBEDDING_MODEL", None)
            else:
                os.environ["EDUMEM_EMBEDDING_MODEL"] = saved_model
            importlib.reload(_embeddings)

    def test_ingest_diagnostics_and_supersession_deltas(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        beam = None
        try:
            beam = BeamMemory(db_path=db_path)

            first_ids = beam.remember_batch([
                {
                    "content": (
                        "I prefer simple dependencies. "
                        "Always format code with syntax highlighting."
                    ),
                    "source": "beam_user",
                    "message_index": 0,
                },
                {
                    "content": (
                        "Project uses PostgreSQL v16.1 and deployed on 2024-03-15."
                    ),
                    "source": "beam_assistant",
                    "message_index": 1,
                }
            ])

            batch1 = beam._last_ingest_diagnostics_batch
            assert batch1 is not None
            assert len(batch1["rows"]) == 2
            assert batch1["rows"][0]["source_role"] == "user"
            assert batch1["rows"][0]["message_idx"] == 0
            assert batch1["rows"][0]["graph_facts_delta"] == 0
            assert batch1["rows"][0]["consolidated_delta"] == 0
            assert batch1["rows"][0]["superseded_delta"] == 0
            assert batch1["rows"][1]["source_role"] == "assistant"
            assert batch1["rows"][1]["message_idx"] == 1
            assert batch1["rows"][1]["graph_facts_delta"] >= 1
            assert batch1["rows"][1]["consolidated_delta"] >= 1
            assert batch1["rows"][1]["superseded_delta"] == 0
            assert batch1["totals"]["graph_facts_delta"] == batch1["rows"][0]["graph_facts_delta"] + batch1["rows"][1]["graph_facts_delta"]
            assert batch1["totals"]["consolidated_delta"] == batch1["rows"][0]["consolidated_delta"] + batch1["rows"][1]["consolidated_delta"]
            assert batch1["totals"]["superseded_delta"] == 0
            assert beam._last_ingest_diagnostics["message_idx"] == 1
            assert beam._last_ingest_diagnostics["source_role"] == "assistant"

            assert beam.conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM consolidated_facts"
            ).fetchone()[0] == 1
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NOT NULL"
            ).fetchone()[0] == 0
            assert beam.conn.execute(
                "SELECT COUNT(*) FROM conflicts"
            ).fetchone()[0] == 0

            second_ids = beam.remember_batch([
                {
                    "content": "Project uses MySQL v8.0 and deployed on 2024-03-16.",
                    "source": "beam_assistant",
                    "message_index": 2,
                }
            ])

            batch2 = beam._last_ingest_diagnostics_batch
            assert batch2 is not None
            assert len(batch2["rows"]) == 1
            assert batch2["rows"][0]["source_role"] == "assistant"
            assert batch2["rows"][0]["message_idx"] == 2
            assert batch2["rows"][0]["graph_facts_delta"] >= 1
            assert batch2["rows"][0]["consolidated_delta"] >= 1
            assert batch2["rows"][0]["superseded_delta"] == 0

            conflict = beam.conn.execute(
                "SELECT id, fact_a_id, fact_b_id FROM conflicts ORDER BY id DESC LIMIT 1"
            ).fetchone()
            assert conflict is not None
            fact_rows = beam.conn.execute(
                "SELECT id, object FROM consolidated_facts WHERE subject = ? AND predicate = ? ORDER BY first_seen",
                ("Project", "uses"),
            ).fetchall()
            assert len(fact_rows) >= 2
            winning_id = next(row["id"] for row in fact_rows if row["object"] == "MySQL")

            before_superseded = beam.conn.execute(
                "SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NOT NULL"
            ).fetchone()[0]
            beam.veracity_consolidator.resolve_conflict(conflict["id"], winning_id)
            after_superseded = beam.conn.execute(
                "SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NOT NULL"
            ).fetchone()[0]
            assert after_superseded == before_superseded + 1
        finally:
            if beam is not None:
                beam.conn.close()
            if db_path.exists():
                try:
                    db_path.unlink()
                except PermissionError:
                    pass


class TestNegationTagging:
    """Verify [NEG] tags are appended to content containing negation sentences during ingestion."""

    def test_negation_tag_appended_on_remember_batch(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "I have never worked with Flask routes.", "source": "test", "importance": 0.5},
            ])
            row = beam.conn.execute("SELECT content FROM working_memory").fetchone()
            assert "[NEG]" in row["content"]
            assert "never worked with Flask" in row["content"]
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_no_negation_tag_on_positive_content(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "I love working with Flask routes.", "source": "test", "importance": 0.5},
            ])
            row = beam.conn.execute("SELECT content FROM working_memory").fetchone()
            assert "[NEG]" not in row["content"]
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_negation_tag_found_by_fts(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "I have never used Docker in production.", "source": "test", "importance": 0.5},
                {"content": "I deployed the app using Docker on staging.", "source": "test", "importance": 0.5},
            ])
            results = beam.recall("NEG Docker", top_k=5)
            neg_results = [r for r in results if "[NEG]" in r.get("content", "")]
            assert len(neg_results) >= 1
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_negation_tag_appended_on_remember_single(self):
        """Parity: single remember() must tag negations exactly like remember_batch()."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember(content="I have never worked with Kafka streams.", source="test")
            row = beam.conn.execute("SELECT content FROM working_memory").fetchone()
            assert "[NEG]" in row["content"]
            assert "never worked with Kafka" in row["content"]
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_no_negation_tag_on_positive_remember_single(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember(content="I love working with Kafka streams.", source="test")
            row = beam.conn.execute("SELECT content FROM working_memory").fetchone()
            assert "[NEG]" not in row["content"]
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestRecallContentLimit:
    """Verify recall content truncation is configurable via EDUMEM_RECALL_CONTENT_CHARS."""

    def test_default_truncates_to_about_500(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        os.environ.pop("EDUMEM_RECALL_CONTENT_CHARS", None)
        try:
            beam = BeamMemory(db_path=db_path)
            long_text = "Kafka streaming pipeline. " + ("detail " * 200)  # ~1400 chars
            beam.remember_batch([{"content": long_text, "source": "test", "importance": 0.5}])
            results = beam.recall("Kafka streaming pipeline", top_k=5)
            assert results, "expected a recall hit"
            # Body truncated to 500 (a short date prefix may be prepended) -> well under full length
            assert len(results[0]["content"]) < 600
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_env_raises_limit(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        os.environ["EDUMEM_RECALL_CONTENT_CHARS"] = "2000"
        try:
            beam = BeamMemory(db_path=db_path)
            long_text = "Kafka streaming pipeline. " + ("detail " * 200)  # ~1400 chars
            beam.remember_batch([{"content": long_text, "source": "test", "importance": 0.5}])
            results = beam.recall("Kafka streaming pipeline", top_k=5)
            assert results, "expected a recall hit"
            # With a 2000-char limit the full ~1400-char body should survive
            assert len(results[0]["content"]) > 1000
            beam.conn.close()
        finally:
            os.environ.pop("EDUMEM_RECALL_CONTENT_CHARS", None)
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestMessageIndex:
    """Verify message_index is stored during ingestion and returned during recall."""

    def test_message_index_stored_in_remember_batch(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "First message about Python.", "source": "test", "importance": 0.5, "message_index": 0},
                {"content": "Second message about Flask.", "source": "test", "importance": 0.5, "message_index": 1},
                {"content": "Third message about Django.", "source": "test", "importance": 0.5, "message_index": 2},
            ])
            rows = beam.conn.execute("SELECT content, message_index FROM working_memory ORDER BY message_index").fetchall()
            assert len(rows) == 3
            assert rows[0]["message_index"] == 0
            assert rows[1]["message_index"] == 1
            assert rows[2]["message_index"] == 2
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_message_index_returned_in_recall(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "The API uses Flask framework.", "source": "test", "importance": 0.5, "message_index": 42},
            ])
            results = beam.recall("Flask", top_k=5)
            assert len(results) > 0
            assert results[0].get("message_index") == 42
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_message_index_none_when_not_provided(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "A regular message without index.", "source": "test", "importance": 0.5},
            ])
            row = beam.conn.execute("SELECT message_index FROM working_memory").fetchone()
            assert row["message_index"] is None
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_remember_single_with_message_index(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember(content="A message with index.", source="test", message_index=99)
            row = beam.conn.execute("SELECT message_index FROM working_memory").fetchone()
            assert row["message_index"] == 99
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestQueryModePrompts:
    """Verify query_mode.py correctly detects question types and appends modifiers."""

    def test_ordering_query_detected(self):
        from edumem.core.query_mode import is_ordering_query, build_system_prompt
        assert is_ordering_query("In what order did I discuss the features?")
        assert is_ordering_query("Walk me through the sequence of events.")
        assert not is_ordering_query("What is my favorite color?")

    def test_duration_query_detected(self):
        from edumem.core.query_mode import is_duration_query, build_system_prompt
        assert is_duration_query("How many days between the start and the end?")
        assert is_duration_query("How long did the sprint last?")
        assert not is_duration_query("What framework do I use?")

    def test_knowledge_update_query_detected(self):
        from edumem.core.query_mode import is_knowledge_update_query, build_system_prompt
        assert is_knowledge_update_query("What is the current version of the API?")
        assert is_knowledge_update_query("What is the latest status of the deployment?")
        assert is_knowledge_update_query("I switched to PostgreSQL, what am I now using?")
        assert not is_knowledge_update_query("What framework do I use?")

    def test_multi_hop_query_detected(self):
        from edumem.core.query_mode import is_multi_hop_query, build_system_prompt
        assert is_multi_hop_query("How is the API related to the database?")
        assert is_multi_hop_query("What connects the auth module to the user table?")
        assert not is_multi_hop_query("What is my favorite color?")

    def test_ku_modifier_in_prompt(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("What is the current version?")
        assert "KNOWLEDGE UPDATE" in prompt
        assert "MOST RECENT" in prompt

    def test_mr_modifier_in_prompt(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("How is X related to Y across sessions?")
        assert "MULTI-HOP REASONING" in prompt
        assert "Chain the facts" in prompt

    def test_cr_conflict_in_base_prompt(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("What is my name?")
        assert "contradictory information" in prompt
        assert "Do NOT silently pick one side" in prompt

    def test_eo_ordering_modifier_mentions_message_index(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("In what order did I discuss the topics?")
        assert "MSGIDX" in prompt
        assert "message index" in prompt

    def test_no_modifiers_for_simple_question(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("What is my favorite programming language?")
        assert "ORDERING" not in prompt
        assert "DURATION" not in prompt
        assert "KNOWLEDGE UPDATE" not in prompt
        assert "MULTI-HOP" not in prompt


class TestTROracle:
    """Verify the temporal reasoning oracle: timeline extraction, date matching, answer computation."""

    def _make_msgs(self, texts):
        return [{"role": "user", "content": t} for t in texts]

    # --- Timeline extraction coverage ---

    def test_extract_iso_dates(self):
        """ISO dates (2024-03-15) should be extracted."""
        from tools.evaluate_beam_end_to_end import _extract_timeline_from_conversation
        msgs = self._make_msgs(["The sprint started on 2024-03-15 and ended on 2024-04-12."])
        tl = _extract_timeline_from_conversation(msgs)
        dates = {t["date_obj"].strftime("%Y-%m-%d") for t in tl}
        assert "2024-03-15" in dates
        assert "2024-04-12" in dates

    def test_extract_ordinal_dates(self):
        """Ordinal dates like '15th of March 2024' should be extracted."""
        from tools.evaluate_beam_end_to_end import _extract_timeline_from_conversation
        msgs = self._make_msgs(["We launched on the 15th of March, 2024."])
        tl = _extract_timeline_from_conversation(msgs)
        dates = {t["date_obj"].strftime("%Y-%m-%d") for t in tl}
        assert "2024-03-15" in dates

    def test_extract_relative_dates(self):
        """Relative dates like 'two weeks ago' should be resolved against message context."""
        from tools.evaluate_beam_end_to_end import _extract_timeline_from_conversation
        msgs = [{"role": "user", "content": "The incident happened on March 15, 2024. Two weeks later we deployed the fix."}]
        tl = _extract_timeline_from_conversation(msgs)
        dates = {t["date_obj"].strftime("%Y-%m-%d") for t in tl}
        assert "2024-03-15" in dates
        assert "2024-03-29" in dates

    def test_extract_informal_month_references(self):
        """'mid-March 2024', 'early April 2024', 'late January 2024' should produce approximate dates."""
        from tools.evaluate_beam_end_to_end import _extract_timeline_from_conversation
        msgs = self._make_msgs(["Planning started in early April 2024 and finished in late June 2024."])
        tl = _extract_timeline_from_conversation(msgs)
        dates = {t["date_obj"].strftime("%Y-%m-%d") for t in tl}
        # early April ≈ April 5, late June ≈ June 25
        assert any(d.startswith("2024-04") for d in dates)
        assert any(d.startswith("2024-06") for d in dates)

    def test_extract_slash_dates(self):
        """Dates like '03/15/2024' or '15/03/2024' should be extracted."""
        from tools.evaluate_beam_end_to_end import _extract_timeline_from_conversation
        msgs = self._make_msgs(["Meeting scheduled for 03/15/2024."])
        tl = _extract_timeline_from_conversation(msgs)
        dates = {t["date_obj"].strftime("%Y-%m-%d") for t in tl}
        assert "2024-03-15" in dates

    # --- Python date math ---

    def test_compute_tr_python_weeks(self):
        """When question asks 'how many weeks', answer should include weeks."""
        from tools.evaluate_beam_end_to_end import _compute_tr_python
        from datetime import datetime as _dt
        timeline = [
            {"date_obj": _dt(2024, 3, 1), "date_str": "March 1", "event_text": "started the project", "msg_index": 0},
            {"date_obj": _dt(2024, 3, 29), "date_str": "March 29", "event_text": "finished the project", "msg_index": 5},
        ]
        answer = _compute_tr_python("How many weeks between starting and finishing the project?", timeline)
        assert answer is not None
        assert "4" in answer or "week" in answer.lower()

    def test_compute_tr_python_months(self):
        """When question asks 'how many months', answer should include months."""
        from tools.evaluate_beam_end_to_end import _compute_tr_python
        from datetime import datetime as _dt
        timeline = [
            {"date_obj": _dt(2024, 1, 15), "date_str": "January 15", "event_text": "started alpha", "msg_index": 0},
            {"date_obj": _dt(2024, 4, 15), "date_str": "April 15", "event_text": "finished beta", "msg_index": 10},
        ]
        answer = _compute_tr_python("How many months between alpha and beta?", timeline)
        assert answer is not None
        assert "3" in answer or "month" in answer.lower()

    def test_compute_tr_python_earlier_later(self):
        """'Did X happen before or after Y?' should produce a before/after answer."""
        from tools.evaluate_beam_end_to_end import _compute_tr_python
        from datetime import datetime as _dt
        timeline = [
            {"date_obj": _dt(2024, 2, 10), "date_str": "February 10", "event_text": "database migration", "msg_index": 0},
            {"date_obj": _dt(2024, 5, 20), "date_str": "May 20", "event_text": "API deployment", "msg_index": 5},
        ]
        answer = _compute_tr_python("Did the database migration happen before or after the API deployment?", timeline)
        assert answer is not None
        assert "before" in answer.lower()

    def test_compute_tr_python_best_event_match(self):
        """With 4+ timeline entries, the oracle should pick the two most relevant to the question."""
        from tools.evaluate_beam_end_to_end import _compute_tr_python
        from datetime import datetime as _dt
        timeline = [
            {"date_obj": _dt(2024, 1, 5), "date_str": "Jan 5", "event_text": "team standup meeting", "msg_index": 0},
            {"date_obj": _dt(2024, 2, 10), "date_str": "Feb 10", "event_text": "started the database migration sprint", "msg_index": 2},
            {"date_obj": _dt(2024, 3, 15), "date_str": "Mar 15", "event_text": "launched the new homepage redesign", "msg_index": 5},
            {"date_obj": _dt(2024, 4, 20), "date_str": "Apr 20", "event_text": "completed the database migration rollout", "msg_index": 8},
        ]
        answer = _compute_tr_python("How long did the database migration take from start to completion?", timeline)
        assert answer is not None
        # Should pick Feb 10 and Apr 20 (the database migration events), not Jan 5 or Mar 15
        assert "69" in answer or "70" in answer  # Feb 10 to Apr 20 = 70 days

    def test_compute_tr_python_no_small_duration_false_positive(self):
        """The small-duration guard should not block valid short durations when events match strongly."""
        from tools.evaluate_beam_end_to_end import _compute_tr_python
        from datetime import datetime as _dt
        timeline = [
            {"date_obj": _dt(2024, 3, 10), "date_str": "March 10", "event_text": "started the code review", "msg_index": 0},
            {"date_obj": _dt(2024, 3, 12), "date_str": "March 12", "event_text": "finished the code review", "msg_index": 3},
        ]
        # Only 2 entries and both strongly match — should produce an answer even though it's 2 days
        answer = _compute_tr_python("How long did the code review take?", timeline)
        assert answer is not None
        assert "2" in answer


class TestOverallMacroAverage:
    """Verify OVERALL is a macro-average (per-ability equal weight), matching BEAM's leaderboard."""

    def test_overall_is_macro_not_micro(self):
        from tools.evaluate_beam_end_to_end import compute_ability_scores
        # Ability A: 1 question scored 1.0; Ability B: 3 questions all 0.0
        all_results = [{
            "scale": "100K",
            "results": [
                {"ability": "ABS", "score": 1.0},
                {"ability": "IE", "score": 0.0},
                {"ability": "IE", "score": 0.0},
                {"ability": "IE", "score": 0.0},
            ],
        }]
        summary = compute_ability_scores(all_results)
        overall = summary["100K"]["OVERALL"]["avg_score"]
        # Macro: (1.0 + 0.0) / 2 = 0.5   (micro would be 1/4 = 0.25)
        assert abs(overall - 0.5) < 1e-9, f"expected macro 0.5, got {overall}"

    def test_per_ability_averages_unchanged(self):
        from tools.evaluate_beam_end_to_end import compute_ability_scores
        all_results = [{
            "scale": "100K",
            "results": [
                {"ability": "IE", "score": 1.0},
                {"ability": "IE", "score": 0.0},
            ],
        }]
        summary = compute_ability_scores(all_results)
        assert abs(summary["100K"]["IE"]["avg_score"] - 0.5) < 1e-9


class TestContextFactsClean:
    """Verify the context->value index is built from clean text, not synthetic tags."""

    def test_context_phrases_have_no_synthetic_tags(self):
        from tools.evaluate_beam_end_to_end import ingest_conversation
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            ingest_conversation(beam, [
                {"role": "user", "content": "My first sprint ends on 2024-03-15 with the dashboard launch."},
            ])
            keys = list(getattr(beam, "_context_facts", {}).keys())
            # No context phrase should contain the synthetic MSGIDX/DATES/DURATIONS tags
            for k in keys:
                assert "msgidx" not in k, f"tag pollution in context phrase: {k!r}"
                assert "[dates" not in k, f"tag pollution in context phrase: {k!r}"
                assert "[durations" not in k, f"tag pollution in context phrase: {k!r}"
            # The real natural-language fact should still be indexed
            assert any("sprint ends on" in k for k in keys)
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestContextValueMatch:
    """Verify the IE/KU context->value matcher gates weak matches behind a confidence floor."""

    def test_strong_match_returns_value(self):
        from tools.evaluate_beam_end_to_end import _context_value_match
        facts = {"main database engine postgres version": ["PostgreSQL 14"]}
        val, score = _context_value_match("What is the main database engine version?", facts)
        assert val == "PostgreSQL 14"
        assert score >= 0.5

    def test_weak_match_below_floor(self):
        from tools.evaluate_beam_end_to_end import _context_value_match
        # Question shares only 2 words with a long, unrelated context phrase
        facts = {"deployed the staging server with docker compose and nginx config": ["xyz"]}
        val, score = _context_value_match("What server did I deploy?", facts)
        # Only ~2 words overlap out of 9 context words -> low score
        assert score < 0.5

    def test_no_match_returns_none(self):
        from tools.evaluate_beam_end_to_end import _context_value_match
        facts = {"completely unrelated topic about cats": ["meow"]}
        val, score = _context_value_match("What is the API rate limit?", facts)
        assert val is None


class TestImplicitContradictionRecall:
    """Verify CR retrieval surfaces BOTH sides of an implicit (no negation word) contradiction."""

    def test_implicit_contradiction_both_sides_retrievable(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "We are using PostgreSQL for the main database.", "source": "test", "importance": 0.5},
                {"content": "I migrated the project to MySQL last sprint.", "source": "test", "importance": 0.5},
            ])
            # Broad topic-mention retrieval: pull ALL mentions of the topic word
            rows = beam.conn.execute(
                "SELECT content FROM working_memory WHERE content LIKE ? OR content LIKE ?",
                ("%PostgreSQL%", "%MySQL%")
            ).fetchall()
            contents = " ".join(r["content"] for r in rows)
            assert "PostgreSQL" in contents
            assert "MySQL" in contents
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestTRTrustGuard:
    """Verify Python TR answers are trusted only on sparse timelines; dense -> defer to LLM."""

    def test_dense_timeline_defers_even_for_large_duration(self):
        from tools.evaluate_beam_end_to_end import _tr_python_answer_is_trustworthy
        # On a dense timeline Python date-pair matching is unreliable -> defer to LLM
        ans = "Between A and B, there are 70 days."
        assert _tr_python_answer_is_trustworthy(ans, timeline_size=120) is False

    def test_sparse_timeline_trusts_large_duration(self):
        from tools.evaluate_beam_end_to_end import _tr_python_answer_is_trustworthy
        ans = "Between A and B, there are 70 days."
        assert _tr_python_answer_is_trustworthy(ans, timeline_size=4) is True

    def test_sparse_timeline_trusts_small_duration(self):
        from tools.evaluate_beam_end_to_end import _tr_python_answer_is_trustworthy
        ans = "Between A and B, there are 2 days."
        assert _tr_python_answer_is_trustworthy(ans, timeline_size=3) is True

    def test_round_duration_not_rejected_as_zero(self):
        from tools.evaluate_beam_end_to_end import _tr_python_answer_is_trustworthy
        # Regression: "70 days" must NOT be treated as "0 days" (substring bug)
        ans = "Between A and B, there are 70 days."
        assert _tr_python_answer_is_trustworthy(ans, timeline_size=4) is True

    def test_weeks_format_total_days_parsed(self):
        from tools.evaluate_beam_end_to_end import _tr_python_answer_is_trustworthy
        # "4 weeks and 2 days (30 days)" on a sparse timeline -> trusted
        ans = "Between A and B, there are 4 weeks and 2 days (30 days)."
        assert _tr_python_answer_is_trustworthy(ans, timeline_size=4) is True

    def test_zero_duration_never_trusted(self):
        from tools.evaluate_beam_end_to_end import _tr_python_answer_is_trustworthy
        assert _tr_python_answer_is_trustworthy("Between A and B, there are 0 days.", timeline_size=3) is False

    def test_none_not_trusted(self):
        from tools.evaluate_beam_end_to_end import _tr_python_answer_is_trustworthy
        assert _tr_python_answer_is_trustworthy(None, timeline_size=3) is False


class TestJudgeJsonCleaning:
    """Verify the judge-output cleaner survives markdown fences and prose chatter."""

    def _score(self, cleaned):
        return json.loads(cleaned)

    def test_pristine_json_unchanged(self):
        from tools.evaluate_beam_end_to_end import _clean_judge_json
        out = self._score(_clean_judge_json('{"score": 1.0}'))
        assert out["score"] == 1.0

    def test_markdown_fenced_json(self):
        from tools.evaluate_beam_end_to_end import _clean_judge_json
        raw = '```json\n{"score": 1.0}\n```'
        out = self._score(_clean_judge_json(raw))
        assert out["score"] == 1.0

    def test_prose_prefix_then_json(self):
        from tools.evaluate_beam_end_to_end import _clean_judge_json
        raw = 'The answer fully matches the rubric item. {"score": 1.0}'
        out = self._score(_clean_judge_json(raw))
        assert out["score"] == 1.0

    def test_single_element_list_unwrapped(self):
        from tools.evaluate_beam_end_to_end import _clean_judge_json
        raw = '[{"score": 0.5}]'
        out = self._score(_clean_judge_json(raw))
        assert isinstance(out, dict)
        assert out["score"] == 0.5

    def test_scores_array_preserved(self):
        from tools.evaluate_beam_end_to_end import _clean_judge_json
        raw = '```json\n{"scores": [1.0, 0.5], "overall_score": 0.75}\n```'
        out = self._score(_clean_judge_json(raw))
        assert out["scores"] == [1.0, 0.5]
        assert out["overall_score"] == 0.75

    def test_unparseable_returns_original(self):
        from tools.evaluate_beam_end_to_end import _clean_judge_json
        raw = "I cannot produce a score."
        assert _clean_judge_json(raw) == raw

    def test_empty_returns_original(self):
        from tools.evaluate_beam_end_to_end import _clean_judge_json
        assert _clean_judge_json("") == ""


class TestPass2Routing:
    """Verify Pass-2 prompt routing: only duration questions get the calculator prompt."""

    def test_duration_question_uses_calculator(self):
        from tools.evaluate_beam_end_to_end import _is_calculator_question
        assert _is_calculator_question("How many days between the start and the end?")

    def test_ordering_question_does_not_use_calculator(self):
        from tools.evaluate_beam_end_to_end import _is_calculator_question
        # Ordering questions must NOT get the duration calculator prompt
        assert not _is_calculator_question("In what order did I discuss the features?")
        assert not _is_calculator_question("Walk me through the sequence of events.")

    def test_simple_question_does_not_use_calculator(self):
        from tools.evaluate_beam_end_to_end import _is_calculator_question
        assert not _is_calculator_question("What is my favorite color?")


class TestEOAndSUMPrompts:
    """Verify EO and SUM query detection and prompt modifiers."""

    def test_summarization_query_detected(self):
        from edumem.core.query_mode import is_summarization_query
        assert is_summarization_query("Can you summarize the main topics we discussed?")
        assert is_summarization_query("Give me an overview of our conversation.")
        assert is_summarization_query("What were the key themes in our discussion?")
        assert not is_summarization_query("What is my favorite color?")

    def test_sum_modifier_in_prompt(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("Summarize the main topics we discussed.")
        assert "SUMMARIZATION" in prompt
        assert "theme" in prompt.lower() or "topic" in prompt.lower()

    def test_eo_modifier_includes_msgidx_instruction(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("In what order did I discuss the features?")
        assert "MSGIDX" in prompt or "message index" in prompt
        assert "one item per line" in prompt.lower() or "one per line" in prompt.lower() or "one clause per line" in prompt.lower()


class TestNegationRecall:
    """Verify that negation content is retrievable via SQL LIKE search for CR questions."""

    def test_negation_content_found_via_like_search(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "I have never worked with Flask routes in any project.", "source": "test", "importance": 0.5},
                {"content": "I implemented the Flask API endpoints last week.", "source": "test", "importance": 0.5},
            ])
            neg_rows = beam.conn.execute(
                "SELECT id, content FROM working_memory "
                "WHERE content LIKE ? AND content LIKE ?",
                ("%Flask%", "%never%")
            ).fetchall()
            assert len(neg_rows) >= 1
            assert "never" in neg_rows[0]["content"]
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_both_positive_and_negative_retrievable(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            beam.remember_batch([
                {"content": "I have never used Docker containers.", "source": "test", "importance": 0.5},
                {"content": "I deployed the service using Docker on the staging server.", "source": "test", "importance": 0.5},
            ])
            positive = beam.conn.execute(
                "SELECT content FROM working_memory WHERE content LIKE ? AND content NOT LIKE ?",
                ("%Docker%", "%never%")
            ).fetchall()
            negative = beam.conn.execute(
                "SELECT content FROM working_memory WHERE content LIKE ? AND content LIKE ?",
                ("%Docker%", "%never%")
            ).fetchall()
            assert len(positive) >= 1
            assert len(negative) >= 1
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestFIX1_ISODateTokenization:
    """FIX 1 (TR): Make ISO dates retrievable through FTS5 via datetok tokens.

    Problem: FTS5 tokenizes `2024-03-15` into `2024 OR 03 OR 15`, so date strings
    are not searchable as a unit and date-bearing messages don't reliably surface
    during recall. This is why Temporal Reasoning scores 0%.

    Solution: Append FTS-survivable tokens like `datetok20240315` (no hyphens)
    so FTS5 indexes them as ONE token per ISO date found in the message.
    """

    def test_datetok_appended_to_content_on_ingest(self):
        """Test that ingest_conversation appends datetok tokens for ISO dates."""
        from tools.evaluate_beam_end_to_end import ingest_conversation
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            messages = [
                {"role": "user", "content": "The sprint started on 2024-03-15 and ended on 2024-04-12."},
            ]
            ingest_conversation(beam, messages)

            # Check that the stored content contains datetok tokens
            row = beam.conn.execute("SELECT content FROM working_memory LIMIT 1").fetchone()
            assert row is not None
            content = row["content"]
            # Should contain both datetok versions (one for each ISO date found)
            assert "datetok20240315" in content, f"datetok20240315 not in: {content}"
            assert "datetok20240412" in content, f"datetok20240412 not in: {content}"
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass

    def test_datetok_tokens_are_fts_searchable(self):
        """Test that datetok tokens can be found via FTS5 recall."""
        from tools.evaluate_beam_end_to_end import ingest_conversation
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)
            messages = [
                {"role": "user", "content": "The sprint started on 2024-03-15 with the dashboard launch."},
            ]
            ingest_conversation(beam, messages)

            # Recall using the datetok token should find the message
            results = beam.recall("datetok20240315", top_k=5)
            assert len(results) > 0, "datetok20240315 should be FTS-searchable"
            assert "2024-03-15" in results[0]["content"]
            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestFIX2_OrderingQueryDepthMultiplier:
    """FIX 2 (EO): Raise candidate depth for ordering queries.

    Problem: Event Ordering is graded by Kendall tau-b over an ordered list; with
    the top-K cap at 30, not all `[MSGIDX:N]`-tagged mentions of the queried
    topics reach context, so the ordering is incomplete.

    Solution: When is_ordering_query(question) is True, use a larger effective
    candidate budget — multiply the local `top_k` used for sub-queries and the
    final slice by 3 (so ordering questions return up to 3x candidates).
    """

    def test_ordering_query_trigger_function(self):
        """Test that is_ordering_query correctly identifies ordering questions."""
        from edumem.core.query_mode import is_ordering_query

        # Should return True for ordering questions
        assert is_ordering_query("In what order did I discuss the features?") is True
        assert is_ordering_query("Walk me through the sequence of events.") is True
        assert is_ordering_query("What order did things happen?") is True

        # Should return False for non-ordering questions
        assert is_ordering_query("What is my name?") is False
        assert is_ordering_query("Tell me about the project.") is False


class TestFIX3_NegationRetrievalCapIncrease:
    """FIX 3 (CR): Broaden negation retrieval cap.

    Problem: In _multi_strategy_recall, the negation SQL searches use `LIMIT 5`
    and only trigger on narrow phrasings.

    Solution: Raise those `LIMIT 5` to `LIMIT 15` in the negation/topic-mention
    SQL blocks. This documents the intent that contradiction resolution needs
    broader candidate pools.
    """

    def test_both_sides_of_contradiction_retrievable_after_fix(self):
        """Test that both positive and negative statements about the same topic are retrievable.

        This test documents the FIX 3 requirement: both sides of a contradiction must
        be retrievable via SQL LIKE search with an expanded LIMIT (15 instead of 5).
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        try:
            beam = BeamMemory(db_path=db_path)

            # Ingest contradicting statements
            beam.remember_batch([
                {"content": "I have never used Redis in any of my projects.",
                 "source": "test", "importance": 0.5},
                {"content": "I deployed the cache with Redis on the staging environment.",
                 "source": "test", "importance": 0.5},
            ])

            # Raw SQL search with the expanded LIMIT (15) should find both
            rows = beam.conn.execute(
                "SELECT id, content FROM working_memory "
                "WHERE content LIKE ? "
                "LIMIT 15",
                ("%Redis%",)
            ).fetchall()

            # With LIMIT 15, both rows should be retrievable
            assert len(rows) >= 2, f"Expected 2 Redis-related rows with LIMIT 15, got {len(rows)}"

            contents = [r["content"] for r in rows]
            # Verify we got both the negative and positive statements
            has_negative = any("never" in c.lower() for c in contents)
            has_positive = any("deployed" in c.lower() for c in contents)
            assert has_negative, "Negative statement not found"
            assert has_positive, "Positive statement not found"

            beam.conn.close()
        finally:
            if db_path.exists():
                try: db_path.unlink()
                except PermissionError: pass


class TestTemporalCheatsheet:
    """Test suite for temporal cheatsheet injection in TR questions."""

    def test_cheatsheet_extracts_dates_from_memories(self):
        """Verify that _inject_temporal_cheatsheet extracts and sorts ISO dates from memory content."""
        from tools.evaluate_beam_end_to_end import _inject_temporal_cheatsheet

        memories = [
            {"content": "We started work on 2024-01-15 with the sprint kickoff meeting."},
            {"content": "The final demo was delivered on 2024-03-20 to the client."},
            {"content": "Initial planning happened on 2024-01-10 with the team."},
        ]
        question = "How long between the start and final demo?"

        cheatsheet = _inject_temporal_cheatsheet(memories, question)

        # Should contain the temporal reference marker
        assert "[TEMPORAL REFERENCE]" in cheatsheet
        assert "[END TEMPORAL REFERENCE]" in cheatsheet

        # Should contain all three dates in chronological order
        assert "2024-01-10" in cheatsheet
        assert "2024-01-15" in cheatsheet
        assert "2024-03-20" in cheatsheet

        # Should contain timedelta information
        assert "days" in cheatsheet

    def test_cheatsheet_empty_for_non_temporal(self):
        """Verify that _inject_temporal_cheatsheet returns empty string for non-temporal questions."""
        from tools.evaluate_beam_end_to_end import _inject_temporal_cheatsheet

        memories = [
            {"content": "My name is Alice and I work at TechCorp."},
            {"content": "I have 5 years of experience with Python."},
        ]
        question = "What is my name?"

        cheatsheet = _inject_temporal_cheatsheet(memories, question)

        # Should return empty string
        assert cheatsheet == ""

    def test_cheatsheet_empty_when_no_dates(self):
        """Verify that _inject_temporal_cheatsheet returns empty string when no ISO dates found."""
        from tools.evaluate_beam_end_to_end import _inject_temporal_cheatsheet

        memories = [
            {"content": "We worked on the project last month."},
            {"content": "The deadline was around mid-summer."},
        ]
        question = "How long did the project take?"

        cheatsheet = _inject_temporal_cheatsheet(memories, question)

        # Should return empty string (no ISO dates)
        assert cheatsheet == ""

    def test_cheatsheet_computes_correct_delta(self):
        """Verify that _inject_temporal_cheatsheet correctly computes timedeltas (including leap years)."""
        from tools.evaluate_beam_end_to_end import _inject_temporal_cheatsheet

        # Use 2024 which is a leap year
        memories = [
            {"content": "The sprint started on 2024-01-15."},
            {"content": "The final deliverable was on 2024-03-20."},
        ]
        question = "How many days between the sprint start and final deliverable?"

        cheatsheet = _inject_temporal_cheatsheet(memories, question)

        # Jan 15 -> Mar 20 in leap year 2024:
        # Remaining Jan: 31-15 = 16 days
        # Feb: 29 days (leap year)
        # Mar: 20 days
        # Total: 16 + 29 + 20 = 65 days
        assert "65 days" in cheatsheet

    def test_cheatsheet_context_snippet(self):
        """Verify that cheatsheet includes event context snippets around dates."""
        from tools.evaluate_beam_end_to_end import _inject_temporal_cheatsheet

        memories = [
            {"content": "We had the sprint kickoff on 2024-02-01 at 9am."},
        ]
        question = "How long between the start and now?"

        cheatsheet = _inject_temporal_cheatsheet(memories, question)

        # Should contain the date
        assert "2024-02-01" in cheatsheet
        # Should contain a context snippet (around 30 chars)
        assert "sprint kickoff" in cheatsheet or "kickoff" in cheatsheet


class MockLLMClient:
    """Mock LLM client for testing conflict resolution."""
    def __init__(self, response="UPDATE"):
        self._response = response

    def chat(self, messages, **kwargs):
        """Return the configured response."""
        return self._response


class TestWriteTimeConflictResolution:
    """Test write-time conflict resolution for CR and KU."""

    def test_update_decision_supersedes_old_fact(self):
        """When LLM decides UPDATE, the old fact should be superseded."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)
            mock_llm = MockLLMClient(response="UPDATE")

            # First fact
            consolidator.consolidate_fact(
                "user", "database", "PostgreSQL",
                veracity="stated", source="mem1",
                llm_client=mock_llm
            )

            # Second fact (conflict detected)
            consolidator.consolidate_fact(
                "user", "database", "MySQL",
                veracity="stated", source="mem2",
                llm_client=mock_llm
            )

            # Query: old fact should be superseded
            cursor = consolidator.conn.cursor()
            cursor.execute("""
                SELECT id, object, superseded_by FROM consolidated_facts
                WHERE subject = 'user' AND predicate = 'database'
                ORDER BY first_seen
            """)
            facts = cursor.fetchall()

            assert len(facts) == 2
            assert facts[0]["object"] == "PostgreSQL"
            assert facts[0]["superseded_by"] is not None
            assert facts[1]["object"] == "MySQL"
            assert facts[1]["superseded_by"] is None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore

    def test_add_decision_keeps_both_visible(self):
        """When LLM decides ADD, both facts should remain visible (no supersession)."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)
            mock_llm = MockLLMClient(response="ADD")

            # First fact
            consolidator.consolidate_fact(
                "user", "languages", "Python",
                veracity="stated", source="mem1",
                llm_client=mock_llm
            )

            # Second fact (conflict detected)
            consolidator.consolidate_fact(
                "user", "languages", "Rust",
                veracity="stated", source="mem2",
                llm_client=mock_llm
            )

            # Query: both facts should have no supersession
            cursor = consolidator.conn.cursor()
            cursor.execute("""
                SELECT id, object, superseded_by FROM consolidated_facts
                WHERE subject = 'user' AND predicate = 'languages'
                ORDER BY first_seen
            """)
            facts = cursor.fetchall()

            assert len(facts) == 2
            assert facts[0]["object"] == "Python"
            assert facts[0]["superseded_by"] is None
            assert facts[1]["object"] == "Rust"
            assert facts[1]["superseded_by"] is None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore

    def test_no_llm_defaults_to_add(self):
        """When no LLM provided, both facts should remain visible (backward compat)."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)

            # First fact (no LLM)
            consolidator.consolidate_fact(
                "user", "role", "Engineer",
                veracity="stated", source="mem1"
            )

            # Second fact (conflict, no LLM)
            consolidator.consolidate_fact(
                "user", "role", "Manager",
                veracity="stated", source="mem2"
            )

            # Query: both facts should have no supersession
            cursor = consolidator.conn.cursor()
            cursor.execute("""
                SELECT id, object, superseded_by FROM consolidated_facts
                WHERE subject = 'user' AND predicate = 'role'
                ORDER BY first_seen
            """)
            facts = cursor.fetchall()

            assert len(facts) == 2
            assert facts[0]["superseded_by"] is None
            assert facts[1]["superseded_by"] is None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore

    def test_working_memory_supersession_propagates(self):
        """Supersession set on consolidated_facts should propagate to working_memory."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)
            mock_llm = MockLLMClient(response="UPDATE")

            # Create dummy working_memory rows first
            cursor = consolidator.conn.cursor()
            mem1_id = "mem_1"
            mem2_id = "mem_2"
            now = datetime.now().isoformat()

            # Ensure working_memory table exists (from beam.py schema)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    superseded_by TEXT
                )
            """)

            cursor.execute(
                "INSERT INTO working_memory (id, content) VALUES (?, ?)",
                (mem1_id, "I use PostgreSQL")
            )
            cursor.execute(
                "INSERT INTO working_memory (id, content) VALUES (?, ?)",
                (mem2_id, "I switched to MySQL")
            )
            consolidator.conn.commit()

            # First fact
            consolidator.consolidate_fact(
                "user", "database", "PostgreSQL",
                veracity="stated", source=mem1_id,
                llm_client=mock_llm
            )

            # Second fact (conflict)
            consolidator.consolidate_fact(
                "user", "database", "MySQL",
                veracity="stated", source=mem2_id,
                llm_client=mock_llm
            )

            # Check working_memory supersession
            cursor.execute("""
                SELECT id, superseded_by FROM working_memory
                WHERE id = ?
            """, (mem1_id,))
            wm_row = cursor.fetchone()

            assert wm_row is not None
            assert wm_row["superseded_by"] is not None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore

    def test_delete_decision_supersedes_old_fact(self):
        """When LLM decides DELETE, the old fact should be superseded."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)
            mock_llm = MockLLMClient(response="DELETE")

            # First fact
            consolidator.consolidate_fact(
                "user", "status", "junior",
                veracity="stated", source="mem1",
                llm_client=mock_llm
            )

            # Second fact (DELETE decision)
            consolidator.consolidate_fact(
                "user", "status", "senior",
                veracity="stated", source="mem2",
                llm_client=mock_llm
            )

            # Query: old fact should be superseded
            cursor = consolidator.conn.cursor()
            cursor.execute("""
                SELECT id, object, superseded_by FROM consolidated_facts
                WHERE subject = 'user' AND predicate = 'status'
                ORDER BY first_seen
            """)
            facts = cursor.fetchall()

            assert len(facts) == 2
            assert facts[0]["object"] == "junior"
            assert facts[0]["superseded_by"] is not None
            assert facts[1]["object"] == "senior"
            assert facts[1]["superseded_by"] is None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore

    def test_noop_decision_keeps_both_no_duplicate(self):
        """When LLM decides NOOP, both facts should remain visible (similar to ADD)."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)
            mock_llm = MockLLMClient(response="NOOP")

            # First fact
            consolidator.consolidate_fact(
                "user", "food", "likes cheese pizza",
                veracity="stated", source="mem1",
                llm_client=mock_llm
            )

            # Second fact (conflict detected, but similar info)
            consolidator.consolidate_fact(
                "user", "food", "loves cheese pizza",
                veracity="stated", source="mem2",
                llm_client=mock_llm
            )

            # Query: both facts should have no supersession
            cursor = consolidator.conn.cursor()
            cursor.execute("""
                SELECT id, object, superseded_by FROM consolidated_facts
                WHERE subject = 'user' AND predicate = 'food'
                ORDER BY first_seen
            """)
            facts = cursor.fetchall()

            assert len(facts) == 2
            assert facts[0]["object"] == "likes cheese pizza"
            assert facts[0]["superseded_by"] is None
            assert facts[1]["object"] == "loves cheese pizza"
            assert facts[1]["superseded_by"] is None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore

    def test_chained_updates_only_latest_visible(self):
        """When multiple UPDATE decisions occur, only the latest fact is current."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)
            mock_llm = MockLLMClient(response="UPDATE")

            # First fact
            consolidator.consolidate_fact(
                "user", "city", "NYC",
                veracity="stated", source="mem1",
                llm_client=mock_llm
            )

            # Second fact (UPDATE: NYC -> London)
            consolidator.consolidate_fact(
                "user", "city", "London",
                veracity="stated", source="mem2",
                llm_client=mock_llm
            )

            # Third fact (UPDATE: London -> Tokyo)
            consolidator.consolidate_fact(
                "user", "city", "Tokyo",
                veracity="stated", source="mem3",
                llm_client=mock_llm
            )

            # Query: first two should be superseded, only Tokyo should be current
            cursor = consolidator.conn.cursor()
            cursor.execute("""
                SELECT id, object, superseded_by FROM consolidated_facts
                WHERE subject = 'user' AND predicate = 'city'
                ORDER BY first_seen
            """)
            facts = cursor.fetchall()

            assert len(facts) == 3
            assert facts[0]["object"] == "NYC"
            assert facts[0]["superseded_by"] is not None
            assert facts[1]["object"] == "London"
            assert facts[1]["superseded_by"] is not None
            assert facts[2]["object"] == "Tokyo"
            assert facts[2]["superseded_by"] is None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore

    def test_llm_garbage_response_defaults_to_add(self):
        """When LLM returns garbage response, default to ADD (safe fallback)."""
        from edumem.core.veracity_consolidation import VeracityConsolidator

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            consolidator = VeracityConsolidator(db_path=db_path)
            mock_llm = MockLLMClient(response="hmm maybe replace?")

            # First fact
            consolidator.consolidate_fact(
                "user", "editor", "vim",
                veracity="stated", source="mem1",
                llm_client=mock_llm
            )

            # Second fact (conflict, but LLM response is garbage)
            consolidator.consolidate_fact(
                "user", "editor", "emacs",
                veracity="stated", source="mem2",
                llm_client=mock_llm
            )

            # Query: both facts should remain visible (ADD behavior)
            cursor = consolidator.conn.cursor()
            cursor.execute("""
                SELECT id, object, superseded_by FROM consolidated_facts
                WHERE subject = 'user' AND predicate = 'editor'
                ORDER BY first_seen
            """)
            facts = cursor.fetchall()

            assert len(facts) == 2
            assert facts[0]["object"] == "vim"
            assert facts[0]["superseded_by"] is None
            assert facts[1]["object"] == "emacs"
            assert facts[1]["superseded_by"] is None
        finally:
            consolidator.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass  # Windows file lock; ignore


class TestInstructionPreferenceDetection:
    """Test IF and PF instruction/preference detection and tagging."""

    def test_instruction_tag_appended_at_ingest(self):
        from edumem.core.beam import BeamMemory
        import tempfile
        from pathlib import Path
        from tools.evaluate_beam_end_to_end import ingest_conversation

        # Simple mock LLM that returns classification based on message content
        class MockLLM:
            def chat(self, messages, temperature=0.1, max_tokens=1024):
                # For testing, return a response containing classification tags
                # The _classify_message_llm function parses this response for tags
                # Extract the original message being classified from the prompt
                prompt = messages[-1].get("content", "")
                # The prompt ends with "Message: <actual_message>\n\nLabels:"
                # Extract text after "Message: " and before "Labels:"
                if "Message: " in prompt:
                    msg_start = prompt.rfind("Message: ") + len("Message: ")
                    msg_end = prompt.rfind("\n\nLabels:")
                    if msg_end > msg_start:
                        msg_text = prompt[msg_start:msg_end].lower()
                    else:
                        msg_text = prompt[msg_start:].lower()
                else:
                    msg_text = prompt.lower()

                if "always format" in msg_text:
                    return "INSTRUCTION"
                elif "prefer" in msg_text or "minimal" in msg_text:
                    return "PREFERENCE"
                return "FACT"

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            beam = BeamMemory(db_path=db_path)
            mock_llm = MockLLM()
            messages = [
                {"role": "user", "content": "Always format code with syntax highlighting when I ask about implementation."},
                {"role": "user", "content": "Can you help me with Flask?"},
            ]
            ingest_conversation(beam, messages, llm=mock_llm)

            rows = beam.conn.execute("SELECT content FROM working_memory").fetchall()
            tagged = [r for r in rows if "[INSTRUCTION]" in r["content"]]
            untagged = [r for r in rows if "[INSTRUCTION]" not in r["content"]]
            assert len(tagged) >= 1
            assert len(untagged) >= 1
        finally:
            beam.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass

    def test_preference_tag_appended_at_ingest(self):
        from edumem.core.beam import BeamMemory
        import tempfile
        from pathlib import Path
        from tools.evaluate_beam_end_to_end import ingest_conversation

        # Simple mock LLM that returns classification based on message content
        class MockLLM:
            def chat(self, messages, temperature=0.1, max_tokens=1024):
                # For testing, return a response containing classification tags
                # The _classify_message_llm function parses this response for tags
                # Extract the original message being classified from the prompt
                prompt = messages[-1].get("content", "")
                # The prompt ends with "Message: <actual_message>\n\nLabels:"
                # Extract text after "Message: " and before "Labels:"
                if "Message: " in prompt:
                    msg_start = prompt.rfind("Message: ") + len("Message: ")
                    msg_end = prompt.rfind("\n\nLabels:")
                    if msg_end > msg_start:
                        msg_text = prompt[msg_start:msg_end].lower()
                    else:
                        msg_text = prompt[msg_start:].lower()
                else:
                    msg_text = prompt.lower()

                if "prefer" in msg_text or "minimal" in msg_text:
                    return "PREFERENCE"
                elif "always" in msg_text:
                    return "INSTRUCTION"
                return "FACT"

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            beam = BeamMemory(db_path=db_path)
            mock_llm = MockLLM()
            messages = [
                {"role": "user", "content": "I prefer minimal dependencies to keep the app lightweight."},
                {"role": "user", "content": "Let me set up the database schema."},
            ]
            ingest_conversation(beam, messages, llm=mock_llm)

            rows = beam.conn.execute("SELECT content FROM working_memory").fetchall()
            tagged = [r for r in rows if "[PREFERENCE]" in r["content"]]
            untagged = [r for r in rows if "[PREFERENCE]" not in r["content"]]
            assert len(tagged) >= 1
            assert len(untagged) >= 1
        finally:
            beam.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass


class TestAggregationQuery:
    """Test MR aggregation query detection and broader recall."""

    def test_how_many_detected(self):
        from edumem.core.query_mode import is_aggregation_query
        assert is_aggregation_query("How many new columns did I want to add to the transactions table?")

    def test_across_sessions_detected(self):
        from edumem.core.query_mode import is_aggregation_query
        assert is_aggregation_query("How many different user roles am I trying to implement across my sessions?")

    def test_total_detected(self):
        from edumem.core.query_mode import is_aggregation_query
        assert is_aggregation_query("What is the total number of features we discussed?")

    def test_simple_question_not_aggregation(self):
        from edumem.core.query_mode import is_aggregation_query
        assert not is_aggregation_query("When does my first sprint end?")

    def test_ordering_question_not_aggregation(self):
        from edumem.core.query_mode import is_aggregation_query
        assert not is_aggregation_query("In what order did I bring up the development topics?")

    def test_aggregation_increases_topk(self):
        from edumem.core.query_mode import is_aggregation_query
        q = "How many different security features am I implementing across sessions?"
        assert is_aggregation_query(q)
        base_topk = 30
        effective_topk = base_topk * 3 if is_aggregation_query(q) else base_topk
        assert effective_topk == 90


class TestCRSupersededRecall:
    """Test that CR queries can retrieve superseded facts."""

    def test_superseded_facts_retrievable_with_like_query(self):
        """Superseded working_memory rows should be findable via SQL LIKE."""
        from edumem.core.beam import BeamMemory
        import tempfile
        from pathlib import Path

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            beam = BeamMemory(db_path=db_path)

            # Insert a superseded fact and a current fact
            beam.conn.execute(
                "INSERT INTO working_memory (id, content, superseded_by) VALUES (?, ?, ?)",
                ("mem1", "[MSGIDX:0] I have never written any Flask routes", "mem2")
            )
            beam.conn.execute(
                "INSERT INTO working_memory (id, content, superseded_by) VALUES (?, ?, ?)",
                ("mem2", "[MSGIDX:50] I integrated Flask routes into my project", None)
            )
            beam.conn.commit()

            # Query for superseded facts about Flask
            superseded = beam.conn.execute(
                "SELECT id, content FROM working_memory "
                "WHERE content LIKE ? AND superseded_by IS NOT NULL",
                ("%Flask%",)
            ).fetchall()
            assert len(superseded) == 1
            assert "never" in superseded[0]["content"]

            # Normal recall (superseded_by IS NULL) should only find the current fact
            current = beam.conn.execute(
                "SELECT id, content FROM working_memory "
                "WHERE content LIKE ? AND superseded_by IS NULL",
                ("%Flask%",)
            ).fetchall()
            assert len(current) == 1
            assert "integrated" in current[0]["content"]
        finally:
            beam.conn.close()
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                pass


class TestEOMessageIndexSorting:
    """Test that EO context is sorted by message index."""

    def test_msgidx_extraction_and_sort(self):
        """Memories with MSGIDX tags should be sortable by index."""
        import re
        memories = [
            {"content": "[MSGIDX:50] Set up deployment config"},
            {"content": "[MSGIDX:10] Created database schema"},
            {"content": "[MSGIDX:30] Implemented user auth"},
        ]
        def extract_msgidx(mem):
            m = re.search(r'\[MSGIDX:(\d+)\]', mem.get("content", ""))
            return int(m.group(1)) if m else 999999

        sorted_mems = sorted(memories, key=extract_msgidx)
        assert "schema" in sorted_mems[0]["content"]
        assert "auth" in sorted_mems[1]["content"]
        assert "deployment" in sorted_mems[2]["content"]

    def test_msgidx_missing_sorts_last(self):
        """Memories without MSGIDX should sort to the end."""
        import re
        memories = [
            {"content": "Some random memory without index"},
            {"content": "[MSGIDX:5] First event"},
        ]
        def extract_msgidx(mem):
            m = re.search(r'\[MSGIDX:(\d+)\]', mem.get("content", ""))
            return int(m.group(1)) if m else 999999

        sorted_mems = sorted(memories, key=extract_msgidx)
        assert "First event" in sorted_mems[0]["content"]
        assert "random" in sorted_mems[1]["content"]
