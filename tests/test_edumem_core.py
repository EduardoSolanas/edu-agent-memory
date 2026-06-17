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
    generate_derived_temporal_facts
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
        assert "FIRST MENTION" in prompt
        assert "message index" in prompt

    def test_no_modifiers_for_simple_question(self):
        from edumem.core.query_mode import build_system_prompt
        prompt = build_system_prompt("What is my favorite programming language?")
        assert "ORDERING" not in prompt
        assert "DURATION" not in prompt
        assert "KNOWLEDGE UPDATE" not in prompt
        assert "MULTI-HOP" not in prompt


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
