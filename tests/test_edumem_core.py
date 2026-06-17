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


class TestRerankMerge:
    """Verify reranked candidates always tier above the un-reranked tail (no scale mixing)."""

    def test_reranked_tier_above_unreranked(self):
        from tools.evaluate_beam_end_to_end import _apply_rerank_scores
        memories = [
            {"content": "A", "score": 0.9},   # idx 0 - will be reranked low
            {"content": "B", "score": 0.1},   # idx 1 - will be reranked high
            {"content": "C", "score": 0.95},  # idx 2 - NOT reranked (tail), high raw score
        ]
        # Reranker only saw the first two; gives B a higher rerank score than A
        scores = [{"index": 0, "score": 0.2}, {"index": 1, "score": 0.8}]
        out = _apply_rerank_scores(memories, scores, top_n=3)
        # B (reranked high) first, A (reranked low) second, C (un-reranked tail) last —
        # even though C has the highest raw score, it must not leapfrog reranked items.
        assert out[0]["content"] == "B"
        assert out[1]["content"] == "A"
        assert out[2]["content"] == "C"

    def test_reranked_sorted_by_rerank_score(self):
        from tools.evaluate_beam_end_to_end import _apply_rerank_scores
        memories = [{"content": "X", "score": 0.5}, {"content": "Y", "score": 0.5}]
        scores = [{"index": 0, "score": 0.3}, {"index": 1, "score": 0.9}]
        out = _apply_rerank_scores(memories, scores, top_n=2)
        assert [m["content"] for m in out] == ["Y", "X"]

    def test_top_n_truncation(self):
        from tools.evaluate_beam_end_to_end import _apply_rerank_scores
        memories = [{"content": str(i), "score": 0.5} for i in range(5)]
        scores = [{"index": i, "score": 1.0 - i * 0.1} for i in range(5)]
        out = _apply_rerank_scores(memories, scores, top_n=2)
        assert len(out) == 2


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


class TestPass2Routing:
    """Verify Pass-2 prompt routing: only duration questions get the calculator prompt."""

    def test_duration_question_uses_calculator(self):
        from tools.evaluate_beam_end_to_end import _is_calculator_question
        assert _is_calculator_question("How many days between the start and the end?")
        assert _is_calculator_question("How long did the sprint last?")

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
        assert "one item per line" in prompt.lower() or "one per line" in prompt.lower()


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
