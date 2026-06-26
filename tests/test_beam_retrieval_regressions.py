from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.evaluate_beam_end_to_end import (
    _is_calculator_question,
    _multi_strategy_recall,
    ingest_conversation,
)


def _make_beam(tmp_path: Path):
    beam_mod = pytest.importorskip("edumem.core.beam")
    return beam_mod.BeamMemory(db_path=tmp_path / "beam.db", session_id="test-session")


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How many days between 2024-03-01 and 2024-03-05?", True),
        ("How many days passed between when I planned peer review and when I completed final review?", True),
        ("How many days did I say the project would take?", False),
        ("How long did I say the project is expected to take?", False),
    ],
)
def test_calculator_routing_is_limited_to_true_date_intervals(question, expected):
    assert _is_calculator_question(question) is expected


def test_stated_duration_questions_do_not_enable_negation_recall(tmp_path, monkeypatch):
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "the project would take three days."},
                {"role": "user", "content": "i said it might take five days."},
                {"role": "user", "content": "the project finished in four days."},
            ],
        )

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            "How many days did I say the project would take?",
            top_k=5,
            ability=None,
            diag=diag,
        )

        assert memories
        assert diag["strategies"].get("S2", {}).get("activated") is False
    finally:
        beam.conn.close()


def test_point_in_time_when_question_does_not_generate_date_pair_memories(
    tmp_path,
    monkeypatch,
):
    """Regression for BEAM 100K conversation 1, q6."""
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        beam.remember_batch(
            [
                {
                    "content": (
                        "I'm working on a project with scheduled two-week sprints, "
                        "and the first sprint ends on March 29, focusing on user "
                        "registration and login."
                    ),
                    "source": "beam_user",
                    "occurred_at": "2024-03-29",
                },
                {
                    "content": (
                        "I'm working on sprint 2 which targets analytics by April 19, "
                        "and I've already completed sprint 1 on March 29 with user auth "
                        "and basic transaction CRUD, so I need to implement analytics "
                        "features, can you help me with that?"
                    ),
                    "source": "beam_user",
                    "occurred_at": "2024-04-19",
                },
            ],
        )

        memories = beam.recall("When does my first sprint end?", top_k=10)

        assert any("March 29" in memory["content"] for memory in memories)
        assert not any(
            memory.get("source") == "derived_temporal" for memory in memories
        )
    finally:
        beam.conn.close()


def test_q6_time_anchor_normalizes_and_retrieves_yearless_sprint_date(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {
                    "role": "user",
                    "time_anchor": "March-15-2024",
                    "content": (
                        "I'm working on a project with a Time Anchor of March 15, "
                        "2024, and I need to plan my tasks accordingly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "I'm working on a project with scheduled two-week sprints, "
                        "and the first sprint ends on March 29, focusing on user "
                        "registration and login."
                    ),
                },
            ],
        )

        stored = beam.conn.execute(
            "SELECT content, occurred_at, recorded_at FROM working_memory "
            "WHERE message_index = 1"
        ).fetchone()
        assert stored["recorded_at"].startswith("2024-03-15")
        assert stored["occurred_at"] == "2024-03-29"
        assert "first sprint ends on March 29" in stored["content"]
        assert "[ISO_DATES: 2024-03-29]" in stored["content"]
        assert "datetok20240329" in stored["content"]

        memories = beam.recall("When does my first sprint end?", top_k=10)
        assert any(
            memory.get("occurred_at") == "2024-03-29"
            and "first sprint ends on March 29" in memory["content"]
            for memory in memories
        )
    finally:
        beam.conn.close()


def test_time_anchor_normalizes_relative_weekday_during_real_ingestion(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {
                    "role": "user",
                    "time_anchor": "March-15-2024",
                    "content": "The planning reference date is March 15, 2024.",
                },
                {
                    "role": "user",
                    "content": "I will review the sprint backlog next Tuesday.",
                },
            ],
        )

        stored = beam.conn.execute(
            "SELECT content, occurred_at FROM working_memory WHERE message_index = 1"
        ).fetchone()
        assert stored["occurred_at"] == "2024-03-19"
        assert "[ISO_DATES: 2024-03-19]" in stored["content"]
        assert "datetok20240319" in stored["content"]
    finally:
        beam.conn.close()


def test_tr_timeline_anchors_dates_to_the_queried_schedule_events(
    tmp_path,
    monkeypatch,
):
    """Regression for BEAM 100K conversation 1, q18."""
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {
                    "role": "user",
                    "time_anchor": "March-15-2024",
                    "content": (
                        "November 1 - November 15, 2023: set up the Flask project "
                        "and initial database schema. November 16 - December 15, "
                        "2023: implement user authentication. December 16, 2023 - "
                        "January 15, 2024: develop transaction management features. "
                        "January 16 - February 15, 2024: integrate basic analytics. "
                        "February 16 - March 15, 2024: final adjustments, testing, "
                        "and deployment."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "A separate revised MVP plan has a final deployment "
                        "deadline of April 15, 2024."
                    ),
                },
            ],
        )

        result = beam.memoria_retrieve(
            "How many weeks are between finishing transaction management "
            "and the final deployment deadline?",
            intent="timeline",
            top_k=10,
        )

        context = result["context"].lower()
        assert "january 15, 2024" in context
        assert "march 15, 2024" in context
        # With fused retrieval, multiple lines may contain a date (duration
        # facts, timeline entries). Find the line where the date AND the
        # matching event co-occur, not just the first line with the date.
        january_line = next(
            line for line in context.splitlines()
            if "january 15, 2024" in line and "transaction management" in line
        )
        march_line = next(
            line for line in context.splitlines()
            if "march 15, 2024" in line and "deployment" in line
        )
        assert "transaction management" in january_line
        assert "deployment" in march_line
        assert "april 15, 2024" not in context
    finally:
        beam.conn.close()


@pytest.mark.parametrize(
    "question, expected_mr",
    [
        ("How many project cards are there?", False),
        ("How many project cards across all sessions are there?", True),
        ("How many project cards in total are there?", True),
    ],
)
def test_project_card_counts_only_expand_for_broad_aggregation_language(
    tmp_path,
    monkeypatch,
    question,
    expected_mr,
):
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "project cards are three."},
                {"role": "user", "content": "there are four project cards in the backlog."},
                {"role": "user", "content": "the cards are red, blue, and green."},
            ],
        )

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            question,
            top_k=5,
            ability=None,
            diag=diag,
        )

        assert memories
        assert diag["strategies"].get("MR", {}).get("activated", False) is expected_mr
    finally:
        beam.conn.close()


def test_ku_question_triggers_knowledge_update_modifier():
    """Verify that KU questions trigger the KU modifier with MSGIDX tags."""
    from edumem.core.query_mode import build_system_prompt, is_knowledge_update_query

    question = "What is the deadline for completing the first sprint?"

    # Test that the question is recognized as a knowledge update query
    assert is_knowledge_update_query(question) is True

    # Test that the prompt contains the KU modifier
    prompt = build_system_prompt(question)
    assert "KNOWLEDGE UPDATE" in prompt
    assert "MSGIDX" in prompt


def test_ku_updated_fact_supersedes_old_in_retrieval(tmp_path, monkeypatch):
    """Verify that KU retrieval surfaces both old and new facts for comparison."""
    from edumem.core.query_mode import is_knowledge_update_query
    from datetime import datetime, timezone

    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        # Insert two working_memory rows with different message indices
        # The higher MSGIDX represents more recent information
        now = datetime.now(timezone.utc).isoformat()
        beam.conn.execute(
            """INSERT INTO working_memory
               (id, message_index, content, veracity, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("old-sprint-deadline", 13,
             "[MSGIDX:13] The first sprint deadline is April 1, 2024.",
             "known", now)
        )
        beam.conn.execute(
            """INSERT INTO working_memory
               (id, message_index, content, veracity, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("new-sprint-deadline", 95,
             "[MSGIDX:95] The deadline for the first sprint has been updated to April 5, 2024.",
             "known", now)
        )
        beam.conn.commit()

        question = "What is the deadline for the first sprint?"
        assert is_knowledge_update_query(question) is True

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            question,
            top_k=5,
            ability=None,
            diag=diag,
        )

        # Both memories should be retrieved so the LLM can see the update
        assert len(memories) >= 2, f"Expected at least 2 memories, got {len(memories)}"
        contents = [m.get("content", "") for m in memories]
        assert any("MSGIDX:13" in c and "April 1" in c for c in contents), \
            f"Old deadline not found in contents: {contents}"
        assert any("MSGIDX:95" in c and "April 5" in c for c in contents), \
            f"Updated deadline not found in contents: {contents}"
    finally:
        beam.conn.close()


def test_ku_contradictory_counts_resolved_by_msgidx(tmp_path, monkeypatch):
    """Verify that contradictory counts are both retrieved with MSGIDX for resolution."""
    from edumem.core.query_mode import is_knowledge_update_query
    from datetime import datetime, timezone

    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        # Insert two contradictory facts about project cards count
        now = datetime.now(timezone.utc).isoformat()
        beam.conn.execute(
            """INSERT INTO working_memory
               (id, message_index, content, veracity, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("project-cards-initial", 48,
             "[MSGIDX:48] The project gallery has 6 items in the MVP.",
             "known", now)
        )
        beam.conn.execute(
            """INSERT INTO working_memory
               (id, message_index, content, veracity, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("project-cards-updated", 116,
             "[MSGIDX:116] I now have 10 project cards after adding two new projects.",
             "known", now)
        )
        beam.conn.commit()

        question = "How many project cards do I have?"
        assert is_knowledge_update_query(question) is True

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            question,
            top_k=5,
            ability=None,
            diag=diag,
        )

        # Both contradictory memories should be retrieved
        assert len(memories) >= 2, f"Expected at least 2 memories, got {len(memories)}"
        contents = [m.get("content", "") for m in memories]
        assert any("MSGIDX:48" in c and "6 items" in c for c in contents), \
            f"Initial count not found: {contents}"
        assert any("MSGIDX:116" in c and "10 project cards" in c for c in contents), \
            f"Updated count not found: {contents}"
    finally:
        beam.conn.close()


def test_eo_question_triggers_ordering_modifier_with_msgidx():
    """Verify that EO questions trigger the ordering modifier with MSGIDX."""
    from edumem.core.query_mode import build_system_prompt, is_ordering_query

    question = "Can you list the order in which I brought up different aspects?"

    # Test that the question is recognized as an ordering query
    assert is_ordering_query(question) is True

    # Test that the prompt contains the ordering modifier and MSGIDX reference
    prompt = build_system_prompt(question)
    assert "ORDERING" in prompt
    assert "MSGIDX" in prompt


def test_eo_retrieval_preserves_msgidx_tags(tmp_path, monkeypatch):
    """Verify that EO retrieval preserves MSGIDX tags for LLM ordering."""
    from edumem.core.query_mode import is_ordering_query
    import re

    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        # Ingest messages which will get proper MSGIDX tags and FTS5 indexing
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "We discussed the project timeline first."},
                {"role": "user", "content": "Then we talked about resource allocation."},
                {"role": "user", "content": "Finally, we addressed budget constraints."},
            ],
        )

        question = "In what order did I bring up the different aspects of the project?"
        assert is_ordering_query(question) is True

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            question,
            top_k=10,
            ability=None,
            diag=diag,
        )

        # Ordering queries should retrieve memories (at least 1)
        assert len(memories) >= 1, f"Expected at least 1 memory, got {len(memories)}"
        contents = [m.get("content", "") for m in memories]

        # Verify MSGIDX tags are present in retrieved memories
        msgidx_tags = []
        for content in contents:
            matches = re.findall(r'\[MSGIDX:(\d+)\]', content)
            msgidx_tags.extend(matches)
        assert len(msgidx_tags) > 0, \
            f"No MSGIDX tags found in retrieved memories: {contents}"

        # Verify the EO modifier was activated (top_k multiplied for ordering queries)
        assert diag["strategies"].get("EO", {}).get("activated") is True, \
            "EO strategy should be activated for ordering queries"
    finally:
        beam.conn.close()


def test_change_over_time_precedes_conflicts_in_prompt():
    """Verify that CHANGE OVER TIME appears before CONFLICTS in the prompt."""
    from edumem.core.query_mode import build_system_prompt

    # Any question works; we're checking prompt structure
    question = "What is the current status?"
    prompt = build_system_prompt(question)

    # Both should be in the base prompt
    assert "CHANGE OVER TIME" in prompt
    assert "CONFLICTS" in prompt

    # Verify ordering: CHANGE OVER TIME should come before CONFLICTS
    change_pos = prompt.find("CHANGE OVER TIME")
    conflicts_pos = prompt.find("CONFLICTS")
    assert change_pos < conflicts_pos, \
        f"CHANGE OVER TIME should appear before CONFLICTS in prompt. " \
        f"CHANGE OVER TIME at {change_pos}, CONFLICTS at {conflicts_pos}"
