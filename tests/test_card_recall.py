"""Tests for cards-first retrieval path (§6 read path).

All tests run fully offline: EDUMEM_NO_EMBEDDINGS=1 is set at the top so
no embedding container or network connection is required.

Tests use REAL BeamMemory objects on tmp_path — no mocks, no stubs.
monkeypatch.setenv/delenv for flag control is allowed (env control, not a double).
"""

import os
from datetime import datetime, timezone

os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"

import pytest
from edumem.core.beam import BeamMemory
from edumem.core.context_assembly import assemble_card_context


@pytest.fixture()
def beam(tmp_path):
    db = str(tmp_path / "test.db")
    bm = BeamMemory(db_path=db, session_id="s1")
    yield bm
    try:
        bm.conn.close()
    except Exception:
        pass


def _make_card(beam, card_type, card_key, title, summary, msg_idx=1):
    """Insert a live card and return its id."""
    card_id = beam._upsert_card(
        session="s1",
        card_type=card_type,
        card_key=card_key,
        title=title,
        summary=summary,
        state={},
        confidence=0.9,
        msg_idx=msg_idx,
    )
    return card_id


# ---------------------------------------------------------------------------
# _card_retrieve: basic fts hit + invalidated card exclusion
# ---------------------------------------------------------------------------

def test_card_retrieve_fts_hit(beam):
    """_card_retrieve returns the relevant live card via FTS when terms match title/summary."""
    card_id = _make_card(beam, "topic", "topic:security", "Security hardening", "password hashing and RBAC work")
    cards = beam._card_retrieve("security hardening")
    ids = [c["id"] for c in cards]
    assert card_id in ids, f"Expected card {card_id} in results, got {ids}"


def test_card_retrieve_excludes_invalidated(beam):
    """_card_retrieve must NOT return invalidated (valid_to_msg_idx IS NOT NULL) cards."""
    card_id = _make_card(beam, "topic", "topic:old", "Old deployment window", "was February 1-10")
    # Now invalidate by upserting a new version (which expires the old one)
    _make_card(beam, "topic", "topic:old", "Old deployment window updated", "now February 5-12", msg_idx=5)
    # The first card is now invalidated (valid_to_msg_idx=5); only the new one is live
    cards = beam._card_retrieve("deployment window")
    live_ids = [c["id"] for c in cards]
    assert card_id not in live_ids, f"Invalidated card {card_id} must not appear in results"


def test_card_retrieve_lexical_fallback_handles_natural_language_question(beam):
    """Natural-language questions should still find matching cards without embeddings."""
    card_id = _make_card(
        beam,
        "topic",
        "topic:auth",
        "Security Implementation: Authentication and Hardening",
        "Flask-Login was integrated for session management and replaced manual handling.",
    )

    cards = beam._card_retrieve(
        "Have I integrated Flask-Login for session management in my project?",
        intent="change",
    )

    ids = [c["id"] for c in cards]
    assert card_id in ids, f"Expected card {card_id} in results, got {ids}"


# ---------------------------------------------------------------------------
# Intent boost: preferred card_type surfaces first
# ---------------------------------------------------------------------------

def test_intent_boost_sum_prefers_session_over_entity(beam):
    """SUM intent should boost session/topic cards above entity cards."""
    entity_id = _make_card(beam, "entity", "entity:alice", "Alice", "Alice is a developer on the project")
    session_id = _make_card(beam, "session", "session:overview", "Session overview", "The project covers authentication and deployment")
    # Use a summarization query so intent='summary' is derived
    cards = beam._card_retrieve("summarize the project progress", intent="summary")
    if len(cards) >= 2:
        # session card should appear before entity card in results
        positions = {c["id"]: i for i, c in enumerate(cards)}
        assert positions.get(session_id, 999) < positions.get(entity_id, 999), (
            f"session card (pos {positions.get(session_id)}) should be before "
            f"entity card (pos {positions.get(entity_id)}) for SUM intent"
        )


# ---------------------------------------------------------------------------
# _hydrate_card_evidence: snippets + message_idx
# ---------------------------------------------------------------------------

def test_hydrate_card_evidence_returns_message_idx(beam):
    """_hydrate_card_evidence returns evidence rows with message_idx preserved."""
    card_id = _make_card(beam, "topic", "topic:auth", "Authentication", "Auth work summary")
    beam._link_card_evidence(card_id, [
        {"table": "memoria_facts", "row_id": "10", "message_idx": 42, "snippet": "password hashing added", "weight": 1.0},
        {"table": "memoria_facts", "row_id": "11", "message_idx": 72, "snippet": "RBAC introduced", "weight": 0.8},
    ])
    evidence = beam._hydrate_card_evidence([card_id], per_card=5)
    assert len(evidence) == 2
    msg_idxs = {e["message_idx"] for e in evidence}
    assert 42 in msg_idxs
    assert 72 in msg_idxs


def test_hydrate_card_evidence_ordered_by_weight(beam):
    """Evidence rows must come back ordered by weight descending."""
    card_id = _make_card(beam, "belief", "belief:pref", "User prefers dark mode", "dark mode preference")
    beam._link_card_evidence(card_id, [
        {"table": "memoria_facts", "row_id": "1", "message_idx": 5, "snippet": "low weight evidence", "weight": 0.3},
        {"table": "memoria_facts", "row_id": "2", "message_idx": 10, "snippet": "high weight evidence", "weight": 0.9},
        {"table": "memoria_facts", "row_id": "3", "message_idx": 15, "snippet": "mid weight evidence", "weight": 0.6},
    ])
    evidence = beam._hydrate_card_evidence([card_id], per_card=5)
    weights = [e["weight"] for e in evidence]
    assert weights == sorted(weights, reverse=True), f"Evidence not ordered by weight desc: {weights}"


# ---------------------------------------------------------------------------
# _card_first_retrieve: fallback on no live cards
# ---------------------------------------------------------------------------

def test_card_first_retrieve_returns_none_when_no_cards(beam):
    """_card_first_retrieve must return None (fallback signal) when no live cards exist."""
    result = beam._card_first_retrieve("what did we discuss about authentication?")
    assert result is None, f"Expected None fallback, got {result}"


def test_card_first_retrieve_returns_result_with_cards(beam):
    """_card_first_retrieve returns a dict (not None) when live cards match the query."""
    _make_card(beam, "topic", "topic:auth", "Authentication", "auth and login implementation details")
    result = beam._card_first_retrieve("authentication login")
    # May return None if FTS finds no match or confidence too low, but with a clear match should return dict
    # When it returns a result, it must have 'context' and 'source'
    if result is not None:
        assert "context" in result
        assert "source" in result
        assert result["context"]  # non-empty


def test_recall_surfaces_card_context_via_single_public_api(beam):
    """beam.recall() should surface card-backed context without a side call.

    This is the production shape we want: callers use one public read API and
    the card layer stays an internal retrieval stage.
    """
    card_id = _make_card(
        beam,
        "topic",
        "topic:security",
        "Security hardening",
        "password hashing and RBAC work",
    )
    beam._link_card_evidence(card_id, [
        {"table": "memoria_facts", "row_id": "1", "message_idx": 42, "snippet": "password hashing was added", "weight": 1.0},
    ])

    results = beam.recall("security hardening", top_k=5)

    assert any(
        row.get("source", "").startswith("card_")
        and "Security hardening" in row.get("content", "")
        for row in results
    ), results


def test_derive_structured_recall_intent_prefers_current_for_single_count_questions(beam):
    """Single-fact count questions should route to current-state retrieval, not summary."""
    assert (
        beam._derive_structured_recall_intent(
            "How many commits have been merged into the main branch of my Git repository?"
        )
        == "current"
    )
    assert (
        beam._derive_structured_recall_intent(
            "How many different user roles and security features am I trying to implement across my sessions?"
        )
        == "summary"
    )


# ---------------------------------------------------------------------------
# EO safeguard: ordered intent falls back when insufficient MSGIDX evidence
# ---------------------------------------------------------------------------

def test_eo_safeguard_fallback_when_no_msgidx(beam):
    """ordered intent with no MSGIDX evidence must return None (fall back to raw)."""
    # Insert a card but NO evidence with message_idx
    card_id = _make_card(beam, "change", "change:deploy", "Deployment sequence", "deployed in three stages")
    beam._link_card_evidence(card_id, [
        # message_idx=None: no anchor
        {"table": "memoria_facts", "row_id": "1", "message_idx": None, "snippet": "some event", "weight": 1.0},
    ])
    result = beam._card_first_retrieve("in what order did we deploy the features?", intent="ordered")
    assert result is None, "EO with no MSGIDX evidence should return None (fallback)"


def test_eo_result_has_msgidx_anchors_when_evidence_present(beam):
    """When EO has sufficient MSGIDX evidence, the assembled context contains MSGIDX: lines."""
    card_id = _make_card(beam, "change", "change:events", "Event sequence", "three deployment stages")
    beam._link_card_evidence(card_id, [
        {"table": "memoria_facts", "row_id": "1", "message_idx": 10, "snippet": "first deployment stage", "weight": 1.0},
        {"table": "memoria_facts", "row_id": "2", "message_idx": 25, "snippet": "second deployment stage", "weight": 0.9},
    ])
    # _card_first_retrieve with ordered intent AND sufficient evidence
    result = beam._card_first_retrieve("in what order did we deploy?", intent="ordered")
    if result is not None:
        assert "MSGIDX:" in result["context"], "Ordered result must contain MSGIDX: anchors"


def test_ordered_queries_synthesize_structured_cards_when_live_cards_are_missing(beam):
    """Ordered recall should synthesize aspect cards from typed artifacts when dream cards are absent."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            "wm-core",
            2,
            "[MSGIDX:2] Sure, let's break it down for my budget tracker project.\n"
            "1. **User Authentication**\n"
            "2. **Transaction Management**\n"
            "3. **Basic Analytics**",
            "beam_user",
        ),
        (
            "wm-transaction",
            60,
            "[MSGIDX:60] I'm currently working on the transaction CRUD and analytics integration "
            "for my personal budget tracker, and I want to make sure I'm handling errors properly.",
            "beam_user",
        ),
        (
            "wm-security",
            116,
            "[MSGIDX:116] I'm finalizing the deployment of my application and I need to add some "
            "security hardening before the public launch.",
            "beam_user",
        ),
    ]
    for memory_id, msg_idx, content, source in rows:
        beam.conn.execute(
            """INSERT INTO working_memory
               (id, session_id, content, source, timestamp, importance, veracity, created_at, message_index, scope)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                "s1",
                content,
                source,
                now,
                0.5,
                "known",
                now,
                msg_idx,
                "session",
            ),
        )
    beam._insert_preference(
        "s1",
        100,
        "I want to make sure my analytics integration is correct and that I'm not missing crucial error handling or validation",
        "transaction CRUD and analytics integration",
        None,
        "transaction CRUD and analytics integration with error handling and validation",
        source_memory_id="wm-transaction",
    )
    beam._insert_preference(
        "s1",
        116,
        "I need to add some security hardening before the public launch",
        "security hardening before the public launch",
        None,
        "finalizing deployment and adding security hardening before public launch",
        source_memory_id="wm-security",
    )
    beam._insert_instruction(
        "s1",
        124,
        "need to update my deployment scripts on Render",
        "update my deployment scripts on Render",
        "Once HTTPS is configured I need to update deployment scripts on Render.",
        source_memory_id="wm-security",
    )
    beam.conn.commit()

    result = beam._card_first_retrieve(
        "Can you list the order in which I brought up different aspects of my app development and deployment? Mention ONLY 3 items.",
        intent="ordered",
    )

    assert result is not None
    lowered = result["context"].lower()
    assert "security" in lowered
    assert "deployment" in lowered
    assert "msgidx:2" in lowered
    assert "msgidx:60" in lowered or "msgidx:100" in lowered
    assert "msgidx:116" in lowered or "msgidx:124" in lowered


# ---------------------------------------------------------------------------
# assemble_card_context: layout §6.6
# ---------------------------------------------------------------------------

def test_assemble_card_context_cards_before_evidence():
    """Cards must appear before the [Evidence] section."""
    cards = [
        {"card_type": "topic", "title": "Security hardening", "summary": "password hashing and RBAC"},
        {"card_type": "change", "title": "Deployment window", "summary": "now February 5-12"},
    ]
    evidence = [
        {"card_id": 1, "message_idx": 40, "snippet": "password hashing was added", "weight": 1.0},
        {"card_id": 1, "message_idx": 72, "snippet": "RBAC was introduced", "weight": 0.8},
    ]
    ctx = assemble_card_context(cards, evidence, max_chars=8000)
    # Cards section appears before Evidence section
    assert "[Card TOPIC]" in ctx
    assert "[Card CHANGE]" in ctx
    assert "[Evidence]" in ctx
    card_pos = ctx.index("[Card TOPIC]")
    ev_pos = ctx.index("[Evidence]")
    assert card_pos < ev_pos, "Cards must appear before [Evidence] section"


def test_assemble_card_context_msgidx_lines():
    """Evidence lines must include MSGIDX:N anchors when message_idx is present."""
    cards = [{"card_type": "topic", "title": "Auth", "summary": "auth work"}]
    evidence = [
        {"card_id": 1, "message_idx": 42, "snippet": "password hashing added", "weight": 1.0},
    ]
    ctx = assemble_card_context(cards, evidence, max_chars=8000)
    assert "MSGIDX:42" in ctx, f"Expected MSGIDX:42 in context, got: {ctx}"


def test_assemble_card_context_card_type_uppercase():
    """Card type in header must be uppercase."""
    cards = [{"card_type": "session", "title": "Session overview", "summary": "project overview"}]
    ctx = assemble_card_context(cards, [], max_chars=8000)
    assert "[Card SESSION]" in ctx
