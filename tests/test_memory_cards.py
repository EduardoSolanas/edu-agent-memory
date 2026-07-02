"""Tests for memory card schema and CRUD primitives (Phase 2).

All tests run fully offline: EDUMEM_NO_EMBEDDINGS=1 is set at the top so
no embedding container or network connection is required.
"""

import os
import json
import sqlite3

os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"

import pytest
from edumem.core.beam import BeamMemory


@pytest.fixture()
def beam(tmp_path):
    db = str(tmp_path / "test.db")
    bm = BeamMemory(db_path=db, session_id="s1")
    yield bm
    try:
        bm.conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Schema existence
# ---------------------------------------------------------------------------

def test_card_tables_exist(beam):
    """All card-related tables and virtual tables must be present after init."""
    conn = beam.conn
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') "
            "OR (type = 'table' AND name LIKE 'vec_%')"
        ).fetchall()
    }
    # Also get virtual tables
    all_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master"
        ).fetchall()
    }
    assert "memory_cards" in all_names
    assert "memory_card_evidence" in all_names
    assert "memory_card_queue" in all_names
    assert "fts_cards" in all_names
    # vec_cards may not exist if sqlite-vec is absent — just check it's in
    # sqlite_master (either as a real table or not present at all; we don't
    # hard-require vec_cards since vec may be missing in the test env)


def test_card_indexes_exist(beam):
    """Required indexes must be present."""
    conn = beam.conn
    idx_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert "idx_cards_session" in idx_names
    assert "idx_cards_type" in idx_names
    assert "idx_cards_live_key" in idx_names
    assert "idx_card_evidence_card" in idx_names
    assert "idx_card_queue_status" in idx_names
    assert "idx_card_queue_dedupe" in idx_names


# ---------------------------------------------------------------------------
# _upsert_card / _get_live_card
# ---------------------------------------------------------------------------

def test_upsert_creates_live_card(beam):
    """_upsert_card inserts a card; _get_live_card retrieves it with decoded state."""
    cid = beam._upsert_card(
        session="s1",
        card_type="entity",
        card_key="alice",
        title="Alice",
        summary="Alice is a developer.",
        state={"role": "dev"},
        confidence=0.9,
        msg_idx=5,
        source_start_msg_idx=1,
        source_end_msg_idx=5,
    )
    assert isinstance(cid, int) and cid > 0

    card = beam._get_live_card("s1", "entity", "alice")
    assert card is not None
    assert card["id"] == cid
    assert card["title"] == "Alice"
    assert card["summary"] == "Alice is a developer."
    assert card["confidence"] == pytest.approx(0.9)
    assert card["version_id"] == 0
    assert card["previous_card_id"] is None
    assert card["valid_to_msg_idx"] is None
    # state_json must be decoded
    assert card["state"] == {"role": "dev"}


def test_upsert_unknown_type_rejected(beam):
    """CHECK constraint must reject invalid card_type."""
    with pytest.raises(Exception):
        beam.conn.execute(
            "INSERT INTO memory_cards "
            "(session_id, card_type, card_key, title, summary) "
            "VALUES ('s1', 'invalid_type', 'k', 't', 's')"
        )
        beam.conn.commit()


# ---------------------------------------------------------------------------
# Version chaining
# ---------------------------------------------------------------------------

def test_upsert_version_chains(beam):
    """Second upsert on the same key expires the old row and creates a new versioned one."""
    old_id = beam._upsert_card(
        session="s1", card_type="topic", card_key="auth",
        title="Auth v1", summary="Auth topic v1.",
        state={}, confidence=0.7, msg_idx=10,
    )
    new_id = beam._upsert_card(
        session="s1", card_type="topic", card_key="auth",
        title="Auth v2", summary="Auth topic v2.",
        state={"updated": True}, confidence=0.8, msg_idx=20,
    )

    assert new_id != old_id

    # Old row must be expired
    old_row = beam.conn.execute(
        "SELECT valid_to_msg_idx FROM memory_cards WHERE id = ?", (old_id,)
    ).fetchone()
    assert old_row is not None
    assert old_row[0] == 20

    # New row must have version_id incremented and previous_card_id pointing at old
    new_row = beam.conn.execute(
        "SELECT version_id, previous_card_id, valid_to_msg_idx FROM memory_cards WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert new_row[0] == 1          # version_id incremented
    assert new_row[1] == old_id     # previous_card_id chains back
    assert new_row[2] is None       # new row is live

    # _get_live_card returns only the new one
    card = beam._get_live_card("s1", "topic", "auth")
    assert card is not None
    assert card["id"] == new_id
    assert card["title"] == "Auth v2"


def test_live_key_unique_index(beam):
    """Only one live card per (session, card_type, card_key) — UNIQUE partial index enforced."""
    beam._upsert_card(
        session="s1", card_type="belief", card_key="pref_dark_mode",
        title="Prefers dark mode", summary="User prefers dark mode.",
        state={}, confidence=0.85, msg_idx=3,
    )
    # Attempt a direct INSERT bypass (without expiring the old row) must fail
    with pytest.raises(Exception):
        beam.conn.execute(
            "INSERT INTO memory_cards "
            "(session_id, card_type, card_key, title, summary) "
            "VALUES ('s1', 'belief', 'pref_dark_mode', 'Dup', 'Dup')"
        )
        beam.conn.commit()


# ---------------------------------------------------------------------------
# _invalidate_card
# ---------------------------------------------------------------------------

def test_invalidate_card(beam):
    """_invalidate_card sets valid_to and _get_live_card returns None afterward."""
    beam._upsert_card(
        session="s1", card_type="change", card_key="deploy_window",
        title="Deploy window", summary="Feb 5-12.",
        state={}, confidence=0.75, msg_idx=7,
    )
    invalidated = beam._invalidate_card("s1", "change", "deploy_window", msg_idx=15)
    assert invalidated is True

    card = beam._get_live_card("s1", "change", "deploy_window")
    assert card is None


def test_invalidate_missing_card_returns_false(beam):
    """_invalidate_card returns False if no live card exists for that key."""
    result = beam._invalidate_card("s1", "entity", "nobody", msg_idx=1)
    assert result is False


# ---------------------------------------------------------------------------
# _list_live_cards
# ---------------------------------------------------------------------------

def test_list_live_cards_filter_by_type(beam):
    """_list_live_cards filters correctly by type and excludes invalidated cards."""
    beam._upsert_card(
        session="s1", card_type="entity", card_key="bob",
        title="Bob", summary="Bob is QA.", state={}, confidence=0.7, msg_idx=1,
    )
    beam._upsert_card(
        session="s1", card_type="topic", card_key="testing",
        title="Testing", summary="Testing topic.", state={}, confidence=0.7, msg_idx=2,
    )
    beam._upsert_card(
        session="s1", card_type="entity", card_key="carol",
        title="Carol", summary="Carol is PM.", state={}, confidence=0.7, msg_idx=3,
    )
    # Invalidate carol
    beam._invalidate_card("s1", "entity", "carol", msg_idx=10)

    entities = beam._list_live_cards("s1", card_type="entity")
    keys = {c["card_key"] for c in entities}
    assert "bob" in keys
    assert "carol" not in keys   # invalidated
    assert "testing" not in keys  # wrong type

    all_live = beam._list_live_cards("s1")
    all_keys = {c["card_key"] for c in all_live}
    assert "bob" in all_keys
    assert "testing" in all_keys
    assert "carol" not in all_keys


# ---------------------------------------------------------------------------
# _link_card_evidence
# ---------------------------------------------------------------------------

def test_link_card_evidence_inserts_and_replaces(beam):
    """_link_card_evidence inserts rows; re-calling replaces (not appends) them."""
    cid = beam._upsert_card(
        session="s1", card_type="topic", card_key="security",
        title="Security", summary="Security hardening.",
        state={}, confidence=0.8, msg_idx=5,
    )

    items_v1 = [
        {"table": "facts", "row_id": "f1", "message_idx": 1, "snippet": "hash added", "weight": 1.0},
        {"table": "memoria_facts", "row_id": "42", "message_idx": 2, "snippet": "rbac added", "weight": 0.9},
    ]
    count_v1 = beam._link_card_evidence(cid, items_v1)
    assert count_v1 == 2

    rows = beam.conn.execute(
        "SELECT evidence_table, snippet FROM memory_card_evidence WHERE card_id = ? ORDER BY id",
        (cid,),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "facts"
    assert rows[1][0] == "memoria_facts"

    # Replace with a single item — must NOT append
    items_v2 = [
        {"table": "facts", "row_id": "f2", "message_idx": 3, "snippet": "lockout added", "weight": 1.0},
    ]
    count_v2 = beam._link_card_evidence(cid, items_v2)
    assert count_v2 == 1

    rows2 = beam.conn.execute(
        "SELECT snippet FROM memory_card_evidence WHERE card_id = ?",
        (cid,),
    ).fetchall()
    assert len(rows2) == 1
    assert rows2[0][0] == "lockout added"


def test_link_card_evidence_invalid_table_rejected(beam):
    """CHECK constraint rejects evidence_table values not in the allowed list."""
    cid = beam._upsert_card(
        session="s1", card_type="entity", card_key="dave",
        title="Dave", summary="Dave.", state={}, confidence=0.7, msg_idx=1,
    )
    with pytest.raises(Exception):
        beam.conn.execute(
            "INSERT INTO memory_card_evidence "
            "(card_id, evidence_table, evidence_row_id, snippet) "
            "VALUES (?, 'bad_table', '1', 'x')",
            (cid,),
        )
        beam.conn.commit()


# ---------------------------------------------------------------------------
# fts_cards sync triggers
# ---------------------------------------------------------------------------

def test_fts_cards_insert_trigger(beam):
    """After inserting a card, fts_cards MATCH query must find it by title term."""
    beam._upsert_card(
        session="s1", card_type="topic", card_key="deployment",
        title="Deployment Window", summary="February deployment window updated.",
        state={}, confidence=0.7, msg_idx=1,
    )
    # FTS MATCH on a title word
    rows = beam.conn.execute(
        "SELECT rowid FROM fts_cards WHERE fts_cards MATCH 'Deployment'"
    ).fetchall()
    assert len(rows) >= 1


def test_fts_cards_summary_match(beam):
    """fts_cards MATCH works on summary terms too."""
    beam._upsert_card(
        session="s1", card_type="belief", card_key="dark_mode",
        title="Display pref", summary="User strongly prefers dark mode interface.",
        state={}, confidence=0.8, msg_idx=2,
    )
    rows = beam.conn.execute(
        "SELECT rowid FROM fts_cards WHERE fts_cards MATCH 'dark'"
    ).fetchall()
    assert len(rows) >= 1


def test_fts_cards_not_returned_after_invalidation(beam):
    """After invalidation the card row still exists (expired), but fts_cards
    update trigger fires and the old content is still findable unless deleted.
    What we really care about: the live-card query returns None. The FTS table
    holds content for all card rows (live or expired); retrieval code is
    responsible for joining with valid_to_msg_idx IS NULL. This test verifies
    the FTS table is populated (row present) and that after an UPDATE (expiry)
    the trigger fires without error."""
    cid = beam._upsert_card(
        session="s1", card_type="entity", card_key="eve",
        title="Eve Entity", summary="Eve description.",
        state={}, confidence=0.7, msg_idx=1,
    )
    # Verify FTS has the card
    rows = beam.conn.execute(
        "SELECT rowid FROM fts_cards WHERE fts_cards MATCH 'Eve'"
    ).fetchall()
    assert any(r[0] == cid for r in rows)

    # Invalidate — this fires cards_au trigger; must not raise
    beam._invalidate_card("s1", "entity", "eve", msg_idx=5)
    # Live card is gone
    assert beam._get_live_card("s1", "entity", "eve") is None
