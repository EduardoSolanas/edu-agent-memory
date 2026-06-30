"""Tests for the WRITE PATH card layer (Phases D, E, F).

All offline tests run with:
  EDUMEM_NO_EMBEDDINGS=1  — no embedding container needed

Uses REAL BeamMemory objects on tmp_path DBs. NO mocks/stubs/monkeypatching.
"""

import os
import pytest

# Set before any edumem import so the module-level checks see the right env.
os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"

from edumem.core.beam import BeamMemory


@pytest.fixture()
def beam_card_on(tmp_path):
    """BeamMemory with the default always-on card layer, no embeddings."""
    db = str(tmp_path / "beam_card_on.db")
    bm = BeamMemory(db_path=db, session_id="s1", use_cloud=False)
    yield bm
    try:
        bm.conn.close()
    except Exception:
        pass


@pytest.fixture()
def beam_no_flag(tmp_path):
    """BeamMemory with no card env var set — the production default."""
    db = str(tmp_path / "beam_no_flag.db")
    bm = BeamMemory(db_path=db, session_id="s1", use_cloud=False)
    yield bm
    try:
        bm.conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# _card_layer_enabled — production default is always on, no flag needed
# ---------------------------------------------------------------------------

def test_card_layer_on_without_flag(beam_no_flag):
    """Card layer is the production default — on even with no env var."""
    assert beam_no_flag._card_layer_enabled() is True


def test_card_layer_on(beam_card_on):
    """Card layer stays on in the default BeamMemory path."""
    assert beam_card_on._card_layer_enabled() is True


# ---------------------------------------------------------------------------
# _enqueue_card_updates + dedupe (UNIQUE index coalesces duplicates)
# ---------------------------------------------------------------------------

def test_enqueue_inserts_rows(beam_card_on):
    """_enqueue_card_updates inserts rows into memory_card_queue."""
    items = [
        {
            "agenda_type": "entity",
            "agenda_key": "entity:alice",
            "trigger_table": "memoria_facts",
            "trigger_row_id": 42,
        }
    ]
    n = beam_card_on._enqueue_card_updates("s1", items)
    assert n == 1
    rows = beam_card_on.conn.execute(
        "SELECT agenda_type, agenda_key, status FROM memory_card_queue WHERE session_id='s1'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "entity"
    assert rows[0][1] == "entity:alice"
    assert rows[0][2] == "pending"


def test_enqueue_deduplicates(beam_card_on):
    """Enqueuing the same (session, agenda_type, agenda_key, trigger_table, trigger_row_id)
    twice coalesces: second insert adds 0 rows."""
    item = {
        "agenda_type": "topic",
        "agenda_key": "topic:security",
        "trigger_table": "memoria_facts",
        "trigger_row_id": 7,
    }
    n1 = beam_card_on._enqueue_card_updates("s1", [item])
    n2 = beam_card_on._enqueue_card_updates("s1", [item])
    assert n1 == 1
    assert n2 == 0  # Proves UNIQUE index coalescing
    count = beam_card_on.conn.execute(
        "SELECT COUNT(*) FROM memory_card_queue WHERE session_id='s1'"
    ).fetchone()[0]
    assert count == 1


def test_enqueue_coalesces_same_live_card_key(beam_card_on):
    """Different raw triggers for one pending card update one agenda row."""
    item1 = {
        "agenda_type": "topic",
        "agenda_key": "topic:security",
        "trigger_table": "memoria_facts",
        "trigger_row_id": 7,
    }
    item2 = dict(item1, trigger_row_id=8)

    n1 = beam_card_on._enqueue_card_updates("s1", [item1])
    n2 = beam_card_on._enqueue_card_updates("s1", [item2])

    assert n1 == 1
    assert n2 == 0
    row = beam_card_on.conn.execute(
        "SELECT COUNT(*), MAX(trigger_row_id) FROM memory_card_queue "
        "WHERE session_id='s1' AND agenda_key='topic:security'"
    ).fetchone()
    assert row[0] == 1
    assert row[1] == "8"



# ---------------------------------------------------------------------------
# _agenda_from_raw_write
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ftype,expected_card_type", [
    ("entity", "entity"),
    ("conclusion", "topic"),
    ("version", "change"),
    ("change", "change"),
    ("decision", "change"),
    ("preference", "belief"),
    ("instruction", "belief"),
    ("metric", "topic"),
    ("state", "topic"),
    ("fact", "topic"),
])
def test_agenda_from_raw_write_mapping(beam_card_on, ftype, expected_card_type):
    """_agenda_from_raw_write maps fact_type to the correct agenda_type."""
    items = beam_card_on._agenda_from_raw_write(ftype, "some_key", "some_value", 5, 99)
    assert len(items) == 1
    assert items[0]["agenda_type"] == expected_card_type
    assert items[0]["trigger_table"] == "memoria_facts"
    assert items[0]["trigger_row_id"] == 99
    # agenda_key should include the card_type prefix
    assert items[0]["agenda_key"].startswith(expected_card_type + ":")


# ---------------------------------------------------------------------------
# _apply_card_patch
# ---------------------------------------------------------------------------

def test_apply_card_patch_add(beam_card_on):
    """ADD patch creates a live card with title, summary, and returns card_id."""
    patch = {
        "action": "ADD",
        "card_type": "entity",
        "card_key": "entity:alice",
        "title": "Alice",
        "summary": "Alice is a developer.",
        "state": {"role": "dev"},
        "confidence": 0.9,
        "evidence": [],
    }
    cid = beam_card_on._apply_card_patch("s1", patch, msg_idx=5)
    assert isinstance(cid, int) and cid > 0
    card = beam_card_on._get_live_card("s1", "entity", "entity:alice")
    assert card is not None
    assert card["title"] == "Alice"
    assert card["confidence"] == pytest.approx(0.9)


def test_apply_card_patch_update_version_chains(beam_card_on):
    """UPDATE on existing card expires the old row and creates a new version."""
    # First ADD
    beam_card_on._apply_card_patch("s1", {
        "action": "ADD",
        "card_type": "topic",
        "card_key": "topic:security",
        "title": "Security v1",
        "summary": "Initial security summary.",
        "state": {},
        "confidence": 0.7,
        "evidence": [],
    }, msg_idx=1)

    # Then UPDATE
    cid2 = beam_card_on._apply_card_patch("s1", {
        "action": "UPDATE",
        "card_type": "topic",
        "card_key": "topic:security",
        "title": "Security v2",
        "summary": "Updated security summary.",
        "state": {},
        "confidence": 0.85,
        "evidence": [],
    }, msg_idx=5)
    assert isinstance(cid2, int) and cid2 > 0

    # Live card should be the updated one
    card = beam_card_on._get_live_card("s1", "topic", "topic:security")
    assert card is not None
    assert card["title"] == "Security v2"
    assert card["version_id"] == 1  # version-chained

    # Old card should be expired
    expired = beam_card_on.conn.execute(
        "SELECT COUNT(*) FROM memory_cards WHERE session_id='s1' "
        "AND card_key='topic:security' AND valid_to_msg_idx IS NOT NULL"
    ).fetchone()[0]
    assert expired == 1


def test_apply_card_patch_delete_invalidates(beam_card_on):
    """DELETE patch invalidates the live card (sets valid_to_msg_idx)."""
    beam_card_on._apply_card_patch("s1", {
        "action": "ADD",
        "card_type": "belief",
        "card_key": "belief:diet",
        "title": "Diet preference",
        "summary": "User prefers vegetarian food.",
        "state": {},
        "confidence": 0.8,
        "evidence": [],
    }, msg_idx=2)

    result = beam_card_on._apply_card_patch("s1", {
        "action": "DELETE",
        "card_type": "belief",
        "card_key": "belief:diet",
    }, msg_idx=10)
    assert result is None

    # No live card remains
    card = beam_card_on._get_live_card("s1", "belief", "belief:diet")
    assert card is None


def test_apply_card_patch_noop_returns_none(beam_card_on):
    """NOOP patch returns None and creates no cards."""
    result = beam_card_on._apply_card_patch("s1", {"action": "NOOP"}, msg_idx=0)
    assert result is None
    count = beam_card_on.conn.execute(
        "SELECT COUNT(*) FROM memory_cards WHERE session_id='s1'"
    ).fetchone()[0]
    assert count == 0


def test_apply_card_patch_malformed_returns_none(beam_card_on):
    """Malformed or incomplete patch returns None without raising."""
    # Missing required fields
    assert beam_card_on._apply_card_patch("s1", {}, msg_idx=0) is None
    assert beam_card_on._apply_card_patch("s1", {"action": "ADD"}, msg_idx=0) is None
    assert beam_card_on._apply_card_patch("s1", None, msg_idx=0) is None
    assert beam_card_on._apply_card_patch("s1", {"action": "ADD", "card_type": "entity", "card_key": "k"}, msg_idx=0) is None


def test_apply_card_patch_with_evidence(beam_card_on):
    """ADD patch with evidence links creates card_evidence rows."""
    patch = {
        "action": "ADD",
        "card_type": "topic",
        "card_key": "topic:perf",
        "title": "Performance",
        "summary": "System perf improved.",
        "state": {},
        "confidence": 0.75,
        "evidence": [
            {
                "table": "memoria_facts",
                "row_id": "10",
                "message_idx": 3,
                "snippet": "latency dropped",
                "weight": 1.0,
            }
        ],
    }
    cid = beam_card_on._apply_card_patch("s1", patch, msg_idx=5)
    assert cid is not None
    ev_count = beam_card_on.conn.execute(
        "SELECT COUNT(*) FROM memory_card_evidence WHERE card_id=?",
        (cid,),
    ).fetchone()[0]
    assert ev_count == 1


def test_apply_card_patch_invalid_evidence_table_skipped(beam_card_on):
    """Evidence with a table name not in the CHECK constraint must be silently
    skipped instead of raising IntegrityError."""
    patch = {
        "action": "ADD",
        "card_type": "topic",
        "card_key": "topic:bogus-ev",
        "title": "Bogus evidence",
        "summary": "LLM returned an invalid evidence_table name.",
        "state": {},
        "confidence": 0.8,
        "evidence": [
            {
                "table": "memory_cards",  # not in allowlist
                "row_id": "99",
                "message_idx": 5,
                "snippet": "should be skipped",
                "weight": 1.0,
            },
            {
                "table": "memoria_facts",  # valid
                "row_id": "10",
                "message_idx": 3,
                "snippet": "should be kept",
                "weight": 1.0,
            },
        ],
    }
    cid = beam_card_on._apply_card_patch("s1", patch, msg_idx=5)
    assert cid is not None
    ev_rows = beam_card_on.conn.execute(
        "SELECT evidence_table FROM memory_card_evidence WHERE card_id=?",
        (cid,),
    ).fetchall()
    assert len(ev_rows) == 1
    assert ev_rows[0][0] == "memoria_facts"


# ---------------------------------------------------------------------------
# Card enqueue on raw write (production default — no flag)
# ---------------------------------------------------------------------------

def test_card_layer_on_enqueues_on_fact_write(beam_card_on):
    """The default card layer enqueues a card agenda item on fact write."""
    beam_card_on._store_memoria_fact(
        "s1", 1, "preference", "diet", "vegetarian", "ctx", 0.8
    )
    beam_card_on.conn.commit()

    queue_count = beam_card_on.conn.execute(
        "SELECT COUNT(*) FROM memory_card_queue WHERE session_id='s1'"
    ).fetchone()[0]
    assert queue_count >= 1, "card_queue must have at least one row after a successful fact write"

    # Verify the agenda_type is 'belief' (preference maps to belief)
    row = beam_card_on.conn.execute(
        "SELECT agenda_type FROM memory_card_queue WHERE session_id='s1' LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "belief"


# ---------------------------------------------------------------------------
# Evidence lookup: slug extraction must use content terms, not the type prefix
# ---------------------------------------------------------------------------

def test_card_queue_evidence_lookup_finds_matching_facts(beam_card_on):
    """The evidence lookup in _process_card_queue must find facts whose key/value
    match the content part of the agenda_key, NOT the type prefix.

    Regression: slug_terms[0] was always 'topic'/'belief'/etc., so LIKE '%topic%'
    never matched any actual fact content → 0 evidence → LLM returned NOOP.
    """
    # Store a fact about weather units
    beam_card_on._store_memoria_fact(
        "s1", 1, "preference", "weather_units", "celsius", "User prefers celsius", 0.8
    )
    beam_card_on.conn.commit()

    # The agenda_key for this would be "belief:weather_units"
    agenda_key = "belief:weather_units"
    evidence = beam_card_on._card_queue_evidence("s1", agenda_key)
    assert len(evidence) >= 1, (
        f"Evidence lookup for '{agenda_key}' should find the weather_units fact, "
        f"got {len(evidence)} rows"
    )
    assert any("weather" in (e.get("key", "") + e.get("value", "")) for e in evidence)


def test_card_queue_evidence_lookup_ignores_type_prefix(beam_card_on):
    """Searching for 'topic' (the type prefix) alone must NOT be the lookup strategy."""
    beam_card_on._store_memoria_fact(
        "s1", 1, "state", "deploy_status", "production", "Deployed to prod", 0.9
    )
    beam_card_on.conn.commit()

    evidence = beam_card_on._card_queue_evidence("s1", "topic:deploy_status")
    assert len(evidence) >= 1, (
        "Evidence lookup for 'topic:deploy_status' should find the deploy_status fact"
    )


# ---------------------------------------------------------------------------
# _process_card_queue / _refresh_session_overview_card: no-op when use_cloud=False
# ---------------------------------------------------------------------------

def test_process_card_queue_noop_without_cloud(beam_card_on):
    """_process_card_queue with use_cloud=False returns graceful no-op, queue left pending."""
    # Manually insert a pending queue row
    beam_card_on.conn.execute(
        "INSERT INTO memory_card_queue (session_id, agenda_type, agenda_key, "
        "trigger_table, trigger_row_id, status) VALUES (?, ?, ?, ?, ?, 'pending')",
        ("s1", "topic", "topic:test", "memoria_facts", "1"),
    )
    beam_card_on.conn.commit()

    result = beam_card_on._process_card_queue("s1")
    # use_cloud is False, so no processing occurs
    assert result["processed"] == 0
    assert result["applied"] == 0
    assert result["errors"] == 0

    # Row should still be pending
    status = beam_card_on.conn.execute(
        "SELECT status FROM memory_card_queue WHERE session_id='s1'"
    ).fetchone()[0]
    assert status == "pending"


def test_refresh_session_overview_card_noop_without_cloud(beam_card_on):
    """_refresh_session_overview_card with use_cloud=False returns None, no exception."""
    result = beam_card_on._refresh_session_overview_card("s1")
    assert result is None


# ---------------------------------------------------------------------------
# OPTIONAL: live round-trip (gated on EDUMEM_E2E=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("EDUMEM_E2E") != "1",
    reason="Set EDUMEM_E2E=1 to run live LLM tests",
)
def test_process_card_queue_live(tmp_path, monkeypatch):
    """Live end-to-end: _process_card_queue against the real LLM."""
    db = str(tmp_path / "live_e2e.db")
    bm = BeamMemory(db_path=db, session_id="s_e2e", use_cloud=True)
    try:
        # Seed a fact and enqueue it
        bm._store_memoria_fact("s_e2e", 1, "preference", "diet", "vegan", "ctx", 0.8)
        bm.conn.commit()

        result = bm._process_card_queue("s_e2e")
        assert isinstance(result, dict)
        assert "processed" in result
        assert result.get("errors", 0) == 0
    finally:
        try:
            bm.conn.close()
        except Exception:
            pass
