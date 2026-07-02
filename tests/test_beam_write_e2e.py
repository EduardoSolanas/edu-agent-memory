import hashlib
import json
import shutil
from pathlib import Path

import pytest

from edumem.core.beam import BeamMemory
from tools.beam_write_cache import (
    ReplayLLMClient,
    capture_write_contract,
    load_generated_write_cache,
    materialize_generated_write_cache_db,
    replay_generated_write_cache,
)
from tools.evaluate_beam_end_to_end import ingest_conversation


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "beam_write_cache_minimal"


def _expected_contract() -> dict:
    return json.loads((FIXTURE_DIR / "write_contract.json").read_text(encoding="utf-8"))


def _cached_live_payload() -> dict:
    payload = load_generated_write_cache(FIXTURE_DIR)
    assert payload["manifest"].get("replay_mode") == "cached_live_ingest"
    op_kinds = {op.get("op") for op in payload["operations"]}
    assert "llm_chat_response" in op_kinds
    assert "sleep_summary_response" in op_kinds
    assert "store_spo_facts" in op_kinds
    assert "store_conclusions" in op_kinds
    assert "apply_card_patch" in op_kinds
    return payload


def _classification_prompt_sha1(content: str) -> str:
    prompt = f"""Classify this user message. Reply with one or more labels (comma-separated): INSTRUCTION, PREFERENCE, or FACT.

INSTRUCTION = user telling the system what to do, how to behave, formatting rules, technical requirements, imperatives
PREFERENCE = user expressing likes, dislikes, style choices, priorities, personal taste
FACT = plain information sharing, no directive intent

Message: {content}

Labels:"""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def test_replay_generated_write_cache_matches_cached_contract(tmp_path):
    _cached_live_payload()
    beam = replay_generated_write_cache(
        FIXTURE_DIR,
        db_path=tmp_path / "write-cache.db",
    )
    try:
        assert capture_write_contract(beam) == _expected_contract()
    finally:
        beam.conn.close()


def test_replay_generated_write_cache_contract_is_stable_when_rowids_shift(tmp_path):
    beam = BeamMemory(db_path=tmp_path / "shifted.db", session_id="write-cache-session")
    try:
        beam.conn.execute(
            "INSERT INTO facts (fact_id, session_id, subject, predicate, object, timestamp, source_msg_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("seed-fact", "seed-session", "seed", "uses", "sqlite", "2024-01-01T00:00:00Z", "seed-0", 0.5),
        )
        beam.conn.execute(
            "INSERT INTO memory_cards (session_id, card_type, card_key, title, summary, state_json, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("seed-session", "topic", "topic:seed", "Seed", "Seed summary", "{}", 0.5),
        )
        beam.conn.commit()

        replay_generated_write_cache(FIXTURE_DIR, beam=beam)

        assert capture_write_contract(beam) == _expected_contract()
    finally:
        beam.conn.close()


def test_replay_generated_write_cache_works_before_contract_is_written(tmp_path):
    partial_cache = tmp_path / "partial-cache"
    shutil.copytree(FIXTURE_DIR, partial_cache)
    (partial_cache / "write_contract.json").unlink()

    beam = replay_generated_write_cache(
        partial_cache,
        db_path=tmp_path / "partial.db",
    )
    try:
        assert capture_write_contract(beam) == _expected_contract()
    finally:
        beam.conn.close()


def test_replay_generated_write_cache_can_prime_base_ingest(tmp_path):
    primed_cache = tmp_path / "primed-cache"
    shutil.copytree(FIXTURE_DIR, primed_cache)
    manifest_path = primed_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("replay_mode", None)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    beam = replay_generated_write_cache(
        primed_cache,
        db_path=tmp_path / "primed.db",
        prime_conversation=True,
    )
    try:
        contract = capture_write_contract(beam)
        assert contract["memoria_timelines"], "expected deterministic timeline writes from base ingest"
        assert any(
            fact["fact_type"] == "version" and fact["key"] == "flask_version"
            for fact in contract["memoria_facts"]
        ), "expected deterministic fact writes from base ingest"
    finally:
        beam.conn.close()


def test_replay_llm_matches_cached_classification_by_prompt_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("EDUMEM_MAX_CONCURRENT_LLM", "1")
    conversation = [
        {"role": "user", "content": "Please format responses as bullet lists."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "Return a project plan with milestones."},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": "Add charts to the final version."},
    ]
    recorded_events = [
        {
            "op": "llm_chat_response",
            "prompt_sha1": _classification_prompt_sha1(conversation[4]["content"]),
            "response": "INSTRUCTION",
        },
        {
            "op": "llm_chat_response",
            "prompt_sha1": _classification_prompt_sha1(conversation[0]["content"]),
            "response": "INSTRUCTION",
        },
        {
            "op": "llm_chat_response",
            "prompt_sha1": _classification_prompt_sha1(conversation[2]["content"]),
            "response": "INSTRUCTION",
        },
    ]

    beam = BeamMemory(db_path=tmp_path / "replay-llm-order.db", session_id="replay-llm-order")
    replay_llm = ReplayLLMClient(recorded_events)
    try:
        ingest_conversation(beam, conversation, llm=replay_llm)
        user_rows = beam.conn.execute(
            "SELECT content FROM working_memory "
            "WHERE session_id = ? AND source = 'beam_user' "
            "ORDER BY message_index ASC",
            ("replay-llm-order",),
        ).fetchall()
        assert len(user_rows) == 3
        assert all("[INSTRUCTION]" in row["content"] for row in user_rows)
        replay_llm.assert_drained()
    finally:
        beam.conn.close()


def test_replay_generated_write_cache_recall_surfaces_written_card_context(tmp_path):
    beam = replay_generated_write_cache(
        FIXTURE_DIR,
        db_path=tmp_path / "write-cache-recall.db",
    )
    try:
        results = beam.recall("deployment planning", top_k=5)
        assert any(
            row.get("source", "").startswith("card_")
            and "deployment planning" in row.get("content", "").lower()
            for row in results
        ), results
    finally:
        beam.conn.close()


def test_materialize_generated_write_cache_db_writes_final_snapshot(tmp_path):
    final_db = materialize_generated_write_cache_db(
        FIXTURE_DIR,
        db_path=tmp_path / "final.db",
    )
    assert final_db.exists()

    beam = BeamMemory(db_path=final_db, session_id="write-cache-session")
    try:
        assert capture_write_contract(beam) == _expected_contract()
    finally:
        beam.conn.close()


def test_materialized_final_snapshot_recall_uses_cached_session_context(tmp_path):
    final_db = materialize_generated_write_cache_db(
        FIXTURE_DIR,
        db_path=tmp_path / "final.db",
    )

    beam = BeamMemory(db_path=final_db)
    try:
        results = beam.recall("deployment planning", top_k=5)
        assert any(
            row.get("source", "").startswith("card_")
            and "deployment planning" in row.get("content", "").lower()
            for row in results
        ), results
    finally:
        beam.conn.close()


def test_materialize_generated_write_cache_db_requires_overwrite_for_existing_snapshot(tmp_path):
    final_db = tmp_path / "final.db"
    final_db.write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        materialize_generated_write_cache_db(
            FIXTURE_DIR,
            db_path=final_db,
        )
