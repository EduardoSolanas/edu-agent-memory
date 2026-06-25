import os

import pytest

from edumem.core.beam import BeamMemory
from tools.evaluate_beam_end_to_end import LLMClient


def test_metric_consolidation_zero_disables_llm(tmp_path, monkeypatch):
    """The documented false value must bypass the LLM path entirely."""
    monkeypatch.setenv("EDUMEM_LLM_FACT_CONSOLIDATION", "0")
    client = LLMClient(
        model="qwen3.6",
        api_key="unused",
        base_url="http://127.0.0.1:1",
    )
    beam = BeamMemory(
        db_path=tmp_path / "beam.db",
        session_id="disabled",
        llm_client=client,
    )
    try:
        candidates = [
            {
                "raw_key": "dashboard_latency_ms",
                "value": "250ms",
                "context": "Dashboard latency is 250ms.",
            }
        ]

        assert beam._llm_canonicalize_facts("disabled", candidates) == [
            "dashboard_latency_ms"
        ]
        assert client.last_error_class is None
    finally:
        beam.conn.close()


def test_existing_metric_key_does_not_call_llm(tmp_path, monkeypatch):
    """An exact existing key is already canonical and needs no network call."""
    monkeypatch.setenv("EDUMEM_LLM_FACT_CONSOLIDATION", "1")
    client = LLMClient(
        model="qwen3.6",
        api_key="unused",
        base_url="http://127.0.0.1:1",
    )
    beam = BeamMemory(
        db_path=tmp_path / "beam.db",
        session_id="existing",
        llm_client=client,
    )
    try:
        beam._insert_fact(
            "existing",
            0,
            "metric",
            "dashboard_latency_ms",
            "300ms",
            "Dashboard latency is 300ms.",
            0.8,
        )
        candidates = [
            {
                "raw_key": "dashboard_latency_ms",
                "value": "250ms",
                "context": "Dashboard latency is 250ms.",
            }
        ]

        assert beam._llm_canonicalize_facts("existing", candidates) == [
            "dashboard_latency_ms"
        ]
        assert client.last_error_class is None
    finally:
        beam.conn.close()


def test_canonical_metric_key_rejects_different_subject(tmp_path):
    beam = BeamMemory(db_path=tmp_path / "beam.db", session_id="subjects")
    try:
        candidate = {
            "raw_key": "query_response_time_ms",
            "value": "450ms",
            "context": "The database query response time is 450ms.",
        }
        existing = {
            "dashboard_api_response_time_ms": (
                "The dashboard API response time is 800ms."
            )
        }

        assert beam._validated_canonical_metric_key(
            candidate, "dashboard_api_response_time_ms", existing
        ) == "query_response_time_ms"
    finally:
        beam.conn.close()


def test_canonical_metric_key_keeps_dashboard_wording_variants(tmp_path):
    beam = BeamMemory(db_path=tmp_path / "beam.db", session_id="dashboard")
    try:
        existing = {
            "dashboard_api_response_time_ms": (
                "The dashboard API response time is 800ms."
            )
        }

        assert beam._validated_canonical_metric_key(
            {
                "raw_key": "dashboard_latency_improved_ms",
                "value": "300ms",
                "context": "Dashboard latency improved to 300ms.",
            },
            "dashboard_api_response_time_ms",
            existing,
        ) == "dashboard_api_response_time_ms"
        assert beam._validated_canonical_metric_key(
            {
                "raw_key": "api_response_now_ms",
                "value": "250ms",
                "context": "The API response is now 250ms.",
            },
            "dashboard_api_response_time_ms",
            existing,
        ) == "dashboard_api_response_time_ms"
    finally:
        beam.conn.close()


def test_canonical_metric_key_separates_target_from_observation(tmp_path):
    beam = BeamMemory(db_path=tmp_path / "beam.db", session_id="target")
    try:
        candidate = {
            "raw_key": "response_time_under_ms",
            "value": "200ms",
            "context": "Our target is API response time under 200ms.",
        }
        existing = {
            "dashboard_api_response_time_ms": (
                "The dashboard API response time is 250ms."
            )
        }

        result = beam._validated_canonical_metric_key(
            candidate, "dashboard_api_response_time_ms", existing
        )
        assert result != "dashboard_api_response_time_ms"
        assert "target" in result
    finally:
        beam.conn.close()


@pytest.mark.skipif(
    os.environ.get("EDUMEM_RUN_LIVE_LLM_TESTS") != "1",
    reason="requires explicit live-LLM opt-in",
)
def test_live_metric_consolidation_preserves_subjects_and_target(tmp_path):
    api_key = os.environ.get("EDUMEM_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("EDUMEM_LLM_API_KEY is not configured")
    client = LLMClient(
        model="qwen3.6",
        api_key=api_key,
        base_url=os.environ.get(
            "OPENROUTER_BASE_URL", "https://api.nan.builders/v1"
        ),
    )
    beam = BeamMemory(
        db_path=tmp_path / "beam.db",
        session_id="live-subjects",
        llm_client=client,
    )
    messages = [
        "The database query response time is 450ms.",
        "After indexing, the database query latency is 120ms.",
        "Bcrypt hashing takes 500ms.",
        "The homepage loads in 500ms.",
        "The dashboard API response time is 800ms.",
        "Dashboard latency improved to 300ms.",
        "The API response is now 250ms.",
        "Our target is API response time under 200ms.",
    ]
    try:
        for index, message in enumerate(messages):
            beam.extract_and_store_facts(message, message_idx=index)
        key_by_context = {
            row[1]: row[0]
            for row in beam.conn.execute(
                "SELECT key, context_snippet FROM memoria_facts "
                "WHERE session_id = ? AND fact_type = 'metric'",
                ("live-subjects",),
            )
        }

        assert key_by_context[messages[0]] == key_by_context[messages[1]]
        assert key_by_context[messages[2]] != key_by_context[messages[3]]
        assert len({key_by_context[m] for m in messages[4:7]}) == 1
        assert key_by_context[messages[7]] != key_by_context[messages[6]]
    finally:
        beam.conn.close()
