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
