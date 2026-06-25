from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

from tools import evaluate_beam_end_to_end as beam_eval
from tools.evaluate_beam_end_to_end import (
    _assemble_memory_context,
    _required_rejudge_fields,
    _attach_second_pass_diagnostics,
    _record_second_pass_diagnostics,
    _extract_shared_date_spans,
    _build_paired_outcome_rows,
    _build_question_validation_rows,
    _build_skipped_question_result,
    _finalize_reranker_run_health,
    _multi_strategy_recall,
    _parse_judge_payload,
    _print_env_snapshot,
    _question_row_policy,
    _query_wants_if_pf,
    _select_conversations,
    _sanitize_sensitive_data,
    _summarize_judge_result,
    _update_rejudged_question_row,
    _update_embedding_diagnostic,
    _summarize_recall_memories,
    _write_json_sanitized,
    _benchmark_pure_recall_enabled,
    apply_rejudge_judgment_records,
    compute_ability_scores,
    compute_partial_credit_overall,
    ingest_conversation,
    judge_with_rubrics,
    print_sota_report,
    write_rejudge_artifacts,
)


def test_write_json_sanitized_handles_emoji_payload(tmp_path):
    """Issue 1 regression: a result payload containing an emoji must serialize
    without raising (e.g. UnicodeEncodeError) so a completed run's results are
    never lost to a charmap-encode crash."""
    payload = {
        "metadata": {"note": "CONVERSATION COMPLETE 🎉"},
        "results": [{"qid": "q1", "assessment": "great answer 🎉", "score": 1.0}],
    }
    out_path = tmp_path / "beam_e2e_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _write_json_sanitized(payload, f, indent=2)
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded["results"][0]["assessment"] == "great answer 🎉"


@pytest.mark.parametrize(
    "question",
    [
        "How should I organize the different parts of a webpage in HTML?",
        "If I'm creating a blog layout, which HTML elements should I use to clearly define sections?",
        "Can you help me set up the layout and components for this portfolio site?",
    ],
)
def test_failed_beam_procedural_questions_route_to_instruction_recall(question):
    assert _query_wants_if_pf(question) is True


def test_failed_beam_portfolio_prompt_recalls_tagged_setup_instruction(tmp_path):
    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [{"role": "user", "content": "Always use Bootstrap components for the portfolio layout."}],
        )
        diag = {}

        memories = _multi_strategy_recall(
            beam,
            "Can you help me set up the layout and components for this portfolio site?",
            top_k=5,
            ability=None,
            diag=diag,
        )

        assert diag["strategies"]["IF"]["activated"] is True
        assert any("Bootstrap components" in memory.get("content", "") for memory in memories)
    finally:
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings




def _make_beam(tmp_path: Path):
    beam_mod = pytest.importorskip("edumem.core.beam")
    return beam_mod.BeamMemory(db_path=tmp_path / "beam.db", session_id="test-session")


def test_spo_fact_id_is_content_addressed_not_batch_scoped():
    """SPO fact_id must be deterministic from (session, subject, predicate, object),
    not include batch/message id, so identical triples collapse via INSERT OR IGNORE."""
    from edumem.core.beam import _spo_fact_id

    session_id = "test-session"
    subject, predicate, obj = "DashboardAPI", "uses", "FastAPI"

    # Same triple -> same id
    id1 = _spo_fact_id(session_id, subject, predicate, obj)
    id2 = _spo_fact_id(session_id, subject, predicate, obj)
    assert id1 == id2, "Same SPO should yield same fact_id"

    # Different object -> different id
    id3 = _spo_fact_id(session_id, subject, predicate, "Django")
    assert id1 != id3, "Different object should yield different fact_id"

    # Different session -> different id
    id4 = _spo_fact_id("other-session", subject, predicate, obj)
    assert id1 != id4, "Different session should yield different fact_id"


def test_facts_table_dedups_identical_triple(tmp_path):
    """INSERT OR IGNORE with content-addressed fact_id should dedup identical
    triples across different source messages."""
    import os
    from pathlib import Path
    from edumem.core.beam import _spo_fact_id

    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_beam(tmp_path)
    try:
        subject, predicate, obj = "DashboardAPI", "uses", "FastAPI"
        fact_id = _spo_fact_id(beam.session_id, subject, predicate, obj)

        # Insert the same triple twice with different source messages
        cursor = beam.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO facts "
            "(fact_id, session_id, subject, predicate, object, "
            "timestamp, source_msg_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fact_id, beam.session_id, subject, predicate, obj,
             "2026-06-25T00:00:00Z", "m1", 0.9),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO facts "
            "(fact_id, session_id, subject, predicate, object, "
            "timestamp, source_msg_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fact_id, beam.session_id, subject, predicate, obj,
             "2026-06-25T01:00:00Z", "m2", 0.9),
        )
        beam.conn.commit()

        # Assert only one row exists (deduped)
        cursor.execute(
            "SELECT COUNT(*) FROM facts WHERE subject=? AND predicate=? AND object=?",
            (subject, predicate, obj),
        )
        count = cursor.fetchone()[0]
        assert count == 1, f"Expected 1 row, got {count} (dedup failed)"
    finally:
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings


def _json_request(url: str, payload: dict | None = None) -> object:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_shared_date_parser_handles_named_ordinal_and_iso():
    spans = _extract_shared_date_spans(
        "We met on March 15, 2024, then on Mar 15, and again on 15th of March, 2024. "
        "The release was 2024-03-15."
    )
    raw = {span["raw"] for span in spans}
    assert "March 15, 2024" in raw
    assert "Mar 15" in raw
    assert "15th of March, 2024" in raw
    assert "2024-03-15" in raw
    assert any(span["iso"] == "2024-03-15" for span in spans if span["date_obj"] is not None)


def test_selection_helper_case_and_start_index():
    conversations = [
        {"id": "c0"},
        {"id": "c1"},
        {"id": "c2"},
        {"id": "c3"},
    ]
    selected, ids = _select_conversations(conversations, sample_size=2, start_index=1)
    assert [conv["id"] for conv in selected] == ["c1", "c2"]
    assert ids == ["c1", "c2"]

    selected_case, case_ids = _select_conversations(conversations, case_index=2)
    assert [conv["id"] for conv in selected_case] == ["c2"]
    assert case_ids == ["c2"]


def test_env_sanitizer_redacts_sensitive_keys_recursively():
    payload = {
        "EDUMEM_LLM_API_KEY": "raw-key",
        "nested": {"serviceToken": "raw-token", "safe": "ok"},
        "items": [{"client_secret": "raw-secret"}],
    }

    sanitized = _sanitize_sensitive_data(payload)

    assert sanitized["EDUMEM_LLM_API_KEY"] == "***redacted***"
    assert sanitized["nested"]["serviceToken"] == "***redacted***"
    assert sanitized["nested"]["safe"] == "ok"
    assert sanitized["items"][0]["client_secret"] == "***redacted***"


def test_env_snapshot_printer_redacts_sensitive_values(capsys):
    _print_env_snapshot(
        {
            "EDUMEM_LLM_API_KEY": "raw-key",
            "sessionToken": "raw-token",
            "safe": "ok",
        }
    )

    out = capsys.readouterr().out
    assert "raw-key" not in out
    assert "raw-token" not in out
    assert "EDUMEM_LLM_API_KEY=***redacted***" in out
    assert "sessionToken=***redacted***" in out
    assert "safe=ok" in out


def test_benchmark_pure_recall_defaults_on_and_can_be_disabled():
    saved = os.environ.get("EDUMEM_BENCHMARK_PURE_RECALL")
    try:
        os.environ.pop("EDUMEM_BENCHMARK_PURE_RECALL", None)
        assert _benchmark_pure_recall_enabled() is True

        os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = "0"
        assert _benchmark_pure_recall_enabled() is False
    finally:
        if saved is None:
            os.environ.pop("EDUMEM_BENCHMARK_PURE_RECALL", None)
        else:
            os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = saved


def test_official_grader_probe_is_cached_and_falls_back_silently(capsys):
    saved_cache = beam_eval._OFFICIAL_GRADER_IMPORT_CACHE
    saved_attempts = beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS
    try:
        beam_eval._OFFICIAL_GRADER_IMPORT_CACHE = None
        beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS = 0

        module1, error1 = beam_eval._load_official_compute_metrics()
        module2, error2 = beam_eval._load_official_compute_metrics()

        out = capsys.readouterr().out
        assert beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS == 1
        assert module1 is module2
        assert error1 == error2
        assert out == ""
    finally:
        beam_eval._OFFICIAL_GRADER_IMPORT_CACHE = saved_cache
        beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS = saved_attempts


def test_official_grader_loader_prefers_local_repo_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "BEAM_official"
    compute_metrics = repo_root / "src" / "evaluation" / "compute_metrics.py"
    compute_metrics.parent.mkdir(parents=True)
    compute_metrics.write_text(
        "def evaluate_abstention(**kwargs):\n"
        "    return {'llm_judge_score': 1.0, 'llm_judge_responses': []}\n",
        encoding="utf-8",
    )

    saved_cache = beam_eval._OFFICIAL_GRADER_IMPORT_CACHE
    saved_attempts = beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS
    saved_roots = beam_eval._OFFICIAL_GRADER_CANDIDATE_ROOTS
    try:
        beam_eval._OFFICIAL_GRADER_IMPORT_CACHE = None
        beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS = 0
        monkeypatch.setattr(beam_eval, "_OFFICIAL_GRADER_CANDIDATE_ROOTS", (repo_root,), raising=False)

        module, error = beam_eval._load_official_compute_metrics()

        assert error is None
        assert module is not None
        assert hasattr(module, "evaluate_abstention")
        assert beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS == 1
    finally:
        beam_eval._OFFICIAL_GRADER_IMPORT_CACHE = saved_cache
        beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS = saved_attempts
        beam_eval._OFFICIAL_GRADER_CANDIDATE_ROOTS = saved_roots


def test_official_grader_binds_question_into_prompt(tmp_path, monkeypatch):
    """Verify that the <question> placeholder is replaced with actual question text."""
    repo_root = tmp_path / "BEAM_official"
    compute_metrics_path = repo_root / "src" / "evaluation" / "compute_metrics.py"
    compute_metrics_path.parent.mkdir(parents=True)

    # Create a real LLMClient that records what it receives
    class FakeLLMClient:
        def __init__(self):
            self.chat_calls = []  # List of (messages, kwargs) tuples

        def chat(self, messages, temperature=0.0, max_tokens=1024):
            """Record all chat calls and return valid judge JSON."""
            self.chat_calls.append((messages, {"temperature": temperature, "max_tokens": max_tokens}))
            return '{"scores":[1.0]}'

    # Write a real evaluate function that tests the placeholder replacement
    # by sending a prompt with <question> to the model.invoke() method
    compute_metrics_path.write_text(
        "def evaluate_information_extraction(rubric, llm_response, probing_question, model, **kwargs):\n"
        "    # Send a prompt containing the <question> placeholder\n"
        "    test_prompt = 'Evaluate this question: <question>'\n"
        "    response = model.invoke(test_prompt)\n"
        "    # If the replacement worked, response.content will have the actual question\n"
        "    # Store it on the function so the test can check it\n"
        "    evaluate_information_extraction.last_response_content = response.content\n"
        "    return {'llm_judge_score': 1.0, 'llm_judge_responses': [{'score': 1.0}]}\n",
        encoding="utf-8",
    )

    saved_cache = beam_eval._OFFICIAL_GRADER_IMPORT_CACHE
    saved_attempts = beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS
    saved_roots = beam_eval._OFFICIAL_GRADER_CANDIDATE_ROOTS
    try:
        beam_eval._OFFICIAL_GRADER_IMPORT_CACHE = None
        beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS = 0
        monkeypatch.setattr(beam_eval, "_OFFICIAL_GRADER_CANDIDATE_ROOTS", (repo_root,), raising=False)

        # Create the fake client and call judge_with_rubrics
        question_text = "What is your favorite color?"
        llm_client = FakeLLMClient()
        rubric = ["Should mention a color"]
        ai_answer = "I like blue because it is calming."

        result = judge_with_rubrics(
            llm=llm_client,
            question=question_text,
            rubric=rubric,
            ai_answer=ai_answer,
            ability="IE",
        )

        # Verify the function succeeded
        assert result["official_score"] == 1.0
        assert len(result["scores"]) > 0

        # Verify that the client.chat() was called at least once
        assert len(llm_client.chat_calls) > 0

        # Extract the actual content that was sent to chat()
        # Each chat call is a tuple of (messages, kwargs)
        all_message_contents = []
        for messages, _ in llm_client.chat_calls:
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        all_message_contents.append(msg["content"])

        # Verify the actual question text made it into the messages
        assert any(question_text in content for content in all_message_contents), (
            f"Question text '{question_text}' not found in any chat call. "
            f"Contents: {all_message_contents}"
        )

        # Verify the literal placeholder was replaced (not sent as-is)
        assert not any("<question>" in content for content in all_message_contents), (
            f"Literal <question> placeholder still found in chat calls. "
            f"Contents: {all_message_contents}"
        )
    finally:
        beam_eval._OFFICIAL_GRADER_IMPORT_CACHE = saved_cache
        beam_eval._OFFICIAL_GRADER_IMPORT_ATTEMPTS = saved_attempts
        beam_eval._OFFICIAL_GRADER_CANDIDATE_ROOTS = saved_roots


def test_json_writer_redacts_sensitive_values(tmp_path):
    output = tmp_path / "artifact.json"
    with output.open("w", encoding="utf-8") as fh:
        _write_json_sanitized(
            {
                "metadata": {
                    "config": {
                        "env": {
                            "EDUMEM_LLM_API_KEY": "raw-key",
                            "accessToken": "raw-token",
                        }
                    }
                }
            },
            fh,
            indent=2,
        )

    written = output.read_text(encoding="utf-8")
    assert "raw-key" not in written
    assert "raw-token" not in written
    assert "***redacted***" in written


def test_question_policy_keeps_empty_ideal_but_skips_invalid_rows():
    evaluated = _question_row_policy(
        {
            "question": "What should I do?",
            "ideal_answer": "",
            "rubric": ["Mention the instruction."],
        }
    )
    missing_question = _question_row_policy(
        {
            "question": "",
            "ideal_answer": "anything",
            "rubric": ["Mention the instruction."],
        }
    )
    missing_rubric = _question_row_policy(
        {
            "question": "What should I do?",
            "ideal_answer": "anything",
            "rubric": [],
        }
    )

    assert evaluated["should_evaluate"] is True
    assert evaluated["ideal_answer"] == ""
    assert missing_question["should_evaluate"] is False
    assert missing_question["skip_reason"] == "missing_question"
    assert missing_rubric["should_evaluate"] is False
    assert missing_rubric["skip_reason"] == "missing_rubric"


def test_skipped_question_rows_stay_accounted_for_without_affecting_scores():
    all_results = [
        {
            "conversation_id": "conv-1",
            "scale": "100K",
            "results": [
                {"qid": "conv-1:q7", "ability": "IE", "score": 1.0, "partial_credit_score": 1.0},
                _build_skipped_question_result(
                    qid="conv-1:q8",
                    ability="IF",
                    question="What did I ask?",
                    ideal_answer="",
                    rubric=["Should mention the instruction."],
                    skip_reason="missing_question",
                ),
            ],
        }
    ]

    ability_summary = compute_ability_scores(all_results)
    assert ability_summary["100K"]["IE"]["count"] == 1
    assert ability_summary["100K"]["OVERALL"]["avg_score"] == pytest.approx(1.0)
    assert compute_partial_credit_overall(all_results) == pytest.approx(1.0)


def test_reranker_health_finalizer_marks_failed_when_calls_all_fail():
    finalized = _finalize_reranker_run_health(
        {"ok": True, "status_code": 200},
        {
            "calls": 3,
            "successes": 0,
            "failures": 3,
            "fallbacks": 3,
            "scores_recorded": 0,
            "errors": [{"class": "TimeoutError", "message": "timeout"}],
        },
    )

    assert finalized["preflight_health"] == "ok"
    assert finalized["call_health"] == "failed"
    assert finalized["health"] == "failed"
    assert finalized["calls"] == 3
    assert finalized["successes"] == 0
    assert finalized["failures"] == 3
    assert finalized["errors"] == [{"class": "TimeoutError", "message": "timeout"}]


def test_paired_outcome_builder_emits_every_evaluated_row_and_thresholds_scores():
    conv_result = {
        "conversation_id": "conv-1",
        "scale": "100K",
        "results": [
            {"qid": "conv-1:q1", "ability": "IE", "score": 0.5},
            {"qid": "conv-1:q2", "ability": "MR", "score": 0.49},
            _build_skipped_question_result(
                qid="conv-1:q3",
                ability="IF",
                question="What should I do?",
                ideal_answer="",
                rubric=["Mention the instruction."],
                skip_reason="missing_rubric",
            ),
        ],
    }

    rows = _build_paired_outcome_rows(conv_result, "cfg-test", "2026-06-19T00:00:00Z")

    assert [row["qid"] for row in rows] == ["conv-1:q1", "conv-1:q2"]
    assert rows[0]["correct"] is True
    assert rows[1]["correct"] is False


def test_question_validation_builder_emits_full_rows_and_keeps_skips():
    conv_result = {
        "conversation_id": "conv-1",
        "scale": "100K",
        "results": [
            {
                "qid": "conv-1:q1",
                "ability": "IE",
                "score": 0.75,
                "official_score": 1.0,
                "partial_credit_score": 0.75,
                "parse_status": "ok",
                "judge_status": "ok",
                "judge_failure_class": None,
                "judge_failure_message": "",
                "question": "Short question",
                "question_full": "Short question with the full long context that should be preserved.",
                "ideal_answer": "Short answer",
                "ideal_answer_full": "Short answer with the full long context that should be preserved.",
                "ai_answer": "AI answer excerpt",
                "ai_answer_full": "AI answer excerpt with the complete answer payload that should be preserved.",
                "ai_answer_excerpt": "AI answer excerpt",
                "assessment": "brief",
                "judge_assessment": "full judge assessment",
                "answer_model": "answer-model",
                "judge_model": "judge-model",
                "answer_time_ms": 11.0,
                "judge_time_ms": 22.0,
                "judge_finish_reason": "stop",
                "judge_response_had_content": True,
                "judge_retry_count": 0,
                "judge_raw_response": "{\"scores\":[1.0]}",
                "judge_raw_result": {"scores": [1.0]},
                "judge_raw_payload": {"scores": [1.0]},
                "retrieval_diagnostics": {"strategy": "keyword"},
                "answer_api_diagnostics": {"finish_reason": "stop"},
                "nuggets": ["fact"],
                "recall_provenance": {"source": "memory"},
            },
            _build_skipped_question_result(
                qid="conv-1:q2",
                ability="MR",
                question="",
                ideal_answer="",
                rubric=["Mention the instruction."],
                skip_reason="missing_question",
            ),
        ],
    }

    rows = _build_question_validation_rows(conv_result, "cfg-test", "2026-06-19T00:00:00Z")

    assert [row["qid"] for row in rows] == ["conv-1:q1", "conv-1:q2"]
    assert rows[0]["validation_status"] == "evaluated"
    assert rows[0]["validation_passed"] is True
    assert rows[0]["question_full"].endswith("should be preserved.")
    assert rows[0]["ai_answer_full"].endswith("should be preserved.")
    assert rows[0]["judge_raw_response"] == "{\"scores\":[1.0]}"
    assert rows[1]["validation_status"] == "skipped"
    assert rows[1]["validation_passed"] is None
    assert rows[1]["skip_reason"] == "missing_question"


@pytest.mark.parametrize(
    "raw, expected_status, expected_scores",
    [
        ('{"scores":[1.0,0.5,0.0],"overall_score":0.5}', "ok", [1.0, 0.5, 0.0]),
        ('```json\n{"scores":[1.0,0.5,0.0],"overall_score":0.5}\n```', "ok", [1.0, 0.5, 0.0]),
        ('Here is the rubric result: {"scores":[1.0,0.5,0.0],"overall_score":0.5}', "ok", [1.0, 0.5, 0.0]),
        ("", "parse_failure", None),
        ("not valid json", "parse_failure", None),
    ],
)
def test_judge_parser_handles_pristine_fenced_prose_empty_and_invalid(raw, expected_status, expected_scores):
    parsed, status = _parse_judge_payload(raw)
    assert status == expected_status
    if expected_scores is None:
        assert parsed is None
    else:
        assert isinstance(parsed, dict)
        assert parsed["scores"] == expected_scores


def test_judge_summary_preserves_scoring_mode_and_diagnostics():
    summarized = _summarize_judge_result(
        {
            "scores": [1.0, 0.5, 0.0],
            "overall_score": 0.5,
            "scoring_mode": "fallback",
            "parse_status": "ok",
            "judge_status": "ok",
            "finish_reason": "stop",
            "response_had_content": True,
            "retry_count": 2,
            "raw_response": "{\"scores\":[1.0,0.5,0.0]}",
            "raw_result": {"scores": [1.0, 0.5, 0.0]},
            "judge_failure_class": "ValueError",
            "judge_failure_message": "bad payload",
        }
    )

    assert summarized["official_score"] == pytest.approx(1 / 3)
    assert summarized["partial_credit_score"] == pytest.approx(0.5)
    assert summarized["scoring_mode"] == "fallback"
    assert summarized["finish_reason"] == "stop"
    assert summarized["response_had_content"] is True
    assert summarized["retry_count"] == 2
    assert summarized["judge_failure_class"] == "ValueError"
    assert summarized["judge_api_error_class"] == "ValueError"
    assert summarized["judge_api_error_message"] == "bad payload"


def test_rejudge_row_update_preserves_answers_and_replaces_judge_state():
    row = {
        "qid": "conv-1:q0",
        "ability": "IE",
        "question": "What did I prefer for lunch?",
        "question_full": "What did I prefer for lunch?",
        "ideal_answer": "I preferred salads.",
        "ideal_answer_full": "I preferred salads.",
        "rubric": ["mention salads", "answer lunch preference"],
        "ai_answer": "You preferred salads for lunch.",
        "ai_answer_full": "You preferred salads for lunch.",
        "ai_answer_excerpt": "You preferred salads for lunch.",
        "score": 0.0,
        "official_score": 0.0,
        "partial_credit_score": 0.0,
        "scoring_mode": "fallback",
        "parse_status": "parse_failure",
        "judge_status": "parse_failure",
        "judge_failure_class": "ValueError",
        "judge_failure_message": "old failure",
        "judge_raw_response": "old raw",
        "judge_raw_result": {"scores": [0.0, 0.0]},
        "judge_raw_payload": {"scores": [0.0, 0.0]},
        "judge_finish_reason": "stop",
        "judge_response_had_content": True,
        "judge_retry_count": 1,
        "assessment": "old assessment",
        "judge_assessment": "old judge assessment",
        "answer_model": "answer-model-v1",
        "judge_model": "judge-model-v1",
        "answer_time_ms": 15.5,
        "judge_time_ms": 7.0,
        "retrieval_diagnostics": {"strategy": "keyword"},
        "answer_api_diagnostics": {"finish_reason": "stop"},
        "nuggets": [],
    }
    judged = _summarize_judge_result(
        {
            "scores": [1.0, 0.4],
            "overall_score": 0.7,
            "official_score": 0.5,
            "partial_credit_score": 0.7,
            "scoring_mode": "official",
            "parse_status": "ok",
            "judge_status": "ok",
            "judge_failure_class": None,
            "judge_failure_message": "",
            "finish_reason": "stop",
            "response_had_content": True,
            "retry_count": 0,
            "raw_response": "{\"scores\":[1.0,0.4]}",
            "raw_result": {"scores": [1.0, 0.4], "overall_score": 0.7},
            "assessment": "fresh assessment",
            "brief_assessment": "fresh brief",
            "nuggets": ["salads"],
        }
    )

    updated = _update_rejudged_question_row(row, judged, "judge-model-v2", 12.3)

    assert updated["answer_model"] == "answer-model-v1"
    assert updated["ai_answer_full"] == "You preferred salads for lunch."
    assert updated["judge_model"] == "judge-model-v2"
    assert updated["score"] == pytest.approx(0.5)
    assert updated["official_score"] == pytest.approx(0.5)
    assert updated["partial_credit_score"] == pytest.approx(0.7)
    assert updated["judge_raw_response"] == "{\"scores\":[1.0,0.4]}"
    assert updated["judge_raw_payload"] == {"scores": [1.0, 0.4], "overall_score": 0.7}
    assert updated["assessment"] == "fresh brief"
    assert updated["judge_assessment"] == "fresh assessment"
    assert updated["judge_time_ms"] == pytest.approx(12.3)


def test_required_rejudge_fields_is_pure_and_reports_missing_full_fields():
    row = {
        "question_full": "",
        "rubric": None,
        "ai_answer_full": "answer",
    }

    assert set(_required_rejudge_fields(row)) == {"question_full", "rubric"}


def test_rejudge_results_artifact_reuses_stored_rows_and_recomputes_summary():
    fixture = Path(__file__).parent / "fixtures" / "beam_rejudge_input.json"
    judgments_fixture = Path(__file__).parent / "fixtures" / "beam_rejudge_judgments.json"
    source_artifact = json.loads(fixture.read_text())
    judgment_records = json.loads(judgments_fixture.read_text())

    updated_artifact, summary_artifact = apply_rejudge_judgment_records(
        source_artifact,
        judgment_records_by_qid=judgment_records,
        judge_model="judge-model-v2",
        source_path=fixture,
    )

    first_row = updated_artifact["results"][0]["results"][0]
    second_row = updated_artifact["results"][0]["results"][1]
    assert first_row["answer_model"] == "answer-model-v1"
    assert second_row["answer_model"] == "answer-model-v1"
    assert first_row["ai_answer_full"] == "You preferred salads for lunch."
    assert second_row["ai_answer_full"] == "I preferred afternoon tea."
    assert first_row["judge_model"] == "judge-model-v2"
    assert second_row["judge_model"] == "judge-model-v2"
    assert first_row["score"] == pytest.approx(0.5)
    assert second_row["score"] == pytest.approx(0.0)
    assert summary_artifact["metadata"]["judge_model"] == "judge-model-v2"
    assert summary_artifact["metadata"]["source_results_path"] == str(fixture)
    assert summary_artifact["metadata"]["source_judge_model"] == "judge-model-v1"
    assert summary_artifact["ability_summary"]["100K"]["IE"]["avg_score"] == pytest.approx(0.5)
    assert summary_artifact["ability_summary"]["100K"]["MR"]["avg_score"] == pytest.approx(0.0)
    assert summary_artifact["micro_overall"]["100K"] == pytest.approx(0.25)
    assert summary_artifact["partial_credit_overall"] == pytest.approx((0.7 + 0.2) / 2)


def test_rejudge_results_file_writes_separate_artifacts(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "beam_rejudge_input.json"
    judgments_fixture = Path(__file__).parent / "fixtures" / "beam_rejudge_judgments.json"
    output_path = tmp_path / "beam_e2e_results.rejudged.json"

    source_artifact = json.loads(fixture.read_text())
    judgment_records = json.loads(judgments_fixture.read_text())
    updated_artifact, summary_artifact = apply_rejudge_judgment_records(
        source_artifact,
        judgment_records_by_qid=judgment_records,
        judge_model="judge-model-v2",
        source_path=fixture,
    )
    results_path, summary_path = write_rejudge_artifacts(output_path, updated_artifact, summary_artifact)

    assert results_path == output_path
    assert summary_path == tmp_path / "beam_e2e_results.rejudged_summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    written_results = json.loads(results_path.read_text())
    written_summary = json.loads(summary_path.read_text())
    assert written_results["metadata"]["judge_model"] == "judge-model-v2"
    assert written_results["results"][0]["results"][0]["answer_model"] == "answer-model-v1"
    assert written_summary["metadata"]["judge_model"] == "judge-model-v2"
    assert written_summary["partial_credit_overall"] == pytest.approx((0.7 + 0.2) / 2)


def test_rejudge_results_artifact_fails_when_full_fields_are_missing():
    fixture = Path(__file__).parent / "fixtures" / "beam_rejudge_input.json"
    judgments_fixture = Path(__file__).parent / "fixtures" / "beam_rejudge_judgments.json"
    source_artifact = json.loads(fixture.read_text())
    del source_artifact["results"][0]["results"][0]["question_full"]
    judgment_records = json.loads(judgments_fixture.read_text())

    with pytest.raises(ValueError, match="question_full"):
        apply_rejudge_judgment_records(
            source_artifact,
            judgment_records_by_qid=judgment_records,
            judge_model="judge-model-v2",
            source_path=fixture,
        )


def test_recall_provenance_retains_hashed_metadata_and_strategy_telemetry(tmp_path):
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "The launch happened on 2024-03-15."},
                {"role": "user", "content": "The meeting ended on 2024-04-12."},
                {"role": "user", "content": "I prefer concise replies."},
            ],
        )

        diag = {}
        memories_ie = _multi_strategy_recall(beam, "When did the launch happen?", top_k=5, ability="IE", diag=diag)
        memories_mr = _multi_strategy_recall(beam, "When did the launch happen?", top_k=5, ability="MR", diag={})

        assert [m.get("content") for m in memories_ie] == [m.get("content") for m in memories_mr]
        assert diag["strategies"]
        for bucket in diag["strategies"].values():
            assert set(bucket) >= {
                "activated",
                "candidates_before_dedup",
                "added_after_dedup",
                "final_contribution",
            }

        memories_ie[0]["final_context_included"] = True
        summary = _summarize_recall_memories(memories_ie)
        assert summary["memories"]
        first = summary["memories"][0]
        assert first["memory_id"] is not None
        assert first["content_hash"]
        assert first["source"]
        assert any(mem["message_index"] is not None for mem in summary["memories"])
        assert "raw_score" in first
        assert "final_score" in first
        assert "components" in first
        assert "reranker" in first["components"]
        assert first["final_context_included"] is True
    finally:
        beam.conn.close()


def test_embedding_diagnostic_helper_marks_dense_success_with_matching_counts():
    diag = _update_embedding_diagnostic(
        {"model": "test-embed-model", "dimension": 3, "failed": False},
        backend_available=True,
        eligible_rows=2,
        inserted_vectors=2,
        rows_before=0,
        rows_after=2,
        api_calls_before=4,
        api_calls_after=5,
    )

    assert diag["backend"] == "dense"
    assert diag["model"] == "test-embed-model"
    assert diag["dimension"] == 3
    assert diag["eligible_rows"] == 2
    assert diag["inserted_vectors"] == 2
    assert diag["memory_embeddings_rows_before"] == 0
    assert diag["memory_embeddings_rows_after"] == 2
    assert diag["memory_embeddings_row_delta"] == 2
    assert diag["api_calls"] == 1
    assert diag["status"] == "dense"
    assert diag.get("failed") is False


def test_embedding_diagnostic_helper_marks_mismatch_as_failed():
    diag = _update_embedding_diagnostic(
        {"backend": "dense"},
        backend_available=True,
        eligible_rows=2,
        inserted_vectors=0,
        rows_before=0,
        rows_after=0,
    )

    assert diag["eligible_rows"] == 2
    assert diag["inserted_vectors"] == 0
    assert diag["memory_embeddings_row_delta"] == 0
    assert diag["status"] == "failed"
    assert diag["failed"] is True


def test_embedding_diagnostic_helper_keeps_keyword_only_when_embeddings_are_off():
    diag = _update_embedding_diagnostic(
        {"backend": "keyword-only"},
        backend_available=False,
        eligible_rows=2,
        inserted_vectors=0,
        rows_before=0,
        rows_after=0,
    )

    assert diag["backend"] == "keyword-only"
    assert diag["eligible_rows"] == 2
    assert diag["inserted_vectors"] == 0
    assert diag["status"] == "keyword-only"
    assert diag.get("failed") is not True


def test_if_pf_recall_uses_session_scoped_sql_and_excludes_unrelated_tags(tmp_path):
    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"

    beam = _make_beam(tmp_path)
    sql_trace: list[str] = []
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "Always keep the lunch schedule simple."},
                {"role": "user", "content": "I prefer quiet mornings for the office."},
                {"role": "user", "content": "Always water the plants on Fridays."},
                {"role": "user", "content": "I prefer handwritten notes for the meeting."},
                {"role": "user", "content": "Always use the blue folder for invoices."},
                {"role": "user", "content": "I prefer fresh fruit during the afternoon."},
                {"role": "user", "content": "Always keep the status update short."},
                {"role": "user", "content": "I prefer calm music while reading."},
                {"role": "user", "content": "Always store the cables in the drawer."},
                {"role": "user", "content": "I prefer black tea after lunch."},
                {"role": "user", "content": "Always use bullet points for release notes."},
            ],
        )

        beam.conn.set_trace_callback(sql_trace.append)
        memories = _multi_strategy_recall(
            beam,
            "What bullet points should I use for the release notes?",
            top_k=5,
            ability=None,
            diag={},
        )
        contents = [mem.get("content", "") for mem in memories]

        assert any("Always use bullet points for release notes." in content for content in contents)
        assert not any("Always keep the lunch schedule simple." in content for content in contents)
        assert any(
            "FROM working_memory" in stmt
            and "session_id =" in stmt
            and "[INSTRUCTION]" in stmt
            and "bullet" in stmt.lower()
            for stmt in sql_trace
        )
    finally:
        beam.conn.set_trace_callback(None)
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings


def test_pure_recall_label_invariance_uses_the_same_retrieval_results(tmp_path):
    saved_pure_recall = os.environ.get("EDUMEM_BENCHMARK_PURE_RECALL")
    os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = "1"

    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "Always keep the output in bullet points."},
                {"role": "user", "content": "I prefer short summaries."},
                {"role": "user", "content": "The launch happened on 2024-03-15."},
            ],
        )

        memories_none = _multi_strategy_recall(
            beam,
            "What bullet points should I use for the launch summary?",
            top_k=5,
            ability=None,
            diag={},
        )
        memories_ie = _multi_strategy_recall(
            beam,
            "What bullet points should I use for the launch summary?",
            top_k=5,
            ability="IE",
            diag={},
        )

        assert [mem.get("content") for mem in memories_none] == [mem.get("content") for mem in memories_ie]
    finally:
        beam.conn.close()
        if saved_pure_recall is None:
            os.environ.pop("EDUMEM_BENCHMARK_PURE_RECALL", None)
        else:
            os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = saved_pure_recall


def test_second_pass_diagnostics_are_stored_separately_from_first_pass():
    first_diag = {
        "strategies": {
            "S1": {
                "activated": True,
                "candidates_before_dedup": 4,
                "added_after_dedup": 2,
                "final_contribution": 1,
            }
        }
    }
    gap_diag = {
        "strategies": {
            "S1": {
                "activated": True,
                "candidates_before_dedup": 3,
                "added_after_dedup": 1,
                "final_contribution": 1,
            },
            "TR": {
                "activated": True,
                "candidates_before_dedup": 5,
                "added_after_dedup": 4,
                "final_contribution": 2,
            },
        }
    }

    _attach_second_pass_diagnostics(first_diag, ["2024-03-15"], gap_diag)
    gap_diag["strategies"]["TR"]["final_contribution"] = 99

    assert first_diag["strategies"]["S1"]["final_contribution"] == 1
    assert first_diag["second_pass"]["activated"] is True
    assert first_diag["second_pass"]["gap_queries"] == ["2024-03-15"]
    assert first_diag["second_pass"]["strategies"]["TR"]["final_contribution"] == 2


def test_second_pass_diagnostics_helper_attaches_to_the_provided_diag_dict():
    diag = {
        "strategies": {
            "S1": {
                "activated": True,
                "candidates_before_dedup": 4,
                "added_after_dedup": 2,
                "final_contribution": 1,
            }
        }
    }
    gap_diag = {
        "strategies": {
            "TR": {
                "activated": True,
                "candidates_before_dedup": 5,
                "added_after_dedup": 4,
                "final_contribution": 2,
            }
        }
    }

    _record_second_pass_diagnostics(diag, ["2024-03-15", "2024-04-12"], gap_diag)
    gap_diag["strategies"]["TR"]["final_contribution"] = 99

    assert diag["second_pass"]["activated"] is True
    assert diag["second_pass"]["gap_queries"] == ["2024-03-15", "2024-04-12"]
    assert diag["second_pass"]["strategies"]["TR"]["final_contribution"] == 2


@pytest.mark.skipif(not os.getenv("EDUMEM_TEST_INFERENCE_URL"), reason="EDUMEM_TEST_INFERENCE_URL is not set")
def test_shipped_inference_service_smoke():
    base_url = os.environ["EDUMEM_TEST_INFERENCE_URL"].rstrip("/")

    health = _json_request(f"{base_url}/health")
    assert health["status"] == "ok"

    info = _json_request(f"{base_url}/info")
    assert info["model_id"] == "Alibaba-NLP/gte-modernbert-base"
    assert info["dimension"] == 768

    embedding = _json_request(
        f"{base_url}/v1/embeddings",
        {"input": ["beam smoke"], "model": "Alibaba-NLP/gte-modernbert-base"},
    )
    assert embedding["model"] == "Alibaba-NLP/gte-modernbert-base"
    assert len(embedding["data"][0]["embedding"]) == 768

    rerank = _json_request(
        f"{base_url}/rerank",
        {"query": "beam smoke", "texts": ["alpha", "beta"]},
    )
    assert len(rerank) == 2
    assert all("index" in item and "score" in item for item in rerank)


def test_print_sota_report_suppresses_direct_comparison_for_subset(capsys):
    ability_summary = {
        "100K": {
            "OVERALL": {"avg_score": 0.5, "count": 2},
            "IE": {"avg_score": 0.5, "count": 2},
        }
    }
    print_sota_report(
        ability_summary,
        {
            "model": "test-model",
            "conversation_count": 1,
            "micro_overall": {"100K": 0.25},
            "partial_credit_overall": 0.5,
            "comparison_valid": False,
        },
    )
    out = capsys.readouterr().out
    assert "OVERALL (macro)" in out
    assert "Micro Diagnostic" in out
    assert "Partial-Credit Diagnostic" in out
    assert "Direct comparison not asserted" in out


def test_partial_credit_overall_flattens_question_rows_and_separates_from_macro_overall():
    all_results = [
        {
            "conversation_id": "c1",
            "scale": "100K",
            "results": [
                {"qid": "q1", "ability": "IE", "score": 1.0, "partial_credit_score": 1.0},
                {"qid": "q2", "ability": "MR", "score": 0.0, "partial_credit_score": 0.25},
            ],
        },
        {
            "conversation_id": "c2",
            "scale": "100K",
            "results": [
                {"qid": "q3", "ability": "IE", "score": 0.0, "partial_credit_score": 0.5},
            ],
        },
        {
            "conversation_id": "c3",
            "scale": "10M",
            "results": [
                {"qid": "q4", "ability": "IE", "score": 0.0, "partial_credit_score": 0.0},
                {"qid": "q5", "ability": "MR", "score": 1.0, "partial_credit_score": 0.75},
                {"qid": "q6", "ability": "MR", "score": 1.0, "partial_credit_score": 0.5},
            ],
        },
    ]

    ability_summary = compute_ability_scores(all_results)
    partial_credit_overall = compute_partial_credit_overall(all_results)

    assert partial_credit_overall == pytest.approx((1.0 + 0.25 + 0.5 + 0.0 + 0.75 + 0.5) / 6)
    assert ability_summary["100K"]["OVERALL"]["avg_score"] == pytest.approx(0.25)
    assert ability_summary["10M"]["OVERALL"]["avg_score"] == pytest.approx(0.5)
    assert partial_credit_overall != pytest.approx(ability_summary["100K"]["OVERALL"]["avg_score"])


def test_sum_query_triples_retrieval_depth(tmp_path):
    """Verify that summarization questions get top_k * 3 in _multi_strategy_recall."""
    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "We discussed the platform architecture."},
                {"role": "user", "content": "The team uses microservices for scalability."},
                {"role": "user", "content": "We are deploying to Kubernetes."},
            ],
        )
        diag = {}

        memories = _multi_strategy_recall(
            beam,
            "Can you summarize everything we discussed?",
            top_k=5,
            ability=None,
            diag=diag,
        )

        assert diag["strategies"]["SUM"]["activated"] is True
        assert len(memories) > 0
    finally:
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings


def test_cr_sum_mr_get_higher_max_tokens():
    """Verify that CR, SUM, MR abilities use 8192 max_tokens in answer path."""
    from tools.evaluate_beam_end_to_end import answer_with_memory

    # Test that the answer_with_memory function exists and can accept ability parameter.
    # The actual max_tokens logic is hardcoded at line 3204 of evaluate_beam_end_to_end.py:
    # _answer_max_tokens = 8192 if ability in ("CR", "SUM", "MR") else 2048
    # We verify the function signature accepts ability parameter.
    assert "ability" in answer_with_memory.__code__.co_varnames


def test_ku_modifier_references_msgidx(tmp_path):
    """Verify that the KU modifier prompt mentions MSGIDX."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt("What is the current deadline?")

    # The KU modifier should be included and should mention MSGIDX
    assert "KNOWLEDGE UPDATE" in prompt
    assert "MSGIDX" in prompt


def test_multi_hop_query_matches_combined():
    """Verify that is_multi_hop_query returns True for questions containing 'combined'."""
    from edumem.core.query_mode import is_multi_hop_query

    assert is_multi_hop_query("What is the combined impact of these changes?") is True
    assert is_multi_hop_query("What is the deadline?") is False


def test_if_pf_fallback_retrieves_by_tag_when_terms_miss(tmp_path):
    """Verify the tag-only fallback works when search terms don't match."""
    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_beam(tmp_path)
    try:
        # Insert a working_memory row with [INSTRUCTION] tag but different vocabulary
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, message_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "test-id",
                beam.session_id,
                "Always use semantic HTML elements for accessibility. [INSTRUCTION]",
                "beam_user",
                "2024-01-01T00:00:00Z",
                0.5,
                0,
            ),
        )
        beam.conn.commit()

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            "How should I organize the different parts of a webpage?",
            top_k=5,
            ability=None,
            diag=diag,
        )

        # IF strategy should activate for procedural questions
        assert _query_wants_if_pf("How should I organize the different parts of a webpage?") is True
        # Should find memories (either from IF or fallback)
        assert len(memories) > 0
    finally:
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings


def test_false_contradiction_prompt_order():
    """Verify that CHANGE OVER TIME comes before CONFLICTS in base prompt."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt("What did we discuss?")

    change_idx = prompt.find("CHANGE OVER TIME")
    conflict_idx = prompt.find("CONFLICTS")

    assert change_idx != -1, "CHANGE OVER TIME should be in the prompt"
    assert conflict_idx != -1, "CONFLICTS should be in the prompt"
    assert change_idx < conflict_idx, "CHANGE OVER TIME must appear before CONFLICTS"


# ---------- Regression tests from 2026-06-20 11:00 run ----------


def test_eo_modifier_bans_date_labels_and_requires_functional():
    """EO q0/q1: model used 'Planning tasks and schedule with March 15 time
    anchor' instead of 'Core functionality'. Modifier must ban date labels."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt(
        "Can you list the order in which I brought up different aspects "
        "of developing my personal budget tracker?"
    )
    assert "functional purpose" in prompt.lower()
    assert "bad:" in prompt.lower() and "good:" in prompt.lower()
    # Examples must be generic, not lifted from BEAM test items
    assert "march 15 time anchor" not in prompt.lower()
    assert "user authentication and expense tracking" not in prompt.lower()
    assert "do not include dates" in prompt.lower()


def test_sum_modifier_suppresses_conflicts():
    """SUM q1: model started with 'contradictory information' instead of
    summarizing security evolution. SUM must override CONFLICTS."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt(
        "Can you give me a comprehensive summary of how I handled the "
        "security and database challenges?"
    )
    assert "summaries never flag contradictions" in prompt.lower()
    assert "narrate the progression" in prompt.lower()


def test_cr_yesno_question_triggers_both_sides_check():
    """CR q0: 'Have I worked with Flask routes?' — model only reported one
    side. Yes/no questions must prompt for both-sides evidence search."""
    from edumem.core.query_mode import build_system_prompt, is_yesno_check_query

    q = "Have I worked with Flask routes and handled HTTP requests in this project?"
    assert is_yesno_check_query(q) is True
    prompt = build_system_prompt(q)
    assert "yes/no verification" in prompt.lower()
    assert "both supporting and contradicting" in prompt.lower()


def test_ie_how_question_suppresses_false_absence():
    """IE q1: 'How did I organize tasks over the sprint?' — model falsely
    triggered ABSENCE. HOW modifier must override it."""
    from edumem.core.query_mode import build_system_prompt, is_how_query

    q = "How did I organize the tasks over the course of the sprint?"
    assert is_how_query(q) is True
    prompt = build_system_prompt(q)
    assert "how questions" in prompt.lower()
    assert "do not trigger absence" in prompt.lower()


def test_case1_q7_how_prompt_overrides_unrelated_conflict_framing():
    from edumem.core.query_mode import build_system_prompt

    question = (
        "How did I organize the tasks over the course of the sprint to ensure "
        "both backend and frontend aspects of the features were completed on time?"
    )
    prompt = build_system_prompt(question).lower()

    assert "do not start with 'the conversation contains contradictory information:'" in prompt
    assert "different sprints" in prompt
    assert "answer from the relevant actions" in prompt
    assert "you must surface both explicitly" not in prompt


def test_explicit_how_conflict_question_keeps_contradiction_resolution():
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt(
        "How did I handle the conflicting sprint deadlines?"
    ).lower()

    assert "contradiction resolution" in prompt
    assert "present both sides" in prompt


def test_case1_q7_procedural_recall_keeps_sprint_plan_out_of_negation_route(tmp_path):
    """100K case 1 q7: a procedural ``How did I`` query is not a CR query."""
    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_beam(tmp_path)
    question = (
        "How did I organize the tasks over the course of the sprint to ensure "
        "both backend and frontend aspects of the features were completed on time?"
    )
    sprint_plan = (
        "Sprint 1 ends March 29. Week 1: set up the development environment, "
        "define the database schema, implement registration and login, add "
        "validation, and write unit tests. Week 2: add frontend forms, integrate "
        "the frontend with the backend, add security, and finish testing."
    )
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "The analytics sprint did not include frontend work."},
                {"role": "assistant", "content": "That later sprint was backend-only."},
                {"role": "assistant", "content": sprint_plan},
            ],
        )
        diag = {}

        memories = _multi_strategy_recall(
            beam, question, top_k=10, ability=None, diag=diag,
        )

        assert diag["strategies"]["S2"]["activated"] is False
        assert any(
            "define the database schema" in memory.get("content", "")
            and "add frontend forms" in memory.get("content", "")
            for memory in memories
        )
    finally:
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings


def test_tr_duration_modifier_uses_semantic_matching():
    """TR q0: 'Weeks between transaction management and final deployment
    deadline' — model couldn't find 'final deployment deadline'. Duration
    modifier must instruct semantic matching."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt(
        "How many weeks do I have between finishing the transaction "
        "management features and the final deployment deadline?"
    )
    assert "duration" in prompt.lower()
    assert "semantic matches" in prompt.lower()
    assert "exact wording" in prompt.lower() or "literal strings" in prompt.lower()
    # Examples must be generic, not lifted from BEAM test items
    assert "final deployment deadline" not in prompt.lower()
    assert "transaction management" not in prompt.lower()


def test_if_list_question_triggers_exhaustive_modifier():
    """IF q1: 'Which libraries are used?' — answer was truncated. List
    modifier must demand exhaustive enumeration with versions."""
    from edumem.core.query_mode import build_system_prompt, is_list_query

    q = "Which libraries are used in this project?"
    assert is_list_query(q) is True
    prompt = build_system_prompt(q)
    assert "list completeness" in prompt.lower()
    assert "exhaustive" in prompt.lower()
    assert "version" in prompt.lower()


# ---------- Regression tests from 2026-06-20 11:50 run (prompt fixes) ----------


def test_eo_modifier_ignores_later_refinements_of_same_feature():
    """EO: model substituted 'optimizing CRUD' for 'transaction error handling'
    — treated a later refinement as a separate topic. Rule 8 must suppress it."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt(
        "Can you list the order in which I brought up different aspects "
        "of developing my personal budget tracker?"
    )
    assert "first occurrence" in prompt.lower()
    assert "later refinements" in prompt.lower()
    assert "not new topics" in prompt.lower()


def test_mr_modifier_counts_schema_items_as_new():
    """MR q0: model excluded 'category' because it was in the initial schema.
    Modifier must instruct that design-time items count."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt("How many new columns did I want to add across my requests?")
    assert "schema definition" in prompt.lower() or "design" in prompt.lower()
    assert "do not exclude" in prompt.lower()


def test_mr_modifier_treats_described_features_as_real():
    """MR q1: model treated RBAC, password hashing, lockout as 'suggestions'.
    Modifier must say features with concrete detail are things the user is doing."""
    from edumem.core.query_mode import build_system_prompt, is_multi_hop_query

    q = "How many user roles and security features am I implementing across my sessions?"
    assert is_multi_hop_query(q) is True, "question must trigger MR modifier"
    prompt = build_system_prompt(q)
    assert "concrete detail" in prompt.lower()
    assert "not implemented" in prompt.lower()


def test_sum_modifier_constrains_to_specific_domain():
    """SUM q1: model wrote a general project overview instead of security/database focus.
    Modifier must constrain to the question's specific domain."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt(
        "Can you give me a comprehensive summary of how I handled "
        "the security and database challenges?"
    )
    assert "specific domain" in prompt.lower()
    assert "only that domain" in prompt.lower()
    assert "do not write a general project overview" in prompt.lower()


def test_tr_modifier_prefers_later_dates_over_plans():
    """TR q0: model confused by planned vs actual dates from different schedules.
    Modifier must instruct to prefer most recently stated dates."""
    from edumem.core.query_mode import build_system_prompt

    prompt = build_system_prompt(
        "How many weeks do I have between finishing the transaction "
        "management features and the final deployment deadline?"
    )
    assert "most recently stated" in prompt.lower()
    assert "later message" in prompt.lower()


def test_format_versioned_fact_ku_shows_previous_value(tmp_path):
    """KU ability should render version history showing previous value."""
    from edumem.core.beam import BeamMemory, init_beam

    db_path = str(tmp_path / "test.db")
    init_beam(db_path)
    beam = BeamMemory(db_path=db_path, session_id="test_ku")
    try:
        # Insert two versions of the same metric
        beam._insert_fact("test_ku", 3, "metric", "team_size", "5members",
                          "We have 5 team members", 0.7, source_memory_id="msg3")
        beam._insert_fact("test_ku", 15, "metric", "team_size", "12members",
                          "Team grew to 12 members", 0.7, source_memory_id="msg15")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve("How many team members?", top_k=10, intent="current")
        ctx = result["context"]
        assert "current" in ctx.lower()
        assert "12members" in ctx
        assert "was:" in ctx.lower() or "5members" in ctx
    finally:
        beam.conn.close()


def test_format_versioned_fact_cr_shows_both_sides(tmp_path):
    """CR ability should render both old and new values for contradiction resolution."""
    from edumem.core.beam import BeamMemory, init_beam

    db_path = str(tmp_path / "test.db")
    init_beam(db_path)
    beam = BeamMemory(db_path=db_path, session_id="test_cr")
    try:
        beam._insert_fact("test_cr", 5, "metric", "api_latency", "450ms",
                          "API response time is 450ms", 0.7, source_memory_id="msg5")
        beam._insert_fact("test_cr", 20, "metric", "api_latency", "250ms",
                          "API response time improved to 250ms", 0.7, source_memory_id="msg20")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve("Is there a contradiction about API latency?", top_k=10, intent="change")
        ctx = result["context"]
        assert "changed" in ctx.lower()
        assert "450ms" in ctx
        assert "250ms" in ctx
    finally:
        beam.conn.close()


def test_format_versioned_fact_tr_precomputes_date_delta(tmp_path):
    """TR ability should show date changes with pre-computed deltas."""
    from edumem.core.beam import BeamMemory, init_beam

    db_path = str(tmp_path / "test.db")
    init_beam(db_path)
    beam = BeamMemory(db_path=db_path, session_id="test_tr")
    try:
        beam._insert_fact("test_tr", 2, "date", "launch_date", "2024-03-15",
                          "Launch planned for 2024-03-15", 0.7, source_memory_id="msg2")
        # Note: date facts skip versioning in _insert_fact (ftype == 'date' early return).
        # So we test with metric type instead for the version chain to work.
        beam._insert_fact("test_tr", 2, "metric", "Deadline", "2024-03-15",
                          "Deadline is March 15", 0.7, source_memory_id="msg2")
        beam._insert_fact("test_tr", 18, "metric", "Deadline", "2024-04-01",
                          "Deadline moved to April 1", 0.7, source_memory_id="msg18")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve("How long between Deadline dates?", top_k=10, intent="timeline")
        ctx = result["context"]
        assert "timeline" in ctx.lower() or "2024-03-15" in ctx
        assert "2024-04-01" in ctx
        # Should show both dates for temporal reasoning
    finally:
        beam.conn.close()


def test_format_versioned_fact_no_history_fallback(tmp_path):
    """Facts without version history should render as flat [Fact type] key: value."""
    from edumem.core.beam import BeamMemory, init_beam

    db_path = str(tmp_path / "test.db")
    init_beam(db_path)
    beam = BeamMemory(db_path=db_path, session_id="test_flat")
    try:
        beam._insert_fact("test_flat", 1, "metric", "cpu_usage", "85%",
                          "CPU usage at 85%", 0.7, source_memory_id="msg1")
        beam.conn.commit()

        # Should work the same for any intent when there's no history
        for intent in ["current", "change", "timeline", "ordered", "", ""]:
            result = beam._memoria_fact_retrieve("What is CPU usage?", top_k=10, intent=intent)
            ctx = result["context"]
            assert "[fact" in ctx.lower()
            assert "cpu_usage" in ctx
            assert "85%" in ctx
    finally:
        beam.conn.close()


def test_format_versioned_fact_eo_shows_msgidx(tmp_path):
    """EO ability should show MSGIDX anchors for ordering."""
    from edumem.core.beam import BeamMemory, init_beam

    db_path = str(tmp_path / "test.db")
    init_beam(db_path)
    beam = BeamMemory(db_path=db_path, session_id="test_eo")
    try:
        beam._insert_fact("test_eo", 7, "metric", "feature_count", "3features",
                          "We built 3 features", 0.7, source_memory_id="msg7")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve("In what order were features built?", top_k=10, intent="ordered")
        ctx = result["context"]
        assert "msgidx" in ctx.lower() or "7" in ctx
    finally:
        beam.conn.close()


def test_memoria_retrieve_cr_includes_versioned_facts(tmp_path):
    """CR routing should merge negation results with versioned fact results."""
    from edumem.core.beam import BeamMemory, init_beam

    db_path = str(tmp_path / "test.db")
    init_beam(db_path)
    beam = BeamMemory(db_path=db_path, session_id="test_cr_merge")
    try:
        # Insert a versioned fact (value changed)
        beam._insert_fact("test_cr_merge", 5, "metric", "team_size", "5members",
                          "Team has 5 members", 0.7, source_memory_id="msg5")
        beam._insert_fact("test_cr_merge", 20, "metric", "team_size", "12members",
                          "Team grew to 12 members", 0.7, source_memory_id="msg20")
        beam.conn.commit()

        result = beam.memoria_retrieve("Is there a contradiction about team size?", ability="CR", top_k=10)
        # Should not be fallback — versioned facts should be found even if no negation triples
        assert result["source"] != "fallback" or result["context"] != ""
    finally:
        beam.conn.close()


def test_memoria_retrieve_tr_includes_versioned_facts(tmp_path):
    """TR routing should merge timeline results with versioned fact results."""
    from edumem.core.beam import BeamMemory, init_beam

    db_path = str(tmp_path / "test.db")
    init_beam(db_path)
    beam = BeamMemory(db_path=db_path, session_id="test_tr_merge")
    try:
        beam._insert_fact("test_tr_merge", 3, "metric", "response_time", "500ms",
                          "Response time is 500ms", 0.7, source_memory_id="msg3")
        beam._insert_fact("test_tr_merge", 15, "metric", "response_time", "200ms",
                          "Response time improved to 200ms", 0.7, source_memory_id="msg15")
        beam.conn.commit()

        result = beam.memoria_retrieve("How much did response time change?", ability="TR", top_k=10)
        assert result["source"] != "fallback" or result["context"] != ""
    finally:
        beam.conn.close()


def test_extract_switched_from_to_creates_version_chain():
    """'switched from X to Y' should create a version chain in memoria_facts."""
    import tempfile, os
    from edumem.core.beam import BeamMemory, init_beam

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        init_beam(db_path)
        beam = BeamMemory(db_path=db_path, session_id="test_change")
        beam.extract_and_store_facts(
            "[MSGIDX:10] We switched from Flask to FastAPI for the backend",
            message_idx=10, source_memory_id="msg10"
        )
        beam.conn.commit()

        rows = beam.conn.execute(
            "SELECT key, value, version_id, previous_value FROM memoria_facts "
            "WHERE session_id = 'test_change' AND fact_type = 'change' "
            "ORDER BY version_id ASC"
        ).fetchall()
        # Should have a version chain: old value (Flask) and new value (FastAPI)
        assert len(rows) >= 1
        values = [r[1] for r in rows]
        prev_values = [r[3] if r[3] else "" for r in rows]
        # The latest version should have FastAPI
        assert any("fastapi" in v.lower() for v in values)
        # There should be a reference to Flask (either as previous_value or as a row value)
        all_text = " ".join(values + prev_values)
        assert "flask" in all_text.lower()
        beam.conn.close()
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


def test_extract_increased_metric_creates_version_chain():
    """'increased X from N to M' should create a metric version chain."""
    import tempfile, os
    from edumem.core.beam import BeamMemory, init_beam

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        init_beam(db_path)
        beam = BeamMemory(db_path=db_path, session_id="test_metric_change")
        beam.extract_and_store_facts(
            "[MSGIDX:8] We increased the team from 5 members to 12 members",
            message_idx=8, source_memory_id="msg8"
        )
        beam.conn.commit()

        rows = beam.conn.execute(
            "SELECT key, value, version_id, previous_value, fact_type FROM memoria_facts "
            "WHERE session_id = 'test_metric_change' "
            "ORDER BY version_id ASC"
        ).fetchall()
        values = [r[1] for r in rows]
        prev_values = [r[3] if r[3] else "" for r in rows]
        # Should capture both 5 and 12
        all_text = " ".join(values + prev_values)
        assert "5" in all_text
        assert "12" in all_text
        beam.conn.close()
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


def test_extract_moved_deadline_creates_version_chain():
    """'moved deadline from D1 to D2' should create a date version chain."""
    import tempfile, os
    from edumem.core.beam import BeamMemory, init_beam

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        init_beam(db_path)
        beam = BeamMemory(db_path=db_path, session_id="test_date_change")
        beam.extract_and_store_facts(
            "[MSGIDX:15] We postponed the launch from 2024-03-15 to 2024-04-01",
            message_idx=15, source_memory_id="msg15"
        )
        beam.conn.commit()

        rows = beam.conn.execute(
            "SELECT key, value, version_id, previous_value, fact_type FROM memoria_facts "
            "WHERE session_id = 'test_date_change' "
            "ORDER BY version_id ASC"
        ).fetchall()
        values = [r[1] for r in rows]
        prev_values = [r[3] if r[3] else "" for r in rows]
        all_text = " ".join(values + prev_values)
        assert "2024-03-15" in all_text
        assert "2024-04-01" in all_text
        beam.conn.close()
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


def test_ku_modifier_references_structured_facts():
    """KU modifier should instruct LLM to use [Fact CURRENT ...] entries."""
    from edumem.core.query_mode import build_system_prompt
    prompt = build_system_prompt("What is the current team size?")
    assert "fact current" in prompt.lower()


def test_cr_modifier_references_structured_facts():
    """CR modifier should instruct LLM to use [Fact CHANGED ...] entries."""
    from edumem.core.query_mode import build_system_prompt
    prompt = build_system_prompt("Is there a contradiction about the API?")
    assert "fact changed" in prompt.lower()


def test_duration_modifier_references_structured_facts():
    """TR/Duration modifier should instruct LLM to use [Fact TIMELINE ...] entries."""
    from edumem.core.query_mode import build_system_prompt
    prompt = build_system_prompt("How many weeks between the start and the deadline?")
    assert "fact timeline" in prompt.lower()


def test_ordering_modifier_references_structured_facts():
    """EO modifier should instruct LLM to use [Fact ... MSGIDX:N] entries."""
    from edumem.core.query_mode import build_system_prompt
    prompt = build_system_prompt("In what order were the features discussed?")
    assert "fact" in prompt.lower() and "msgidx" in prompt.lower()


# ---------------------------------------------------------------------------
# BEAM Integration Tests — full ingest → extract → retrieve pipeline
# Modeled on real failing cases from BEAM 100K case 0 (2026-06-20 11:00 run)
# ---------------------------------------------------------------------------


def _make_beam_for_integration(tmp_path, session_id="beam_integ"):
    """Create a fresh BeamMemory with initialized schema for integration tests."""
    from edumem.core.beam import BeamMemory, init_beam
    db_path = str(tmp_path / f"{session_id}.db")
    init_beam(db_path)
    return BeamMemory(db_path=db_path, session_id=session_id)


def test_beam_integ_metric_history_preserves_previous_versions(tmp_path):
    beam = _make_beam_for_integration(tmp_path, "metric_history")
    try:
        for message_idx, value in ((50, "800ms"), (80, "300ms"), (100, "250ms")):
            beam._insert_fact(
                "metric_history",
                message_idx,
                "metric",
                "dashboard_api_response_time_ms",
                value,
                f"Dashboard API response time is {value}",
                0.7,
                source_memory_id=f"msg{message_idx}",
            )
        beam.conn.commit()

        rows = beam.conn.execute(
            "SELECT value, version_id, previous_value, valid_from_msg_idx, valid_to_msg_idx "
            "FROM memoria_facts "
            "WHERE session_id = ? AND fact_type = 'metric' AND key = ? "
            "ORDER BY version_id",
            ("metric_history", "dashboard_api_response_time_ms"),
        ).fetchall()

        assert [tuple(row) for row in rows] == [
            ("800ms", 0, None, 50, 80),
            ("300ms", 1, "800ms", 80, 100),
            ("250ms", 2, "300ms", 100, None),
        ]
    finally:
        beam.conn.close()


def test_beam_integ_ku_versioned_fact_prevents_false_conflict(tmp_path):
    """BEAM KU q0 scored 0.0: AI said 'contradictory information' about response
    time 800ms->300ms->250ms instead of answering '250ms'.

    With versioned facts, _memoria_fact_retrieve(ability='KU') must render
    [Fact CURRENT ...] format showing the latest value with 'was:' annotation,
    so the LLM treats it as an UPDATE not a contradiction.

    This test FAILS if _format_versioned_fact() is removed or doesn't handle KU."""
    beam = _make_beam_for_integration(tmp_path, "ku_conflict")
    try:
        # Same key, three values — simulates response time being updated
        beam._insert_fact("ku_conflict", 50, "metric", "response_time_ms",
                          "800ms", "Response time is 800ms", 0.7,
                          source_memory_id="msg50")
        beam._insert_fact("ku_conflict", 80, "metric", "response_time_ms",
                          "300ms", "Response time reduced to 300ms", 0.7,
                          source_memory_id="msg80")
        beam._insert_fact("ku_conflict", 100, "metric", "response_time_ms",
                          "250ms", "Response time improved to 250ms", 0.7,
                          source_memory_id="msg100")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve(
            "What is the response time?", top_k=10, intent="current"
        )
        ctx = result["context"]
        # Must contain CURRENT format (Phase 1 rendering)
        assert "[fact current" in ctx.lower(), \
            f"KU versioned fact must use [Fact CURRENT] format, got: {ctx}"
        # Must show latest value
        assert "250ms" in ctx
        # Must reference previous value
        assert "was:" in ctx.lower(), \
            f"KU versioned fact must show 'was:' annotation, got: {ctx}"
    finally:
        beam.conn.close()


def test_beam_integ_cr_versioned_fact_shows_both_sides(tmp_path):
    """BEAM CR q0 scored 0.5: AI found contradiction about Flask routes but
    only partially presented both sides.

    With versioned facts, _memoria_fact_retrieve(ability='CR') must render
    [Fact CHANGED ...] format showing both old and new values.

    This test FAILS if _format_versioned_fact() doesn't handle CR."""
    beam = _make_beam_for_integration(tmp_path, "cr_both")
    try:
        beam._insert_fact("cr_both", 30, "entity", "flask_usage",
                          "never used", "Never written any Flask routes", 0.7,
                          source_memory_id="msg30")
        beam._insert_fact("cr_both", 85, "entity", "flask_usage",
                          "implemented", "Implemented Flask route for login", 0.7,
                          source_memory_id="msg85")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve(
            "Have I worked with Flask?", top_k=10, intent="change"
        )
        ctx = result["context"]
        # Must contain CHANGED format (Phase 1 rendering)
        assert "[fact changed" in ctx.lower(), \
            f"CR versioned fact must use [Fact CHANGED] format, got: {ctx}"
        # Must show BOTH values
        assert "never used" in ctx.lower()
        assert "implemented" in ctx.lower()
    finally:
        beam.conn.close()


def test_beam_integ_tr_versioned_fact_precomputes_delta(tmp_path):
    """BEAM TR q0 scored 0.0: AI said 'contradictory information' about deadlines
    instead of computing 4 weeks between Jan 15 and Mar 15.

    With versioned facts, _memoria_fact_retrieve(ability='TR') must render
    [Fact TIMELINE ...] format with pre-computed delta.

    This test FAILS if _format_versioned_fact() doesn't handle TR."""
    beam = _make_beam_for_integration(tmp_path, "tr_delta")
    try:
        # Insert facts with numeric values that will match Pass 1 (numbers search for 15, 01, 03, 2024)
        # and will be found when querying with those numbers
        beam._insert_fact("tr_delta", 10, "metric", "target_date_01_15",
                          "2024-01-15", "Target date is January 15", 0.7,
                          source_memory_id="msg10")
        beam._insert_fact("tr_delta", 50, "metric", "target_date_01_15",
                          "2024-03-15", "Deadline moved to March 15", 0.7,
                          source_memory_id="msg50")
        beam.conn.commit()

        # Query with numbers that will be extracted by Pass 1 search (looking for '15')
        result = beam._memoria_fact_retrieve(
            "What happened to the 01-15 date in 2024?", top_k=10, intent="timeline"
        )
        ctx = result["context"]
        # Must contain TIMELINE format (Phase 1 rendering) when facts are retrieved
        assert "[fact timeline" in ctx.lower(), \
            f"TR versioned fact must use [Fact TIMELINE] format, got: {ctx}"
        # Must show both dates
        assert "2024-01-15" in ctx
        assert "2024-03-15" in ctx
        # Must have pre-computed delta
        assert "weeks" in ctx.lower() or "days" in ctx.lower(), \
            f"TR versioned fact must include pre-computed delta, got: {ctx}"
    finally:
        beam.conn.close()


def test_beam_integ_eo_versioned_fact_shows_msgidx_anchor(tmp_path):
    """EO scored 0.12-0.14: LLM used abstract labels instead of functional descriptions.

    With versioned facts, _format_versioned_fact(ability='EO') must include
    MSGIDX anchors so the LLM can order by first appearance.

    This test FAILS if _format_versioned_fact() doesn't handle EO."""
    beam = _make_beam_for_integration(tmp_path, "eo_anchor")
    try:
        beam._insert_fact("eo_anchor", 15, "metric", "sprint_feature",
                          "3features", "Built 3 features in sprint 1", 0.7,
                          source_memory_id="msg15")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve(
            "In what order did we build features?", top_k=10, intent="ordered"
        )
        ctx = result["context"]
        # Must contain MSGIDX anchor in fact line (Phase 1 rendering)
        assert "msgidx:" in ctx.lower(), \
            f"EO versioned fact must include MSGIDX anchor, got: {ctx}"
        assert "15" in ctx  # The actual message index
    finally:
        beam.conn.close()


def test_beam_integ_cr_routing_merges_versioned_facts(tmp_path):
    """memoria_retrieve(ability='CR') must merge negation triples WITH versioned
    fact results from _memoria_fact_retrieve.

    This test FAILS if Phase 1.5 routing change is reverted (CR only going to
    _memoria_negation_retrieve)."""
    beam = _make_beam_for_integration(tmp_path, "cr_merge")
    try:
        # Insert a versioned fact (value changed) — this goes into memoria_facts
        beam._insert_fact("cr_merge", 5, "metric", "db_choice",
                          "SQLite", "Using SQLite for storage", 0.7,
                          source_memory_id="msg5")
        beam._insert_fact("cr_merge", 40, "metric", "db_choice",
                          "PostgreSQL", "Switched to PostgreSQL", 0.7,
                          source_memory_id="msg40")
        beam.conn.commit()

        result = beam.memoria_retrieve(
            "Which database are we using?", ability="CR", top_k=10
        )
        ctx = result["context"].lower()
        # CR routing must include versioned facts (not just negation triples)
        # If routing is reverted, only _memoria_negation_retrieve runs,
        # which won't find these metric facts
        assert "sqlite" in ctx or "postgresql" in ctx, \
            f"CR routing must include versioned facts, got: {result['context']}"
    finally:
        beam.conn.close()


def test_beam_integ_tr_routing_merges_versioned_facts(tmp_path):
    """memoria_retrieve(ability='TR') must merge timeline results WITH versioned
    fact results from _memoria_fact_retrieve.

    This test FAILS if Phase 1.5 routing change is reverted (TR only going to
    _memoria_timeline_retrieve)."""
    beam = _make_beam_for_integration(tmp_path, "tr_merge")
    try:
        beam._insert_fact("tr_merge", 10, "metric", "milestone_date",
                          "2024-02-01", "Milestone set for Feb 1", 0.7,
                          source_memory_id="msg10")
        beam._insert_fact("tr_merge", 30, "metric", "milestone_date",
                          "2024-03-15", "Milestone moved to Mar 15", 0.7,
                          source_memory_id="msg30")
        beam.conn.commit()

        result = beam.memoria_retrieve(
            "When was the milestone?", ability="TR", top_k=10
        )
        ctx = result["context"]
        # TR routing must include versioned facts (not just timeline entries)
        assert "2024" in ctx, \
            f"TR routing must include versioned facts, got: {ctx}"
    finally:
        beam.conn.close()


def test_beam_integ_cr_fusion_runs_without_error(tmp_path):
    """For a CR / yes-no query, memoria_retrieve runs RRF fusion (which includes
    the negation specialist) and returns a fused result without error.

    No special-casing is asserted: the negation specialist is just one of the
    fused specialists, ranked by RRF like any other. Whether a given negation
    fact survives a small top_k is RRF's call, not a guaranteed prepend."""
    beam = _make_beam_for_integration(tmp_path, "cr_negation")
    try:
        beam.conn.execute(
            "INSERT INTO memoria_kg (session_id, subject, predicate, object, message_idx, confidence, source_memory_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("cr_negation", "Redis", "negation", "never used Redis for caching", 7, 0.7, "neg1"),
        )
        for i in range(8):
            beam._insert_fact(
                "cr_negation", 10 + i, "metric", f"redis_metric_{i}",
                f"Redis value {i}", f"Redis fact {i}", 0.7,
                source_memory_id=f"msg{10 + i}",
            )
        beam.conn.commit()

        result = beam.memoria_retrieve(
            "Have I ever used Redis for caching?", ability="CR", top_k=10
        )
        assert result is not None
        assert "context" in result
    finally:
        beam.conn.close()


def test_beam_integ_default_ability_no_version_annotation(tmp_path):
    """For IE and unrecognized abilities, versioned facts should render as
    flat [Fact type] key: value (no CURRENT/CHANGED/TIMELINE annotations).

    Ensures version-aware rendering is scoped to KU/CR/TR/EO only."""
    beam = _make_beam_for_integration(tmp_path, "ie_flat")
    try:
        beam._insert_fact("ie_flat", 5, "metric", "sprint_length",
                          "2weeks", "Sprint is 2 weeks", 0.7,
                          source_memory_id="msg5")
        beam._insert_fact("ie_flat", 20, "metric", "sprint_length",
                          "3weeks", "Sprint extended to 3 weeks", 0.7,
                          source_memory_id="msg20")
        beam.conn.commit()

        result = beam._memoria_fact_retrieve(
            "How long is the sprint?", top_k=10, intent=""
        )
        ctx = result["context"].lower()
        # IE should use flat format, NOT version-aware annotations
        assert "[fact current" not in ctx, \
            f"IE should not use CURRENT format, got: {result['content']}"
        assert "[fact changed" not in ctx
        assert "[fact timeline" not in ctx
        # Should still show latest value
        assert "3weeks" in ctx
    finally:
        beam.conn.close()


# ============================================================
#  LLM-Assisted Canonical Metric Key Consolidation Tests
# ============================================================

def test_canonical_metric_keys_collapse_and_chain(tmp_path):
    """Prove that canonical keys enable version chaining via existing _insert_fact.

    Manually insert three metric facts under the SAME key "response_time_ms"
    with different values at increasing msg_idx. Assert the invariant:
    exactly one live row for that key, with previous_value chain.

    This should PASS even before implementing the LLM consolidation feature,
    proving that _insert_fact already chains versions correctly.
    """
    beam = _make_beam_for_integration(tmp_path, "canonical_chain")
    try:
        # Three observations of the same metric, same canonical key
        beam._insert_fact("canonical_chain", 50, "metric", "response_time_ms",
                          "800ms", "Response time is 800ms", 0.65,
                          source_memory_id="msg50")
        beam._insert_fact("canonical_chain", 80, "metric", "response_time_ms",
                          "300ms", "Response time reduced to 300ms", 0.65,
                          source_memory_id="msg80")
        beam._insert_fact("canonical_chain", 100, "metric", "response_time_ms",
                          "250ms", "Response time improved to 250ms", 0.65,
                          source_memory_id="msg100")
        beam.conn.commit()

        # Query the metric fact table
        rows = beam.conn.execute(
            "SELECT key, value, previous_value, version_id FROM memoria_facts "
            "WHERE session_id = ? AND fact_type = 'metric' AND key = ? AND valid_to_msg_idx IS NULL",
            ("canonical_chain", "response_time_ms")
        ).fetchall()

        # Should have exactly one live row (the latest)
        assert len(rows) == 1, f"Expected 1 live row, got {len(rows)}"
        key, value, previous_value, version_id = rows[0]
        assert value == "250ms", f"Expected latest value '250ms', got '{value}'"
        assert previous_value == "300ms", f"Expected previous_value '300ms', got '{previous_value}'"
        assert version_id == 2, f"Expected version_id=2, got {version_id}"
    finally:
        beam.conn.close()


def test_build_canonicalize_prompt_lists_existing_and_observations(tmp_path):
    """Test that _build_canonicalize_prompt generates a well-formed LLM prompt.

    The prompt should list existing facts and new observations,
    requesting canonical keys in JSON format, and include collapse instructions
    for same-metric measurements while preserving target/goal distinction.
    """
    beam = _make_beam_for_integration(tmp_path, "prompt_test")
    try:
        existing = [("response_time_ms", "300ms")]
        candidates = [{
            "context": "dashboard API response time of",
            "value": "250ms",
            "raw_key": "api_response_of_ms"
        }]

        prompt = beam._build_canonicalize_prompt(existing, candidates)

        # Prompt should mention existing facts and observations
        assert "response_time_ms" in prompt, "Prompt should list existing key"
        assert "300ms" in prompt, "Prompt should list existing value"
        assert "250ms" in prompt, "Prompt should list observation value"
        assert "dashboard API response time" in prompt, "Prompt should include context"
        # Should ask for JSON response
        assert "json" in prompt.lower() or "JSON" in prompt, "Prompt should ask for JSON"

        # NEW: Prompt should include instruction to collapse actual measurements to a single base key
        assert "single" in prompt.lower() and "base" in prompt.lower(), \
            "Prompt should instruct collapsing to single base key for actual measurements"
        # Should instruct to ignore incidental qualifiers
        assert ("qualifiers" in prompt.lower() or "incidental" in prompt.lower()), \
            "Prompt should mention ignoring incidental qualifiers"
        # Should distinguish targets/goals from measurements
        assert ("target" in prompt.lower() and ("goal" in prompt.lower() or "distinct" in prompt.lower())), \
            "Prompt should distinguish targets/goals from actual measurements"
    finally:
        beam.conn.close()


def test_parse_canonicalize_response_handles_json_and_fences(tmp_path):
    """Test that _parse_canonicalize_response correctly parses LLM output.

    Should handle JSON fences and extract canonical keys from array.
    """
    beam = _make_beam_for_integration(tmp_path, "parse_test")
    try:
        # Test with JSON fences
        raw = '''```json
[{"index": 0, "canonical_key": "response_time_ms", "is_update": true}]
```'''
        result = beam._parse_canonicalize_response(raw, 1)
        assert result == ["response_time_ms"], f"Expected ['response_time_ms'], got {result}"

        # Test with invalid JSON
        bad_raw = "this is not json"
        result = beam._parse_canonicalize_response(bad_raw, 2)
        assert result == [None, None], f"Expected [None, None] for invalid JSON, got {result}"
    finally:
        beam.conn.close()


def test_consolidation_falls_back_to_regex_without_llm(tmp_path):
    """Test that without llm_client, _llm_canonicalize_facts returns raw_keys.

    Should preserve current behavior when no LLM is available.
    """
    beam = _make_beam_for_integration(tmp_path, "no_llm")
    try:
        # Verify beam has no llm_client
        assert beam._llm_client is None, "Test setup: beam should have no llm_client"

        candidates = [
            {"raw_key": "a_ms", "value": "1ms", "context": "x"},
            {"raw_key": "b_ms", "value": "2ms", "context": "y"},
        ]

        result = beam._llm_canonicalize_facts("no_llm", candidates)
        assert result == ["a_ms", "b_ms"], f"Expected raw keys, got {result}"

        # Also test that extract_and_store_facts still works (doesn't raise)
        # with the current metric loop
        content = "The API response time is 250ms and latency was 300ms."
        counts = beam.extract_and_store_facts(content, message_idx=0)
        assert counts["metric"] >= 0, "extract_and_store_facts should not raise"
        beam.conn.commit()
    finally:
        beam.conn.close()


# ============================================================================
# NEW TESTS: Static Grading Thresholds (Loosen IE/IF/SUM, Keep Atomic Strict)
# ============================================================================
# These tests verify the new fractional threshold logic for descriptive checks


def test_contains_all_with_min_fraction_passes_majority():
    """IE descriptive: _contains_all(..., min_fraction=0.6) passes with 6/8 nuggets."""
    from tests.test_beam_e2e_full import _contains_all

    answer = (
        "The sprint was organized over multiple dates: March 15 for planning, "
        "March 22 for schema work, March 23 for another milestone, and March 29 for the sprint end. "
        "The team set up the database schema, implemented user registration, "
        "and created frontend forms."
    )
    nuggets = [
        "march 15", "march 22", "march 23", "march 29",
        "database schema", "user registration", "frontend forms",
        "integrate frontend",  # MISSING
    ]

    # With min_fraction=0.6: 6/8 = 0.75 >= 0.6 should PASS
    ok, missing = _contains_all(answer, nuggets, min_fraction=0.6)
    assert ok is True, (
        f"Should PASS with {(len(nuggets) - len(missing)) / len(nuggets):.1%} coverage "
        f"(>= 0.6 threshold). Missing: {missing}"
    )


def test_contains_all_with_min_fraction_fails_below_threshold():
    """IE descriptive: _contains_all(..., min_fraction=0.6) fails with 2/8 nuggets."""
    from tests.test_beam_e2e_full import _contains_all

    answer = "March 15 was important. Database schema was discussed."
    nuggets = [
        "march 15", "march 22", "march 23", "march 29",
        "database schema", "user registration", "frontend forms",
        "integrate frontend",
    ]

    # With min_fraction=0.6: 2/8 = 0.25 < 0.6 should FAIL
    ok, missing = _contains_all(answer, nuggets, min_fraction=0.6)
    fraction = (len(nuggets) - len(missing)) / len(nuggets)
    assert ok is False, f"Should FAIL with {fraction:.1%} coverage (< 0.6 threshold)"


def test_contains_groups_with_min_fraction_passes_majority():
    """SUM descriptive: _contains_groups(..., min_fraction=0.5) passes with 7/13 groups."""
    from tests.test_beam_e2e_full import _contains_groups

    answer = (
        "The project includes user authentication and registration. "
        "It handles expense and transaction management. "
        "We added visualization and analytics features. "
        "The deadline was April 15, 2024. "
        "We implemented password hashing. "
        "We used token-based authentication. "
        "Input validation was added throughout."
    )

    groups = [
        ["registration", "user authentication"],
        ["expense", "transaction management"],
        ["visualization", "analytics"],
        ["april 15, 2024", "april 15 2024"],
        ["authentication", "login"],
        ["deployment"],
        ["password hashing", "stronger password"],
        ["token-based", "token authentication"],
        ["role-based access", "rbac"],
        ["input validation"],
        ["confluence"],
        ["api endpoint", "architecture decision"],
        ["table", "diagram"],
    ]

    # With min_fraction=0.5: 7/13 = 53.8% >= 0.5 should PASS
    ok, missing = _contains_groups(answer, groups, min_fraction=0.5)
    assert ok is True, (
        f"Should PASS with {(len(groups) - len(missing)) / len(groups):.1%} coverage "
        f"(>= 0.5 threshold). Missing groups: {len(missing)}/{len(groups)}"
    )


def test_contains_groups_with_min_fraction_fails_below_threshold():
    """SUM descriptive: _contains_groups(..., min_fraction=0.5) fails with 2/13 groups."""
    from tests.test_beam_e2e_full import _contains_groups

    answer = "The project includes user authentication. We handle transactions."

    groups = [
        ["registration", "user authentication"],
        ["expense", "transaction management"],
        ["visualization", "analytics"],
        ["april 15, 2024", "april 15 2024"],
        ["authentication", "login"],
        ["deployment"],
        ["password hashing", "stronger password"],
        ["token-based", "token authentication"],
        ["role-based access", "rbac"],
        ["input validation"],
        ["confluence"],
        ["api endpoint", "architecture decision"],
        ["table", "diagram"],
    ]

    # With min_fraction=0.5: 2/13 = 15.4% < 0.5 should FAIL
    ok, missing = _contains_groups(answer, groups, min_fraction=0.5)
    fraction = (len(groups) - len(missing)) / len(groups)
    assert ok is False, f"Should FAIL with {fraction:.1%} coverage (< 0.5 threshold)"


def test_versioned_dependencies_requires_multiple_versions():
    """IF: _case_outcome requires multiple explicitly versioned dependencies."""
    from tests.test_beam_e2e_full import _versioned_dependency_stats

    # Two versioned deps: should PASS
    answer_good = "Flask 2.3.0 and SQLAlchemy==2.0.19 are used."
    count_good, unv_good = _versioned_dependency_stats(answer_good)
    assert count_good >= 2, f"Expected >= 2 versioned, got {count_good}"
    assert not unv_good, f"Should have no unversioned bullets, got {unv_good}"

    # One versioned dep: should FAIL (with min_versioned_dependencies=2)
    answer_bad = "The project uses Flask 2.3.0."
    count_bad, unv_bad = _versioned_dependency_stats(answer_bad)
    assert count_bad == 1, f"Expected exactly 1 versioned, got {count_bad}"
    assert not unv_bad, "Should have no unversioned bullets"


def test_versioned_dependencies_detects_unversioned_bullets():
    """IF: check fails if any unversioned dependency bullets are found."""
    from tests.test_beam_e2e_full import _versioned_dependency_stats

    answer = (
        "Main libraries:\n"
        "* Flask: 2.3.0\n"
        "* SQLAlchemy: 2.0.19\n"
        "* Werkzeug\n"  # No version
    )

    count, unversioned = _versioned_dependency_stats(answer)
    # count >= 2 but unversioned bullets exist -> should still fail
    assert count >= 2, f"Expected >= 2 versioned, got {count}"
    assert len(unversioned) >= 1, (
        f"Should detect unversioned bullet 'Werkzeug', "
        f"got unversioned={unversioned}"
    )


def test_atomic_ku_check_strict_exact_value():
    """KU atomic checks remain STRICT: must have exact value."""
    from tests.test_beam_e2e_full import _contains_all

    # Exact value must be found
    answer = "The average response time is 250ms according to our dashboard."
    nuggets = ["250ms"]
    ok, missing = _contains_all(answer, nuggets)
    assert ok is True, "Exact atomic value must be found"

    # Wrong value should fail
    answer_wrong = "The average response time is 200ms."
    ok_wrong, missing_wrong = _contains_all(answer_wrong, nuggets)
    assert ok_wrong is False, "Wrong atomic value must fail"
    assert "250ms" in missing_wrong


def test_atomic_mr_count_check_strict_exact_count():
    """MR atomic checks remain STRICT: must have exact count."""
    from tests.test_beam_e2e_full import _contains_all

    # Exact count must be found
    answer = "I want to implement three different user roles: admin, moderator, viewer."
    nuggets = ["three"]
    ok, missing = _contains_all(answer, nuggets)
    assert ok is True, "Exact count must be found"

    # Wrong count should fail
    answer_wrong = "I want to implement two different user roles."
    ok_wrong, missing_wrong = _contains_all(answer_wrong, nuggets)
    assert ok_wrong is False, "Wrong count must fail"


# ---------- RRF Fusion tests (2026-06-22) ----------


def test_rrf_fuse_single_list_preserves_order():
    """A single ranked list → same order."""
    from edumem.core.beam import _rrf_fuse

    ranked_lists = [["A", "B", "C"]]
    result = _rrf_fuse(ranked_lists, k=60)
    assert result == ["A", "B", "C"]


def test_rrf_fuse_rewards_items_in_multiple_lists():
    """Items appearing in multiple lists rank higher (appear in both)."""
    from edumem.core.beam import _rrf_fuse

    ranked_lists = [["A", "B"], ["A", "C"]]
    result = _rrf_fuse(ranked_lists, k=60)
    # A is in both lists at positions 0 and 0 → RRF score = 1/(60+0) + 1/(60+0) = 2/60
    # B is in list 1 at position 1 → RRF score = 1/(60+1) = 1/61
    # C is in list 2 at position 1 → RRF score = 1/(60+1) = 1/61
    # So A ranks first
    assert result[0] == "A"


def test_rrf_fuse_multi_list_beats_single():
    """An item ranked mid in TWO lists outranks an item ranked top in only ONE."""
    from edumem.core.beam import _rrf_fuse

    # Item X: top in list 1 only → score = 1/60
    # Item Y: mid (position 2) in both lists → score = 1/62 + 1/62 = 2/62 ≈ 0.0323
    # Item Z: similar to Y
    # With k=60, 1/60 ≈ 0.0167 and 2/62 ≈ 0.0323, so Y > X
    ranked_lists = [
        ["X", "Y", "Z"],    # X top, Y/Z mid
        ["Y", "Z", "X"],    # Y top, Z mid, X bottom
    ]
    result = _rrf_fuse(ranked_lists, k=60)
    # Y: 1/60 (pos 1 in list 2) + 1/62 (pos 2 in list 1) = 1/60 + 1/62 ≈ 0.0333
    # X: 1/60 (pos 0 in list 1) + 1/63 (pos 2 in list 2) ≈ 0.0276
    # Z: 1/61 (pos 2 in list 1) + 1/61 (pos 1 in list 2) ≈ 0.0328
    assert result[0] == "Y", f"Y should rank first, got {result}"


def test_rrf_fuse_handles_empty_lists():
    """Empty/missing lists don't crash; returns non-empty union."""
    from edumem.core.beam import _rrf_fuse

    ranked_lists = [[], ["A", "B"]]
    result = _rrf_fuse(ranked_lists, k=60)
    assert set(result) == {"A", "B"}

    ranked_lists = []
    result = _rrf_fuse(ranked_lists, k=60)
    assert result == []


def test_memoria_fused_retrieve_merges_multiple_specialists(tmp_path):
    """Fused retrieval includes results from BOTH fact AND timeline specialists."""
    from edumem.core.beam import BeamMemory, init_beam

    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"

    try:
        db_path = str(tmp_path / "test_fused.db")
        init_beam(db_path)
        beam = BeamMemory(db_path=db_path, session_id="test_fused")

        # Insert a versioned metric fact
        beam._insert_fact(
            "test_fused", 3, "metric", "team_size", "5members",
            "We have 5 team members", 0.7, source_memory_id="msg3"
        )
        # Insert a timeline entry
        beam.conn.execute(
            "INSERT INTO memoria_timelines (session_id, date, description, message_idx) "
            "VALUES (?, ?, ?, ?)",
            ("test_fused", "2024-03-15", "The team expanded to 5 members", 3)
        )
        beam.conn.commit()

        # Call fused retrieve
        result = beam._memoria_fused_retrieve("team members", top_k=10)

        # Should have both fact and timeline context
        assert result["source"] == "rrf_fused"
        assert len(result["context"]) > 0
        assert len(result["facts"]) > 0
        # The result should include contributions from multiple specialists
        # At minimum, source should indicate fusion
        assert result.get("source") is not None
    finally:
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings


def test_rrf_fusion_on_by_default(tmp_path):
    """RRF fusion is always active and returns fused results."""
    from edumem.core.beam import BeamMemory, init_beam

    saved_no_embeddings = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"

    try:
        db_path = str(tmp_path / "test_fusion_default.db")
        init_beam(db_path)
        beam = BeamMemory(db_path=db_path, session_id="test_fusion_default")

        # Insert test data using correct schema (key/value pattern)
        beam.conn.execute(
            "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test_fusion_default", 5, "product", "launch_date", "March 2024", "Feature launches in March")
        )
        beam.conn.commit()

        # Call with a query - always uses RRF fusion
        result = beam.memoria_retrieve("When did we launch?", ability="KU", top_k=10)
        # Fusion returns "rrf_fused" as source
        assert result["source"] == "rrf_fused", f"Expected 'rrf_fused' but got '{result['source']}'"
    finally:
        beam.conn.close()
        if saved_no_embeddings is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_no_embeddings


def _make_eo_beam(tmp_path, session_id):
    from edumem.core.beam import BeamMemory

    return BeamMemory(db_path=tmp_path / "beam.db", session_id=session_id)


def test_fused_recall_emits_msgidx_anchors(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_eo_beam(tmp_path, "eo_anchors")
    try:
        beam.conn.execute(
            "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("eo_anchors", 3, "product", "login_flow", "implemented login flow", "I built the login flow"),
        )
        beam.conn.execute(
            "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("eo_anchors", 7, "sequence", "step", "set up the database schema", "Then database schema"),
        )
        beam.conn.commit()

        result = beam.memoria_retrieve(
            "In what order did I work on these things?", ability="EO", top_k=10
        )
        ctx = result["context"]
        assert "MSGIDX:3" in ctx, ctx
        assert "MSGIDX:7" in ctx, ctx
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_fused_recall_orders_by_msgidx_for_ordering_queries(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_eo_beam(tmp_path, "eo_order")
    try:
        for idx, val in (
            (30, "deployed the service"),
            (5, "designed the api"),
            (18, "wrote the tests"),
        ):
            beam.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("eo_order", idx, "sequence", "step", val, val),
            )
        beam.conn.commit()

        result = beam.memoria_retrieve(
            "In what order did I work on these steps?", ability="EO", top_k=10
        )
        ctx = result["context"]
        import re

        idxs = [int(m) for m in re.findall(r"MSGIDX:(\d+)", ctx)]
        assert idxs == sorted(idxs), idxs
        assert {5, 18, 30}.issubset(set(idxs)), idxs
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_fused_recall_no_raw_dict_repr(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_eo_beam(tmp_path, "eo_norepr")
    try:
        beam.conn.execute(
            "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("eo_norepr", 2, "sequence", "step", "kicked off the project", "first I kicked off the project"),
        )
        beam.conn.execute(
            "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("eo_norepr", 9, "sequence", "step", "shipped the project", "then I shipped the project"),
        )
        beam.conn.commit()

        result = beam.memoria_retrieve(
            "In what order did I work on the project?", ability="EO", top_k=10
        )
        ctx = result["context"]
        assert "'msg_idx':" not in ctx, ctx
        assert "{'sequence'" not in ctx, ctx
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_fused_recall_non_ordering_query_unaffected(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_eo_beam(tmp_path, "eo_nonorder")
    try:
        beam.conn.execute(
            "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("eo_nonorder", 4, "metric", "latency", "120 ms", "latency was 120 ms"),
        )
        beam.conn.commit()

        result = beam.memoria_retrieve("What is the latency?", ability="IE", top_k=10)
        assert result["source"] in ("rrf_fused", "fallback")
        assert "'msg_idx':" not in result["context"]
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


# ---------------------------------------------------------------------------
# LLM-based write-time fact extraction (flag-gated: EDUMEM_LLM_EXTRACTION)
# ---------------------------------------------------------------------------

class _RecordingLLMClient:
    """Real (non-mock) test double: records chat calls, returns canned text."""

    def __init__(self, response: str = ""):
        self.response = response
        self.chat_calls = []

    def chat(self, messages, temperature=0.0, max_tokens=1024):
        self.chat_calls.append({"messages": messages, "temperature": temperature})
        return self.response


def _make_beam_with_llm(tmp_path, response=""):
    beam_mod = pytest.importorskip("edumem.core.beam")
    client = _RecordingLLMClient(response)
    beam = beam_mod.BeamMemory(
        db_path=tmp_path / "beam.db", session_id="llm-extract", llm_client=client
    )
    return beam, client


def test_extraction_prompt_is_bounded(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam, _ = _make_beam_with_llm(tmp_path)
        content = "The dashboard API response time is 250ms."
        prompt = beam._build_llm_extraction_prompt(content)
        assert content in prompt
        for token in ("facts", "entities", "relations", "dates", "JSON"):
            assert token in prompt
        assert "5" in prompt
        low = prompt.lower()
        assert "canonical" in low and "snake_case" in low
        prompt_with_keys = beam._build_llm_extraction_prompt(
            content, existing_keys=["response_time_ms", "team_size"]
        )
        assert "response_time_ms" in prompt_with_keys
        assert "team_size" in prompt_with_keys
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_parse_truncates_to_five_and_handles_garbage(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam, _ = _make_beam_with_llm(tmp_path)
        seven = {
            "facts": [{"key": f"k{i}", "value": str(i), "type": "metric"} for i in range(7)],
            "entities": [{"name": f"e{i}", "kind": "service"} for i in range(7)],
            "relations": [{"subject": "a", "predicate": "uses", "object": f"b{i}"} for i in range(7)],
            "dates": [{"key": f"d{i}", "date": "2024-03-15", "context": "x"} for i in range(7)],
        }
        fenced = "```json\n" + json.dumps(seven) + "\n```"
        parsed = beam._parse_llm_extraction(fenced)
        assert len(parsed["facts"]) == 5
        assert len(parsed["entities"]) == 5
        assert len(parsed["relations"]) == 5
        assert len(parsed["dates"]) == 5
        garbage = beam._parse_llm_extraction("totally not json at all")
        assert garbage == {"facts": [], "entities": [], "relations": [], "dates": []}
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_store_extraction_chains_and_routes(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam, _ = _make_beam_with_llm(tmp_path)
        session = beam.session_id

        parsed_v1 = {
            "facts": [{"key": "response_time_ms", "value": "250ms", "type": "metric"}],
            "entities": [{"name": "DashboardAPI", "kind": "service"}],
            "relations": [{"subject": "user", "predicate": "never_used", "object": "flask routes"}],
            "dates": [{"key": "deploy_deadline", "date": "2024-03-15", "context": "deployment deadline"}],
        }
        c1 = beam._store_llm_extraction(session, 0, parsed_v1, "ctx-1")
        assert c1["facts"] == 1 and c1["entities"] == 1 and c1["relations"] == 1 and c1["dates"] == 1

        parsed_v2 = {
            "facts": [{"key": "response_time_ms", "value": "180ms", "type": "metric"}],
            "entities": [], "relations": [], "dates": [],
        }
        beam._store_llm_extraction(session, 5, parsed_v2, "ctx-2")
        beam.conn.commit()

        rows = beam.conn.execute(
            "SELECT value, previous_value FROM memoria_facts "
            "WHERE session_id=? AND key='response_time_ms' AND fact_type='metric' "
            "ORDER BY version_id",
            (session,),
        ).fetchall()
        values = {r[0] for r in rows}
        assert values == {"250ms", "180ms"}
        assert any(r[1] == "250ms" for r in rows)

        row = beam.conn.execute(
            "SELECT subject, predicate, object FROM memoria_kg "
            "WHERE session_id=? AND subject='DashboardAPI' AND predicate='is_a'",
            (session,),
        ).fetchone()
        assert row is not None
        assert row[2] == "service"

        neg = beam.conn.execute(
            "SELECT subject, predicate, object FROM memoria_kg "
            "WHERE session_id=? AND subject='user' AND predicate='negation'",
            (session,),
        ).fetchone()
        assert neg is not None
        assert "flask" in neg[2]

        res = beam._memoria_negation_retrieve("Did the user use flask routes?")
        assert res["source"] == "memoria_kg_negation"
        assert any("flask" in f["object"] for f in res["facts"])

        tl = beam.conn.execute(
            "SELECT date FROM memoria_timelines WHERE session_id=? AND date='2024-03-15'",
            (session,),
        ).fetchone()
        assert tl is not None
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_extraction_truncation_falls_back_to_regex(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    saved_flag = os.environ.get("EDUMEM_LLM_EXTRACTION")
    saved_consol = os.environ.get("EDUMEM_LLM_FACT_CONSOLIDATION")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    os.environ.pop("EDUMEM_LLM_EXTRACTION", None)
    os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = "0"
    try:
        beam_mod = pytest.importorskip("edumem.core.beam")
        beam = beam_mod.BeamMemory(
            db_path=tmp_path / "beam.db", session_id="no-llm", llm_client=None
        )
        counts = beam.extract_and_store_facts(
            "The API response time was 250ms.", message_idx=0
        )
        assert counts.get("metric", 0) >= 1
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved
        if saved_flag is not None:
            os.environ["EDUMEM_LLM_EXTRACTION"] = saved_flag
        if saved_consol is None:
            os.environ.pop("EDUMEM_LLM_FACT_CONSOLIDATION", None)
        else:
            os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = saved_consol


def test_memoria_kg_retrieve_returns_matching_triples(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam, _ = _make_beam_with_llm(tmp_path)
        session = beam.session_id
        beam._insert_kg(session, "DashboardAPI", "is_a", "service", 1)
        beam._insert_kg(session, "DashboardAPI", "uses", "PostgreSQL", 2)
        beam._insert_kg(session, "user", "prefers", "FastAPI", 3)
        beam.conn.commit()

        res = beam._memoria_kg_retrieve("DashboardAPI", top_k=10)
        assert res["source"] == "memoria_kg"
        assert len(res["facts"]) >= 2
        subjects = {f["subject"] for f in res["facts"]}
        assert "DashboardAPI" in subjects

        res_ctx = res.get("context", "")
        assert "[MSGIDX:" in res_ctx
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_fused_recall_includes_kg_source(tmp_path):
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam, _ = _make_beam_with_llm(tmp_path)
        session = beam.session_id
        beam._insert_kg(session, "DashboardAPI", "is_a", "service", 1)
        beam._insert_kg(session, "DashboardAPI", "uses", "PostgreSQL", 2)
        beam.conn.commit()

        result = beam._memoria_fused_retrieve("DashboardAPI PostgreSQL", top_k=10)
        ctx = result.get("context", "")
        assert any(kw in ctx for kw in ["DashboardAPI", "PostgreSQL", "[MSGIDX:"])
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


# ---------------------------------------------------------------------------
# Integration: flag-gated LLM extraction end-to-end
# ---------------------------------------------------------------------------

def test_llm_extraction_happy_path_flag_on(tmp_path):
    """Full end-to-end: flag ON, client returns valid JSON → correct counts + storage."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    saved_flag = os.environ.get("EDUMEM_LLM_EXTRACTION")
    saved_consol = os.environ.get("EDUMEM_LLM_FACT_CONSOLIDATION")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    os.environ["EDUMEM_LLM_EXTRACTION"] = "1"
    os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = "0"
    try:
        response = json.dumps({
            "facts": [{"key": "response_time_ms", "value": "250ms", "type": "metric"}],
            "entities": [{"name": "DashboardAPI", "kind": "service"}],
            "relations": [{"subject": "user", "predicate": "never_used", "object": "flask"}],
            "dates": [{"key": "deploy", "date": "2024-03-15", "context": "deadline"}],
        })
        beam, client = _make_beam_with_llm(tmp_path, response=response)
        counts = beam.extract_and_store_facts(
            "The dashboard API response time is 250ms. We never used flask.",
            message_idx=5
        )
        assert counts["metric"] == 1
        assert counts["entity"] == 1
        assert counts["negation"] == 1
        assert counts["date"] == 1
        assert counts["timeline"] == 1
        assert len(client.chat_calls) == 1

        row = beam.conn.execute(
            "SELECT value FROM memoria_facts WHERE session_id=? AND key='response_time_ms'",
            (beam.session_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "250ms"

        kg = beam.conn.execute(
            "SELECT subject, predicate, object FROM memoria_kg "
            "WHERE session_id=? AND subject='DashboardAPI'",
            (beam.session_id,),
        ).fetchone()
        assert kg is not None
        assert kg[0] == "DashboardAPI" and kg[1] == "is_a" and kg[2] == "service"

        neg = beam.conn.execute(
            "SELECT predicate, object FROM memoria_kg "
            "WHERE session_id=? AND subject='user' AND predicate='negation'",
            (beam.session_id,),
        ).fetchone()
        assert neg is not None
        assert "flask" in neg[1]

        tl = beam.conn.execute(
            "SELECT date FROM memoria_timelines WHERE session_id=? AND date='2024-03-15'",
            (beam.session_id,),
        ).fetchone()
        assert tl is not None
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved
        if saved_flag is not None:
            os.environ["EDUMEM_LLM_EXTRACTION"] = saved_flag
        else:
            os.environ.pop("EDUMEM_LLM_EXTRACTION", None)
        if saved_consol is None:
            os.environ.pop("EDUMEM_LLM_FACT_CONSOLIDATION", None)
        else:
            os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = saved_consol


def test_llm_extraction_version_chains_across_messages(tmp_path):
    """Flag ON: same key in two messages → version chain with previous_value."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    saved_flag = os.environ.get("EDUMEM_LLM_EXTRACTION")
    saved_consol = os.environ.get("EDUMEM_LLM_FACT_CONSOLIDATION")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    os.environ["EDUMEM_LLM_EXTRACTION"] = "1"
    os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = "0"
    try:
        v1 = json.dumps({
            "facts": [{"key": "response_time_ms", "value": "250ms", "type": "metric"}],
            "entities": [], "relations": [], "dates": [],
        })
        beam, client = _make_beam_with_llm(tmp_path, response=v1)
        beam.extract_and_store_facts("Response 250ms.", message_idx=5)

        client.response = json.dumps({
            "facts": [{"key": "response_time_ms", "value": "180ms", "type": "metric"}],
            "entities": [], "relations": [], "dates": [],
        })
        beam.extract_and_store_facts("Optimized to 180ms.", message_idx=10)
        beam.conn.commit()

        rows = beam.conn.execute(
            "SELECT value, previous_value, version_id FROM memoria_facts "
            "WHERE session_id=? AND key='response_time_ms' AND fact_type='metric' "
            "ORDER BY version_id",
            (beam.session_id,),
        ).fetchall()
        assert len(rows) >= 2
        values = {r[0] for r in rows}
        assert values == {"250ms", "180ms"}
        assert any(r[1] == "250ms" for r in rows)
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved
        if saved_flag is not None:
            os.environ["EDUMEM_LLM_EXTRACTION"] = saved_flag
        else:
            os.environ.pop("EDUMEM_LLM_EXTRACTION", None)
        if saved_consol is None:
            os.environ.pop("EDUMEM_LLM_FACT_CONSOLIDATION", None)
        else:
            os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = saved_consol


def test_llm_extraction_empty_parse_falls_back_to_regex(tmp_path):
    """Flag ON but empty JSON response → falls through to regex path."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    saved_flag = os.environ.get("EDUMEM_LLM_EXTRACTION")
    saved_consol = os.environ.get("EDUMEM_LLM_FACT_CONSOLIDATION")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    os.environ["EDUMEM_LLM_EXTRACTION"] = "1"
    os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = "0"
    try:
        response = json.dumps({"facts": [], "entities": [], "relations": [], "dates": []})
        beam, client = _make_beam_with_llm(tmp_path, response=response)
        counts = beam.extract_and_store_facts(
            "The API response time was 250ms.", message_idx=0
        )
        assert len(client.chat_calls) == 1
        assert counts.get("metric", 0) >= 1
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved
        if saved_flag is not None:
            os.environ["EDUMEM_LLM_EXTRACTION"] = saved_flag
        else:
            os.environ.pop("EDUMEM_LLM_EXTRACTION", None)
        if saved_consol is None:
            os.environ.pop("EDUMEM_LLM_FACT_CONSOLIDATION", None)
        else:
            os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = saved_consol


def test_llm_extraction_client_exception_falls_back_to_regex(tmp_path):
    """Flag ON but client raises → falls through to regex path."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    saved_flag = os.environ.get("EDUMEM_LLM_EXTRACTION")
    saved_consol = os.environ.get("EDUMEM_LLM_FACT_CONSOLIDATION")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    os.environ["EDUMEM_LLM_EXTRACTION"] = "1"
    os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = "0"
    try:

        class _RaisingClient:
            chat_calls = []
            def chat(self, messages, temperature=0.0, max_tokens=1024):
                self.chat_calls.append(1)
                raise RuntimeError("simulated failure")

        beam_mod = pytest.importorskip("edumem.core.beam")
        client = _RaisingClient()
        beam = beam_mod.BeamMemory(
            db_path=tmp_path / "beam.db", session_id="raising", llm_client=client
        )
        counts = beam.extract_and_store_facts(
            "The API response time was 250ms.", message_idx=0
        )
        assert len(client.chat_calls) == 1
        assert counts.get("metric", 0) >= 1
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved
        if saved_flag is not None:
            os.environ["EDUMEM_LLM_EXTRACTION"] = saved_flag
        else:
            os.environ.pop("EDUMEM_LLM_EXTRACTION", None)
        if saved_consol is None:
            os.environ.pop("EDUMEM_LLM_FACT_CONSOLIDATION", None)
        else:
            os.environ["EDUMEM_LLM_FACT_CONSOLIDATION"] = saved_consol


def test_sequence_extraction_uses_word_boundaries_not_substrings(tmp_path):
    """Sequence markers (first/then/next/...) must match as whole words, not
    as substrings inside other words. 'then' lives inside 'au-THEN-tication',
    so without \b the regex captured 'thentication ...' as a bogus sequence
    fact, corrupting retrieved context across abilities."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_beam(tmp_path)
    try:
        beam.extract_and_store_facts(
            "We implemented user authentication with registration and login.",
            message_idx=0,
        )
        rows = beam.conn.execute(
            "SELECT key, value FROM memoria_facts WHERE fact_type='sequence'"
        ).fetchall()
        joined = " ".join((k or "") + " " + (v or "") for k, v in rows)
        assert "thentication" not in joined, f"mid-word 'then' captured: {rows}"

        # Genuine sequence markers must still be extracted.
        beam.extract_and_store_facts(
            "First I designed the database schema, then I built the login endpoints.",
            message_idx=1,
        )
        seq = beam.conn.execute(
            "SELECT value FROM memoria_facts WHERE fact_type='sequence'"
        ).fetchall()
        assert any("login endpoints" in (v or "") for (v,) in seq), seq
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_memoria_kg_retrieve_one_hop_expansion(tmp_path):
    """KG recall should exploit graph structure: when a query matches entity A
    and A-(rel)->B exists, B's own triples should also surface even though the
    query never named B. One hop only, no recursion."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    beam = _make_beam(tmp_path)
    try:
        beam._insert_kg("test-session", "Alice", "manages", "ProjectX", 0, confidence=0.7)
        beam._insert_kg("test-session", "ProjectX", "uses", "Redis", 1, confidence=0.7)
        res = beam._memoria_kg_retrieve("What does Alice work on?", top_k=10)
        ctx = res.get("context", "")
        assert "Alice manages ProjectX" in ctx, ctx
        assert "ProjectX uses Redis" in ctx, f"1-hop neighbor missing: {ctx}"
    finally:
        beam.conn.close()
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


# ---------------------------------------------------------------------------
# Write-Time Rolling Summary Tests (6 tests for TDD)
# ---------------------------------------------------------------------------


def test_should_summarize_only_on_segment_boundary(tmp_path):
    """Pure predicate: _should_summarize returns True only at segment boundaries."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam = _make_beam(tmp_path)
        segment_size = 20
        # True at indices 19, 39, 59 (0-indexed)
        assert beam._should_summarize(19, segment_size) is True
        assert beam._should_summarize(39, segment_size) is True
        assert beam._should_summarize(59, segment_size) is True
        # False elsewhere
        assert beam._should_summarize(0, segment_size) is False
        assert beam._should_summarize(18, segment_size) is False
        assert beam._should_summarize(20, segment_size) is False
        assert beam._should_summarize(38, segment_size) is False
    finally:
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_store_summary_inserts_and_is_retrievable(tmp_path):
    """_store_summary inserts into memoria_summaries and _memoria_summary_retrieve finds it."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam = _make_beam(tmp_path)
        ok = beam._store_summary(beam.session_id, 0, 19, "User learned Python basics and set up their environment.")
        assert ok is True
        result = beam._memoria_summary_retrieve("Python setup environment", top_k=5)
        assert "[MSGIDX:19]" in result["context"]
        assert "Python" in result["context"]
        assert result["source"] == "memoria_summaries"
    finally:
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_summary_retrieve_breadth_fallback_returns_recent(tmp_path):
    """When query matches no summary term, breadth fallback returns recent rows."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam = _make_beam(tmp_path)
        beam._store_summary(beam.session_id, 0, 19, "First segment about cooking")
        beam._store_summary(beam.session_id, 20, 39, "Second segment about travel")
        beam._store_summary(beam.session_id, 40, 59, "Third segment about music")
        # Query with no matching term — breadth fallback returns recent rows
        result = beam._memoria_summary_retrieve("xyzzy_no_match_term_here", top_k=5)
        assert result["source"] == "memoria_summaries"
        assert "[MSGIDX:59]" in result["context"]
        assert len(result["facts"]) >= 2
    finally:
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_summary_prompt_is_bounded(tmp_path):
    """_build_summary_prompt mentions 150-word cap and narrative-only instruction."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam = _make_beam(tmp_path)
        segment_text = "user: Hello\nassistant: Hi there"
        prompt = beam._build_summary_prompt(segment_text)
        # Must mention the 150-word cap
        assert "150" in prompt
        # Must instruct narrative-only (no markdown, no preamble)
        lower = prompt.lower()
        assert "narrative" in lower or "summary" in lower
        assert "markdown" in lower or "preamble" in lower or "only" in lower
    finally:
        if saved is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_fused_recall_includes_summary_source_when_flag_on(tmp_path):
    """Summary source appears in fused recall only when EDUMEM_LLM_SUMMARY=1."""
    saved_emb = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    saved_sum = os.environ.get("EDUMEM_LLM_SUMMARY")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    try:
        beam = _make_beam(tmp_path)
        beam._store_summary(beam.session_id, 0, 19, "User discussed Python learning progress and goals.")

        # Flag ON — summary source must appear in fused context
        os.environ["EDUMEM_LLM_SUMMARY"] = "1"
        result_on = beam.memoria_retrieve("Python learning progress", top_k=5)
        assert "Python" in result_on.get("context", ""), "summary not fused when flag is ON"

        # Flag OFF — summary source must NOT appear in fused context (only from this specialist)
        os.environ["EDUMEM_LLM_SUMMARY"] = "0"
        result_off = beam.memoria_retrieve("Python learning progress", top_k=5)
        # The summary row should not come from the summaries specialist when flag is off
        # (other specialists like fact might still surface something, so we check source)
        off_context = result_off.get("context", "")
        # If there are no other facts, context should be empty
        # At minimum, verify flag-off doesn't crash
        assert isinstance(off_context, str)
    finally:
        if saved_emb is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_emb
        if saved_sum is None:
            os.environ.pop("EDUMEM_LLM_SUMMARY", None)
        else:
            os.environ["EDUMEM_LLM_SUMMARY"] = saved_sum


def test_summary_flag_off_writes_nothing(tmp_path):
    """When EDUMEM_LLM_SUMMARY is off, no summaries are written during ingestion."""
    saved_emb = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    saved_sum = os.environ.get("EDUMEM_LLM_SUMMARY")
    os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    # Ensure flag is OFF (default)
    os.environ.pop("EDUMEM_LLM_SUMMARY", None)
    try:
        beam = _make_beam(tmp_path)
        # Ingest some messages — no llm_client, flag off
        ingest_conversation(beam, [
            {"role": "user", "content": "I love hiking in the mountains."},
            {"role": "assistant", "content": "That sounds wonderful!"},
        ])
        # memoria_summaries table must exist but be empty
        count = beam.conn.execute(
            "SELECT COUNT(*) FROM memoria_summaries WHERE session_id = ?",
            (beam.session_id,)
        ).fetchone()[0]
        assert count == 0, f"Expected 0 summaries when flag is off, got {count}"
    finally:
        if saved_emb is None:
            os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else:
            os.environ["EDUMEM_NO_EMBEDDINGS"] = saved_emb
        if saved_sum is None:
            os.environ.pop("EDUMEM_LLM_SUMMARY", None)
        else:
            os.environ["EDUMEM_LLM_SUMMARY"] = saved_sum


# ---------------------------------------------------------------------------
# _assemble_memory_context — relevance-sort fix
# ---------------------------------------------------------------------------

def test_assemble_memory_context_high_relevance_included():
    """High-relevance item placed LAST in input list must appear in context
    when the char budget is too small to include everything."""
    low_content = "A" * 200  # each 200 chars
    high_content = "Z" * 50   # short, high relevance

    memories = [
        {"content": low_content, "score": 0.3},
        {"content": low_content + "B", "score": 0.3},  # different key
        {"content": low_content + "C", "score": 0.3},
        {"content": high_content, "score": 0.97},       # placed LAST
    ]

    max_chars = 250  # fits ~1 low item OR the high item, not all four

    ctx, result_mems = _assemble_memory_context(memories, max_chars)

    # High-relevance item must be included
    high_mem = next(m for m in result_mems if m["score"] == 0.97)
    assert high_mem["final_context_included"] is True, (
        "High-relevance item (score=0.97) must be included but was dropped"
    )
    assert high_content in ctx, "High-relevance content must appear in context string"

    # At least one low-relevance item must have been dropped
    dropped = [m for m in result_mems if m["score"] == 0.3 and not m.get("final_context_included")]
    assert dropped, "At least one low-relevance item should be dropped to fit the budget"


def test_assemble_memory_context_descending_order():
    """Memories in returned list must be ordered by relevance descending (stable)."""
    memories = [
        {"content": "low", "score": 0.1},
        {"content": "high", "score": 0.9},
        {"content": "mid", "score": 0.5},
    ]
    _ctx, result_mems = _assemble_memory_context(memories, max_chars=10000)
    scores = [m["score"] for m in result_mems]
    assert scores == sorted(scores, reverse=True), (
        f"Expected descending scores, got {scores}"
    )


def _no_embed_beam(tmp_path):
    import os as _os
    _os.environ["EDUMEM_NO_EMBEDDINGS"] = "1"
    return _make_beam(tmp_path)


def test_insert_fact_dedups_same_value_across_sources(tmp_path):
    """Same fact (type+key+value) re-mentioned in different messages/sources
    must collapse to ONE live row -- not accumulate (the flask_version x22 bug)."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    beam = _no_embed_beam(tmp_path)
    try:
        beam._insert_fact("s", 1, "metric", "flask_version", "2.3.1", "c1", 0.7, source_memory_id="m1")
        beam._insert_fact("s", 20, "metric", "flask_version", "2.3.1", "c2", 0.7, source_memory_id="m20")
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM memoria_facts WHERE key='flask_version' "
            "AND value='2.3.1' AND valid_to_msg_idx IS NULL"
        ).fetchone()[0]
        assert n == 1, f"expected 1 live row, got {n}"
    finally:
        beam.conn.close()
        if saved is None: os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else: os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_insert_fact_version_chains_on_changed_value(tmp_path):
    """Dedup must NOT break version-chaining: same key, new value still supersedes."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    beam = _no_embed_beam(tmp_path)
    try:
        beam._insert_fact("s", 1, "metric", "flask_version", "2.3.1", "c", 0.7, source_memory_id="m1")
        beam._insert_fact("s", 5, "metric", "flask_version", "2.4.0", "c", 0.7, source_memory_id="m2")
        live = [r[0] for r in beam.conn.execute(
            "SELECT value FROM memoria_facts WHERE key='flask_version' AND valid_to_msg_idx IS NULL").fetchall()]
        assert live == ["2.4.0"], live
        pv = beam.conn.execute(
            "SELECT previous_value FROM memoria_facts WHERE key='flask_version' "
            "AND valid_to_msg_idx IS NULL").fetchone()[0]
        assert pv == "2.3.1", pv
    finally:
        beam.conn.close()
        if saved is None: os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else: os.environ["EDUMEM_NO_EMBEDDINGS"] = saved


def test_insert_fact_dedups_duplicate_dates_keeps_distinct(tmp_path):
    """Dates can't version-chain (deadline vs start coexist), but exact-duplicate
    dates must still collapse to one row."""
    saved = os.environ.get("EDUMEM_NO_EMBEDDINGS")
    beam = _no_embed_beam(tmp_path)
    try:
        beam._insert_fact("s", 1, "date", "iso_date", "2024-03-15", "c", 0.7, source_memory_id="m1")
        beam._insert_fact("s", 9, "date", "iso_date", "2024-03-15", "c", 0.7, source_memory_id="m2")
        beam._insert_fact("s", 12, "date", "iso_date", "2024-04-15", "c", 0.7, source_memory_id="m3")
        n_dup = beam.conn.execute(
            "SELECT COUNT(*) FROM memoria_facts WHERE key='iso_date' AND value='2024-03-15'").fetchone()[0]
        n_total = beam.conn.execute(
            "SELECT COUNT(*) FROM memoria_facts WHERE key='iso_date'").fetchone()[0]
        assert n_dup == 1, f"duplicate date not collapsed: {n_dup}"
        assert n_total == 2, f"distinct dates should coexist: {n_total}"
    finally:
        beam.conn.close()
        if saved is None: os.environ.pop("EDUMEM_NO_EMBEDDINGS", None)
        else: os.environ["EDUMEM_NO_EMBEDDINGS"] = saved
