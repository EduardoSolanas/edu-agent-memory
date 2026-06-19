from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

from tools.evaluate_beam_end_to_end import (
    _required_rejudge_fields,
    _attach_second_pass_diagnostics,
    _record_second_pass_diagnostics,
    _extract_shared_date_spans,
    _build_paired_outcome_rows,
    _build_skipped_question_result,
    _finalize_reranker_run_health,
    _multi_strategy_recall,
    _parse_judge_payload,
    _print_env_snapshot,
    _question_row_policy,
    _select_conversations,
    _sanitize_sensitive_data,
    _summarize_judge_result,
    _update_rejudged_question_row,
    _update_embedding_diagnostic,
    _summarize_recall_memories,
    _write_json_sanitized,
    apply_rejudge_judgment_records,
    compute_ability_scores,
    compute_partial_credit_overall,
    ingest_conversation,
    print_sota_report,
    write_rejudge_artifacts,
)


def _make_beam(tmp_path: Path):
    beam_mod = pytest.importorskip("edumem.core.beam")
    return beam_mod.BeamMemory(db_path=tmp_path / "beam.db", session_id="test-session")


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
    assert info["model_id"] == "sentence-transformers/all-mpnet-base-v2"
    assert info["dimension"] == 768

    embedding = _json_request(
        f"{base_url}/v1/embeddings",
        {"input": ["beam smoke"], "model": "sentence-transformers/all-mpnet-base-v2"},
    )
    assert embedding["model"] == "sentence-transformers/all-mpnet-base-v2"
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
