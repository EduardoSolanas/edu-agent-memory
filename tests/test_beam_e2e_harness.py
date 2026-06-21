import io
import json

import pytest

from tests.test_beam_e2e_full import (
    _build_e2e_artifact,
    _format_metric_chain,
    _llm_errors,
    _write_e2e_artifact,
)


def test_metric_chain_reporting_is_ascii_safe_under_cp1252():
    rendered = _format_metric_chain("dashboard_latency", "300ms", "250ms", 2)

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    stream.write(rendered)
    stream.flush()

    assert rendered == "  dashboard_latency: 300ms -> 250ms (version 2)"


def test_e2e_artifact_persists_answers_and_static_outcomes(tmp_path):
    cases = [
        {
            "qid": "1:q0",
            "ability": "IE",
            "expectation": "pass",
            "check": "contains_all",
            "nuggets": ["alpha"],
        },
        {
            "qid": "1:q1",
            "ability": "IF",
            "expectation": "skip",
            "check": "skip",
            "nuggets": [],
        },
    ]
    artifact = _build_e2e_artifact(cases, {"1:q0": "Alpha found", "1:q1": "answer"})
    output = tmp_path / "nested" / "answers.json"

    _write_e2e_artifact(output, artifact)

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == {
        "cases": [
            {
                "qid": "1:q0",
                "ability": "IE",
                "expectation": "pass",
                "check": "contains_all",
                "outcome": "passed",
                "missing": [],
                "answer": "Alpha found",
            },
            {
                "qid": "1:q1",
                "ability": "IF",
                "expectation": "skip",
                "check": "skip",
                "outcome": "ungraded",
                "detail": "not statically gradable",
                "answer": "answer",
            },
        ]
    }


def test_llm_errors_are_detected_independently_of_case_xfail_marks():
    errors = _llm_errors({"1:q0": "ok", "1:q7": "[LLM_ERROR: timeout]"})

    assert errors == {"1:q7": "[LLM_ERROR: timeout]"}
