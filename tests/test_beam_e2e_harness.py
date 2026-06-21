import io
import json

import pytest

from tests.test_beam_e2e_full import (
    FIXTURE_PATH,
    _build_e2e_artifact,
    _case_outcome,
    _contains_all,
    _format_metric_chain,
    _llm_errors,
    _write_e2e_artifact,
)


def _fixture_case(qid):
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return next(case for case in cases if case["qid"] == qid)


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


def test_q7_fixture_accepts_complete_backend_and_frontend_sprint_answer():
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case = next(case for case in cases if case["qid"] == "1:q7")
    answer = """The tasks were organized into a two-week sprint (March 15 - March 29).
Week 1 (March 15 - March 22): Backend Foundation. Define the database schema
for users and implement user registration.
Week 2 (March 23 - March 29): Frontend and Integration. Add frontend forms and
integrate frontend forms with the backend."""

    assert case["expectation"] == "hard"
    assert _contains_all(answer, case["nuggets"]) == (True, [])


@pytest.mark.parametrize(
    "incomplete_answer",
    [
        (
            "Week 1, March 15 - March 22: define the database schema and "
            "implement user registration."
        ),
        (
            "Week 2, March 23 - March 29: add frontend forms and integrate "
            "frontend forms with the backend."
        ),
    ],
    ids=["missing_frontend_phase", "missing_backend_phase"],
)
def test_q7_fixture_rejects_answer_missing_a_sprint_phase(incomplete_answer):
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case = next(case for case in cases if case["qid"] == "1:q7")

    ok, missing = _contains_all(incomplete_answer, case["nuggets"])

    assert not ok
    assert missing


def test_q8_static_grade_requires_a_language_tagged_code_fence():
    case = _fixture_case("1:q8")

    assert _case_outcome(case, "Use this:\n```python\ndef login():\n    pass\n```")["outcome"] == "passed"
    assert _case_outcome(case, "Use def login() and validate the password.")["outcome"] == "failed"
    assert _case_outcome(case, "```\ndef login(): pass\n```")["outcome"] == "failed"


def test_q9_static_grade_requires_multiple_explicit_dependency_versions():
    case = _fixture_case("1:q9")

    assert _case_outcome(case, "Flask 2.3.1 and SQLAlchemy==2.0.19 are used.")["outcome"] == "passed"
    assert _case_outcome(case, "**Flask**: 2.3.1\n* **SQLAlchemy**: 2.0.19")["outcome"] == "passed"
    assert _case_outcome(case, "The project uses Flask and SQLAlchemy.")["outcome"] == "failed"
    assert _case_outcome(case, "The project uses Flask 2.3.1 and SQLAlchemy.")["outcome"] == "failed"
    assert _case_outcome(case, "* Flask: 2.3.1\n* SQLAlchemy: 2.0.19\n* Werkzeug")["outcome"] == "failed"


@pytest.mark.parametrize(
    "qid,complete,incomplete",
    [
        (
            "1:q14",
            "Use Flask-Login for sessions, Flask-SQLAlchemy with SQLite for storage, and Chart.js for analytics.",
            "Use Flask-Login for sessions and Flask-SQLAlchemy with SQLite for storage.",
        ),
        (
            "1:q15",
            "Start with password hashing and CSRF protection, then incrementally add rate limiting.",
            "Start with password hashing.",
        ),
        (
            "1:q16",
            "Registration, login, expenses, and visualization came first. The April 15, 2024 MVP plan phased authentication, transactions, analytics, and deployment. Security added password hashing, token authentication, role-based access control, and input validation. Confluence documented API endpoints and architecture decisions with tables and diagrams.",
            "Registration, login, expenses, and visualization came first. The April 15, 2024 MVP plan phased authentication, transactions, analytics, and deployment.",
        ),
        (
            "1:q17",
            "Werkzeug password hashing was followed by SQLite UUID uniqueness checks, OperationalError handling, CSRF token fixes, and a Redis login lockout with expiry.",
            "Werkzeug password hashing was followed by SQLite UUID uniqueness checks and CSRF token fixes.",
        ),
    ],
)
def test_grouped_static_grades_accept_core_evidence_and_reject_missing_facts(qid, complete, incomplete):
    case = _fixture_case(qid)

    assert _case_outcome(case, complete)["outcome"] == "passed"
    outcome = _case_outcome(case, incomplete)
    assert outcome["outcome"] == "failed"
    assert outcome["missing_groups"]


def test_new_static_grades_preserve_evidence_based_expectations():
    assert _fixture_case("1:q8")["expectation"] == "hard"
    assert _fixture_case("1:q9")["expectation"] == "hard"
    assert _fixture_case("1:q14")["expectation"] == "hard"
    assert _fixture_case("1:q15")["expectation"] == "xfail"
    assert _fixture_case("1:q16")["expectation"] == "hard"
    assert _fixture_case("1:q17")["expectation"] == "xfail"
