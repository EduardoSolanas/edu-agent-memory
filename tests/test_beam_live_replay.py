from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from edumem.core.beam import BeamMemory
from tools.evaluate_beam_end_to_end import (
    LLMClient,
    answer_with_memory,
    ingest_conversation,
    judge_with_rubrics,
    normalize_for_judge,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "beam_live_replay_cases.json"

pytestmark = pytest.mark.skipif(
    os.getenv("BEAM_LIVE_REPLAY") != "1",
    reason="set BEAM_LIVE_REPLAY=1 to run the live replay integration test",
)


def _load_cases() -> list[dict]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise TypeError("beam live replay fixture must contain a list of cases")
    return cases


def _select_cases(cases: list[dict]) -> list[dict]:
    raw = os.getenv("BEAM_LIVE_CASES", "").strip()
    if not raw:
        return cases[:2]

    by_qid = {case["qid"]: case for case in cases}
    selected: list[dict] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        case = by_qid.get(token)
        if case is None:
            suffix = token if token.startswith(":") else f":{token}"
            matches = [item for item in cases if item["qid"].endswith(suffix)]
            if len(matches) == 1:
                case = matches[0]
        if case is None:
            raise AssertionError(
                f"Unknown BEAM_LIVE_CASES entry {token!r}; available qids: "
                + ", ".join(case["qid"] for case in cases)
            )
        selected.append(case)
    return selected


def _build_messages(context_evidence: list[str]) -> list[dict]:
    return [{"role": "user", "content": evidence} for evidence in context_evidence if evidence.strip()]


def _assert_expected_terms(answer: str, expected_terms: list[str]) -> None:
    lower_answer = answer.lower()
    missing = [term for term in expected_terms if term.lower() not in lower_answer]
    assert not missing, f"answer missing expected term(s): {missing}"


def _resolve_model(env_name: str, default: str) -> str:
    return os.getenv(env_name) or default


def _judgment_score(judgment: dict) -> float:
    for key in ("overall_score", "official_score", "partial_credit_score", "score"):
        value = judgment.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    scores = judgment.get("scores")
    if isinstance(scores, list) and scores:
        try:
            return float(sum(float(item) for item in scores) / len(scores))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def test_beam_live_replay(tmp_path):
    cases = _select_cases(_load_cases())
    assert cases, "beam live replay fixture did not yield any cases"

    answer_model = _resolve_model("EDUMEM_LLM_MODEL", "deepseek-v4-flash")
    judge_model = _resolve_model("EDUMEM_JUDGE_MODEL", answer_model)
    should_judge = os.getenv("BEAM_LIVE_JUDGE") == "1"
    min_score = float(os.getenv("BEAM_LIVE_MIN_SCORE", "0.5"))

    llm = LLMClient(model=answer_model)
    judge_llm = LLMClient(model=judge_model) if should_judge else None

    try:
        for case in cases:
            beam = BeamMemory(
                db_path=tmp_path / f"{case['qid'].replace(':', '_')}.db",
                session_id=f"beam-live-replay-{case['qid']}",
            )
            try:
                conversation_messages = _build_messages(case["context_evidence"])
                ingest_conversation(beam, conversation_messages)

                answer, _memories = answer_with_memory(
                    llm,
                    beam,
                    case["question"],
                    conversation_messages=conversation_messages,
                    ability=case["ability"],
                    return_memories=True,
                )

                assert isinstance(answer, str)
                assert answer.strip()
                _assert_expected_terms(answer, case["expected_terms"])

                if should_judge:
                    normalized = normalize_for_judge(answer, case["ability"])
                    judgment = judge_with_rubrics(
                        judge_llm,
                        case["question"],
                        case["rubric"],
                        normalized,
                        ability=case["ability"],
                    )
                    assert isinstance(judgment, dict)
                    assert judgment["scores"]
                    assert len(judgment["scores"]) == len(case["rubric"])
                    assert _judgment_score(judgment) >= min_score
            finally:
                beam.conn.close()
    finally:
        llm.close()
        if judge_llm is not None:
            judge_llm.close()
