"""
BEAM End-to-End Full Test Suite
================================
Real e2e test against the live Docker stack (reranker + LLM via NAN endpoint).
This test requires the Docker stack to be running and is gated by EDUMEM_E2E=1.

Runs the full pipeline with STATIC grading (no judge LLM):
  1. Load BEAM 100K conversation
  2. Ingest into BeamMemory
  3. For each question: retrieve → LLM answer (recall + synthesis only)
  4. Grade answer against STATIC expectations mined offline from prior judge results
  5. No live judge LLM involved

The test is skipped by default (EDUMEM_E2E != 1) to keep the fast suite fast.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import pytest

# Gate: skip this entire module if not explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("EDUMEM_E2E") != "1",
    reason="real e2e; set EDUMEM_E2E=1 and have the Docker stack up"
)

# Load fixture at module import time to build parametrize list
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "beam_e2e_100k_case1_fixture.json"
_FIXTURE_CASES = []
if FIXTURE_PATH.exists():
    try:
        _FIXTURE_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass


# ============================================================
#  Static Grading Helpers
# ============================================================

def _norm(s: str) -> tuple[str, str]:
    """
    Normalize a string for comparison.
    Returns: (whitespace-normalized, space-stripped)
    - Lowercase, collapse whitespace runs to single space
    - Also return variant with all spaces removed for unit matching
    """
    s = (s or "").lower()
    ws_norm = " ".join(s.split())
    space_stripped = ws_norm.replace(" ", "")
    return ws_norm, space_stripped


def _contains_all(answer: str, nuggets: list[str]) -> tuple[bool, list[str]]:
    """
    Check if answer contains all nuggets.
    A nugget matches if its normalized form is a substring of the normalized answer
    OR its space-stripped form is a substring of the space-stripped answer.
    Returns: (ok, missing_list)
    """
    ans_norm, ans_stripped = _norm(answer)
    missing = []

    for nugget in nuggets:
        nug_norm, nug_stripped = _norm(nugget)
        # Match if either: ws-normalized substring OR space-stripped substring
        if nug_norm not in ans_norm and nug_stripped not in ans_stripped:
            missing.append(nugget)

    return len(missing) == 0, missing


def _absence(answer: str) -> bool:
    """
    Check if answer indicates absence of information.
    Returns True if normalized answer contains any of the absence markers.
    """
    ans_norm, _ = _norm(answer)
    markers = [
        "no information",
        "does not contain",
        "doesn't contain",
        "not contain",
        "no mention",
        "not provide",
        "no record",
        "isn't any information",
        "no relevant information",
        "cannot find",
        "no details",
        "not mentioned",
        "not discussed",
        "no specific information",
        "wasn't mentioned",
        "not available",
        "no data",
    ]
    return any(marker in ans_norm for marker in markers)


def _order(answer: str, nuggets: list[str]) -> tuple[bool, str]:
    """
    Check if answer contains topic phrases in the correct order.
    For each topic phrase, pick the longest word (>3 chars) as its keyword.
    Find each keyword's first index in the normalized answer.
    Returns: (ok, detail_string)

    Keywords must all be present AND indices must be strictly increasing.
    """
    if not nuggets:
        return True, "no topics to order"

    ans_norm, _ = _norm(answer)

    # Extract keyword (longest word >3 chars) from each topic
    keywords = []
    for topic in nuggets:
        words = _norm(topic)[0].split()
        words_filtered = [w for w in words if len(w) > 3]
        if words_filtered:
            keyword = max(words_filtered, key=len)  # longest
            keywords.append(keyword)

    if not keywords:
        return True, "no keywords extracted from topics"

    # Find indices
    indices = []
    for kw in keywords:
        idx = ans_norm.find(kw)
        if idx == -1:
            return False, f"keyword '{kw}' not found"
        indices.append(idx)

    # Check strictly increasing
    for i in range(1, len(indices)):
        if indices[i] <= indices[i-1]:
            return False, f"keywords not in order: {keywords}"

    return True, f"keywords in order: {keywords}"


@pytest.fixture(scope="module")
def e2e_answers():
    """
    Module-scope fixture: generates answers for all 20 questions.

    - Preflight: checks reranker availability
    - Ingest: loads BEAM 100K conversation, ingest into BeamMemory
    - Answer: run all 20 questions through recall + LLM answer (no judge)
    - Yield: answers dict {qid: answer_str} and beam for storage test
    """
    from tools.evaluate_beam_end_to_end import (
        _probe_reranker,
        load_beam_dataset,
        ingest_conversation,
        answer_with_memory,
        LLMClient,
        ABILITY_MAP,
    )
    from edumem.core.beam import init_beam, BeamMemory

    # Preflight: check reranker
    reranker_url = os.environ.get("EDUMEM_RERANKER_URL", "http://localhost:3002/rerank")
    reranker_health = _probe_reranker(reranker_url)
    if not reranker_health.get("ok"):
        pytest.skip(f"reranker endpoint unavailable: {reranker_health}")

    # Configuration
    answer_model = os.environ.get("EDUMEM_E2E_ANSWER_MODEL", "qwen3.6")

    # Create LLM client (answer only, no judge)
    llm = LLMClient(model=answer_model)

    # Load BEAM dataset
    print(f"\n[E2E] Loading BEAM 100K...")
    data = load_beam_dataset(["100K"], max_conversations=1)
    conv = data["100K"][0]

    # Create temporary directory for beam DB (keep it open for the fixture scope)
    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / "beam.db"

    try:
        # Initialize beam
        print(f"[E2E] Initializing BeamMemory at {db_path}")
        init_beam(str(db_path))
        beam = BeamMemory(session_id="e2e-test", db_path=str(db_path), llm_client=llm)

        # Ingest conversation
        print(f"[E2E] Ingesting {len(conv['messages'])} messages...")
        ingest_conversation(beam, conv["messages"], diag={}, llm=llm)

        # Generate answers for all questions (recall + synthesis, no judge)
        print(f"[E2E] Generating answers for {len(conv['questions'])} questions...")
        answers = {}
        for i, q in enumerate(conv['questions']):
            qid = f"1:q{i}"
            ability_name = q.get('ability', 'IE')
            ability = ABILITY_MAP.get(ability_name, ability_name)

            result = answer_with_memory(
                llm, beam, q['question'],
                conversation_messages=conv['messages'],
                ability=ability,
                diag={},
                return_memories=False
            )

            # Handle tuple return in case return_memories was True
            if isinstance(result, tuple):
                answer = result[0]
            else:
                answer = result

            answer = answer or ""
            answers[qid] = answer
            print(f"[E2E] {qid}: {answer[:80]}")

        print(f"\n[E2E] Answer generation complete. Generated: {len(answers)}")

        yield {"answers": answers, "beam": beam}

    finally:
        # Teardown: close beam connection and clean tmpdir
        try:
            beam.conn.close()
        except Exception:
            pass
        tmpdir.cleanup()


# Build parametrize list: hard cases with hard assertions, xfail cases marked for expected failure
_PARAMS = []
for case in _FIXTURE_CASES:
    if case["expectation"] == "xfail":
        _PARAMS.append(
            pytest.param(
                case,
                id=case["qid"],
                marks=pytest.mark.xfail(reason="known BEAM failure - target to flip", strict=False),
            )
        )
    else:
        _PARAMS.append(pytest.param(case, id=case["qid"]))


@pytest.mark.parametrize("case", _PARAMS)
def test_beam_e2e_answer_meets_static_expectation(case, e2e_answers):
    """
    Parametrized test: check each answer against static expectations.

    For each check type:
    - skip: pytest.skip (not statically gradable)
    - contains_all: assert all nuggets are in the answer
    - absence: assert answer indicates absence of information
    - order: assert nuggets appear in order with increasing indices

    Assertion includes qid, ability, check type, missing nuggets, and first 200 chars of answer.
    """
    qid = case["qid"]
    ability = case["ability"]
    check = case["check"]
    nuggets = case.get("nuggets", [])

    answer = e2e_answers["answers"].get(qid)
    assert answer is not None, f"{qid}: no answer generated"

    # Skip checks that are not statically gradable
    if check == "skip":
        pytest.skip(f"{ability} not statically gradable")

    # Dispatch on check type
    if check == "contains_all":
        ok, missing = _contains_all(answer, nuggets)
        assert ok, (
            f"{qid} [{ability}] contains_all failed: missing {missing}\n"
            f"Answer (first 200 chars): {answer[:200]}"
        )
    elif check == "absence":
        ok = _absence(answer)
        assert ok, (
            f"{qid} [{ability}] absence check failed: answer does not indicate absence\n"
            f"Answer (first 200 chars): {answer[:200]}"
        )
    elif check == "order":
        ok, detail = _order(answer, nuggets)
        assert ok, (
            f"{qid} [{ability}] order check failed: {detail}\n"
            f"Answer (first 200 chars): {answer[:200]}"
        )


def test_beam_e2e_storage(e2e_answers):
    """
    Storage test (not parametrized): verify facts and version chains.

    - Assert total facts > 0
    - Count and report version-chained facts (previous_value IS NOT NULL)
    """
    beam = e2e_answers["beam"]
    cursor = beam.conn.cursor()

    # Total facts count
    cursor.execute("SELECT COUNT(*) FROM memoria_facts")
    total_facts = cursor.fetchone()[0]
    assert total_facts > 0, "Expected at least 1 fact in storage"

    # Version-chained facts
    cursor.execute("SELECT COUNT(*) FROM memoria_facts WHERE previous_value IS NOT NULL")
    versioned_facts = cursor.fetchone()[0]

    print(f"\n[E2E Storage] Total facts: {total_facts}, Versioned (with previous_value): {versioned_facts}")
