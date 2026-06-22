"""Opt-in integration tests for the real reranker HTTP service."""

from concurrent.futures import ThreadPoolExecutor
import os
import random
import string
from urllib.parse import urlsplit, urlunsplit

import pytest
import requests

from tools.evaluate_beam_end_to_end import _intent_from_reranker_scores


_LIVE_ENABLED = os.environ.get("BEAM_LIVE_RERANKER_TEST") == "1"
_RERANKER_URL = os.environ.get("EDUMEM_RERANKER_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not _LIVE_ENABLED or not _RERANKER_URL,
    reason=(
        "set BEAM_LIVE_RERANKER_TEST=1 and EDUMEM_RERANKER_URL "
        "to run the live reranker integration"
    ),
)


def _post(query: str, texts: list[str], timeout: float = 15.0) -> list[dict]:
    response = requests.post(
        _RERANKER_URL,
        json={"query": query, "texts": texts},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload, list)
    assert {item["index"] for item in payload} == set(range(len(texts)))
    assert len(payload) == len(texts)
    return payload


def _top_index(results: list[dict]) -> int:
    return max(results, key=lambda item: float(item["score"]))["index"]


def _health_url() -> str:
    parsed = urlsplit(_RERANKER_URL)
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


@pytest.fixture(scope="module", autouse=True)
def live_reranker_available():
    try:
        _post("preflight", ["preflight"], timeout=5.0)
    except (requests.RequestException, ValueError, AssertionError) as exc:
        pytest.skip(f"live reranker unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize(
    ("query", "texts", "relevant_index"),
    [
        (
            "What is the current dashboard API response time?",
            [
                "The PostgreSQL pool contains 20 connections.",
                "With caching, the dashboard API response time is now 250ms.",
                "The homepage bundle was reduced to 180 KB.",
            ],
            1,
        ),
        (
            "How did I plan the two-week authentication sprint?",
            [
                "The dashboard latency fell after enabling a response cache.",
                "Sprint plan: week one implements login and sessions; week two adds tests and documentation.",
                "The production deployment uses three Docker containers.",
            ],
            1,
        ),
        (
            "How long was it between the January 15 kickoff and March 15 launch?",
            [
                "An unrelated maintenance window is scheduled for April 4.",
                "The project kicked off on January 15 and launched on March 15.",
                "The team selected PostgreSQL for durable storage.",
            ],
            1,
        ),
    ],
)
def test_relevant_memory_ranks_above_realistic_distractors(query, texts, relevant_index):
    assert _top_index(_post(query, texts)) == relevant_index


@pytest.mark.parametrize(
    ("query", "texts"),
    [
        ("", ["", "   \n\t"]),
        ("repeated grid", ["=" * 20_000, "+---+" * 5_000]),
        (
            "long identifier",
            [
                "".join(random.Random(seed).choices(string.ascii_letters + string.digits, k=20_000))
                for seed in (7, 11)
            ],
        ),
        (
            "Which document contains the launch date?",
            [
                "HEAD launch notes " + ("ordinary filler " * 4_000) + " TAIL launched March 15",
                "HEAD database notes " + ("unrelated filler " * 4_000) + " TAIL pool size 20",
            ],
        ),
    ],
)
def test_tokenizer_edge_cases_return_a_complete_permutation(query, texts):
    _post(query, texts, timeout=30.0)


def test_ambiguous_intent_obeys_confidence_gate(monkeypatch):
    question = "What can you tell me about how the project evolved?"
    hypotheses = [
        "This query asks about the chronological order, sequence, or ordering of events.",
        "This query asks about dates, deadlines, duration, intervals, or timeline of events.",
        "This query asks about a change of state, switched preference, contradictions, or previous versus current values.",
        "This query asks for general factual details, current versions, or standard information of an entity.",
    ]
    scores = _post(question, hypotheses)
    accepted = _intent_from_reranker_scores(scores)
    assert accepted is None or accepted in {"ordered", "timeline", "change", "current"}


def test_concurrent_requests_complete_while_health_remains_reachable():
    requests_to_send = [
        (f"dashboard response time request {index}", ["dashboard response time is 250ms", "pool size is 20"])
        for index in range(6)
    ]

    def rerank_one(item):
        return _post(*item, timeout=30.0)

    with ThreadPoolExecutor(max_workers=7) as executor:
        rerank_futures = [executor.submit(rerank_one, item) for item in requests_to_send]
        health_future = executor.submit(requests.get, _health_url(), timeout=10.0)
        results = [future.result(timeout=35.0) for future in rerank_futures]
        health = health_future.result(timeout=15.0)

    assert health.status_code == 200
    assert all(_top_index(result) == 0 for result in results)
