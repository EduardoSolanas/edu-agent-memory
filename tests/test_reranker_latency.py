"""
Reranker latency guard (gated live test)
=========================================
Measures steady-state latency of the live reranker endpoint and asserts it is
far below the ~2s per-shape cold-start regression that the length-bucketing +
startup-warmup fix in server_nvidia.py removes.

Gated by EDUMEM_E2E=1 and skipped automatically when the endpoint is unreachable
(e.g. no GPU/Docker box), so the offline suite stays green. Uses stdlib only.

Run (Git Bash), with the Docker stack up:
    export EDUMEM_E2E=1
    python -m pytest tests/test_reranker_latency.py -v -s
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EDUMEM_E2E") != "1",
    reason="live reranker test; set EDUMEM_E2E=1 and have the reranker up",
)

# 127.0.0.1, not localhost: on Windows/WSL localhost resolves to IPv6 ::1 first
# and stalls ~2s before falling back to the IPv4-only published port.
RERANK_URL = os.environ.get("EDUMEM_RERANKER_URL", "http://127.0.0.1:3002/rerank")
# Generous ceiling: well under the ~2000ms cold-start regression, loose enough
# not to flake across GPUs. Override per box with EDUMEM_RERANK_MAX_MS.
MAX_MS = float(os.environ.get("EDUMEM_RERANK_MAX_MS", "500"))


def _rerank(query: str, texts: list[str], timeout: float = 30.0) -> tuple[list, float]:
    """POST to the reranker; return (results, wall_ms). Raises on transport error."""
    payload = json.dumps({"query": query, "texts": texts}).encode("utf-8")
    req = urllib.request.Request(
        RERANK_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        results = json.loads(resp.read().decode("utf-8"))
    return results, (time.perf_counter() - start) * 1000.0


def test_reranker_steady_state_latency_is_well_under_regression():
    query = "what database does the project use"
    texts = [
        "We migrated the primary store to PostgreSQL last quarter.",
        "The cache layer is backed by Redis for hot keys.",
        "Analytics events are streamed into BigQuery nightly.",
        "Legacy reports still read from an old MySQL replica.",
    ]

    try:
        warm_results, _ = _rerank(query, texts)
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        pytest.skip(f"reranker not reachable at {RERANK_URL}: {exc}")

    # Sanity: endpoint returns one scored entry per text.
    assert len(warm_results) == len(texts)

    # Steady-state: shape is now a reused bucket, so these should be fast.
    samples = []
    for _ in range(5):
        _, ms = _rerank(query, texts)
        samples.append(ms)

    median_ms = statistics.median(samples)
    print(f"\nReranker latency: median={median_ms:.1f}ms samples={[round(s, 1) for s in samples]}")

    assert median_ms < MAX_MS, (
        f"reranker median latency {median_ms:.1f}ms exceeds {MAX_MS:.0f}ms — "
        f"per-shape cold-start may have regressed (bucketing/warmup not effective)"
    )
