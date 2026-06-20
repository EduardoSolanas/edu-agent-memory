from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.evaluate_beam_end_to_end import (
    _is_calculator_question,
    _multi_strategy_recall,
    ingest_conversation,
)


def _make_beam(tmp_path: Path):
    beam_mod = pytest.importorskip("edumem.core.beam")
    return beam_mod.BeamMemory(db_path=tmp_path / "beam.db", session_id="test-session")


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How many days between 2024-03-01 and 2024-03-05?", True),
        ("How many days passed between when I planned peer review and when I completed final review?", True),
        ("How many days did I say the project would take?", False),
        ("How long did I say the project is expected to take?", False),
    ],
)
def test_calculator_routing_is_limited_to_true_date_intervals(question, expected):
    assert _is_calculator_question(question) is expected


def test_stated_duration_questions_do_not_enable_negation_recall(tmp_path, monkeypatch):
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "the project would take three days."},
                {"role": "user", "content": "i said it might take five days."},
                {"role": "user", "content": "the project finished in four days."},
            ],
        )

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            "How many days did I say the project would take?",
            top_k=5,
            ability=None,
            diag=diag,
        )

        assert memories
        assert diag["strategies"].get("S2", {}).get("activated") is False
    finally:
        beam.conn.close()


@pytest.mark.parametrize(
    "question, expected_mr",
    [
        ("How many project cards are there?", False),
        ("How many project cards across all sessions are there?", True),
        ("How many project cards in total are there?", True),
    ],
)
def test_project_card_counts_only_expand_for_broad_aggregation_language(
    tmp_path,
    monkeypatch,
    question,
    expected_mr,
):
    monkeypatch.setenv("EDUMEM_NO_EMBEDDINGS", "1")
    beam = _make_beam(tmp_path)
    try:
        ingest_conversation(
            beam,
            [
                {"role": "user", "content": "project cards are three."},
                {"role": "user", "content": "there are four project cards in the backlog."},
                {"role": "user", "content": "the cards are red, blue, and green."},
            ],
        )

        diag = {}
        memories = _multi_strategy_recall(
            beam,
            question,
            top_k=5,
            ability=None,
            diag=diag,
        )

        assert memories
        assert diag["strategies"].get("MR", {}).get("activated", False) is expected_mr
    finally:
        beam.conn.close()
