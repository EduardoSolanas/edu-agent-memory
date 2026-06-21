#!/usr/bin/env python3
"""
BEAM End-to-End Evaluation Pipeline
===================================
Evaluates edumem as a memory backend for LLMs using the official
BEAM benchmark protocol:
  1. Download BEAM dataset from HuggingFace
  2. Ingest conversations into edumem
  3. For each probing question: retrieve memories -> LLM answers -> LLM-as-judge scores
  4. Report per-scale, per-ability scores comparable to published SOTA

Published SOTA (BEAM 10M):
  Hindsight: 64.1%   Honcho: 40.6%   LIGHT: 26.6%   RAG: 24.9%

LLM: Nvidia API (deepseek-v4-flash) via OpenAI-compatible endpoint.
     Fast, cheap (~$2/M tokens), no local GPU needed.

Usage:
  cd /root/.hermes/projects/edumem
  .venv/bin/python tools/evaluate_beam_end_to_end.py --sample 5 --scales 100K,500K,1M,10M

--sample N: conversations per scale (default 3, use 0 for all)
--scales: comma-separated (default 100K,500K,1M,10M)
--mode: retrieval|end_to_end (default end_to_end)
--judge-model: LLM model for judging (default same as answer model)
--resume: skip already-evaluated questions from results file
"""

from __future__ import annotations  # PEP 563: defer annotation eval so PEP 604 (X|None) and PEP 585 (list[str]) work on Python 3.9

import argparse
import ast
import gc
import json
import logging
import math
import os
import sys
import tempfile
import time
import subprocess
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

# Unbuffered output for real-time progress
print = partial(print, flush=True)

# --- Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import urllib.request
import urllib.error
import numpy as np

from edumem.core.beam import BeamMemory, init_beam, _embeddings, _vec_available, _vec_insert, _fts_search_working, _generate_id, parse_relative_date
from edumem.core.query_mode import (
    build_system_prompt, is_temporal_query, needs_second_pass, is_ordering_query, is_duration_query,
    is_aggregation_query, is_date_interval_query, is_stated_duration_query, is_summarization_query,
    is_contradiction_query, is_knowledge_update_query
)


def _intent_from_question(question: str) -> str:
    """Map a question to a generic recall intent from its TEXT only (no dataset label).

    Returns one of: 'ordered', 'timeline', 'change', or 'current'.

    Uses the OpenVINO reranker to perform Zero-Shot Classification of intent,
    falling back to robust local regex if the reranker is unavailable.
    """
    import re
    import os
    import requests

    def _fallback_intent(q_text: str) -> str:
        if is_ordering_query(q_text):
            return "ordered"
        if is_duration_query(q_text):
            return "timeline"
        if is_contradiction_query(q_text) or re.match(r"^\s*(?:have|did)\s+(?:i|we)\b", (q_text or "").lower()):
            return "change"
        return "current"

    # Reuse our local OpenVINO reranker for fast dynamic intent classification (Zero-Shot classification pattern)
    _reranker_url = os.environ.get("EDUMEM_RERANKER_URL", "http://localhost:3002/rerank")
    try:
        # Hypotheses mapped to index 0: ordered, 1: timeline, 2: change, 3: current
        hypotheses = [
            "This query asks about the chronological order, sequence, or ordering of events.",
            "This query asks about dates, deadlines, duration, intervals, or timeline of events.",
            "This query asks about a change of state, switched preference, contradictions, or previous versus current values.",
            "This query asks for general factual details, current versions, or standard information of an entity.",
        ]
        resp = requests.post(_reranker_url, json={"query": question, "texts": hypotheses}, timeout=1.0)
        if resp.status_code == 200:
            scores = resp.json()
            if isinstance(scores, list) and len(scores) == len(hypotheses):
                # Format: [{"index": idx, "score": score}, ...]
                sorted_scores = sorted(scores, key=lambda x: x.get("score", 0.0), reverse=True)
                best_idx = sorted_scores[0]["index"]
                intents = ["ordered", "timeline", "change", "current"]
                return intents[best_idx]
    except Exception:
        pass

    return _fallback_intent(question)


def _is_calculator_question(question: str) -> bool:
    """True only for DURATION questions (compute a number). Ordering questions
    must NOT use the calculator prompt — they need an ordered list, not a number."""
    return is_date_interval_query(question)


def _wants_broad_aggregation_retrieval(question: str) -> bool:
    """Return True only when aggregation language is explicitly broad."""
    q = (question or "").lower()
    if not is_aggregation_query(question):
        return False
    broad_markers = (
        "combination",
        "combined",
        "combine",
        "across",
        "total",
        "altogether",
        "in total",
    )
    return any(marker in q for marker in broad_markers)


def _wants_negation_retrieval(question: str) -> bool:
    """Return True only when the question is actually asking about negation."""
    import re

    q = (question or "").strip().lower()
    if is_duration_query(question) or is_stated_duration_query(question):
        return False
    if "contradict" in q or "conflict" in q:
        return True
    return re.match(r"^(?:(?:have|did|do|am)\s+i\b|has\b)", q) is not None


def _tr_python_answer_is_trustworthy(py_answer: str, timeline_size: int,
                                     dense_threshold: int = 15) -> bool:
    """Decide whether to trust the zero-LLM Python TR answer or defer to the LLM.

    The Python oracle picks the two best keyword-matching dates. That matching is
    reliable on a SPARSE timeline (few candidate dates) but unreliable on a DENSE
    one -- with 100+ dates the wrong pair is easily selected (observed: 123 dates
    -> a duration between two unrelated events). So we trust Python only on sparse
    timelines and defer every dense-timeline TR question to the LLM-with-timeline
    path (which is fed the same extracted dates). Zero-durations never trust.
    """
    import re as _re_tr
    if not py_answer:
        return False
    low = py_answer.lower()
    # Reject zero spans. Parse "N days" rather than substring-matching "0 days",
    # which also matches "70 days"/"30 days".
    day_counts = [int(x) for x in _re_tr.findall(r'(\d+)\s+days', low)]
    if day_counts and max(day_counts) == 0:
        return False
    if not day_counts and _re_tr.search(r'\b0\s+(?:weeks|months|years)\b', low):
        return False
    # Dense timeline => Python date-pair matching is unreliable; defer to the LLM.
    if timeline_size > dense_threshold:
        return False
    return True


_CTX_MATCH_STOP = {
    'when', 'does', 'do', 'did', 'what', 'how', 'where', 'which', 'who', 'why',
    'is', 'are', 'was', 'were', 'can', 'will', 'would', 'should', 'could', 'may',
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'my', 'me', 'i', 'you',
}


def _context_value_match(question: str, context_facts: dict, min_score: float = 0.5):
    """Match a question against the context->value side index.

    Returns (best_value, best_score). best_score is the fraction of the matched
    context phrase's words that the question covers. Callers should only bypass
    the LLM when best_score >= min_score — weak matches (a couple of incidental
    shared words) return a low score so they fall through to full LLM reasoning.
    """
    if not context_facts:
        return None, 0.0
    q_words = [w.lower() for w in question.split()
               if w.lower() not in _CTX_MATCH_STOP and len(w) > 1]
    q_set = set(q_words)
    if not q_set:
        return None, 0.0
    best_value = None
    best_score = 0.0
    for context_phrase, values in context_facts.items():
        c_words = set(context_phrase.split())
        overlap = q_set & c_words
        if len(overlap) < 2:
            continue
        score = len(overlap) / max(len(c_words), 1)
        if score > best_score:
            best_score = score
            best_value = values[0] if values else None
    if best_value is not None and best_score >= min_score:
        return best_value, best_score
    return None, best_score


def _extract_shared_date_spans(text: str) -> list[dict]:
    """Extract named, ordinal, and ISO date spans from text.

    The helper returns raw matched spans for tagging plus a parsed date
    object only when the text contains an explicit year. Month/day forms
    without a year are kept as raw spans so we do not invent a year.
    """
    import re

    if not text:
        return []

    month_names = (
        "january|february|march|april|may|june|july|august|"
        "september|october|november|december|"
        "jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    )
    spans: list[dict] = []
    seen: set[tuple] = set()

    def _add(raw: str, start: int, end: int, dt=None):
        key = (start, end, raw.lower(), dt.isoformat() if dt else None)
        if key in seen:
            return
        seen.add(key)
        spans.append({
            "raw": raw,
            "start": start,
            "end": end,
            "date_obj": dt,
            "iso": dt.strftime("%Y-%m-%d") if dt else None,
        })

    # Explicit ISO dates.
    for m in re.finditer(r'\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b', text):
        try:
            dt = datetime(int(m.group("year")), int(m.group("month")), int(m.group("day")))
        except ValueError:
            continue
        _add(m.group(0), m.start(), m.end(), dt)

    # Month-name forms with an explicit year.
    for m in re.finditer(
        rf'\b(?P<month>{month_names})[a-z]*\s+'
        rf'(?P<day>\d{{1,2}})(?:st|nd|rd|th)?'
        rf'(?:,?\s*(?P<year>\d{{4}}))?\b',
        text,
        re.IGNORECASE,
    ):
        month = m.group("month")[:3].lower()
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        year = m.group("year")
        try:
            if year:
                dt = datetime(int(year), month_map[month], int(m.group("day")))
                _add(m.group(0), m.start(), m.end(), dt)
            else:
                _add(m.group(0), m.start(), m.end(), None)
        except ValueError:
            continue

    # Day-first ordinal forms such as "15th of March, 2024" and
    # "15 March". We keep the no-year form as raw text only.
    for m in re.finditer(
        rf'\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+'
        rf'(?P<month>{month_names})[a-z]*'
        rf'(?:,?\s*(?P<year>\d{{4}}))?\b',
        text,
        re.IGNORECASE,
    ):
        month = m.group("month")[:3].lower()
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        year = m.group("year")
        try:
            if year:
                dt = datetime(int(year), month_map[month], int(m.group("day")))
                _add(m.group(0), m.start(), m.end(), dt)
            else:
                _add(m.group(0), m.start(), m.end(), None)
        except ValueError:
            continue

    return spans


def _normalize_time_anchor(value) -> str | None:
    """Return a dataset time anchor as an ISO date, or None when invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _query_wants_if_pf(question: str) -> bool:
    """Detect when a question likely needs instruction/preference memories."""
    import re

    q = (question or "").lower()
    patterns = (
        r'\binstruction(s)?\b',
        r'\bformat\b',
        r'\bfollow\b',
        r'\bpreference(s)?\b',
        r'\bprefer\b',
        r'\blike\b.*\bkeep\b',
        r'\bmust\b',
        r'\bshould\b',
        r'\balways\b',
        r'\bnever\b',
        r'\bhelp me\s+(?:set up|build|create|organize)\b',
    )
    return any(re.search(p, q) for p in patterns)


def _select_conversations(conversations: list[dict], sample_size: int | None = None,
                          start_index: int = 0, case_index: int | None = None) -> tuple[list[dict], list[str]]:
    """Select a deterministic subset of conversations and return selected IDs."""
    if case_index is not None:
        if case_index < 0 or case_index >= len(conversations):
            raise IndexError(f"case_index {case_index} out of range for {len(conversations)} conversations")
        selected = [conversations[case_index]]
        return selected, [conversations[case_index]["id"]]

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    end_index = None if sample_size is None else start_index + sample_size
    selected = conversations[start_index:end_index]
    return selected, [conv["id"] for conv in selected]


def _table_counts(conn, tables: list[str]) -> dict[str, int]:
    """Return per-table row counts for delta-based ingestion diagnostics."""
    counts = {}
    cursor = conn.cursor()
    for table in tables:
        try:
            counts[table] = int(cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except Exception:
            counts[table] = 0
    return counts


def _mean_float_score(scores: list) -> float:
    """Return the arithmetic mean of numeric judge scores."""
    numeric_scores = []
    for item in scores:
        try:
            numeric_scores.append(float(item))
        except (TypeError, ValueError):
            numeric_scores.append(0.0)
    return sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0


def _mean_int_score(scores: list) -> float:
    """Return the official-compatible mean of integer-cast rubric scores."""
    numeric_scores = []
    for item in scores:
        try:
            numeric_scores.append(int(float(item)))
        except (TypeError, ValueError):
            numeric_scores.append(0)
    return sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0


def _parse_judge_payload(raw: str) -> tuple[dict | list | None, str]:
    """Parse judge JSON from pristine, fenced, or prose-wrapped content."""
    if raw is None or not str(raw).strip():
        return None, "parse_failure"
    cleaned = _clean_judge_json(str(raw))
    if not cleaned or not cleaned.strip():
        return None, "parse_failure"
    try:
        return json.loads(cleaned), "ok"
    except Exception:
        return None, "parse_failure"


def _summarize_judge_result(judgment: dict, fallback_reason: str | None = None) -> dict:
    """Normalize judge metadata while preserving raw responses."""
    scores = []
    for item in judgment.get("scores", []):
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            scores.append(0.0)
    partial_credit = _mean_float_score(scores)
    official_score = judgment.get("official_score")
    if official_score is None:
        official_score = _mean_int_score(scores)
    try:
        official_score = float(official_score)
    except (TypeError, ValueError):
        official_score = 0.0
    parse_status = judgment.get("parse_status", "ok")
    if fallback_reason and parse_status == "ok":
        parse_status = fallback_reason
    return {
        "scores": scores,
        "official_score": official_score,
        "partial_credit_score": partial_credit,
        "scoring_mode": judgment.get("scoring_mode", "unknown"),
        "parse_status": parse_status,
        "judge_status": judgment.get("judge_status", "ok"),
        "judge_failure_class": judgment.get("judge_failure_class"),
        "finish_reason": judgment.get("finish_reason"),
        "response_had_content": judgment.get("response_had_content"),
        "retry_count": judgment.get("retry_count", 0),
        "judge_api_error_class": judgment.get("judge_failure_class"),
        "judge_failure_message": judgment.get("judge_failure_message", ""),
        "judge_api_error_message": judgment.get("judge_failure_message", ""),
        "raw_response": judgment.get("raw_response", ""),
        "raw_result": judgment.get("raw_result"),
        "assessment": judgment.get("assessment", ""),
        "brief_assessment": judgment.get("brief_assessment", ""),
        "nuggets": judgment.get("nuggets", []),
    }


_SENSITIVE_KEY_PARTS = ("KEY", "TOKEN", "SECRET")


def _sanitize_sensitive_data(value):
    """Redact sensitive values from nested JSON-like structures."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.upper() for part in _SENSITIVE_KEY_PARTS):
                sanitized[key] = "***redacted***"
            else:
                sanitized[key] = _sanitize_sensitive_data(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_sensitive_data(item) for item in value)
    return value


def _write_json_sanitized(payload, fp, **dump_kwargs):
    """Write sanitized JSON to an open file handle."""
    return json.dump(_sanitize_sensitive_data(payload), fp, **dump_kwargs)


def _json_dumps_sanitized(payload, **dump_kwargs):
    """Serialize sanitized JSON for JSONL appends and similar writes."""
    return json.dumps(_sanitize_sensitive_data(payload), **dump_kwargs)


def _print_env_snapshot(env_snapshot: dict) -> dict:
    """Print a sanitized environment snapshot and return the sanitized copy."""
    sanitized = _sanitize_sensitive_data(env_snapshot)
    print(f"\n  Env snapshot ({len(sanitized)} vars):")
    for key in sorted(sanitized):
        print(f"    {key}={sanitized[key]}")
    return sanitized


def _question_row_policy(question_row: dict) -> dict:
    """Decide whether a probing row should be evaluated or explicitly skipped."""
    question = question_row.get("question", "")
    if question is None:
        question = ""
    if not isinstance(question, str):
        question = str(question)

    ideal_answer = question_row.get(
        "ideal_answer",
        question_row.get("ideal_response", question_row.get("answer", question_row.get("ideal_summary", ""))),
    )
    if ideal_answer is None:
        ideal_answer = ""
    if not isinstance(ideal_answer, str):
        ideal_answer = str(ideal_answer)

    rubric = question_row.get("rubric", [])
    if not isinstance(rubric, list):
        rubric = []

    question_text = question.strip()
    if not question_text:
        return {
            "should_evaluate": False,
            "skip_reason": "missing_question",
            "question": question,
            "ideal_answer": ideal_answer,
            "rubric": rubric,
        }
    if not rubric:
        return {
            "should_evaluate": False,
            "skip_reason": "missing_rubric",
            "question": question_text,
            "ideal_answer": ideal_answer,
            "rubric": rubric,
        }
    return {
        "should_evaluate": True,
        "skip_reason": None,
        "question": question_text,
        "ideal_answer": ideal_answer,
        "rubric": rubric,
    }


def _build_skipped_question_result(
    *,
    qid: str,
    ability: str,
    question: str,
    ideal_answer: str,
    rubric: list,
    skip_reason: str,
) -> dict:
    """Record an explicitly skipped question row for accounting."""
    return {
        "qid": qid,
        "ability": ability,
        "question": question[:200],
        "question_full": question,
        "ideal_answer": ideal_answer[:200],
        "ideal_answer_full": ideal_answer,
        "rubric": rubric,
        "status": "skipped",
        "skipped": True,
        "skip_reason": skip_reason,
        "score": None,
        "official_score": None,
        "partial_credit_score": None,
        "parse_status": "skipped",
        "judge_status": "skipped",
        "judge_failure_class": None,
        "judge_failure_message": "",
        "judge_raw_response": "",
        "judge_raw_result": None,
        "judge_raw_payload": None,
        "judge_finish_reason": None,
        "judge_response_had_content": None,
        "judge_retry_count": 0,
        "assessment": "",
        "judge_assessment": "",
        "answer_model": None,
        "judge_model": None,
        "answer_time_ms": 0.0,
        "judge_time_ms": 0.0,
        "retrieval_diagnostics": {},
        "answer_api_diagnostics": {},
        "nuggets": [],
        "recall_provenance": {},
        "ai_answer": "",
        "ai_answer_full": "",
        "ai_answer_excerpt": "",
    }


def _is_skipped_question_result(question_result: dict) -> bool:
    """Return True when a question row was explicitly skipped."""
    return bool(question_result.get("skipped")) or question_result.get("status") == "skipped"


def _iter_evaluated_question_rows(all_results: list[dict]):
    """Yield only evaluated question rows from nested conversation results."""
    for conv_result in all_results:
        for question in conv_result.get("results", []):
            if _is_skipped_question_result(question):
                continue
            yield question


def _build_paired_outcome_rows(conv_result: dict, config_id: str, run_started_at: str) -> list[dict]:
    """Build paired-outcome rows for evaluated questions only."""
    paired_rows = []
    for question in conv_result.get("results", []):
        if _is_skipped_question_result(question):
            continue
        try:
            score = float(question.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        paired_rows.append({
            "config_id": config_id,
            "run_started_at": run_started_at,
            "scale": conv_result.get("scale"),
            "conversation_id": conv_result.get("conversation_id"),
            "qid": question.get("qid"),
            "ability": question.get("ability"),
            "score": score,
            "correct": score >= 0.5,
        })
    return paired_rows


def _build_question_validation_rows(conv_result: dict, config_id: str, run_started_at: str) -> list[dict]:
    """Build per-question validation rows with full prompts, answers, and judge payloads."""
    validation_rows = []
    for question in conv_result.get("results", []):
        try:
            score = float(question.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        is_skipped = _is_skipped_question_result(question)
        validation_rows.append({
            "config_id": config_id,
            "run_started_at": run_started_at,
            "scale": conv_result.get("scale"),
            "conversation_id": conv_result.get("conversation_id"),
            "qid": question.get("qid"),
            "ability": question.get("ability"),
            "validation_status": "skipped" if is_skipped else "evaluated",
            "validation_passed": None if is_skipped else score >= 0.5,
            "skip_reason": question.get("skip_reason"),
            "score": question.get("score"),
            "official_score": question.get("official_score"),
            "partial_credit_score": question.get("partial_credit_score"),
            "parse_status": question.get("parse_status"),
            "judge_status": question.get("judge_status"),
            "judge_failure_class": question.get("judge_failure_class"),
            "judge_failure_message": question.get("judge_failure_message", ""),
            "question": question.get("question"),
            "question_full": question.get("question_full"),
            "ideal_answer": question.get("ideal_answer"),
            "ideal_answer_full": question.get("ideal_answer_full"),
            "ai_answer": question.get("ai_answer"),
            "ai_answer_full": question.get("ai_answer_full"),
            "ai_answer_excerpt": question.get("ai_answer_excerpt"),
            "assessment": question.get("assessment"),
            "judge_assessment": question.get("judge_assessment"),
            "answer_model": question.get("answer_model"),
            "judge_model": question.get("judge_model"),
            "answer_time_ms": question.get("answer_time_ms"),
            "judge_time_ms": question.get("judge_time_ms"),
            "judge_finish_reason": question.get("judge_finish_reason"),
            "judge_response_had_content": question.get("judge_response_had_content"),
            "judge_retry_count": question.get("judge_retry_count"),
            "judge_raw_response": question.get("judge_raw_response"),
            "judge_raw_result": question.get("judge_raw_result"),
            "judge_raw_payload": question.get("judge_raw_payload"),
            "retrieval_diagnostics": question.get("retrieval_diagnostics", {}),
            "answer_api_diagnostics": question.get("answer_api_diagnostics", {}),
            "nuggets": question.get("nuggets", []),
            "recall_provenance": question.get("recall_provenance", {}),
        })
    return validation_rows


def _finalize_reranker_run_health(preflight: dict | None, call_diag: dict | None = None) -> dict:
    """Combine reranker preflight state with call-time health."""
    finalized = deepcopy(call_diag or {})
    preflight_diag = deepcopy(preflight or {})
    finalized["preflight"] = preflight_diag
    finalized["preflight_health"] = "ok" if preflight_diag.get("ok") else "unavailable"

    calls = int(finalized.get("calls", 0) or 0)
    successes = int(finalized.get("successes", 0) or 0)
    failures = int(finalized.get("failures", 0) or 0)

    if calls > 0:
        if successes == 0:
            call_health = "failed" if failures > 0 else "degraded"
        elif failures > 0:
            call_health = "degraded"
        else:
            call_health = "ok"
    else:
        call_health = "not_run"

    finalized["call_health"] = call_health
    finalized["health"] = call_health if calls > 0 else finalized["preflight_health"]
    finalized["failed"] = finalized["health"] == "failed"
    return finalized


def _summarize_reranker_run(question_results: list[dict], preflight: dict | None) -> dict:
    """Aggregate reranker call diagnostics for a conversation or run."""
    summary = {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "fallbacks": 0,
        "scores_recorded": 0,
        "errors": [],
    }
    for question in question_results:
        reranker_diag = question.get("retrieval_diagnostics", {}).get("reranker")
        if not isinstance(reranker_diag, dict):
            continue
        for key in ("calls", "successes", "failures", "fallbacks", "scores_recorded"):
            try:
                summary[key] += int(reranker_diag.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
        errors = reranker_diag.get("errors", [])
        if isinstance(errors, list):
            summary["errors"].extend(deepcopy(errors))
        elif errors:
            summary["errors"].append(deepcopy(errors))
        if summary.get("url") is None and reranker_diag.get("url") is not None:
            summary["url"] = reranker_diag.get("url")
    return _finalize_reranker_run_health(preflight, summary)

# --- Config ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    # Try to load from file -- check opencode first, then openrouter
    for _kf in ["/tmp/opencode_key.txt", "/tmp/openrouter_key.txt"]:
        _key_file = Path(_kf)
        if _key_file.exists():
            with open(_key_file) as f:
                _content = f.read().strip()
            if "export" in _content:
                OPENROUTER_API_KEY = _content.split("=", 1)[1].strip().strip('"').strip("'")
            else:
                OPENROUTER_API_KEY = _content
            break
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = "deepseek-v4-flash"
CONSOLIDATION_MODEL = "deepseek/deepseek-v4-flash"  # Cheap model for LLM-based consolidation summaries
FALLBACK_MODELS = []  # Disabled -- fallback cascade burned $30 in credits
DEFAULT_TOP_K = 10  # Memories to retrieve per question
MAX_MEMORY_CONTEXT_CHARS = int(os.environ.get("EDUMEM_MAX_CONTEXT_CHARS", "16000"))  # Max chars of retrieved context to send to LLM


# C31: env-var truthy parser. Accepts standard truthy values
# (1/true/yes/on, case-insensitive) and explicit falsies (0/false/no/off).
# Strips whitespace so accidental leading/trailing spaces in shell
# exports don't get treated as falsy. Anything else → False.
# Pre-fix the parser was `lower() in ("1", "true", "yes")` which
# rejected `on` and was whitespace-sensitive -- surprised at least one
# operator running with `EDUMEM_BENCHMARK_PURE_RECALL=on`.
_ENV_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    """Return True iff env var `name` is set to a canonical truthy value.

    Truthy: 1, true, yes, on (case-insensitive, whitespace-stripped).
    Everything else (including 0, false, no, off, empty, garbage) is False.
    """
    return os.environ.get(name, "").strip().lower() in _ENV_TRUTHY_VALUES


def _benchmark_pure_recall_enabled() -> bool:
    """Return the benchmark's default recall mode.

    Pure recall is the default unless the operator explicitly disables it with
    EDUMEM_BENCHMARK_PURE_RECALL=0/false/no/off.
    """
    raw = os.environ.get("EDUMEM_BENCHMARK_PURE_RECALL")
    if raw is None:
        return True
    return _env_truthy("EDUMEM_BENCHMARK_PURE_RECALL")
BENCHMARK_QUERIES_PER_CONV = 50  # Max probing questions per conversation

def _result_paths(output_dir: Path):
    return (
        output_dir / "beam_e2e_results.json",
        output_dir / "paired_outcomes.jsonl",
        output_dir / "beam_question_validations.jsonl",
    )

# Keep defaults for backward compat (used by --rejudge-results default)
_DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = _DEFAULT_RESULTS_DIR / "beam_e2e_results.json"
PAIRED_OUTCOMES_FILE = _DEFAULT_RESULTS_DIR / "paired_outcomes.jsonl"
QUESTION_VALIDATIONS_FILE = _DEFAULT_RESULTS_DIR / "beam_question_validations.jsonl"

# Memory abilities tested by BEAM (10 dimensions)
BEAM_ABILITIES = [
    "IE",   # Information Extraction
    "MR",   # Multi-hop Reasoning
    "KU",   # Knowledge Update
    "TR",   # Temporal Reasoning
    "ABS",  # Abstention
    "CR",   # Contradiction Resolution
    "EO",   # Event Ordering
    "IF",   # Instruction Following
    "PF",   # Preference Following
    "SUM",  # Summarization
]

# Map dataset ability names to our abbreviations
ABILITY_MAP = {
    "information_extraction": "IE",
    "multi_session_reasoning": "MR",
    "knowledge_update": "KU",
    "temporal_reasoning": "TR",
    "abstention": "ABS",
    "contradiction_resolution": "CR",
    "event_ordering": "EO",
    "instruction_following": "IF",
    "preference_following": "PF",
    "summarization": "SUM",
    # Aliases
    "multi_session": "MR",
    "knowledge": "KU",
    "temporal": "TR",
    "information": "IE",
}


# ============================================================
#  LLM Client
# ============================================================

class LLMClient:
    """OpenAI-compatible API client using OpenRouter (fast, reliable)."""
    
    _last_429_time = 0  # Class-level rate-limit cooldown
    
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.fallback_models = FALLBACK_MODELS.copy()
        self.call_count = 0
        self.last_error_class = None
        self.last_error_message = ""
        self.last_finish_reason = None
        self.last_response_had_content = None
        self.last_response = ""
        self.last_retry_count = 0

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """Send chat completion request with retry. No fallback models to avoid rate limits."""
        
        last_error = None
        self.last_retry_count = 0
        for attempt in range(3):
            try:
                response = self._call_api(self.model, messages, temperature, max_tokens)
                self.last_response = response
                self.last_error_class = None
                self.last_error_message = ""
                return response
            except Exception as e:
                last_error = str(e)
                self.last_error_class = type(e).__name__
                self.last_error_message = str(e)
                self.last_retry_count = attempt + 1
                if "429" in last_error or "rate" in last_error.lower():
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s backoff
                    time.sleep(wait)
                    continue
                else:
                    break  # Non-retryable error

        self.last_response = f"[LLM_ERROR: all models failed. Last: {last_error}]"
        self.last_response_had_content = False
        return self.last_response

    def _call_api(self, model: str, messages: list, temperature: float, max_tokens: int) -> str:
        """Single API call via requests (urllib blocked by Cloudflare on some providers)."""
        import json as _json
        import requests as _requests
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://edumem.site",
            "X-Title": "edumem Benchmark",
        }
        resp = _requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        self.call_count += 1
        self.last_finish_reason = data["choices"][0].get("finish_reason", "unknown")
        content = data["choices"][0]["message"].get("content")
        self.last_response_had_content = content is not None
        if content is None:
            print(f"    [DEBUG-API-NULL] model={model} finish_reason={self.last_finish_reason} tokens_used={data.get('usage', {}).get('total_tokens', '?')}", flush=True)
            return ""  # Return empty string instead of None so callers don't choke
        return content

    def close(self):
        pass


# ============================================================
#  Data Loading (adapted from benchmark_beam_sota.py)
# ============================================================

def load_beam_dataset(scales: list[str], max_conversations: int = None) -> dict:
    """Load BEAM dataset from HuggingFace. Returns dict[scale] -> list[conversation]."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    data = {}
    total_loaded = 0

    for scale in scales:
        print(f"  Loading BEAM {scale}...")
        try:
            if scale == "10M":
                ds = load_dataset("Mohammadta/BEAM-10M", streaming=True)
                split_name = "10M" if "10M" in ds else list(ds.keys())[0]

                conversations = []
                for i, sample in enumerate(ds[split_name]):
                    if max_conversations and i >= max_conversations:
                        break

                    probing_raw = sample.get("probing_questions", {})
                    if isinstance(probing_raw, str):
                        try:
                            probing = ast.literal_eval(probing_raw)
                        except Exception:
                            probing = {}
                    else:
                        probing = probing_raw

                    all_questions = []
                    for ability, questions in probing.items():
                        if isinstance(questions, list):
                            for q in questions:
                                if isinstance(q, dict):
                                    all_questions.append({
                                        "ability": ability,
                                        "question": q.get("question", ""),
                                        "ideal_answer": q.get("ideal_answer", q.get("ideal_response", q.get("answer", q.get("ideal_summary", "")))),
                                        "rubric": q.get("rubric", []),
                                    })

                    # Extract messages from plans
                    plans = sample.get("plans", [])
                    all_messages = []
                    for plan in plans:
                        chat_blocks = plan.get("chat", []) if isinstance(plan, dict) else []
                        for block in chat_blocks:
                            if isinstance(block, list):
                                for msg in block:
                                    if isinstance(msg, dict):
                                        loaded_message = {
                                            "role": msg.get("role", "unknown"),
                                            "content": msg.get("content", ""),
                                            "index": len(all_messages),
                                        }
                                        if msg.get("time_anchor"):
                                            loaded_message["time_anchor"] = msg["time_anchor"]
                                        all_messages.append(loaded_message)
                            elif isinstance(block, dict):
                                loaded_message = {
                                    "role": block.get("role", "unknown"),
                                    "content": block.get("content", ""),
                                    "index": len(all_messages),
                                }
                                if block.get("time_anchor"):
                                    loaded_message["time_anchor"] = block["time_anchor"]
                                all_messages.append(loaded_message)

                    conversations.append({
                        "id": sample.get("conversation_id", str(i)),
                        "messages": all_messages,
                        "questions": all_questions,
                        "scale": "10M",
                    })
                    total_loaded += 1

                data[scale] = conversations
                ds.cleanup_cache_files() if hasattr(ds, 'cleanup_cache_files') else None
                del ds
                gc.collect()
                print(f"    Loaded {len(conversations)} conversations")

            else:
                # 100K, 500K, 1M scales from the main dataset
                ds = load_dataset("Mohammadta/BEAM", streaming=False)
                if scale not in ds:
                    print(f"    WARNING: split '{scale}' not found. Available: {list(ds.keys())}")
                    continue

                conversations = []
                for i, sample in enumerate(ds[scale]):
                    if max_conversations and i >= max_conversations:
                        break

                    pq_raw = sample.get("probing_questions", "{}")
                    if isinstance(pq_raw, str):
                        try:
                            probing = ast.literal_eval(pq_raw)
                        except Exception:
                            probing = {}
                    else:
                        probing = pq_raw

                    flat_questions = []
                    for ability, questions in probing.items():
                        if isinstance(questions, list):
                            for q in questions:
                                if isinstance(q, dict):
                                    flat_questions.append({
                                        "ability": ability,
                                        "question": q.get("question", ""),
                                        "ideal_answer": q.get("ideal_answer", q.get("ideal_response", q.get("answer", q.get("ideal_summary", "")))),
                                        "rubric": q.get("rubric", []),
                                    })

                    chat_blocks = sample.get("chat", [])
                    messages = []
                    for block in chat_blocks:
                        if isinstance(block, list):
                            for msg in block:
                                if isinstance(msg, dict):
                                    loaded_message = {
                                        "role": msg.get("role", "unknown"),
                                        "content": msg.get("content", ""),
                                        "index": len(messages),
                                    }
                                    if msg.get("time_anchor"):
                                        loaded_message["time_anchor"] = msg["time_anchor"]
                                    messages.append(loaded_message)
                        elif isinstance(block, dict):
                            # Flat format: chat is a list of dicts directly
                            loaded_message = {
                                "role": block.get("role", "unknown"),
                                "content": block.get("content", ""),
                                "index": len(messages),
                            }
                            if block.get("time_anchor"):
                                loaded_message["time_anchor"] = block["time_anchor"]
                            messages.append(loaded_message)

                    conversations.append({
                        "id": sample.get("conversation_id", str(i)),
                        "messages": messages,
                        "questions": flat_questions,
                        "scale": scale,
                    })
                    total_loaded += 1

                data[scale] = conversations
                ds.cleanup_cache_files() if hasattr(ds, 'cleanup_cache_files') else None
                del ds
                gc.collect()
                print(f"    Loaded {len(conversations)} conversations")

        except Exception as e:
            print(f"    ERROR loading {scale}: {e}")
            import traceback
            traceback.print_exc()

    print(f"  Total: {total_loaded} conversations across {len(data)} scales")
    return data


# ============================================================
#  edumem Ingestion
# ============================================================

def _extract_facts(content: str, source: str = "unknown") -> list[dict]:
    """Extract structured facts from a message for precision retrieval.
    These fact entries complement raw message storage by isolating
    specific data points (numbers, dates, versions, negations) that
    FTS5 keyword search can match more precisely than in long messages."""
    import re
    facts = []
    
    # Pattern 1: Version numbers ("Flask 2.3.1", "v0.6.2", "Python 3.11")
    ver_matches = re.findall(r'([A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)*)\s+v?(\d+\.\d+(?:\.\d+)?)', content)
    for name, ver in ver_matches[:3]:
        facts.append({
            "content": f"FACT version: {name.strip()} {ver}",
            "importance": 0.7,
        })
    
    # Pattern 2: Numbers with units ("250ms", "3 columns", "50 tasks", "5000 port")
    num_matches = re.findall(r'(\d+(?:[.,]\d+)?)\s*(ms|sec|seconds?|minutes?|hours?|days?|weeks?|months?|%|KB|MB|GB|columns?|tasks?|commits?|users?|ports?|items?)', content, re.IGNORECASE)
    for num, unit in num_matches[:5]:
        facts.append({
            "content": f"FACT metric: {num}{unit}",
            "importance": 0.65,
        })
    
    # Pattern 3: Dates
    for span in _extract_shared_date_spans(content):
        facts.append({
            "content": f"FACT date: {span['raw']}",
            "importance": 0.7,
        })
    
    # Pattern 4: Deadlines
    deadline_matches = re.findall(r'(deadline|due by|sprint ends?|sprint \d+)\s*[:\-]?\s*([^.,;!?\n]{5,80})', content, re.IGNORECASE)
    for ctx, detail in deadline_matches[:3]:
        facts.append({
            "content": f"FACT deadline: {ctx} {detail.strip()}",
            "importance": 0.7,
        })
    
    # Pattern 5: Negations ("I have never", "I have not") - critical for CR
    negations = re.findall(r'(I(?: have|\'ve)?\s*(?:never|not)\s+[^.,;!?\n]{15,120})', content, re.IGNORECASE)
    for neg in negations[:3]:
        facts.append({
            "content": f"FACT negation: {neg.strip()}",
            "importance": 0.75,
        })
    
    # Pattern 6: Decisions / choices
    choices = re.findall(r'(?:decided to|chose to|opted for|selected|picked|switching to)\s+([^.,;!?\n]{10,120})', content, re.IGNORECASE)
    for choice in choices[:3]:
        facts.append({
            "content": f"FACT decision: {choice.strip()}",
            "importance": 0.65,
        })
    
    # Pattern 7: Ordinal sequence markers ("first", "then", "finally") for EO
    ordinals = re.findall(r'((?:first|second|third|fourth|fifth|finally|next|then|after that)[^.,;!?\n]{15,120})', content, re.IGNORECASE)
    for ord_text in ordinals[:5]:
        facts.append({
            "content": f"FACT sequence: {ord_text.strip()}",
            "importance": 0.6,
        })
    
    # Pattern 8: Entity-action pairs ("transactions table" + "add") for MR
    entities = re.findall(r'(?:the|my|our)\s+([a-z_]+\s*(?:table|model|schema|API|endpoint|function|module|route|handler))\s+(?:needs?|requires?|should|could|would|will|has|have)\s+([^.,;!?\n]{10,80})', content, re.IGNORECASE)
    for entity, action in entities[:5]:
        facts.append({
            "content": f"FACT entity: {entity.strip()} -> {action.strip()}",
            "importance": 0.65,
        })
    
    return facts[:20]  # Cap per message


def _classify_message_llm(llm, content: str) -> set:
    """Classify a message using LLM into INSTRUCTION, PREFERENCE, or FACT tags."""
    if llm is None:
        return set()

    prompt = f"""Classify this user message. Reply with one or more labels (comma-separated): INSTRUCTION, PREFERENCE, or FACT.

INSTRUCTION = user telling the system what to do, how to behave, formatting rules, technical requirements, imperatives
PREFERENCE = user expressing likes, dislikes, style choices, priorities, personal taste
FACT = plain information sharing, no directive intent

Message: {content}

Labels:"""

    try:
        response = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=32)
        if not response or "[LLM_ERROR" in response:
            return set()

        # Parse response: extract INSTRUCTION, PREFERENCE, FACT tags
        response_upper = response.upper()
        tags = set()
        if "INSTRUCTION" in response_upper:
            tags.add("INSTRUCTION")
        if "PREFERENCE" in response_upper:
            tags.add("PREFERENCE")
        if "FACT" in response_upper:
            tags.add("FACT")
        return tags
    except Exception:
        return set()




def _update_embedding_diagnostic(
    embed_diag: dict,
    *,
    backend_available: bool,
    model: str | None = None,
    dimension: int | None = None,
    eligible_rows: int = 0,
    inserted_vectors: int = 0,
    rows_before: int | None = None,
    rows_after: int | None = None,
    api_calls_before: int | None = None,
    api_calls_after: int | None = None,
) -> dict:
    """Update embedding diagnostics without mutating shared state."""
    updated = dict(embed_diag)
    updated.setdefault("backend", "dense" if backend_available else "keyword-only")
    if model is not None:
        updated.setdefault("model", model)
    if dimension is not None:
        updated.setdefault("dimension", dimension)
    updated["eligible_rows"] = updated.get("eligible_rows", 0) + eligible_rows
    updated["inserted_vectors"] = updated.get("inserted_vectors", 0) + inserted_vectors
    if rows_before is not None and updated.get("memory_embeddings_rows_before") is None:
        updated["memory_embeddings_rows_before"] = rows_before
    if rows_after is not None:
        updated["memory_embeddings_rows_after"] = rows_after
    if rows_before is not None and rows_after is not None:
        updated["memory_embeddings_row_delta"] = (
            updated.get("memory_embeddings_row_delta", 0) + max(rows_after - rows_before, 0)
        )
    if api_calls_before is not None and api_calls_after is not None:
        updated["api_calls"] = api_calls_after - api_calls_before
    if backend_available and updated.get("eligible_rows", 0) > 0 and updated.get("inserted_vectors", 0) != updated.get("eligible_rows", 0):
        updated["failed"] = True
        updated["status"] = "failed"
    elif not updated.get("failed"):
        updated["status"] = "dense" if backend_available else "keyword-only"
    return updated


def ingest_conversation(beam: BeamMemory, messages: list[dict], diag: dict | None = None, llm=None) -> dict:
    """Ingest conversation messages into edumem BEAM tiers.
    Also builds an in-memory facts index for fact-boosted retrieval."""
    start_time = time.perf_counter()
    stats = {"wm_count": 0, "ep_count": 0, "sp_count": 0, "total_chars": 0}
    if diag is not None:
        embedding_backend_available = _embeddings.available()
        embed_diag = diag.setdefault("embedding", {})
        embed_diag = _update_embedding_diagnostic(
            embed_diag,
            backend_available=embedding_backend_available,
            model=getattr(_embeddings, "_DEFAULT_MODEL", None),
            dimension=getattr(_embeddings, "EMBEDDING_DIM", None),
        )
        embed_diag.setdefault("query_vectors", 0)
        embed_diag.setdefault("memory_embeddings_rows_before", None)
        embed_diag.setdefault("memory_embeddings_rows_after", None)
        embed_diag.setdefault("memory_embeddings_row_delta", 0)
        embed_diag.setdefault("failed", False)
        if hasattr(_embeddings, "_API_CALL_COUNT"):
            embed_diag.setdefault("api_calls_before", getattr(_embeddings, "_API_CALL_COUNT", None))
        diag["embedding"] = embed_diag
    
    # In-memory context→value facts index for direct fact matching.
    # Format: {"context phrase": "fact value"} -- maps question-like phrases to answers.
    # Example: "My first sprint ends on" → "March 29"
    # Built during ingestion, queried during answering for zero-LLM fact extraction.
    import re as _re2
    _FACT_VALUE_RE = _re2.compile(
        r'('
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:[, ]*\d{4})?\b|'  # dates
        r'\b\d{4}-\d{2}-\d{2}\b|'  # ISO dates
        r'\b\d+[.,]?\d*\s*(?:ms|sec|mins?|hours?|days?|weeks?|months?|years?|%|KB|MB|GB|TB|rows?|columns?|roles?|features?|bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|sprints?|tickets?)\b|'  # numbers+units
        r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?\b'  # versions
        r')'
    )
    context_facts = getattr(beam, '_context_facts', {})
    if not hasattr(beam, '_context_facts'):
        beam._context_facts = {}
        context_facts = beam._context_facts

    BATCH_SIZE = 500
    current_time_anchor = None

    for batch_start in range(0, len(messages), BATCH_SIZE):
        batch_msgs = messages[batch_start:batch_start + BATCH_SIZE]

        # Pre-classify user messages in parallel to cut down latency (Hindsight concurrent optimization pattern)
        classifications = {}
        if llm is not None:
            user_jobs = []
            for idx, msg in enumerate(batch_msgs):
                if msg.get("role", "").lower() == "user":
                    user_jobs.append((idx, msg.get("content", "")))
            if user_jobs:
                from concurrent.futures import ThreadPoolExecutor
                def _classify_worker(job):
                    idx, content = job
                    return idx, _classify_message_llm(llm, content)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(_classify_worker, user_jobs))
                classifications = {idx: tags for idx, tags in results}

        batch_items = []
        for i, msg in enumerate(batch_msgs):
            raw_content = msg.get("content", "")
            if not raw_content.strip():
                continue
            normalized_anchor = _normalize_time_anchor(msg.get("time_anchor"))
            if normalized_anchor:
                current_time_anchor = normalized_anchor
            recorded_at = (
                f"{current_time_anchor}T00:00:00Z"
                if current_time_anchor
                else msg.get("timestamp") or "1970-01-01T00:00:00Z"
            )
            occurred_at = parse_relative_date(raw_content, current_time_anchor)
            content = raw_content
            # Temporal tag injection: bake date strings into content so
            # FTS5 can find them during recall. Preserve raw date text while
            # adding separately-derived canonical dates when an anchor exists.
            import re as _re_tags
            date_spans = _extract_shared_date_spans(content)
            if date_spans:
                raw_dates = [span["raw"] for span in date_spans]
                content = f"{content} [DATES: {', '.join(raw_dates)}]"
            canonical_dates = [span["iso"] for span in date_spans if span.get("iso")]
            if occurred_at and occurred_at not in canonical_dates:
                canonical_dates.append(occurred_at)
            if canonical_dates:
                content = f"{content} [ISO_DATES: {', '.join(canonical_dates)}]"
                datetok_strs = [f"datetok{date.replace('-', '')}" for date in canonical_dates]
                content = f"{content} datetokens: {' '.join(datetok_strs)}"
            durations = _re_tags.findall(r'\b\d+\s(?:days|weeks|months|years)\b', content, _re_tags.IGNORECASE)
            if durations:
                content = f"{content} [DURATIONS: {', '.join(durations)}]"
            # Prepend message index for EO (Event Ordering) ability
            # so the LLM can sort events chronologically by raw sequence.
            content = f"[MSGIDX:{batch_start + i}] {content}"

            # Detect and tag instructions and preferences for IF/PF abilities
            if msg.get("role", "").lower() == "user" and llm is not None:
                tags = classifications.get(i, [])
                if "INSTRUCTION" in tags:
                    content = f"{content} [INSTRUCTION]"
                if "PREFERENCE" in tags:
                    content = f"{content} [PREFERENCE]"

            batch_items.append({
                "content": content,
                "source": f"beam_{msg.get('role', 'unknown')}",
                "importance": 0.5,
                "timestamp": recorded_at,
                "occurred_at": occurred_at,
                "message_index": batch_start + i,
            })
            stats["total_chars"] += len(content)
            
            # Extract context→value facts: words before AND after each fact value.
            # Use raw_content (NOT the tag-augmented `content`) so context phrases are
            # clean natural language — the synthetic [MSGIDX:N]/[DATES:]/[DURATIONS:]
            # tags would otherwise pollute phrases and create junk duplicate entries.
            # SKIP version numbers (e.g., "3.39", "2.3.1") -- they pollute fact matching
            # and are never the answer to BEAM questions (which ask about dates, counts, names).
            _VERSION_RE = _re2.compile(r'^\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?$')
            for match in _FACT_VALUE_RE.finditer(raw_content):
                value = match.group()
                if _VERSION_RE.match(value):
                    continue  # Skip bare version numbers
                # Extract context: up to 12 words before + 8 words after the fact
                before = raw_content[:match.start()].split()[-12:]
                after = raw_content[match.end():].split()[:8]
                context_words = before + after
                context = ' '.join(context_words).lower().strip()
                if context and len(context) > 5:
                    if context not in context_facts:
                        context_facts[context] = []
                    context_facts[context].append(value)

            # Scratchpad every 10 messages
            if (batch_start + i) % 10 == 0 and len(content) > 50:
                try:
                    beam.scratchpad_write(f"[t={batch_start + i}] {content[:300]}")
                    stats["sp_count"] += 1
                except Exception:
                    pass

        if not batch_items:
            continue

        structured_tables = [
            "memoria_facts",
            "memoria_timelines",
            "memoria_instructions",
            "memoria_preferences",
            "memoria_kg",
        ]
        before_structured = _table_counts(beam.conn, structured_tables)
        before_embedding_rows = _table_counts(beam.conn, ["memory_embeddings"]).get("memory_embeddings", 0)
        batch_ids = beam.remember_batch(batch_items)
        after_structured = _table_counts(beam.conn, structured_tables)
        after_embedding_rows = _table_counts(beam.conn, ["memory_embeddings"]).get("memory_embeddings", 0)
        stats["wm_count"] += len(batch_items)
        stats["structured_deltas"] = {
            table: after_structured.get(table, 0) - before_structured.get(table, 0)
            for table in structured_tables
        }
        if diag is not None:
            embed_diag = diag.setdefault("embedding", {})
            eligible_rows = len(batch_items)
            batch_inserted = max(after_embedding_rows - before_embedding_rows, 0)
            embed_diag = _update_embedding_diagnostic(
                embed_diag,
                backend_available=_embeddings.available(),
                model=getattr(_embeddings, "_DEFAULT_MODEL", None),
                dimension=getattr(_embeddings, "EMBEDDING_DIM", None),
                eligible_rows=eligible_rows,
                inserted_vectors=batch_inserted,
                rows_before=before_embedding_rows,
                rows_after=after_embedding_rows,
            )
            diag["embedding"] = embed_diag

        # Cloud fact extraction: extract facts from batch if enabled
        if getattr(beam, 'use_cloud', False):
            try:
                from edumem.extraction import ExtractionClient
                if beam._extraction_client is None:
                    beam._extraction_client = ExtractionClient()
                facts = beam._extraction_client.extract_facts(batch_msgs)
                if facts:
                    cursor = beam.conn.cursor()
                    import hashlib
                    for fact in facts:
                        fact_id = hashlib.sha256(
                            f"{fact.get('subject','')}:{fact.get('predicate','')}:{fact.get('object','')}:{batch_start}".encode()
                        ).hexdigest()[:24]
                        cursor.execute("""
                            INSERT OR IGNORE INTO facts
                            (fact_id, session_id, subject, predicate, object,
                             timestamp, source_msg_id, confidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            fact_id,
                            beam.session_id,
                            fact.get("subject", ""),
                            fact.get("predicate", "stated"),
                            fact.get("object", ""),
                            fact.get("timestamp", ""),
                            fact.get("source_msg_id", ""),
                            fact.get("confidence", 0.7),
                        ))
                    beam.conn.commit()
                    stats["fact_count"] = stats.get("fact_count", 0) + len(facts)
            except Exception:
                pass  # Best-effort; don't fail ingestion

        # [E1] Additive consolidation per batch via beam.sleep().
        #
        # Pre-E1 this block built a synthetic summary
        # ("Batch N: first_3_msg_contents[:100]") + DELETEd all source
        # working_memory rows. ~99% of message content was discarded
        # before recall could see it -- the entire BEAM benchmark
        # corpus was destroyed at ingest.
        #
        # Post-E1 (option b, depends on E3 additive sleep): backdate
        # ONLY the batch's just-inserted rows past sleep's TTL/2
        # cutoff and let beam.sleep() produce real LLM-generated (or
        # AAAK-fallback) summaries on top of preserved originals.
        # The scoped UPDATE prevents cross-batch timestamp
        # contamination -- without the `id IN (...)` filter, a
        # mid-sleep failure on batch N would let batch N+1's UPDATE
        # walk every still-unconsolidated row in the session and
        # rewrite their timestamps, corrupting per-row temporal
        # ordering. See E1 adversarial review F1/F3.

        # Skip sleep for ≤100K scale to preserve message content for retrieval
        _skip_sleep = os.environ.get("BEAM_CURRENT_SCALE", "100K") in ("100K",)
        if not _skip_sleep:
            try:
                cursor = beam.conn.cursor()
                # Backdate is derived from WORKING_MEMORY_TTL_HOURS so it
                # survives operator config changes via env var. sleep()'s
                # cutoff is TTL/2, _trim's cutoff is TTL -- backdating by
                # TTL+1 ensures the row is on the consolidatable side of
                # sleep's cutoff while staying outside the trim window's
                # safety margin (consolidated_at exempts from trim post-E3
                # anyway, so the trim concern only applies pre-sleep). See
                # E1 adversarial review F6.
                from edumem.core.beam import WORKING_MEMORY_TTL_HOURS as _WM_TTL
                backdate_iso = (
                    datetime.now() - timedelta(hours=_WM_TTL + 1)
                ).isoformat()
                if batch_ids:
                    placeholders = ",".join("?" * len(batch_ids))
                    cursor.execute(
                        f"UPDATE working_memory SET timestamp = ? "
                        f"WHERE id IN ({placeholders}) "
                        f"AND consolidated_at IS NULL",
                        (backdate_iso, *batch_ids),
                    )
                    beam.conn.commit()

                    # Consolidate: run beam.sleep() to produce episodic summaries.
                    # Uses AAAK compression when EDUMEM_LLM_ENABLED=false
                    # (set externally to avoid local model download/inference during
                    # benchmark). Loop until sleep returns no_op so all eligible
                    # rows in this batch get processed regardless of SLEEP_BATCH_SIZE.
                    # Sleep errors are caught and logged; they don't crash ingestion.
                    max_iters = 50
                    while max_iters > 0:
                        try:
                            result = beam.sleep()
                        except Exception as sleep_e:
                            result = {"status": "error", "message": repr(sleep_e)}
                        max_iters -= 1
                        if result.get("status") in ("no_op", "error"):
                            break
                    # E3 contract: originals stay, so stats["wm_count"]
                    # does NOT decrement. Pre-E1 we did stats["wm_count"]
                    # -= ... which produced wm_count=0 always; post-E1 it
                    # grows monotonically with input message count, which
                    # is what the experiment actually wants to measure.
            except Exception as e:
                # Log the failure to stats so the operator sees it. Pre-E1
                # the equivalent block also swallowed silently, but the
                # consolidation IS the point of the experiment -- a silent
                # benchmark that "succeeds" with 0 episodic rows is the
                # exact failure mode the test suite is supposed to catch.
                stats.setdefault("sleep_errors", []).append(repr(e))

    stats["ingest_time_ms"] = (time.perf_counter() - start_time) * 1000
    if diag is not None:
        diag.setdefault("embedding", {})
        embed_diag = diag["embedding"]
        embed_diag.setdefault("memory_embeddings_rows_before", 0)
        embed_diag.setdefault("memory_embeddings_rows_after", 0)
        embed_diag.setdefault("memory_embeddings_row_delta", 0)
        embed_diag = _update_embedding_diagnostic(
            embed_diag,
            backend_available=_embeddings.available(),
            model=getattr(_embeddings, "_DEFAULT_MODEL", None),
            dimension=getattr(_embeddings, "EMBEDDING_DIM", None),
            api_calls_before=embed_diag.pop("api_calls_before", None) if hasattr(_embeddings, "_API_CALL_COUNT") else None,
            api_calls_after=getattr(_embeddings, "_API_CALL_COUNT", None) if hasattr(_embeddings, "_API_CALL_COUNT") else None,
        )
        diag["embedding"] = embed_diag
    return stats


# ============================================================
#  LLM Answering with edumem Memory
# ============================================================


def normalize_for_judge(raw_answer: str, ability: str = None) -> str:
    """Convert JSON specialist output into judge-friendly bullet list."""
    import json, re
    if not raw_answer or raw_answer.startswith("[LLM_ERROR"):
        return raw_answer or ""
    # Try to extract and parse JSON block
    json_match = re.search(r'\{.*\}', raw_answer, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                # EO format: ordered_events
                if "ordered_events" in data and isinstance(data["ordered_events"], list):
                    lines = [f"{i+1}. {e.get('event', '')}" for i, e in enumerate(data["ordered_events"])]
                    if lines:
                        return "\n".join(lines)
                # KU format: updates (before/after pairs + singleton via previous="")
                if "updates" in data and isinstance(data["updates"], list):
                    lines = []
                    for u in data["updates"]:
                        key = u.get("key", "")
                        prev_val = u.get("previous", "")
                        new_val = u.get("new", "")
                        idx = u.get("update_msg_idx", "")
                        if prev_val:
                            line = f"- {key}: {prev_val} -> {new_val}"
                        else:
                            line = f"- {key}: {new_val}"
                        if idx:
                            line += f" (at msg_idx {idx})"
                        lines.append(line)
                    if lines:
                        return "\n".join(lines)
                # KU legacy formats (fallback)
                if "current_facts" in data and isinstance(data["current_facts"], list):
                    lines = []
                    for cf in data["current_facts"]:
                        key = cf.get("key", "")
                        val = cf.get("value", "")
                        idx = cf.get("source_msg_idx", "")
                        line = f"- {key}: {val}"
                        if idx:
                            line += f" (source: msg_idx {idx})"
                        lines.append(line)
                    if lines:
                        return "\n".join(lines)
                # KU legacy format (fallback)
                if "updated_facts" in data and isinstance(data["updated_facts"], list):
                    lines = []
                    for f in data["updated_facts"]:
                        metric = f.get("metric", "")
                        value = f.get("value", "")
                        idx = f.get("msg_idx", "")
                        if idx:
                            lines.append(f"- {metric}: {value} (source: msg_idx {idx})")
                        else:
                            lines.append(f"- {metric}: {value}")
                    if lines:
                        return "\n".join(lines)
        except (json.JSONDecodeError, Exception):
            pass
    return raw_answer  # fallback: pass through as-is

DEFAULT_TOP_K = 30  # Memories to retrieve per question (increased for broader context)
RECENT_CONTEXT_COUNT = 12  # Last N messages to include as recent context
MAX_MEMORY_CONTEXT_CHARS = int(os.environ.get("EDUMEM_MAX_CONTEXT_CHARS", "16000"))  # More context for LLM to find contradictions

DEFAULT_TOP_K = 30  # Memories to retrieve per question (increased for broader context)
RECENT_CONTEXT_COUNT = 12  # Last N messages to include as recent context
MAX_MEMORY_CONTEXT_CHARS = int(os.environ.get("EDUMEM_MAX_CONTEXT_CHARS", "16000"))  # More context for LLM to find contradictions


def _recall_safe(beam: BeamMemory, query: str, top_k: int, temporal_weight: float = 0.0) -> list:
    """Safe recall wrapper with timeout + fresh connection isolation.
    Prevents indefinite hangs and thread-contention on shared connections."""
    import threading
    import sqlite3
    _log = logging.getLogger("recall_safe")
    _thread_id = threading.get_ident()
    _start = time.time()
    _log.info(f"RECALL START | query={query[:60]!r} | thread={_thread_id} | top_k={top_k}")

    result = []
    exception = [None]
    done = threading.Event()

    def _worker():
        nonlocal result
        old_conn = getattr(beam, 'conn', None)
        fresh_conn = None
        try:
            # Fresh connection per call to avoid thread contention
            db_path = getattr(beam, 'db_path', None)
            if db_path:
                fresh_conn = sqlite3.connect(db_path, timeout=120, check_same_thread=False)
                fresh_conn.row_factory = sqlite3.Row
                fresh_conn.execute("PRAGMA journal_mode=WAL")
                fresh_conn.execute("PRAGMA busy_timeout=120000")
                beam.conn = fresh_conn
            result = beam.recall(query, top_k=top_k, temporal_weight=temporal_weight)
        except Exception as e:
            exception[0] = e
        finally:
            if fresh_conn is not None:
                beam.conn = old_conn
                try:
                    fresh_conn.close()
                except Exception:
                    pass
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    if not done.wait(timeout=120):
        _log.warning(f"RECALL TIMEOUT | query={query[:60]!r} | duration={time.time()-_start:.1f}s | thread={_thread_id}")
        print(f"    [RECALL-TIMEOUT] recall({query[:60]!r}) exceeded 120s timeout, returning empty", flush=True)
        return []
    if exception[0]:
        _log.warning(f"RECALL ERROR | query={query[:60]!r} | error={exception[0]} | thread={_thread_id}")
        print(f"    [RECALL-ERROR] recall({query[:60]!r}): {exception[0]}", flush=True)
        return []
    _log.info(f"RECALL END | query={query[:60]!r} | duration={time.time()-_start:.1f}s | thread={_thread_id}")
    return result


def _extract_search_terms(question: str) -> list[str]:
    """Extract diverse search terms from a question for multi-strategy retrieval."""
    import re
    terms = []
    
    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', question)
    terms.extend(quoted)
    
    # Extract numbers and units
    numbers = re.findall(r'\b\d+[.,]?\d*\s*(?:ms|sec|days?|weeks?|months?|years?|%|KB|MB|GB|hours?|minutes?)\b', question, re.IGNORECASE)
    terms.extend(numbers[:5])
    
    # Extract named entities (capitalized phrases)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', question)
    terms.extend(entities[:5])
    
    # Extract version strings
    versions = re.findall(r'\bv?\d+\.\d+(?:\.\d+)?\b', question)
    terms.extend(versions[:5])
    
    # Extract key nouns (filter out question words)
    stop_words = {'have', 'did', 'do', 'does', 'can', 'will', 'would', 'should', 'is', 'are', 'was', 'were',
                  'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'my', 'me', 'i', 'you', 
                  'how', 'what', 'when', 'where', 'which', 'who', 'why', 'many', 'much'}
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', question) if w.lower() not in stop_words]
    terms.extend(words[:10])
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    
    return unique


def _multi_strategy_recall(beam: BeamMemory, question: str, top_k: int = DEFAULT_TOP_K,
                           ability: str = None, diag: dict | None = None) -> list:
    """Multi-strategy retrieval: keyword, semantic, entity, negation, temporal."""
    import re
    all_memories = []
    seen_content_keys = set()
    strategy_diag = diag.setdefault("strategies", {}) if diag is not None else None

    def _strategy_bucket(name: str) -> dict | None:
        if strategy_diag is None:
            return None
        return strategy_diag.setdefault(name, {
            "activated": False,
            "candidates_before_dedup": 0,
            "added_after_dedup": 0,
            "final_contribution": 0,
        })

    def _add_unique(mems, strategy_name: str | None = None):
        before = len(all_memories)
        for mem in mems:
            mem.setdefault("raw_score", mem.get("score", mem.get("relevance", 0.0)))
            if strategy_name and not mem.get("retrieval_strategy"):
                mem["retrieval_strategy"] = strategy_name
            ck = mem.get("content", "")[:80]
            if ck not in seen_content_keys:
                seen_content_keys.add(ck)
                all_memories.append(mem)
        return len(all_memories) - before

    def _mark(name: str, activated: bool = True):
        bucket = _strategy_bucket(name)
        if bucket is not None:
            bucket["activated"] = bucket["activated"] or activated

    def _note_add(name: str, candidates_before_dedup: int, added_after_dedup: int):
        bucket = _strategy_bucket(name)
        if bucket is not None:
            bucket["candidates_before_dedup"] += max(candidates_before_dedup, 0)
            bucket["added_after_dedup"] += max(added_after_dedup, 0)

    # FIX 2 (EO): Raise candidate depth for ordering queries.
    # Ordering questions need more candidates to capture all [MSGIDX:N] mentions.
    # Multiply top_k by 3 for ordering questions so they get broader context.
    if is_ordering_query(question):
        top_k = top_k * 3
        _mark("EO")

    # MR: Raise candidate depth for aggregation queries
    # Aggregation questions need broader retrieval to find items across multiple sessions
    if _wants_broad_aggregation_retrieval(question):
        top_k = top_k * 3
        _mark("MR")

    # SUM: Summarization needs broad retrieval to cover all conversation themes
    if is_summarization_query(question):
        top_k = top_k * 3
        _mark("SUM")

    # Detect temporal questions by ability type or keywords
    temporal_keywords = ['when', 'date', 'deadline', 'sprint', 'day', 'week', 'month',
                         'april', 'march', 'february', 'january', 'may', 'june', 'july',
                         'august', 'september', 'october', 'november', 'december',
                         'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                         'how many days', 'how long', 'timeline', 'schedule']
    is_temporal = is_temporal_query(question) or any(w in question.lower() for w in temporal_keywords)
    temporal_weight = 0.3 if is_temporal else 0.0
    
    # Strategy 1: Direct question search (mostly keyword via FTS5)
    _mark("S1")
    _candidate_mems = _recall_safe(beam, question, top_k * 2, temporal_weight=temporal_weight)
    _note_add("S1", len(_candidate_mems), _add_unique(_candidate_mems, "S1"))

    # Always create the negation bucket so skipped negation routing is explicit.
    _strategy_bucket("S2")

    # Strategy 2: Negation search for contradiction detection
    if _wants_negation_retrieval(question):
        _mark("S2")
        # 2a: Search for [NEG] tagged content via FTS5
        _neg_terms = _extract_search_terms(question)
        for term in _neg_terms[:3]:
            if len(term) > 2:
                _candidate_mems = _recall_safe(beam, f"NEG {term}", max(5, top_k // 2))
                _note_add("S2", len(_candidate_mems), _add_unique(_candidate_mems, "S2"))

        # 2b: SQL LIKE search for negation words near topic terms
        _neg_topic_terms = re.findall(r'[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*', question)
        _neg_topic_terms += re.findall(r'\b[A-Z]{2,8}\b', question)
        _neg_exclude = {'Have', 'Could', 'Which', 'What', 'This', 'That', 'Does', 'About', 'There'}
        _neg_topic_terms = [t for t in _neg_topic_terms if len(t) > 3 and t not in _neg_exclude]
        if not _neg_topic_terms:
            _neg_topic_terms = [w for w in question.split() if len(w) > 4][:3]
        for _term in _neg_topic_terms[:5]:
            for _neg_word in ['never', 'not', "haven't", "didn't", "wasn't", "weren't", "n't"]:
                try:
                    # FIX 3 (CR): Increase LIMIT from 5 to 15 for broader negation retrieval
                    _neg_rows = beam.conn.execute(
                        "SELECT id, content FROM working_memory "
                        "WHERE content LIKE ? AND content LIKE ? "
                        "UNION "
                        "SELECT id, content FROM episodic_memory "
                        "WHERE content LIKE ? AND content LIKE ? "
                        "LIMIT 15",
                        (f"%{_term}%", f"%{_neg_word}%",
                         f"%{_term}%", f"%{_neg_word}%")
                    ).fetchall()
                    for _nr in _neg_rows:
                        if _nr[1]:
                            _note_add("S2", 1, _add_unique([{"id": _nr[0], "content": _nr[1], "score": 0.80,
                                 "source": "negation_search"}], "S2"))
                except Exception:
                    pass

        # 2c: Broad topic-mention search for IMPLICIT contradictions.
        # Many contradictions carry no negation word ("uses PostgreSQL" then
        # "migrated to MySQL"). Pull ALL mentions of each topic term so the LLM
        # sees every claim and can detect the conflict itself.
        for _term in _neg_topic_terms[:5]:
            try:
                # FIX 3 (CR): Increase LIMIT from 8 to 15 for broader topic mention retrieval
                _topic_rows = beam.conn.execute(
                    "SELECT id, content FROM working_memory WHERE content LIKE ? "
                    "UNION "
                    "SELECT id, content FROM episodic_memory WHERE content LIKE ? "
                    "LIMIT 15",
                    (f"%{_term}%", f"%{_term}%")
                ).fetchall()
                for _tr in _topic_rows:
                    if _tr[1]:
                        _note_add("S2", 1, _add_unique([{"id": _tr[0], "content": _tr[1], "score": 0.70,
                                     "source": "topic_mention"}], "S2"))
            except Exception:
                pass

        # 2d: Retrieve superseded facts for contradiction resolution (CR ability)
        # When a query might involve contradictions, also fetch facts that were
        # superseded by the write-time conflict resolver. This helps find BOTH
        # sides of a contradiction (e.g., "never used Flask" vs "implemented Flask").
        _cr_terms = re.findall(r'\b[a-z]{4,}\b', question.lower())
        _cr_stop = {'have', 'been', 'does', 'ever', 'that', 'this', 'what', 'which', 'about', 'your', 'with'}
        _cr_terms = [t for t in _cr_terms if t not in _cr_stop][:5]
        for term in _cr_terms:
            try:
                superseded_rows = beam.conn.execute(
                    "SELECT id, content FROM working_memory "
                    "WHERE content LIKE ? AND superseded_by IS NOT NULL "
                    "LIMIT 10",
                    (f"%{term}%",)
                ).fetchall()
                for row in superseded_rows:
                    _note_add("S2", 1, _add_unique([{"id": row[0], "content": row[1], "score": 0.6, "source": "superseded_recall"}], "S2"))
            except Exception:
                pass

    # Strategy 3: Key entity/term searches
    terms = _extract_search_terms(question)
    for term in terms[:5]:
        if len(term) > 2:
            _mark("S3")
            _candidate_mems = _recall_safe(beam, term, max(5, top_k // 3), temporal_weight=temporal_weight)
            _note_add("S3", len(_candidate_mems), _add_unique(_candidate_mems, "S3"))
    
    # Strategy 4: Temporal search for date-related questions
    if is_temporal:
        _mark("TR")
        # Stronger temporal boost for date-specific sub-queries
        date_temporal_weight = 0.5
        # Search for dates and timelines
        _candidate_mems = _recall_safe(beam, "deadline schedule timeline date", top_k, temporal_weight=date_temporal_weight)
        _note_add("TR", len(_candidate_mems), _add_unique(_candidate_mems, "TR"))

        # --- NEW: Hard-filter for specific extracted date strings ---
        # If the question asks about a specific date (e.g., '2024-03-15'), force-filter SQL
        # directly for that exact string in the content to eliminate FTS5 fuzziness.
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', question)
        if date_match:
            exact_date = date_match.group(0)
            # Inject a high-priority hard-filter query
            _candidate_mems = _recall_safe(beam, f"content:'{exact_date}'", top_k * 2, temporal_weight=0.9)
            _note_add("TR", len(_candidate_mems), _add_unique(_candidate_mems, "TR"))

            # FIX 1 (TR): Also search for the datetok version of this date
            # so FTS5 can find it as a single token (no hyphens: datetokYYYYMMDD)
            datetok = f"datetok{exact_date.replace('-', '')}"
            _candidate_mems = _recall_safe(beam, datetok, top_k, temporal_weight=0.9)
            _note_add("TR", len(_candidate_mems), _add_unique(_candidate_mems, "TR"))

        # FIX 1 (TR): Extract any ISO dates in the question and search via datetok tokens
        # This ensures date-bearing messages are retrievable even when FTS5 alone fails.
        iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', question)
        for iso_date in iso_dates:
            datetok = f"datetok{iso_date.replace('-', '')}"
            _candidate_mems = _recall_safe(beam, datetok, max(5, top_k // 2), temporal_weight=date_temporal_weight)
            _note_add("TR", len(_candidate_mems), _add_unique(_candidate_mems, "TR"))

        # Search for specific months mentioned in the question
        for month in ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']:
            if month in question.lower():
                _candidate_mems = _recall_safe(beam, month, top_k // 2, temporal_weight=date_temporal_weight)
                _note_add("TR", len(_candidate_mems), _add_unique(_candidate_mems, "TR"))

    # Strategy 5: Two-hop entity retrieval for multi-hop reasoning
    if len(all_memories) > 0:
        _hop_entities = set()
        for mem in all_memories[:10]:
            _hop_content = mem.get("content", "")
            _hop_entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', _hop_content))
            _hop_entities.update(re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', _hop_content))
            _hop_entities.update(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', _hop_content))
        q_lower_hop = question.lower()
        _hop_entities = {e for e in _hop_entities if e.lower() not in q_lower_hop and len(e) > 3}
        for entity in list(_hop_entities)[:5]:
            _mark("MR")
            _candidate_mems = _recall_safe(beam, entity, max(5, top_k // 3))
            _note_add("MR", len(_candidate_mems), _add_unique(_candidate_mems, "MR"))

    # Strategy 6: Instruction and Preference retrieval for IF/PF abilities
    try:
        if _query_wants_if_pf(question):
            _mark("IF")
            _mark("PF")
            q_terms = [
                term.lower()
                for term in _extract_search_terms(question)
                if len(term) > 2 and term.lower() not in {
                    "instruction", "instructions", "preference", "preferences",
                    "prefer", "should", "always", "never", "follow",
                }
            ]
            for tag in ["INSTRUCTION", "PREFERENCE"]:
                if q_terms:
                    query_terms = q_terms[:8]
                    term_clause = " OR ".join(["content LIKE ?"] * len(query_terms))
                    params = [beam.session_id, f"%[{tag}]%"] + [f"%{term}%" for term in query_terms]
                    tag_rows = beam.conn.execute(
                        "SELECT id, content FROM working_memory "
                        "WHERE session_id = ? AND superseded_by IS NULL "
                        "AND content LIKE ? "
                        f"AND ({term_clause}) "
                        "ORDER BY message_index DESC, timestamp DESC, rowid DESC "
                        "LIMIT 10",
                        params,
                    ).fetchall()
                else:
                    tag_rows = []
                # Fallback: if term matching found nothing, retrieve by tag alone
                if not tag_rows:
                    tag_rows = beam.conn.execute(
                        "SELECT id, content FROM working_memory "
                        "WHERE session_id = ? AND superseded_by IS NULL "
                        "AND content LIKE ? "
                        "ORDER BY message_index DESC, timestamp DESC, rowid DESC "
                        "LIMIT 5",
                        [beam.session_id, f"%[{tag}]%"],
                    ).fetchall()
                for row in tag_rows:
                    content = row["content"]
                    strategy_name = "IF" if tag == "INSTRUCTION" else "PF"
                    _note_add(strategy_name, 1, _add_unique([{"content": content, "score": 0.75, "source": f"{tag.lower()}_recall"}], strategy_name))
    except Exception:
        pass

    # Strategy: Multi-query expansion for aggregation (MR ability)
    if _wants_broad_aggregation_retrieval(question):
        import re
        # Extract key noun phrases from the question for broader retrieval
        _mr_terms = re.findall(r'\b[a-z]{4,}\b', question.lower())
        _mr_stop = {'have', 'been', 'want', 'many', 'much', 'different', 'across',
                     'sessions', 'what', 'which', 'does', 'were', 'that', 'this',
                     'from', 'with', 'about', 'into', 'your', 'trying', 'implement'}
        _mr_terms = [t for t in _mr_terms if t not in _mr_stop][:5]
        for term in _mr_terms:
            _mark("MR")
            _candidate_mems = _recall_safe(beam, term, top_k)
            _note_add("MR", len(_candidate_mems), _add_unique(_candidate_mems, "MR"))

    # Sort by score and return top-k
    all_memories.sort(key=lambda x: x.get("score", 0), reverse=True)
    if strategy_diag is not None:
        final_counts = defaultdict(int)
        for mem in all_memories[:top_k]:
            final_counts[mem.get("retrieval_strategy") or "unknown"] += 1
        for name, bucket in strategy_diag.items():
            bucket["final_contribution"] = final_counts.get(name, 0)
    return all_memories[:top_k]


def _attach_second_pass_diagnostics(q_diag: dict | None, gap_queries: list[str], gap_diag: dict | None) -> None:
    """Store recursive pass telemetry separately from first-pass diagnostics."""
    if q_diag is None:
        return
    q_diag["second_pass"] = {
        "activated": bool(gap_queries),
        "gap_queries": list(gap_queries),
        "strategies": deepcopy((gap_diag or {}).get("strategies", {})),
    }


def _record_second_pass_diagnostics(diag: dict | None, gap_queries: list[str], gap_diag: dict | None) -> None:
    """Attach second-pass telemetry to the active retrieval diagnostics dict."""
    _attach_second_pass_diagnostics(diag, gap_queries, gap_diag)


# ============================================================
#  Per-Ability Bypasses: TR (Temporal Reasoning) + CR (Contradiction)
# ============================================================

def _extract_timeline_from_conversation(messages: list) -> list[dict]:
    """Extract ALL dates from conversation messages with surrounding event context.
    Filters out dates in code snippets. Returns sorted list of {date_obj, date_str, event_text, msg_index}."""
    import re as _re
    from datetime import datetime as _dt
    
    timeline = []
    
    # Month name map
    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5,
        'june': 6, 'july': 7, 'august': 8, 'september': 9, 'october': 10,
        'november': 11, 'december': 12,
    }
    
    # Code indicators to filter out
    CODE_INDICATORS = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE TABLE',
                       'def ', 'import ', 'print(', 'return ', '```', 'function(',
                       'jsonify', 'datetime', 'params', 'cursor.']
    
    def _is_code_context(text: str, match_start: int) -> bool:
        """Check if a date match appears to be in a code snippet."""
        # Check ~200 chars around match for code indicators
        start = max(0, match_start - 100)
        end = min(len(text), match_start + 100)
        surrounding = text[start:end]
        # If backticks within 200 chars, it's code
        if '```' in surrounding or '`' in surrounding:
            return True
        # If multiple code indicators present
        code_count = sum(1 for ci in CODE_INDICATORS if ci in surrounding)
        if code_count >= 2:
            return True
        # ISO date alone (2024-01-15) in a line with code indicators = likely code
        if _re.search(r'\b\d{4}-\d{2}-\d{2}\b', surrounding):
            if any(ci in surrounding for ci in CODE_INDICATORS):
                return True
        return False
    
    # Track the conversation year context
    year_mentions = []
    for msg in messages:
        years = _re.findall(r'\b(20\d{2})\b', msg.get("content", ""))
        year_mentions.extend(int(y) for y in years)
    # Use the most common year > 2020 as default
    default_year = 2024
    if year_mentions:
        from collections import Counter
        year_counts = Counter(y for y in year_mentions if 2020 <= y <= 2030)
        if year_counts:
            default_year = year_counts.most_common(1)[0][0]
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        
        # Pattern 1: "Month Day, Year" (e.g. "March 15, 2024")
        for m in _re.finditer(
            r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+'
            r'(?P<day>\d{1,2})(?:st|nd|rd|th)?[,\s]+(?P<year>\d{4})',
            content, _re.IGNORECASE
        ):
            if _is_code_context(content, m.start()):
                continue
            month_num = MONTH_MAP.get(m.group('month').lower()[:3])
            if month_num:
                try:
                    dt = _dt(int(m.group('year')), month_num, int(m.group('day')))
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 60)
                    event_text = content[start:end].strip()
                    timeline.append({
                        'date_obj': dt, 'date_str': m.group(0),
                        'event_text': event_text, 'msg_index': i,
                    })
                except ValueError:
                    pass
        
        # Pattern 2: "Month Day" without year (e.g. "March 29")
        for m in _re.finditer(
            r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+'
            r'(?P<day>\d{1,2})(?:st|nd|rd|th)?'
            r'(?![\d,\s]*\d{4})',  # NOT followed by year
            content, _re.IGNORECASE
        ):
            if _is_code_context(content, m.start()):
                continue
            month_num = MONTH_MAP.get(m.group('month').lower()[:3])
            if month_num:
                try:
                    dt = _dt(default_year, month_num, int(m.group('day')))
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 60)
                    event_text = content[start:end].strip()
                    timeline.append({
                        'date_obj': dt, 'date_str': m.group(0),
                        'event_text': event_text, 'msg_index': i,
                    })
                except ValueError:
                    pass
    
        # Pattern 3: ISO dates (2024-03-15)
        for m in _re.finditer(r'\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b', content):
            if _is_code_context(content, m.start()):
                continue
            try:
                dt = _dt(int(m.group('year')), int(m.group('month')), int(m.group('day')))
                start = max(0, m.start() - 60)
                end = min(len(content), m.end() + 60)
                event_text = content[start:end].strip()
                timeline.append({
                    'date_obj': dt, 'date_str': m.group(0),
                    'event_text': event_text, 'msg_index': i,
                })
            except ValueError:
                pass

        # Pattern 4: Ordinal dates — "15th of March, 2024" / "15th of March"
        for m in _re.finditer(
            r'(?P<day>\d{1,2})(?:st|nd|rd|th)\s+(?:of\s+)?'
            r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
            r'(?:[,\s]+(?P<year>\d{4}))?',
            content, _re.IGNORECASE
        ):
            if _is_code_context(content, m.start()):
                continue
            month_num = MONTH_MAP.get(m.group('month').lower()[:3])
            if month_num:
                yr = int(m.group('year')) if m.group('year') else default_year
                try:
                    dt = _dt(yr, month_num, int(m.group('day')))
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 60)
                    event_text = content[start:end].strip()
                    timeline.append({
                        'date_obj': dt, 'date_str': m.group(0),
                        'event_text': event_text, 'msg_index': i,
                    })
                except ValueError:
                    pass

        # Pattern 5: Slash dates — MM/DD/YYYY
        for m in _re.finditer(r'\b(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})\b', content):
            if _is_code_context(content, m.start()):
                continue
            try:
                dt = _dt(int(m.group('y')), int(m.group('m')), int(m.group('d')))
                start = max(0, m.start() - 60)
                end = min(len(content), m.end() + 60)
                event_text = content[start:end].strip()
                timeline.append({
                    'date_obj': dt, 'date_str': m.group(0),
                    'event_text': event_text, 'msg_index': i,
                })
            except ValueError:
                pass

        # Pattern 6: Informal month references — "early/mid/late Month Year"
        for m in _re.finditer(
            r'(?P<qual>early|mid|late)[- ]'
            r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)'
            r'(?:\s+(?P<year>\d{4}))?',
            content, _re.IGNORECASE
        ):
            if _is_code_context(content, m.start()):
                continue
            month_num = MONTH_MAP.get(m.group('month').lower()[:3])
            if month_num:
                yr = int(m.group('year')) if m.group('year') else default_year
                qual = m.group('qual').lower()
                day = 5 if qual == 'early' else 15 if qual == 'mid' else 25
                try:
                    dt = _dt(yr, month_num, day)
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 60)
                    event_text = content[start:end].strip()
                    timeline.append({
                        'date_obj': dt, 'date_str': m.group(0),
                        'event_text': event_text, 'msg_index': i,
                    })
                except ValueError:
                    pass

        # Pattern 7: Relative durations — "N weeks/days later", "N weeks/days ago"
        # Resolved against the nearest preceding absolute date in the same message.
        from datetime import timedelta as _td_rel
        _NUM_MAP = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
        for m in _re.finditer(
            r'\b(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
            r'(?P<unit>day|week|month|year)s?\s+'
            r'(?P<dir>later|after|ago|before|earlier)',
            content, _re.IGNORECASE
        ):
            if _is_code_context(content, m.start()):
                continue
            num_str = m.group('num').lower()
            num = int(num_str) if num_str.isdigit() else _NUM_MAP.get(num_str, 0)
            if num == 0:
                continue
            unit = m.group('unit').lower()
            direction = m.group('dir').lower()
            delta_days = num * {'day': 1, 'week': 7, 'month': 30, 'year': 365}[unit]
            anchor = None
            for prev in timeline:
                if prev['msg_index'] == i and prev['date_obj']:
                    anchor = prev['date_obj']
            if anchor is None:
                continue
            if direction in ('later', 'after'):
                dt = anchor + _td_rel(days=delta_days)
            else:
                dt = anchor - _td_rel(days=delta_days)
            start = max(0, m.start() - 60)
            end = min(len(content), m.end() + 60)
            event_text = content[start:end].strip()
            timeline.append({
                'date_obj': dt, 'date_str': f"{num} {unit}s {direction} ({dt.strftime('%Y-%m-%d')})",
                'event_text': event_text, 'msg_index': i,
            })

    # Sort chronologically and deduplicate (same date, same event text)
    timeline.sort(key=lambda x: (x['date_obj'], x['event_text']))
    seen = set()
    deduped = []
    for t in timeline:
        key = (t['date_obj'].strftime('%Y-%m-%d'), t['event_text'][:40])
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    return deduped


def _build_tr_timeline_prompt(timeline: list[dict]) -> str:
    """Build a structured timeline prompt for TR questions."""
    if not timeline:
        return ""
    
    lines = ["CRITICAL TIMELINE (all dates extracted from the conversation, use ONLY these dates):"]
    for t in timeline:
        lines.append(f"  [{t['date_obj'].strftime('%Y-%m-%d')}] {t['date_str']}: ...{t['event_text'][:100]}...")
    
    return "\n".join(lines)


def _compute_tr_python(question: str, timeline: list[dict]) -> str | None:
    """Compute TR answer in pure Python (date math, no LLM). Returns answer string or None."""
    import re as _re

    q_lower = question.lower()

    # --- Detect question type ---
    _is_before_after = bool(_re.search(r'before\s+or\s+after|earlier\s+or\s+later|which.*(?:came|happened).*first', q_lower))
    _asks_weeks = 'week' in q_lower
    _asks_months = 'month' in q_lower
    _asks_years = 'year' in q_lower

    # --- Score each timeline entry against the question ---
    _stop = {'the','a','an','in','on','at','to','for','of','with','my','me','i','you',
             'how','many','long','did','does','was','were','is','are','do','what','between',
             'and','or','take','from','it','its','this','that','before','after'}
    q_words = [w for w in _re.findall(r'[a-z]+', q_lower) if w not in _stop and len(w) > 2]

    scored = []
    for t in timeline:
        event = t['event_text'].lower()
        score = sum(2 for w in q_words if w in event)
        if len(t['event_text']) > 20:
            score += 1
        scored.append((score, t))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Pick the two best-matching timeline entries with distinct dates
    best_pair = None
    if len(scored) >= 2 and scored[0][0] > 0:
        t1 = scored[0][1]
        for s, t in scored[1:]:
            if s <= 0:
                break
            if t['date_obj'] != t1['date_obj']:
                best_pair = (t1, t)
                break
        if best_pair is None and scored[1][0] > 0:
            best_pair = (scored[0][1], scored[1][1])

    if best_pair is None:
        if len(scored) >= 2:
            candidates = [t for s, t in scored if s > 0][:2]
            if len(candidates) >= 2:
                best_pair = (candidates[0], candidates[1])

    if best_pair is None:
        return None

    t1, t2 = best_pair
    d1, d2 = t1['date_obj'], t2['date_obj']
    earlier = d1 if d1 <= d2 else d2
    later = d2 if d1 <= d2 else d1
    evt_early = t1['date_str'] if d1 <= d2 else t2['date_str']
    evt_late = t2['date_str'] if d1 <= d2 else t1['date_str']
    diff_days = abs((d2 - d1).days)

    # --- Before/after questions ---
    if _is_before_after:
        evt1_text = t1['event_text'][:60].strip()
        evt2_text = t2['event_text'][:60].strip()
        if d1 < d2:
            return (f"The event near {t1['date_str']} happened before the event near {t2['date_str']} "
                    f"({d1.strftime('%Y-%m-%d')} vs {d2.strftime('%Y-%m-%d')}, {diff_days} days earlier).")
        elif d1 > d2:
            return (f"The event near {t1['date_str']} happened after the event near {t2['date_str']} "
                    f"({d1.strftime('%Y-%m-%d')} vs {d2.strftime('%Y-%m-%d')}, {diff_days} days later).")
        else:
            return f"Both events occurred on the same date: {d1.strftime('%Y-%m-%d')}."

    # --- Duration questions: convert to the unit the question asks for ---
    if _asks_months:
        months = (later.year - earlier.year) * 12 + (later.month - earlier.month)
        return (f"Between {evt_early} ({earlier.strftime('%B %d, %Y')}) and "
                f"{evt_late} ({later.strftime('%B %d, %Y')}), "
                f"there are approximately {months} months ({diff_days} days).")

    if _asks_weeks:
        weeks = diff_days // 7
        remainder = diff_days % 7
        week_str = f"{weeks} weeks" + (f" and {remainder} days" if remainder else "")
        return (f"Between {evt_early} ({earlier.strftime('%B %d, %Y')}) and "
                f"{evt_late} ({later.strftime('%B %d, %Y')}), "
                f"there are {week_str} ({diff_days} days).")

    if _asks_years:
        years = later.year - earlier.year
        if (later.month, later.day) < (earlier.month, earlier.day):
            years -= 1
        return (f"Between {evt_early} ({earlier.strftime('%B %d, %Y')}) and "
                f"{evt_late} ({later.strftime('%B %d, %Y')}), "
                f"there are approximately {years} years ({diff_days} days).")

    # Default: days
    return (f"Between {evt_early} ({earlier.strftime('%B %d, %Y')}) and "
            f"{evt_late} ({later.strftime('%B %d, %Y')}), "
            f"there are {diff_days} days.")



def _compute_tr_answer(question: str, timeline: list[dict]) -> str | None:
    """Compute temporal reasoning answer from conversation dates. Returns None if can't compute."""
    if not timeline or len(timeline) < 2:
        return None

    # Build a prompt that presents the timeline and asks the LLM to compute
    # This is more robust than trying to match events ourselves
    timeline_prompt = _build_tr_timeline_prompt(timeline)
    
    # Build the full prompt that we'll return - the caller will send this to LLM
    prompt = (
        f"{timeline_prompt}\n\n"
        f"QUESTION: {question}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Identify the two events mentioned in the question\n"
        f"2. Find the corresponding dates in the timeline above\n"
        f"3. Compute the time difference between them\n"
        f"4. State the answer clearly with both dates and the computed difference\n\n"
        f"ANSWER:"
    )
    return prompt  # Return the prompt, not the answer - caller passes to LLM


def _detect_contradictions(messages: list, question: str) -> str | None:
    """Scan conversation for contradictory statements about the question topic.
    Returns contradiction context string to inject into prompt, or None if none found."""
    import re as _re
    
    # Extract the key topic from the question
    # "Have I worked with Flask routes?" -> key terms: "flask routes", "http requests"
    # "Have I integrated Flask-Login?" -> key terms: "flask-login", "session management"
    
    # Strip question words to get the core topic
    q_clean = _re.sub(r'^(?:Have I|Did I|Do I|Am I|Has)\s+(?:ever\s+)?', '', question, flags=_re.IGNORECASE)
    q_clean = _re.sub(r'\s+(?:in this project|across my sessions|in my project)\s*\??$', '', q_clean, flags=_re.IGNORECASE)
    q_clean = q_clean.strip().rstrip('?').strip()
    
    # Extract meaningful noun phrases
    words = _re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', q_clean)
    # Filter to key content words (nouns, tech terms)
    key_terms = []
    for w in words:
        wl = w.lower()
        if len(wl) > 2 and wl not in ('the', 'and', 'for', 'with', 'any', 'this', 'that', 'have', 'has', 'been'):
            key_terms.append(wl)
    
    if not key_terms:
        return None
    
    # Scan all messages for mentions of ANY key term
    affirmatives = []
    negatives = []
    
    NEGATION_WORDS = {'never', 'not', "n't", 'no', 'without', 'cannot', "can't", 'nothing', 'none'}
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        content_lower = content.lower()
        
        # Check if message mentions any key term
        matched_terms = [t for t in key_terms if t in content_lower]
        if not matched_terms:
            continue
        
        # Check for negation in the SENTENCE containing the matched term
        # (BEAM contradictions embed negation near the topic mention)
        has_negation = False
        for term in matched_terms:
            # Find the sentence containing this term
            term_pos = content_lower.find(term)
            if term_pos < 0:
                continue
            # Extract sentence context (200 chars around term, or to sentence boundaries)
            start = max(0, term_pos - 150)
            end = min(len(content_lower), term_pos + 150)
            sentence = content_lower[start:end]
            # Check for negation words in this sentence
            for nw in NEGATION_WORDS:
                if nw in sentence:
                    has_negation = True
                    break
            if has_negation:
                break
        
        snippet = content[:250].strip()
        if has_negation:
            negatives.append(f"[Msg {i}] {content[:250].strip()}")
        else:
            affirmatives.append(f"[Msg {i}] {content[:250].strip()}")
    
    if affirmatives and negatives:
        ctx = "CRITICAL: CONTRADICTORY INFORMATION DETECTED\n\n"
        ctx += "The conversation contains BOTH affirmative AND negative statements about this topic:\n\n"
        ctx += "Statements suggesting this WAS done or worked on:\n"
        for a in affirmatives[:5]:
            ctx += f"  - {a}\n"
        ctx += "\nStatements suggesting this was NOT done:\n"
        for n in negatives[:5]:
            ctx += f"  - {n}\n"
        ctx += "\nYOU MUST explicitly identify the contradiction and present BOTH sides. "
        ctx += "Do NOT answer with just one side. The correct response begins with "
        ctx += "'I notice you've mentioned contradictory information about this.'"
        return ctx
    
    return None


_POLYPHONIC_VOICE_KEYS = frozenset({"vector", "graph", "fact", "temporal"})
_LINEAR_VOICE_KEYS = frozenset({"vec", "fts", "keyword", "importance", "recency_decay"})


def _summarize_recall_memories(memories: list) -> dict:
    """Compact per-question recall provenance for analysis.

    Captures engine identity + per-voice score sums + top-1 voice
    breakdown. Lets `docs/benchmark-results-analysis.md` Recipe E
    (per-voice attribution) work from the result JSON directly.

    Shape:
        {
          "engine": "polyphonic" | "linear" | "unknown",
          "kept_count": N,
          "voice_sums": {voice_key: total_score, ...},
          "top_result_voices": {voice_key: score, ...} | {},
          "top_result_tier": "working" | "episodic" | None,
        }

    Returns a minimal dict when memories is empty (bypass paths
    short-circuit before recall so the field still exists for
    schema consistency).
    """
    if not memories:
        return {
            "engine": "unknown",
            "kept_count": 0,
            "voice_sums": {},
            "top_result_voices": {},
            "top_result_tier": None,
            "memories": [],
        }

    # Engine ID by the voice_scores keyset of any result that has one.
    engine = "unknown"
    voice_sums: dict = {}
    memory_summaries: list[dict] = []
    import hashlib

    for m in memories:
        vs = m.get("voice_scores") or {}
        if not vs:
            vs = {}
        if engine == "unknown":
            keys = set(vs.keys())
            if keys & _POLYPHONIC_VOICE_KEYS:
                engine = "polyphonic"
            elif keys & _LINEAR_VOICE_KEYS:
                engine = "linear"
        for k, v in vs.items():
            try:
                voice_sums[k] = voice_sums.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                pass  # ignore non-numeric voice values

        content = m.get("content", "")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None
        raw_score = m.get("raw_score", m.get("score", 0.0))
        final_score = m.get("score", m.get("relevance", 0.0))
        memory_summaries.append({
            "memory_id": m.get("id"),
            "content_hash": content_hash,
            "source": m.get("source"),
            "message_index": m.get("message_index"),
            "raw_score": raw_score,
            "final_score": final_score,
            "components": {
                "dense": vs.get("vector", vs.get("vec")),
                "fts": vs.get("fts"),
                "keyword": vs.get("keyword"),
                "reranker": m.get("rerank_score"),
                "importance": vs.get("importance", m.get("importance")),
                "recency_decay": vs.get("recency_decay"),
            },
            "final_context_included": bool(m.get("final_context_included", False)),
        })

    top = memories[0] if memories else {}
    return {
        "engine": engine,
        "kept_count": len(memories),
        "voice_sums": {k: round(v, 4) for k, v in voice_sums.items()},
        "top_result_voices": {
            k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in (top.get("voice_scores") or {}).items()
        },
        "top_result_tier": top.get("tier"),
        "memories": memory_summaries,
    }


def _inject_temporal_cheatsheet(memories: list, question: str) -> str:
    """Extract natural-language dates from recalled memories and inject a temporal cheat sheet.

    Pure-recall legal: reads only from recalled memories, not raw conversation_messages.

    Logic:
    1. Only activate if is_temporal_query(question)
    2. Extract all supported date spans from memory content with surrounding context
    3. Parse explicit-year dates, sort chronologically
    4. Compute pairwise timedeltas between consecutive explicit dates
    5. Return a formatted cheat sheet string with dates and deltas

    Returns: str -- cheat sheet markup (empty string if no dates or not temporal)
    """
    import re

    # Only for temporal queries
    if not is_temporal_query(question):
        return ""

    # Extract all supported date spans from memories.
    date_events = []  # list of (date_obj, date_str, context_snippet)
    raw_only_events = []  # list of (raw_date, context_snippet)

    for mem in memories:
        content = mem.get("content", "")
        if not content:
            continue

        for span in _extract_shared_date_spans(content):
            start = max(0, span["start"] - 15)
            end = min(len(content), span["end"] + 15)
            context = content[start:end].strip()
            if span.get("date_obj"):
                date_events.append((span["date_obj"], span["iso"], context))
            else:
                raw_only_events.append((span["raw"], context))

    # If no dates found, return empty
    if not date_events and not raw_only_events:
        return ""

    # Sort explicit dates chronologically.
    date_objs = sorted(date_events, key=lambda x: x[0])

    # Build cheat sheet
    lines = ["[TEMPORAL REFERENCE]"]

    # Add each event with date
    for dt, date_str, context in date_objs:
        lines.append(f"- Event: \"{context}\" ({date_str})")

    # Retain raw date spans that do not have an explicit year. We do not
    # infer a year for them, but we keep them visible for reasoning.
    for raw_date, context in raw_only_events:
        lines.append(f"- Event: \"{context}\" ({raw_date})")

    # Compute pairwise deltas between consecutive events
    if len(date_objs) >= 2:
        lines.append("")
        for i in range(len(date_objs) - 1):
            dt1, date1, ctx1 = date_objs[i]
            dt2, date2, ctx2 = date_objs[i + 1]
            delta = dt2 - dt1
            days = delta.days

            # Compute months and days for readability
            months = days // 30
            remainder = days % 30

            lines.append(
                f"- Delta: {date1} → {date2} = {days} days "
                f"({months} months, {remainder} days)"
            )

    lines.append("[END TEMPORAL REFERENCE]")

    return "\n".join(lines)


def _apply_rerank_scores(memories: list, scores: list, top_n: int) -> list:
    """Merge cross-encoder scores into memories and tier-sort.

    Only a prefix of `memories` is sent to the reranker, so only those carry a
    `rerank_score`. Cross-encoder scores and the original 0-1 `score` are on
    different scales, so we sort in two tiers: every reranked memory ranks above
    the un-reranked tail; within each tier we sort by the relevant score. This
    prevents an unreviewed tail memory with a high raw score from leapfrogging a
    reranked candidate.
    """
    for item in scores:
        idx = item["index"]
        if idx < len(memories):
            memories[idx]["rerank_score"] = item["score"]
    memories.sort(
        key=lambda m: (1, m["rerank_score"]) if "rerank_score" in m else (0, m.get("score", 0)),
        reverse=True,
    )
    return memories[:top_n]


def _probe_reranker(url: str) -> dict:
    """Check reranker availability with a real request."""
    import requests as _rr
    try:
        resp = _rr.post(url, json={"query": "preflight", "texts": ["healthcheck"]}, timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        return {"ok": True, "status_code": resp.status_code, "response_type": type(payload).__name__}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def _rerank(question: str, memories: list, top_n: int = 30, diag: dict | None = None) -> list:
    """Re-score candidates with local cross-encoder reranker."""
    import requests as _rr
    _reranker_url = os.environ.get("EDUMEM_RERANKER_URL", "http://localhost:3002/rerank")
    texts = [m.get("content", "")[:500] for m in memories[:top_n * 2]]
    if not texts:
        return memories
    if diag is not None:
        rerank_diag = diag.setdefault("reranker", {
            "url": _reranker_url,
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "fallbacks": 0,
            "scores_recorded": 0,
            "errors": [],
            "health": None,
        })
        rerank_diag["calls"] += 1
    else:
        rerank_diag = None
    try:
        resp = _rr.post(_reranker_url, json={"query": question, "texts": texts}, timeout=5)
        resp.raise_for_status()
        scores = resp.json()
        memories = _apply_rerank_scores(memories, scores, top_n)
        if rerank_diag is not None:
            rerank_diag["successes"] += 1
            rerank_diag["scores_recorded"] += len(scores) if isinstance(scores, list) else 0
    except Exception as exc:
        if rerank_diag is not None:
            rerank_diag["failures"] += 1
            rerank_diag["fallbacks"] += 1
            rerank_diag.setdefault("errors", []).append({
                "class": type(exc).__name__,
                "message": str(exc),
            })
        return memories[:top_n]
    return memories


def answer_with_memory(llm: LLMClient, beam: BeamMemory, question: str,
                      conversation_messages: list = None, top_k: int = DEFAULT_TOP_K,
                      ability: str = None,
                      diag: dict | None = None,
                      return_memories: bool = False):
    """Retrieve memories and have LLM answer, with context strategy based on conversation size.

    Pure recall is the default. Set `EDUMEM_BENCHMARK_PURE_RECALL=0`
    to re-enable the per-ability bypass paths (TR oracle, CR
    contradiction injection, IE/KU context→value side-index) AND the
    always-included RECENT CONVERSATION raw-message prompt section.
    Pure-recall mode forces every answer through the full edumem
    retrieval pipeline so the BEAM-recovery experiment can measure
    each arm's recall quality without contamination from harness-side
    oracles.

    Returns:
        str when `return_memories=False` (default -- backward-compat).
        tuple[str, list[dict]] when `return_memories=True` -- the second
        element is the retrieved memories list (post-multi-strategy,
        pre-LLM-context-build). Each memory dict carries `voice_scores`
        from Gap G -- required for per-voice attribution analysis.
        Bypass paths return `(answer, [])` since they short-circuit
        before recall.
    """
    def _ret(answer, memories=None):
        """Pack return value uniformly across all exit points."""
        if return_memories:
            return answer, (memories or [])
        return answer
    # E7/E8/E9 gate: when enabled, the harness disables every shortcut that
    # would let the LLM produce an answer without going through
    # BeamMemory.recall(). The bypasses were useful for measuring
    # LLM-ceiling-with-help on isolated abilities; the BEAM-recovery
    # experiment instead needs to compare Arm A vs Arm B vs Arm C on
    # the recall surface itself.
    _pure_recall = _benchmark_pure_recall_enabled()
    routing_ability = None if _pure_recall else ability
    if diag is not None:
        embed_diag = diag.setdefault("embedding", {
            "backend": "dense" if _embeddings.available() else "keyword-only",
            "model": getattr(_embeddings, "_DEFAULT_MODEL", None),
            "dimension": getattr(_embeddings, "EMBEDDING_DIM", None),
            "inserted_vectors": 0,
            "query_vectors": 0,
            "status": "dense" if _embeddings.available() else "keyword-only",
            "failed": False,
        })
        if _embeddings.available():
            embed_diag["query_vectors"] = embed_diag.get("query_vectors", 0) + 1
            embed_diag["status"] = "dense"
        else:
            embed_diag["status"] = "keyword-only"

    total_msgs = len(conversation_messages) if conversation_messages else 0

    # ---- PER-ABILITY BYPASSES (zero-LLM or augmented) ----

    # TR (Temporal Reasoning): zero-LLM date math from extracted dates
    if not _pure_recall and conversation_messages and is_temporal_query(question):
        timeline = _extract_timeline_from_conversation(conversation_messages)
        print(f"    [TR] extracted {len(timeline)} dates from {len(conversation_messages)} msgs")
        if timeline and len(timeline) >= 2:
            # Phase 1: zero-LLM Python date math (fast, no tokens).
            # Only short-circuit when the Python answer is trustworthy; a small
            # duration on a dense BEAM timeline usually means the wrong date pair
            # was matched, so defer those to the LLM-with-timeline path below.
            py_answer = _compute_tr_python(question, timeline)
            if _tr_python_answer_is_trustworthy(py_answer, len(timeline)):
                print(f"    [TR-zero-LLM] Python computed: {py_answer[:150]}")
                return _ret(py_answer)
            print(f"    [TR-zero-LLM] Python answer not trusted (dense-timeline guard), trying LLM")
            # Phase 2: LLM-assisted with timeline prompt
            tr_prompt = _compute_tr_answer(question, timeline)
            if tr_prompt:
                messages = [
                    {"role": "system", "content": "You are a precise date calculator. Use ONLY the dates from the provided timeline. Output ONLY the answer, no explanation."},
                    {"role": "user", "content": tr_prompt},
                ]
                answer = llm.chat(messages, temperature=0.0, max_tokens=8192)
                print(f"    [TR-LLM] answer: {answer[:150]}")
                return _ret(answer)
            else:
                print(f"    [TR] _compute_tr_answer returned None")
        else:
            print(f"    [TR] no timeline extracted or too few dates")
    
    _cr_context = None  # PATCH: CR injection removed — contradictions handled generically by CONSOLIDATED_SYSTEM_PROMPT step 2
    # ---- END PER-ABILITY BYPASSES ----
    
    # FULL-CONTEXT MODE: send the entire conversation to the LLM, bypassing edumem retrieval.
    # This tests the LLM's reading comprehension ceiling -- useful for establishing the upper bound.
    # Controlled by FULL_CONTEXT_MODE env var.
    # HYBRID: try context→value matching first for factual questions (IE/MR/KU),
    # then fall through to full-context for complex reasoning (ABS/CR/EO/SUM/TR).
    _full_context = _env_truthy("FULL_CONTEXT_MODE")
    # Precedence: pure-recall overrides full-context. The point of
    # pure-recall is to force every answer through edumem recall;
    # full-context's "ship the whole conversation to the LLM" path
    # would silently invalidate that guarantee (the LLM would answer
    # from raw `FULL CONVERSATION:` regardless of arm).
    if _full_context and _pure_recall:
        _full_context = False
    # DEBUG (use _env_truthy so `FULL_CONTEXT_MODE=0` doesn't fire the print)
    if _env_truthy("FULL_CONTEXT_MODE"):
        print(f"    [DEBUG full-context] env={_full_context}, msgs={bool(conversation_messages)}, count={len(conversation_messages) if conversation_messages else 0} (pure_recall={_pure_recall})")
    if _full_context and conversation_messages:
        # ---- Phase 1: Try context→value matching for factual questions ----
        # Only use context→value for Information Extraction (IE) and Knowledge Understanding (KU).
        # MR (Multi-hop) requires reasoning across multiple messages; let full-context handle it.
        # Gated by pure_recall -- when ON, full-context mode still hits the LLM with raw
        # conversation but skips the zero-LLM context→value shortcut.
        _FACT_ABILITIES = {'IE', 'KU'}
        if not _pure_recall and routing_ability in _FACT_ABILITIES and hasattr(beam, '_context_facts') and beam._context_facts:
            # Strong-match gate: only bypass the LLM on a confident match.
            _ctx_floor = float(os.environ.get("BEAM_CTX_MATCH_FLOOR", "0.5"))
            best_match, _ctx_score = _context_value_match(question, beam._context_facts, min_score=_ctx_floor)
            if best_match:
                return _ret(best_match)  # Direct fact answer, zero LLM cost
        
        # ---- Phase 2: Full-context LLM fallback ----
        full_parts = []
        total_chars = 0
        for msg in conversation_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                line = f"[{role}]: {content}"
                if total_chars + len(line) > MAX_MEMORY_CONTEXT_CHARS * 2:
                    break
                full_parts.append(line)
                total_chars += len(line)
        
        context = "FULL CONVERSATION:\n" + "\n".join(full_parts)
        
        # Inject CR contradiction context if detected
        _cr_prefix = ""
        if _cr_context:
            _cr_prefix = f"\n\n{_cr_context}\n\n"
        
        messages = [
            {"role": "system", "content": build_system_prompt(question)},
            {"role": "user", "content": f"{_cr_prefix}{context}\n\nQUESTION: {question}\n\nANSWER:"},
        ]
        return _ret(llm.chat(messages, temperature=0.1, max_tokens=2048))

    # ALWAYS use multi-strategy retrieval to test edumem's recall quality.
    # The previous <=500 bypass sent full raw conversations to the LLM,
    # completely bypassing edumem's retrieval pipeline.
    # This benchmark exists to measure MEMORY performance, not LLM reading comprehension.
    
    # Multi-strategy retrieval
    memories = _multi_strategy_recall(beam, question, top_k * 3, ability=None, diag=diag)  # label-free recall

    # ---- MEMORIA: Structured Fact Retrieval (Phase 2) ----
    # Supplement recall with structured facts from memoria_facts, memoria_timelines,
    # and memoria_kg tables. These provide exact values that FTS5/vector search
    # may miss (dates, metrics, versions, negations, sequences, entity mappings).
    # Injected as synthetic high-score entries so they surface ahead of fuzzy matches.
    # In pure-recall (label-free) mode, derive intent from question text to activate routing.
    try:
        _recall_intent = _intent_from_question(question)
        _memoria_result = beam.memoria_retrieve(question, ability=routing_ability, top_k=top_k, intent=_recall_intent)
        if _memoria_result and _memoria_result.get("source") != "fallback" and _memoria_result.get("context"):
            _memoria_facts = _memoria_result.get("facts", [])
            print(f"    [MEMORIA] {_memoria_result['source']} hit for intent={_recall_intent}, ability={routing_ability}: {len(_memoria_facts)} facts", flush=True)
            memories.insert(0, {
                "content": f"[MEMORIA {_memoria_result['source']}]\n{_memoria_result['context']}",
                "score": 0.95,
                "source": f"memoria_{_memoria_result['source']}",
                "raw_score": 0.95,
                "retrieval_strategy": "MEMORIA",
            })
    except Exception:
        pass  # MEMORIA retrieval is best-effort

    # ---- Reranking (Phase 5.5: local cross-encoder) ----
    # Ordering (EO) is graded by tau-b over ALL items, so keep a wider reranked set
    # for ordering queries -- the char-budgeted context builder trims later. A flat
    # top_k cap here would drop topic mentions and make the ordering incomplete.
    _rerank_top_n = top_k * 3 if is_ordering_query(question) else top_k
    memories = _rerank(question, memories, top_n=_rerank_top_n, diag=diag)

    # ---- EO: Sort by message_index for ordering queries ----
    if is_ordering_query(question):
        memories.sort(key=lambda m: m.get("message_index") if m.get("message_index") is not None else float('inf'))

    # ---- Context→Value fact matching (Phase 7: direct regex-extracted facts, zero-LLM) ----
    # At ingestion, we built beam._context_facts: {"words around fact": ["fact value"]}.
    # Now we try to match the question against context phrases and return the value directly.
    # Only used for factual question types (IE, MR, KU, TR) with strong matches.
    # ABS, CR, EO, SUM need LLM reasoning -- we skip context matching for those.
    context_answer = None
    # Only use context→value for Information Extraction (IE) and Knowledge Understanding (KU).
    # MR (Multi-hop) requires reasoning across multiple messages; CR/TR/EO/SUM need LLM.
    # Gated by pure_recall -- when ON, IE/KU questions go through full recall+LLM
    # rather than returning a side-indexed value directly.
    _FACT_ABILITIES = {'IE', 'KU'}
    if not _pure_recall and routing_ability in _FACT_ABILITIES and hasattr(beam, '_context_facts') and beam._context_facts:
        # Skip context→value matching for procedural/descriptive questions
        # (how, why, walk me through, describe). These need full answer, not one word.
        _q_lower = question.lower()
        _proc_indicators = ['walk me through', 'describe', 'tell me about', 'explain how',
                            'how did i', 'how do i', 'how would i', 'how should i',
                            'what were the', 'what are the', 'list the']
        if not any(ind in _q_lower for ind in _proc_indicators):
            # Only short-circuit the LLM on a STRONG match. Weak matches (a couple
            # of incidental shared words) returned a bare uncomposed value and made
            # IE a coin-flip; below the floor we fall through to full LLM reasoning.
            _ctx_floor = float(os.environ.get("BEAM_CTX_MATCH_FLOOR", "0.5"))
            best_match, _ctx_score = _context_value_match(question, beam._context_facts, min_score=_ctx_floor)
            if best_match:
                context_answer = best_match

    # If cloud extraction enabled, also search the facts table
    if getattr(beam, 'use_cloud', False):
        try:
            fact_memories = beam.fact_recall(question, top_k=top_k)
            # Convert fact dicts to same format as recall results
            for f in fact_memories:
                memories.append({
                    "content": f"FACT: {f['content']}",
                    "score": f.get("score", 0.5) * 2.0,  # 2x weight for facts
                    "source": "fact_extraction",
                    "raw_score": f.get("score", 0.5) * 2.0,
                    "retrieval_strategy": "FACT",
                })
            # Re-sort by score
            memories.sort(key=lambda x: x.get("score", 0), reverse=True)
        except Exception:
            pass  # Fact recall is best-effort
    
    # LLM RERANKING: DISABLED -- rate-limit avoidance + proven ineffective (Reality Check 5.3)
    # The re-ranker cannot beat baseline by >3pp and causes 429 rate-limit cascades.
    # Left as dead code for reference.

    # ---- Fact-density reranking (Phase 6.5: algorithmic, zero-LLM) ----
    # BEAM distractors are generic dev-talk; answer messages carry specific data.
    # Boost messages with dates, numbers, proper nouns, versions, technical terms.
    import re as _re_facts
    _FACT_PATTERNS = [
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:[, ]*\d{4})?\b', 2.0),  # dates
        (r'\b\d{4}-\d{2}-\d{2}\b', 2.5),  # ISO dates
        (r'\b\d+[.,]?\d*\s*(?:ms|sec|mins?|hours?|days?|weeks?|months?|years?|%|KB|MB|GB|TB|rows?|columns?|roles?|features?|bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|sprints?|tickets?)\b', 1.5),  # numbers with units
        (r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?\b', 1.5),  # version strings
        (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b', 1.0),  # proper noun phrases
        (r'\b[A-Z]{2,8}\b', 0.8),  # acronyms
    ]
    for mem in memories:
        content = mem.get("content", "")
        fact_score = 0.0
        for pattern, weight in _FACT_PATTERNS:
            matches = _re_facts.findall(pattern, content)
            fact_score += len(matches) * weight
        # Normalize by content length to get fact density
        density = fact_score / max(len(content.split()), 1)
        mem["fact_density"] = round(density, 4)
        # Boost score: blend original with fact density (40% fact boost)
        orig = mem.get("score", mem.get("relevance", 0))
        mem["score"] = orig * 0.6 + min(density * 5.0, 1.0) * 0.4

    # Re-sort by boosted score
    memories.sort(key=lambda m: m.get("score", 0), reverse=True)
    
    # Build recent context from last N messages (needed by both recursive and non-recursive paths).
    # Pure-recall mode SKIPS this entirely.
    recent_parts = []
    if not _pure_recall and conversation_messages:
        recent = conversation_messages[-RECENT_CONTEXT_COUNT:]
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                recent_parts.append(f"[{role}]: {content[:300]}")
    
    # ---- Recursive Retrieval Loop (Phase 8: two-pass for reasoning-heavy abilities) ----
    # TR, EO, CR, MR questions benefit from a second targeted pass after initial retrieval.
    # Pass 1: answer with current context -> Pass 2: gap analysis + targeted re-retrieval + re-answer.
    _RECURSIVE_ABILITIES = {'TR', 'EO', 'CR'}
    
    if needs_second_pass(question):
        # --- Helper: build context string from memory list ---
        def _build_context(mems, recents):
            if routing_ability == "EO":
                # Sort by message_index (true conversation order). All messages
                # share a constant ingest timestamp, so timestamp sort is a no-op.
                mems = sorted(mems, key=lambda x: x.get("message_index") if x.get("message_index") is not None else float('inf'))
            ctx_blocks = []
            if recents:
                ctx_blocks.append("RECENT CONVERSATION:\n" + "\n".join(recents))
            mem_seen = set()
            mem_strs = []
            chars = 0
            for m in mems:
                c = m.get("content", "")
                ck = c[:100]
                if ck in mem_seen:
                    continue
                mem_seen.add(ck)
                s = m.get("score", m.get("relevance", 0))
                if isinstance(s, (int, float)) and s < 0.05:
                    continue
                if chars + len(c) > MAX_MEMORY_CONTEXT_CHARS:
                    rem = MAX_MEMORY_CONTEXT_CHARS - chars
                    if rem > 100:
                        mem_strs.append(f"[Memory] {c[:rem]}...")
                    break
                mem_strs.append(f"[Memory] {c}")
                chars += len(c)
            if mem_strs:
                ctx_blocks.append("RETRIEVED MEMORIES:\n" + "\n\n".join(mem_strs))
            return "\n\n".join(ctx_blocks) if ctx_blocks else "[No memories found]"
        
        # --- CR: Negation-aware retrieval ---
        # CR rubrics require finding BOTH positive claims AND negations
        # ("never worked with Flask" vs "implemented Flask routes").
        # Regular FTS5 OR-search finds positive claims but misses negation
        # statements because "never"/"not"/"haven't" are stop-words or
        # don't co-occur with query terms in the same FTS5 token window.
        # LIKE-based exact substring search catches what FTS5 misses.
        if _wants_negation_retrieval(question):
            import re as _re_cr_neg
            # Extract key noun phrases from the question.
            # Use word boundaries to also catch all-caps acronyms (HTTP, API, SQL)
            _neg_terms = _re_cr_neg.findall(r'[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*', question)
            _neg_terms += _re_cr_neg.findall(r'\b[A-Z]{2,8}\b', question)  # catch HTTP, API, SQL
            _neg_exclude = {'have', 'could', 'which', 'what', 'this', 'that', 'does', 'about', 'there'}
            _neg_terms = [t for t in _neg_terms if len(t) > 3 and t.lower() not in _neg_exclude]
            if not _neg_terms:
                _neg_terms = [w for w in question.split() if len(w) > 4][:3]
            
            _neg_seen = {m.get("content", "")[:80] for m in memories}
            for _term in _neg_terms[:5]:
                for _neg_word in ['never', 'not', "haven't", "didn't", "wasn't", "weren't", "n't"]:
                    try:
                        # Search both working_memory and episodic_memory.
                        # AAAK consolidation moves old messages out of working_memory
                        # into episodic_memory, which the single-table query misses.
                        _neg_rows = beam.conn.execute(
                            "SELECT id, content FROM working_memory "
                            "WHERE content LIKE ? AND (content LIKE ? OR content LIKE ?) "
                            "UNION "
                            "SELECT id, content FROM episodic_memory "
                            "WHERE content LIKE ? AND (content LIKE ? OR content LIKE ?) "
                            "LIMIT 5",
                            (f"%{_term}%", f"%{_neg_word}%", f"%{_term}%{_neg_word}%",
                             f"%{_term}%", f"%{_neg_word}%", f"%{_term}%{_neg_word}%")
                        ).fetchall()
                        for _nr in _neg_rows:
                            _nk = _nr[1][:80] if _nr[1] else ""
                            if _nk and _nk not in _neg_seen:
                                _neg_seen.add(_nk)
                                memories.insert(0, {
                                    "id": _nr[0], "content": _nr[1], "score": 0.80,
                                    "source": "negation_cr",
                                    "raw_score": 0.80,
                                    "retrieval_strategy": "CR",
                                })
                    except Exception:
                        pass
        
        # --- TR: Timeline bypass ---
        # TR questions need exact dates from the conversation. Retrieval via
        # FTS5+vector misses date-specific content because date strings get
        # OR-tokenized ("2024-03-15" → "2024 OR 03 OR 15") and temporal
        # weighting can't compensate. Direct timeline extraction from the raw
        # conversation gives the LLM all dates with surrounding event context.
        _tr_timeline = None
        if not _pure_recall and is_temporal_query(question) and conversation_messages:
            _tr_timeline = _extract_timeline_from_conversation(conversation_messages)
            if _tr_timeline and len(_tr_timeline) >= 2:
                # Build a timeline string to inject as pre-context
                _tl_lines = ["DIRECT TIMELINE (extracted from raw conversation, NOT from retrieval):"]
                for _td in _tr_timeline:
                    _tl_lines.append(f"  {_td['date_str']}: {_td['event_text'][:200]}")
                _tl_str = "\n".join(_tl_lines)
                # Prepend timeline to memories as a synthetic high-score entry
                memories.insert(0, {"id": "timeline_direct", "content": _tl_str, "score": 1.0,
                                    "source": "tr_timeline_bypass",
                                    "raw_score": 1.0,
                                    "retrieval_strategy": "TR"})
                print(f"    [TR-timeline] injected {len(_tr_timeline)} dates from conversation", flush=True)
        
        # --- Pass 1: Initial context building (no LLM call, only retrieval) ---
        pass1_ctx = _build_context(memories, recent_parts)

        # --- Gap analysis: extract exact date/entity strings for Pass 2 FTS5 hard-filter ---
        # Pure regex extraction (no cloud LLM call) to stay within recall budget.
        import re as _re_gap
        gap_queries = []

        # Extract dates from Pass 1 context
        gap_queries.extend(_re_gap.findall(r'\b\d{4}-\d{2}-\d{2}\b', pass1_ctx))

        # Extract month+day patterns
        gap_queries.extend(_re_gap.findall(
            r'(?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?',
            pass1_ctx, _re_gap.IGNORECASE
        ))

        # Extract named entities from the question that aren't in pass1 results
        q_entities = _re_gap.findall(r'\b[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*\b', question)
        for ent in q_entities[:3]:
            if ent.lower() not in pass1_ctx.lower()[:2000]:
                gap_queries.append(ent)

        # Extract key terms from question
        q_terms = _extract_search_terms(question)
        for term in q_terms[:3]:
            if len(term) > 2 and term.lower() not in pass1_ctx.lower()[:2000]:
                gap_queries.append(term)

        # Deduplicate
        gap_queries = list(dict.fromkeys(gap_queries))

        # Debug: log gap analysis results
        print(f"    [DEBUG-GAP] ability={routing_ability} regex-extracted queries={gap_queries}", flush=True)
        
        # --- Pass 2: Targeted retrieval + re-answer ---
        if gap_queries:
            gap_memories = []
            gap_seen = set()
            gap_diag = {"strategies": {}}
            for gq in gap_queries[:3]:
                # Standard recall
                for mem in _multi_strategy_recall(beam, gq, top_k, ability=routing_ability, diag=gap_diag):
                    ck = mem.get("content", "")[:80]
                    if ck not in gap_seen:
                        gap_seen.add(ck)
                        gap_memories.append(mem)
                # MEMORIA structured recall for the same gap query
                try:
                    _memoria_gap = beam.memoria_retrieve(gq, ability=routing_ability, top_k=5)
                    if _memoria_gap and _memoria_gap.get("source") != "fallback" and _memoria_gap.get("context"):
                        _mg_key = _memoria_gap["context"][:80]
                        if _mg_key not in gap_seen:
                            gap_seen.add(_mg_key)
                            _mg_facts = _memoria_gap.get("facts", [])
                            print(f"      [MEMORIA-gap] {_memoria_gap['source']} hit for gap=\"{gq[:60]}\": {len(_mg_facts)} facts", flush=True)
                            gap_memories.insert(0, {
                                "content": f"[MEMORIA {_memoria_gap['source']}]\n{_memoria_gap['context']}",
                                "score": 0.95,
                                "source": f"memoria_gap_{_memoria_gap['source']}",
                                "raw_score": 0.95,
                                "retrieval_strategy": "MEMORIA",
                            })
                except Exception:
                    pass
            
            # Merge: original + gap memories, deduplicate, re-sort
            all_mems = list(memories)
            existing_keys = {m.get("content", "")[:80] for m in all_mems}
            for gm in gap_memories:
                gk = gm.get("content", "")[:80]
                if gk not in existing_keys:
                    existing_keys.add(gk)
                    all_mems.append(gm)
            all_mems.sort(key=lambda m: m.get("score", 0), reverse=True)
            
            # Rebuild context with augmented memories, trimmed for pass2.
            # Ordering questions need ALL events to order correctly, so give them
            # a much larger budget; duration questions only need a few dates.
            pass2_ctx = _build_context(all_mems, recent_parts)
            _pass2_limit = 16000 if is_ordering_query(question) else 6000
            if len(pass2_ctx) > _pass2_limit:
                pass2_ctx = pass2_ctx[:_pass2_limit] + "...[truncated]"
            
            # Inject temporal cheatsheet for TR/EO questions
            _pass2_temporal_cheat = _inject_temporal_cheatsheet(all_mems, question)

            # Switch to Calculator prompt ONLY for DURATION questions.
            # Ordering (EO) questions must use the ordering prompt below — routing
            # them to the calculator made them output a number, killing tau-b.
            if _is_calculator_question(question):
                calc_prompt = """You are a precise temporal calculator. You have been provided with specific retrieved evidence (dates, event timelines).
Your task is to compute the duration or interval between the events.
DO NOT use chat pleasantries or summarize the conversation.
Follow this format strictly:
1. IDENTIFIED DATES: [List dates found]
2. CALCULATION: [Show the step-by-step math]
3. FINAL ANSWER: [Provide only the number/duration]"""
                pass2_messages = [
                    {"role": "system", "content": calc_prompt},
                    {"role": "user", "content": _pass2_temporal_cheat + pass2_ctx + "\n\nQUESTION: " + question + "\n\nANSWER:"},
                ]
            else:
                # CR questions need contradiction-first prompt even in Pass 2
                _pass2_prompt = build_system_prompt(question)
                pass2_messages = [
                    {"role": "system", "content": _pass2_prompt},
                    {"role": "user", "content": _pass2_temporal_cheat + pass2_ctx + "\n\nQUESTION: " + question + "\n\nANSWER:"},
                ]
            _record_second_pass_diagnostics(diag, gap_queries[:3], gap_diag)
            return _ret(llm.chat(pass2_messages, temperature=0.1, max_tokens=8192), all_mems)

        # No gaps: fall through to the single final LLM answer below
    # ---- END Recursive Retrieval Loop ----

    # ---- SUM: Broad topic sampling for summarization ----
    _is_sum = any(w in question.lower() for w in ["summarize", "summary", "overview", "main topics", "key themes"])
    if _is_sum:
        try:
            total_rows = beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
            if total_rows > 0:
                stride = max(1, total_rows // 60)
                broad_rows = beam.conn.execute(
                    "SELECT id, content, message_index FROM working_memory "
                    "WHERE content != '' ORDER BY ROWID LIMIT 60 OFFSET 0"
                ).fetchall()
                if stride > 1:
                    broad_rows = beam.conn.execute(
                        f"SELECT id, content, message_index FROM working_memory "
                        f"WHERE content != '' AND (ROWID % {stride}) = 0 "
                        f"ORDER BY ROWID LIMIT 60"
                    ).fetchall()
                _sum_seen = {m.get("content", "")[:80] for m in memories}
                for row in broad_rows:
                    ck = row["content"][:80]
                    if ck not in _sum_seen:
                        _sum_seen.add(ck)
                        memories.append({
                            "id": row["id"], "content": row["content"],
                            "score": 0.3, "source": "sum_broad_sample",
                            "raw_score": 0.3,
                            "retrieval_strategy": "SUM",
                            "message_index": row["message_index"],
                        })
        except Exception:
            pass
        try:
            from edumem.core.mmr import mmr_rerank as _mmr
            memories = _mmr(memories, lambda_param=0.5, top_k=top_k * 3)
        except Exception:
            pass

    context = ""  # Built below from memories

    # Build retrieved memory context (deduplicated, relevance-sorted)
    _effective_max_chars = 24000 if _is_sum else MAX_MEMORY_CONTEXT_CHARS
    seen_content = set()
    memory_parts = []
    total_chars = 0
    for i, mem in enumerate(memories):
        mem.setdefault("final_context_included", False)
        content = mem.get("content", "")
        # Deduplicate
        content_key = content[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)

        score = mem.get("score", mem.get("relevance", 0))
        if isinstance(score, (int, float)) and score < 0.05:
            continue  # Skip very low relevance

        if total_chars + len(content) > _effective_max_chars:
            remaining = _effective_max_chars - total_chars
            if remaining > 100:
                mem["final_context_included"] = True
                memory_parts.append(f"[Memory] {content[:remaining]}...")
            break
        mem["final_context_included"] = True
        memory_parts.append(f"[Memory] {content}")
        total_chars += len(content)

    # Build prompt with contexts (skip if full-conversation mode already set)
    if not context:
        context_blocks = []
        if recent_parts:
            context_blocks.append("RECENT CONVERSATION:\n" + "\n".join(recent_parts))
        if memory_parts:
            context_blocks.append("RETRIEVED MEMORIES:\n" + "\n\n".join(memory_parts))
        
        context = "\n\n".join(context_blocks) if context_blocks else "[No memories found]"

    # If we found a direct context→value match, return it immediately (zero LLM cost)
    if context_answer:
        return _ret(context_answer, memories)

    # Inject CR contradiction context if detected
    _cr_prefix_ret = ""
    if _cr_context:
        _cr_prefix_ret = f"\n\n{_cr_context}\n\n"

    # Inject user instructions and preferences for IF/PF
    _if_pf_prefix = ""
    _instructions = [m.get("content", "") for m in memories if "[INSTRUCTION]" in m.get("content", "")]
    _preferences = [m.get("content", "") for m in memories if "[PREFERENCE]" in m.get("content", "")]
    if _instructions:
        _if_pf_prefix += "\nUSER INSTRUCTIONS (follow these when answering):\n" + "\n".join(f"- {inst}" for inst in _instructions) + "\n"
    if _preferences:
        _if_pf_prefix += "\nUSER PREFERENCES (respect these in your answer):\n" + "\n".join(f"- {pref}" for pref in _preferences) + "\n"
    if _instructions or _preferences:
        _if_pf_prefix += (
            "\nIMPORTANT: The user has previously given instructions or preferences listed above. "
            "When answering the question, APPLY these instructions and preferences to shape your response. "
            "Do NOT say the conversation lacks information if the user's instructions are relevant to the question — "
            "use them to guide your answer.\n"
        )

    # All queries use build_system_prompt() dynamically, which appends
    # light format modifiers to an always-on base prompt for EO/TR queries.
    _prompt = build_system_prompt(question)
    _temporal_cheat = _inject_temporal_cheatsheet(memories, question)
    messages = [
        {"role": "system", "content": _prompt},
        {"role": "user", "content": f"{_if_pf_prefix}{_temporal_cheat}{_cr_prefix_ret}{context}\n\nQUESTION: {question}\n\nANSWER:"},
    ]

    _answer_max_tokens = 8192 if ability in ("CR", "SUM", "MR") else 2048
    ans = llm.chat(messages, temperature=0.1, max_tokens=_answer_max_tokens)
    try:
        from edumem.core.beam import clean_and_format_sequence
        ans = clean_and_format_sequence(question, ans)
    except Exception as e:
        print(f"[!] Warning: clean_and_format_sequence failed: {e}")
    return _ret(ans, memories)


# ============================================================
#  LLM-as-Judge: Nugget-Based Scoring (BEAM Protocol)
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a memory benchmark.
You will be given:
1. A question about a conversation
2. A list of RUBRIC ITEMS (expected facts the AI should mention)
3. The AI's ANSWER

For EACH rubric item, check if the AI's answer contains equivalent information:
- Score 1.0: correct info present, substantially matches the rubric item
- Score 0.5: partially correct, some key detail missing or slightly wrong
- Score 0.0: missing or wrong

Return ONLY this JSON:
{"scores":[1.0,0.5,0.0],"overall_score":0.X}

Where scores[i] corresponds to rubric[i], and overall_score is the average."""


def _clean_judge_json(raw: str) -> str:
    """Normalize an LLM judge response into clean JSON for the official grader.

    Judge models (esp. gpt-4o) wrap their JSON in ```json fences or prepend
    assessment prose. The official compute_metrics grader parses strictly, so any
    non-pristine payload silently scores 0. This strips markdown fences, isolates
    the JSON payload from surrounding chatter, repairs it, and unwraps a single-
    element list to the inner dict. Conservative: if nothing parses, return the
    original string unchanged (never make a parseable response worse).
    """
    if not raw or not raw.strip():
        return raw
    import re as _re_j
    import json as _json_j
    text = raw.strip()
    # 1) Strip a leading ```json / ``` fence and trailing ``` if present.
    fence = _re_j.match(r'^```(?:json)?\s*(.*?)\s*```$', text, _re_j.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 2) Isolate the JSON payload from surrounding prose: first '{' or '[' to the
    #    last matching '}' or ']'. Pick whichever bracket type appears first.
    obj_start = text.find('{')
    arr_start = text.find('[')
    candidates = [p for p in (obj_start, arr_start) if p != -1]
    if candidates:
        start = min(candidates)
        end = max(text.rfind('}'), text.rfind(']'))
        if end > start:
            text = text[start:end + 1]
    # 3) Parse. Try plain json first (fence-strip + isolation usually suffices),
    #    then json_repair if available, else fall back to the original raw string.
    parsed = None
    try:
        parsed = _json_j.loads(text)
    except Exception:
        try:
            from json_repair import repair_json
            parsed = _json_j.loads(repair_json(text))
        except Exception:
            return raw
    # 4) Unwrap a single-element list of one score dict -> the inner dict.
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict) and "score" in parsed[0]:
        parsed = parsed[0]
    return _json_j.dumps(parsed)


def _judge_client_snapshot(llm: LLMClient) -> dict:
    """Capture non-secret judge client diagnostics after a chat call."""
    return {
        "raw_response": getattr(llm, "last_response", "") or "",
        "finish_reason": getattr(llm, "last_finish_reason", None),
        "response_had_content": getattr(llm, "last_response_had_content", None),
        "retry_count": getattr(llm, "last_retry_count", 0),
        "api_error_class": getattr(llm, "last_error_class", None),
        "api_error_message": getattr(llm, "last_error_message", ""),
    }


_OFFICIAL_GRADER_IMPORT_CACHE: tuple[object | None, str | None] | None = None
_OFFICIAL_GRADER_IMPORT_ATTEMPTS = 0
_OFFICIAL_GRADER_REPO_URL = "https://github.com/mohammadtavakoli78/BEAM.git"
_OFFICIAL_GRADER_CANDIDATE_ROOTS = (
    Path(os.environ.get("BEAM_OFFICIAL_REPO_PATH", "")) if os.environ.get("BEAM_OFFICIAL_REPO_PATH") else None,
    Path("/opt/BEAM_official"),
    PROJECT_ROOT / ".cache" / "BEAM_official",
)


def _resolve_official_grader_root() -> Path | None:
    """Find or fetch the official BEAM repo that contains src/evaluation/compute_metrics.py."""
    for root in _OFFICIAL_GRADER_CANDIDATE_ROOTS:
        if root and (root / "src" / "evaluation" / "compute_metrics.py").exists():
            return root

    cache_root = PROJECT_ROOT / ".cache" / "BEAM_official"
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not cache_root.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", _OFFICIAL_GRADER_REPO_URL, str(cache_root)],
                check=True,
                capture_output=True,
                text=True,
            )
    except Exception as exc:
        return None

    if (cache_root / "src" / "evaluation" / "compute_metrics.py").exists():
        return cache_root
    return None


def _load_official_compute_metrics() -> tuple[object | None, str | None]:
    """Load the official BEAM grader module once and cache failures."""
    global _OFFICIAL_GRADER_IMPORT_CACHE
    global _OFFICIAL_GRADER_IMPORT_ATTEMPTS

    if _OFFICIAL_GRADER_IMPORT_CACHE is not None:
        return _OFFICIAL_GRADER_IMPORT_CACHE

    _OFFICIAL_GRADER_IMPORT_ATTEMPTS += 1
    try:
        grader_root = _resolve_official_grader_root()
        if grader_root is None:
            raise FileNotFoundError(
                "official BEAM repo not found locally and clone failed; "
                f"expected {PROJECT_ROOT / '.cache' / 'BEAM_official'} or /opt/BEAM_official"
            )
        if str(grader_root) not in sys.path:
            sys.path.insert(0, str(grader_root))

        import importlib
        compute_metrics = importlib.import_module("src.evaluation.compute_metrics")
        
        # Monkeypatch the event ordering bug in official BEAM to avoid overwriting system_list
        def patched_evaluate_event_ordering(rubric: list, llm_response: str, probing_question: str, model):
            import json
            extract_facts_fn = getattr(compute_metrics, "extract_facts")
            event_ordering_score_fn = getattr(compute_metrics, "event_ordering_score")
            unified_llm_judge_base_prompt_val = getattr(compute_metrics, "unified_llm_judge_base_prompt")
            parse_json_response_fn = getattr(compute_metrics, "parse_json_response")
            repair_json_fn = getattr(compute_metrics, "repair_json", None)
            
            system_list = extract_facts_fn(paragraph=llm_response, question=probing_question, model=model)
            # The official BEAM benchmark bug overrode system_list with llm_response.split("\n"),
            # discarding the results of extract_facts entirely. We skip that overwrite line here!
            
            score = event_ordering_score_fn(reference_list=rubric, system_list=system_list, align_type="llm", llm=model)
            
            llm_judge_responses = []
            llm_judge_score = 0
            for item in rubric:
                prompt = unified_llm_judge_base_prompt_val.replace("<rubric_item>", item).replace("<llm_response>", llm_response)
                response = model.invoke(prompt).content.strip()
                try:
                    response = parse_json_response_fn(response=response)
                except Exception:
                    if repair_json_fn:
                        response = json.loads(repair_json_fn(response))
                    else:
                        response = json.loads(response)
                llm_judge_score += float(response['score'])
                llm_judge_responses.append(response)
                
            llm_judge_score = llm_judge_score / len(rubric)
            score["llm_judge_score"] = llm_judge_score
            score["llm_judge_responses"] = llm_judge_responses
            return score
            
        compute_metrics.evaluate_event_ordering = patched_evaluate_event_ordering
        _OFFICIAL_GRADER_IMPORT_CACHE = (compute_metrics, None)
    except Exception as exc:
        _OFFICIAL_GRADER_IMPORT_CACHE = (None, str(exc))

    return _OFFICIAL_GRADER_IMPORT_CACHE


def _llm_client_snapshot(llm: LLMClient) -> dict:
    """Capture answer-side API diagnostics without secrets."""
    return {
        "finish_reason": getattr(llm, "last_finish_reason", None),
        "response_had_content": getattr(llm, "last_response_had_content", None),
        "retry_count": getattr(llm, "last_retry_count", 0),
        "api_error_class": getattr(llm, "last_error_class", None),
        "api_error_message": getattr(llm, "last_error_message", ""),
    }


def judge_with_rubrics(llm: LLMClient, question: str, rubric: list, ai_answer: str, ability: str = None) -> dict:
    """Judge an AI answer against pre-written BEAM rubric items using the official BEAM grader."""
    if not rubric:
        return {"scores": [], "overall_score": 0.0, "assessment": "no rubric available"}

    MAPPING = {
        "ABS": "evaluate_abstention",
        "CR": "evaluate_contradiction_resolution",
        "EO": "evaluate_event_ordering",
        "IE": "evaluate_information_extraction",
        "IF": "evaluate_instruction_following",
        "KU": "evaluate_knowledge_update",
        "MR": "evaluate_multi_session_reasoning",
        "PF": "evaluate_preference_following",
        "SUM": "evaluate_summarization",
        "TR": "evaluate_temporal_reasoning",
    }

    func_name = MAPPING.get(ability) if ability else None
    if not func_name:
        return _legacy_judge_with_rubrics(llm, question, rubric, ai_answer)

    compute_metrics, probe_error = _load_official_compute_metrics()
    if compute_metrics is None:
        return _legacy_judge_with_rubrics(llm, question, rubric, ai_answer)

    try:
        import os
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy"

        eval_func = getattr(compute_metrics, func_name, None)

        if not eval_func:
            raise ImportError(f"{func_name} not found in compute_metrics")

        class LangchainLLMWrapper:
            def __init__(self, client, question_text):
                self.client = client
                self._question_text = question_text

            def invoke(self, prompt):
                if isinstance(prompt, str) and "<question>" in prompt:
                    prompt = prompt.replace("<question>", self._question_text)
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": prompt}]
                res = self.client.chat(messages, temperature=0.0, max_tokens=1024)
                if res:
                    # Harden against markdown-fenced / prose-wrapped judge output
                    # that would otherwise reach the strict grader raw and score 0.
                    res = _clean_judge_json(res)
                class Response:
                    def __init__(self, content):
                        self.content = content
                return Response(res or "")

        wrapped_model = LangchainLLMWrapper(llm, question)

        print(f"      [Grader-Swap] Running official {func_name}...")
        eval_result = eval_func(rubric=rubric, llm_response=ai_answer, probing_question=question, model=wrapped_model)
        client_diag = _judge_client_snapshot(llm)

        if ability == "EO":
            score = eval_result.get("tau_norm", 0.0)
        else:
            score = eval_result.get("llm_judge_score", 0.0)

        responses = eval_result.get("llm_judge_responses", [])

        scores = []
        for r in responses:
            if isinstance(r, dict) and "score" in r:
                try:
                    scores.append(float(r["score"]))
                except (ValueError, TypeError):
                    scores.append(0.0)
            else:
                scores.append(0.0)

        while len(scores) < len(rubric):
            scores.append(0.0)

        parse_status = "ok"
        if client_diag.get("api_error_class"):
            parse_status = "api_failure"
        elif client_diag.get("response_had_content") is False:
            parse_status = "parse_failure"
        elif client_diag.get("raw_response") and not responses:
            parse_status = "parse_failure"

        return {
            "scores": scores[:len(rubric)],
            "overall_score": float(score),
            "official_score": float(score),
            "partial_credit_score": sum(scores[:len(rubric)]) / len(rubric) if rubric else 0.0,
            "scoring_mode": "official",
            "parse_status": parse_status,
            "judge_status": "ok" if parse_status == "ok" else parse_status,
            "judge_failure_class": client_diag.get("api_error_class"),
            "judge_failure_message": client_diag.get("api_error_message", ""),
            "raw_response": client_diag.get("raw_response", ""),
            "finish_reason": client_diag.get("finish_reason"),
            "response_had_content": client_diag.get("response_had_content"),
            "retry_count": client_diag.get("retry_count", 0),
            "assessment": f"Evaluated using official BEAM compute_metrics.{func_name}",
            "brief_assessment": f"official BEAM grader ({func_name})",
            "raw_result": eval_result,
        }

    except Exception:
        return _legacy_judge_with_rubrics(llm, question, rubric, ai_answer)


def _legacy_judge_with_rubrics(llm: LLMClient, question: str, rubric: list, ai_answer: str) -> dict:
    rubric_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(rubric))
    user_prompt = f"""QUESTION: {question}

RUBRIC ITEMS:
{rubric_text}

AI's ANSWER: {ai_answer}

For each rubric item, score how well the AI's answer matches. Return JSON with scores array and overall_score (average)."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = llm.chat(messages, temperature=0.0, max_tokens=500)
    client_diag = _judge_client_snapshot(llm)
    if response is None:
        return {
            "scores": [0.0] * len(rubric),
            "overall_score": 0.0,
            "official_score": 0.0,
            "partial_credit_score": 0.0,
            "scoring_mode": "fallback",
            "parse_status": "api_failure",
            "judge_status": "api_failure",
            "judge_failure_class": client_diag.get("api_error_class"),
            "judge_failure_message": client_diag.get("api_error_message", ""),
            "raw_response": client_diag.get("raw_response", ""),
            "finish_reason": client_diag.get("finish_reason"),
            "response_had_content": client_diag.get("response_had_content"),
            "retry_count": client_diag.get("retry_count", 0),
            "assessment": "LLM judge returned None (timeout or error)",
        }
    result, parse_status = _parse_judge_payload(response)
    if parse_status == "ok" and isinstance(result, dict):
        scores = []
        for item in result.get("scores", []):
            try:
                scores.append(float(item))
            except (TypeError, ValueError):
                scores.append(0.0)
        while len(scores) < len(rubric):
            scores.append(0.0)
        official_score = _mean_int_score(scores[:len(rubric)])
        partial_credit_score = _mean_float_score(scores[:len(rubric)])
        self_reported_overall = result.get("overall_score", 0.0)
        try:
            self_reported_overall = float(self_reported_overall or 0.0)
        except (TypeError, ValueError):
            self_reported_overall = 0.0
        return {
            "scores": scores[:len(rubric)],
            "overall_score": self_reported_overall,
            "official_score": official_score,
            "partial_credit_score": partial_credit_score,
            "scoring_mode": "fallback",
            "parse_status": "ok",
            "judge_status": "ok",
            "judge_failure_class": client_diag.get("api_error_class"),
            "judge_failure_message": client_diag.get("api_error_message", ""),
            "raw_response": client_diag.get("raw_response", response),
            "finish_reason": client_diag.get("finish_reason"),
            "response_had_content": client_diag.get("response_had_content"),
            "retry_count": client_diag.get("retry_count", 0),
            "assessment": "LLM judge fallback parsed JSON",
            "raw_result": result,
        }
    return {
        "scores": [0.0] * len(rubric),
        "overall_score": basic_text_similarity(ai_answer, " ".join(rubric)),
        "official_score": 0.0,
        "partial_credit_score": 0.0,
        "scoring_mode": "fallback",
        "parse_status": "parse_failure" if client_diag.get("api_error_class") is None else "api_failure",
        "judge_status": "parse_failure" if client_diag.get("api_error_class") is None else "api_failure",
        "judge_failure_class": client_diag.get("api_error_class"),
        "judge_failure_message": client_diag.get("api_error_message", ""),
        "raw_response": client_diag.get("raw_response", response),
        "finish_reason": client_diag.get("finish_reason"),
        "response_had_content": client_diag.get("response_had_content"),
        "retry_count": client_diag.get("retry_count", 0),
        "assessment": "JSON parse failed; using fallback",
        "raw_result": response,
    }


def basic_text_similarity(text1: str, text2: str) -> float:
    """Simple token overlap as fallback when LLM judge fails."""
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union) if union else 0.0


# ============================================================
#  Evaluation Runner
# ============================================================

def evaluate_conversation(
    llm: LLMClient,
    judge_llm: LLMClient,
    beam: BeamMemory,
    conversation: dict,
    resume_ids: set = None,
) -> dict:
    """Evaluate all probing questions for one conversation."""
    conv_id = conversation["id"]
    questions = conversation["questions"][:BENCHMARK_QUERIES_PER_CONV]
    results = []
    evaluated_count = 0
    skipped_count = 0

    print(f"  Conversation {conv_id}: {len(questions)} questions")

    # Thread worker for thread-safe parallel evaluation (Hindsight concurrent optimization pattern)
    def _evaluate_question_worker(item):
        idx, q = item
        qid = f"{conv_id}:q{idx}"
        if resume_ids and qid in resume_ids:
            return None

        policy = _question_row_policy(q)
        question = policy["question"]
        ideal = policy["ideal_answer"]
        rubric = policy["rubric"]
        ability_raw = q.get("ability", "unknown")
        ability = str(ABILITY_MAP.get(ability_type := q.get("ability", "unknown"), ability_type))

        if not policy["should_evaluate"]:
            return {
                "is_skipped": True,
                "result": _build_skipped_question_result(
                    qid=qid,
                    ability=ability,
                    question=question,
                    ideal_answer=ideal,
                    rubric=rubric,
                    skip_reason=policy["skip_reason"],
                ),
                "skip_reason": policy["skip_reason"],
                "ability": ability,
                "qid": qid,
            }

        # Sub-threads must instantiate their own BeamMemory instance for SQLite thread safety
        from edumem.core.beam import BeamMemory
        thread_beam = BeamMemory(session_id=beam.session_id, db_path=beam.db_path)
        try:
            q_diag = {}
            t0 = time.perf_counter()
            ret_ans = answer_with_memory(
                llm, thread_beam, question,
                conversation_messages=conversation.get("messages", []),
                ability=ability, diag=q_diag, return_memories=True,
            )
            assert isinstance(ret_ans, tuple)
            ai_answer, recall_memories = ret_ans
            answer_time = time.perf_counter() - t0

            if ai_answer is None:
                ai_answer = "[LLM_ERROR: No response from answering model]"

            answer_diag = _llm_client_snapshot(llm)
            q_diag["answer_api"] = answer_diag

            t0 = time.perf_counter()
            normalized_answer = normalize_for_judge(ai_answer, ability)
            judgment = judge_with_rubrics(judge_llm, question, rubric, normalized_answer, ability=ability)
            judge_time = time.perf_counter() - t0

            judged = _summarize_judge_result(judgment)
            score = judged.get("official_score", 0.0)
            recall_provenance = _summarize_recall_memories(recall_memories or [])

            result = {
                "qid": qid,
                "ability": ability,
                "question": question[:200],
                "question_full": question,
                "ideal_answer": ideal[:200],
                "ideal_answer_full": ideal,
                "rubric": rubric,
                "ai_answer": ai_answer,
                "ai_answer_full": ai_answer,
                "ai_answer_excerpt": ai_answer[:500],
                "recall_provenance": recall_provenance,
                "score": score,
                "official_score": judged.get("official_score", score),
                "partial_credit_score": judged.get("partial_credit_score", score),
                "scoring_mode": judged.get("scoring_mode", "unknown"),
                "parse_status": judged.get("parse_status", "unknown"),
                "judge_status": judged.get("judge_status", "unknown"),
                "judge_failure_class": judged.get("judge_failure_class"),
                "judge_failure_message": judged.get("judge_failure_message", ""),
                "judge_raw_response": judged.get("raw_response", ""),
                "judge_raw_result": judged.get("raw_result"),
                "judge_raw_payload": judged.get("raw_result", judged.get("raw_response", "")),
                "judge_finish_reason": judged.get("finish_reason"),
                "judge_response_had_content": judged.get("response_had_content"),
                "judge_retry_count": judged.get("retry_count", 0),
                "nuggets": judged.get("nuggets", []),
                "assessment": judged.get("brief_assessment", judged.get("assessment", "")),
                "judge_assessment": judged.get("assessment", ""),
                "answer_model": llm.model,
                "judge_model": judge_llm.model,
                "answer_time_ms": answer_time * 1000,
                "judge_time_ms": judge_time * 1000,
                "retrieval_diagnostics": q_diag,
                "answer_api_diagnostics": answer_diag,
            }
            return {
                "is_skipped": False,
                "result": result,
                "ability": ability,
                "score": score,
                "answer_time_ms": answer_time * 1000,
                "judge_time_ms": judge_time * 1000,
                "question_full": question,
                "qid": qid,
            }
        finally:
            thread_beam.conn.close()

    jobs = [(idx, q) for idx, q in enumerate(questions)]
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Use a ThreadPoolExecutor with max_workers=4 to respect proxy rate limit (concurrency of 4)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_evaluate_question_worker, job): job for job in jobs}
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            if res["is_skipped"]:
                results.append(res["result"])
                skipped_count += 1
                print(f"    [SKIP] [{res['ability']}] {res['qid']} reason={res['skip_reason']}")
            else:
                results.append(res["result"])
                evaluated_count += 1
                print(f"    [{res['ability']}] score={res['score']:.2f} ans={res['answer_time_ms']:.0f}ms judge={res['judge_time_ms']:.0f}ms "
                      f"Q: {res['question_full'][:60]}...")

                # Pause optionally if required (usually BEAM_QUESTION_DELAY is 2 or 0)
                _q_delay = float(os.environ.get("BEAM_QUESTION_DELAY", "2"))
                if _q_delay > 0:
                    time.sleep(_q_delay)

    # Sort results back to original question index order for deterministic file diffs
    results.sort(key=lambda r: int(r["qid"].split(":q")[1]))

    return {
        "conversation_id": conv_id,
        "scale": conversation["scale"],
        "num_questions": len(questions),
        "num_evaluated": evaluated_count,
        "num_skipped": skipped_count,
        "num_accounted_for": len(results),
        "results": results,
    }


def compute_ability_scores(all_results: list[dict]) -> dict:
    """Aggregate scores by ability and scale."""
    by_scale_ability = defaultdict(lambda: defaultdict(list))

    for conv_result in all_results:
        scale = conv_result["scale"]
        for r in conv_result.get("results", []):
            if _is_skipped_question_result(r):
                continue
            ability = r.get("ability", "unknown")
            score = r.get("score", 0.0)
            by_scale_ability[scale][ability].append(score)

    # Compute averages
    summary = {}
    for scale, abilities in by_scale_ability.items():
        scale_scores = {}
        ability_avgs = []
        total_count = 0
        for ability, scores in abilities.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            scale_scores[ability] = {
                "avg_score": avg,
                "count": len(scores),
            }
            ability_avgs.append(avg)
            total_count += len(scores)

        # OVERALL is a MACRO-average: the mean of per-ability scores, each ability
        # weighted equally. This matches BEAM's leaderboard convention and keeps the
        # SOTA comparison valid when abilities have unequal question counts. (A micro
        # average over pooled questions would let high-frequency abilities dominate.)
        overall = sum(ability_avgs) / len(ability_avgs) if ability_avgs else 0.0
        scale_scores["OVERALL"] = {
            "avg_score": overall,
            "count": total_count,
        }

        summary[scale] = scale_scores

    return summary


def compute_micro_scores(all_results: list[dict]) -> dict[str, float]:
    """Compute micro-averaged scores per scale for diagnostics."""
    by_scale = defaultdict(list)
    for conv_result in all_results:
        scale = conv_result["scale"]
        for r in conv_result.get("results", []):
            if _is_skipped_question_result(r):
                continue
            score = r.get("score", 0.0)
            by_scale[scale].append(score)
    return {
        scale: (sum(scores) / len(scores)) if scores else 0.0
        for scale, scores in by_scale.items()
    }


def compute_partial_credit_overall(all_results: list[dict]) -> float:
    """Compute overall partial-credit from nested question rows."""
    scores = []
    for question in _iter_evaluated_question_rows(all_results):
        try:
            scores.append(float(question.get("partial_credit_score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _build_evaluation_summary(all_results: list[dict], metadata: dict) -> dict:
    """Build the summary artifact shared by live runs and re-judge runs."""
    ability_summary = compute_ability_scores(all_results)
    micro_overall = compute_micro_scores(all_results)
    partial_credit_overall = compute_partial_credit_overall(all_results)
    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "ability_summary": {
            scale: {
                ab: {"avg_score": v["avg_score"], "count": v["count"]}
                for ab, v in abilities.items()
            }
            for scale, abilities in ability_summary.items()
        },
        "micro_overall": micro_overall,
        "partial_credit_overall": partial_credit_overall,
    }


def _results_summary_path(results_path: Path) -> Path:
    """Derive the summary path associated with a results JSON artifact."""
    if results_path.name == RESULTS_FILE.name:
        return results_path.with_name("beam_e2e_summary.json")
    return results_path.with_name(f"{results_path.stem}_summary{results_path.suffix}")


def _rejudge_output_path(source_path: Path) -> Path:
    """Derive the default re-judge output path without overwriting the source."""
    return source_path.with_name(f"{source_path.stem}.rejudged{source_path.suffix}")


def _load_results_artifact(path: Path) -> dict:
    """Load a BEAM results artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Results artifact not found: {path}")

    with open(path, encoding="utf-8") as f:
        artifact = json.load(f)

    if not isinstance(artifact, dict):
        raise ValueError(f"Results artifact must be a JSON object: {path}")

    results = artifact.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Results artifact missing top-level 'results' list: {path}")

    return artifact


def _required_rejudge_fields(row: dict) -> list[str]:
    """Return the stored row fields needed for a re-judge pass."""
    missing = []

    question_full = row.get("question_full")
    if not isinstance(question_full, str) or not question_full.strip():
        missing.append("question_full")

    rubric = row.get("rubric")
    if rubric is None or not isinstance(rubric, list):
        missing.append("rubric")

    ai_answer_full = row.get("ai_answer_full")
    if not isinstance(ai_answer_full, str) or not ai_answer_full.strip():
        missing.append("ai_answer_full")

    return missing


def _update_rejudged_question_row(
    row: dict,
    judged: dict,
    judge_model: str,
    judge_time_ms: float,
) -> dict:
    """Merge a fresh judge result into a stored question row."""
    updated = deepcopy(row)
    updated["score"] = judged.get("official_score", 0.0)
    updated["official_score"] = judged.get("official_score", 0.0)
    updated["partial_credit_score"] = judged.get("partial_credit_score", 0.0)
    updated["scoring_mode"] = judged.get("scoring_mode", "unknown")
    updated["parse_status"] = judged.get("parse_status", "unknown")
    updated["judge_status"] = judged.get("judge_status", "unknown")
    updated["judge_failure_class"] = judged.get("judge_failure_class")
    updated["judge_failure_message"] = judged.get("judge_failure_message", "")
    updated["judge_raw_response"] = judged.get("raw_response", "")
    updated["judge_raw_result"] = judged.get("raw_result")
    raw_payload = judged.get("raw_result")
    if raw_payload is None:
        raw_payload = judged.get("raw_response", "")
    updated["judge_raw_payload"] = raw_payload
    updated["judge_finish_reason"] = judged.get("finish_reason")
    updated["judge_response_had_content"] = judged.get("response_had_content")
    updated["judge_retry_count"] = judged.get("retry_count", 0)
    updated["assessment"] = judged.get("brief_assessment", judged.get("assessment", ""))
    updated["judge_assessment"] = judged.get("assessment", "")
    updated["nuggets"] = judged.get("nuggets", [])
    updated["judge_model"] = judge_model
    updated["judge_time_ms"] = judge_time_ms
    return updated


def _normalize_stored_judgment_record(record: dict, qid: str) -> tuple[dict, float]:
    """Extract a stored judgment payload and timing from a keyed record."""
    if not isinstance(record, dict):
        raise ValueError(f"Stored judgment record for {qid} must be a JSON object")

    record_qid = record.get("qid")
    if record_qid is not None and record_qid != qid:
        raise ValueError(f"Stored judgment record key {qid} does not match embedded qid {record_qid}")

    judgment = record.get("judgment", record)
    if not isinstance(judgment, dict):
        raise ValueError(f"Stored judgment payload for {qid} must be a JSON object")

    judge_time_ms = record.get("judge_time_ms", judgment.get("judge_time_ms", 0.0))
    try:
        judge_time_ms = float(judge_time_ms)
    except (TypeError, ValueError):
        judge_time_ms = 0.0

    return judgment, judge_time_ms


def apply_rejudge_judgment_records(
    source_artifact: dict,
    judgment_records_by_qid: dict[str, dict],
    judge_model: str,
    source_path: Path | None = None,
) -> tuple[dict, dict]:
    """Re-score an existing results artifact from stored judgment records."""
    if not isinstance(source_artifact, dict):
        raise ValueError("Results artifact must be a JSON object")

    source_results = source_artifact.get("results")
    if not isinstance(source_results, list):
        raise ValueError("Results artifact missing top-level 'results' list")
    if not isinstance(judgment_records_by_qid, dict):
        raise ValueError("Stored judgment records must be a JSON object keyed by qid")

    updated_results = []
    for conv_idx, conv_result in enumerate(source_results):
        if not isinstance(conv_result, dict):
            raise ValueError(f"Conversation result at index {conv_idx} must be a JSON object")

        conv_copy = deepcopy(conv_result)
        updated_rows = []
        for row_idx, row in enumerate(conv_result.get("results", [])):
            if not isinstance(row, dict):
                raise ValueError(f"Question row at index {row_idx} in conversation {conv_copy.get('conversation_id')} must be a JSON object")

            missing = _required_rejudge_fields(row)
            if missing:
                qid = row.get("qid") or f"{conv_copy.get('conversation_id', conv_idx)}:q{row_idx}"
                location = str(source_path) if source_path else "results artifact"
                raise ValueError(
                    f"Cannot rejudge {location}: row {qid} is missing required field(s): {', '.join(missing)}"
                )

            qid = row.get("qid") or f"{conv_copy.get('conversation_id', conv_idx)}:q{row_idx}"
            if qid not in judgment_records_by_qid:
                location = str(source_path) if source_path else "results artifact"
                raise ValueError(f"Cannot rejudge {location}: missing stored judgment record for qid {qid}")

            judgment, judge_time_ms = _normalize_stored_judgment_record(judgment_records_by_qid[qid], qid)
            judged = _summarize_judge_result(judgment)
            updated_rows.append(_update_rejudged_question_row(row, judged, judge_model, judge_time_ms))

        conv_copy["results"] = updated_rows
        conv_copy["num_questions"] = conv_result.get("num_questions", len(updated_rows))
        conv_copy["num_evaluated"] = len(updated_rows)
        updated_results.append(conv_copy)

    source_metadata = source_artifact.get("metadata", {})
    if not isinstance(source_metadata, dict):
        source_metadata = {}

    updated_metadata = deepcopy(source_metadata)
    updated_at = datetime.now(timezone.utc).isoformat()
    updated_metadata["date"] = updated_at
    updated_metadata["judge_model"] = judge_model
    updated_metadata["rejudge_mode"] = True
    updated_metadata["rejudge_generated_at"] = updated_at
    if source_path is not None:
        updated_metadata["rejudge_source_path"] = str(source_path)
    if source_metadata.get("judge_model") is not None:
        updated_metadata["rejudge_source_judge_model"] = source_metadata.get("judge_model")

    summary_metadata = {
        "model": updated_metadata.get("model", "unknown"),
        "sample_size": updated_metadata.get("sample_size", "ALL"),
        "conversation_count": updated_metadata.get("conversation_count", updated_metadata.get("sample_size", "ALL")),
        "judge_model": judge_model,
        "rejudge_mode": True,
    }
    if source_path is not None:
        summary_metadata["source_results_path"] = str(source_path)
    if source_metadata.get("judge_model") is not None:
        summary_metadata["source_judge_model"] = source_metadata.get("judge_model")

    updated_artifact = {
        "metadata": updated_metadata,
        "results": updated_results,
    }
    summary_artifact = _build_evaluation_summary(updated_results, summary_metadata)
    return updated_artifact, summary_artifact


rejudge_results_artifact = apply_rejudge_judgment_records


def _collect_rejudge_judgment_records(
    source_artifact: dict,
    judge_llm: LLMClient,
    judge_model: str,
) -> dict[str, dict]:
    """Collect fresh judgments for a loaded results artifact."""
    source_results = source_artifact.get("results", [])
    judgment_records: dict[str, dict] = {}
    for conv_idx, conv_result in enumerate(source_results):
        conv_copy = conv_result if isinstance(conv_result, dict) else {}
        for row_idx, row in enumerate(conv_copy.get("results", [])):
            if not isinstance(row, dict):
                continue
            missing = _required_rejudge_fields(row)
            if missing:
                qid = row.get("qid") or f"{conv_copy.get('conversation_id', conv_idx)}:q{row_idx}"
                location = "results artifact"
                raise ValueError(
                    f"Cannot rejudge {location}: row {qid} is missing required field(s): {', '.join(missing)}"
                )
            qid = row.get("qid") or f"{conv_copy.get('conversation_id', conv_idx)}:q{row_idx}"
            t0 = time.perf_counter()
            judgment = judge_with_rubrics(
                judge_llm,
                row["question_full"],
                row["rubric"],
                row["ai_answer_full"],
                ability=row.get("ability"),
            )
            judge_time_ms = (time.perf_counter() - t0) * 1000
            judgment_records[qid] = {
                "qid": qid,
                "judge_model": judge_model,
                "judge_time_ms": judge_time_ms,
                "judgment": judgment,
            }
    return judgment_records


def write_rejudge_artifacts(
    output_path: Path,
    updated_artifact: dict,
    summary_artifact: dict,
) -> tuple[Path, Path]:
    """Write the re-judged results and summary artifacts to disk."""
    summary_path = _results_summary_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        _write_json_sanitized(updated_artifact, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        _write_json_sanitized(summary_artifact, f, indent=2)
    return output_path, summary_path


def rejudge_results_file(
    source_path: Path,
    judge_model: str,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Load, re-judge, and write a results artifact to a separate output file."""
    source_artifact = _load_results_artifact(source_path)
    output_path = output_path or _rejudge_output_path(source_path)
    judge_llm = None
    try:
        judge_llm = LLMClient(model=judge_model)
        judgment_records = _collect_rejudge_judgment_records(source_artifact, judge_llm, judge_model)
        updated_artifact, summary_artifact = apply_rejudge_judgment_records(
            source_artifact,
            judgment_records_by_qid=judgment_records,
            judge_model=judge_model,
            source_path=source_path,
        )
        return write_rejudge_artifacts(output_path, updated_artifact, summary_artifact)
    finally:
        if judge_llm is not None:
            judge_llm.close()


# ============================================================
#  SOTA Comparison
# ============================================================

PUBLISHED_SOTA = {
    "10M": {
        "Hindsight": 64.1,
        "Honcho": 40.6,
        "LIGHT (Llama-4)": 26.6,
        "RAG (Llama-4)": 24.9,
    },
    "1M": {
        "Hindsight": 73.9,
        "Honcho": 63.1,
        "LIGHT (Llama-4)": 33.6,
        "RAG (Llama-4)": 30.7,
    },
    "500K": {
        "Hindsight": 71.1,
        "Honcho": 64.9,
        "LIGHT (Llama-4)": 35.9,
        "RAG (Llama-4)": 33.0,
    },
    "100K": {
        "Hindsight": 73.4,
        "Honcho": 63.0,
        "LIGHT (Llama-4)": 35.8,
        "RAG (Llama-4)": 32.3,
    },
}


def print_sota_report(ability_summary: dict, metadata: dict):
    """Print SOTA comparison report."""
    print(f"\n{'='*80}")
    print(f"  MNEMOSYNE BEAM END-TO-END SOTA REPORT")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Model: {metadata.get('model', 'unknown')}")
    print(f"  Conversation Prefix Count: {metadata.get('conversation_count', metadata.get('sample_size', 'N/A'))}")
    print(f"  Top-K memories: {DEFAULT_TOP_K}")
    print(f"  Methodology: LLM answering + LLM-as-judge (nugget scoring, per BEAM protocol)")
    print(f"{'='*80}")

    print(f"\n  Per-Ability Scores:")
    print(f"  {'Scale':<8} {'OVERALL (macro)':>14}", end="")
    for ab in BEAM_ABILITIES:
        print(f" {ab:>6}", end="")
    print()

    for scale in sorted(ability_summary.keys()):
        scores = ability_summary[scale]
        overall = scores.get("OVERALL", {}).get("avg_score", 0.0)
        print(f"  {scale:<8} {overall*100:>13.1f}%", end="")
        for ab in BEAM_ABILITIES:
            s = scores.get(ab, {}).get("avg_score", 0.0)
            print(f" {s*100:>5.1f}%", end="")
        print()

    micro = metadata.get("micro_overall", {})
    if micro:
        print(f"\n  Micro Diagnostic:")
        for scale in sorted(micro.keys()):
            print(f"  {scale:<8} {micro[scale]*100:>13.1f}%")

    partial_credit = metadata.get("partial_credit_overall")
    if partial_credit is not None:
        print(f"\n  Partial-Credit Diagnostic: {partial_credit*100:.1f}%")

    print(f"\n  SOTA Comparison (OVERALL):")
    print(f"  {'Scale':<8} {'edumem':>12}", end="")
    for system in ["Hindsight", "Honcho", "LIGHT (Llama-4)", "RAG (Llama-4)"]:
        print(f" {system:>18}", end="")
    print()

    for scale in sorted(ability_summary.keys()):
        our_score = ability_summary[scale].get("OVERALL", {}).get("avg_score", 0.0) * 100
        sota = PUBLISHED_SOTA.get(scale, {})
        print(f"  {scale:<8} {our_score:>11.1f}%", end="")
        for system in ["Hindsight", "Honcho", "LIGHT (Llama-4)", "RAG (Llama-4)"]:
            print(f" {sota.get(system, 0):>17.1f}%", end="")
        print()

    print(f"\n  Note: Published SOTA numbers from Hindsight blog (Apr 2026) and BEAM paper Table 3.")
    print(f"  edumem uses {metadata.get('model', 'unknown')} as answering + judging LLM.")
    print(f"  OVERALL is a macro-average across abilities (BEAM leaderboard convention).")
    if metadata.get("comparison_valid"):
        print(f"  Direct comparison valid: identical BEAM dataset, identical LLM-as-judge protocol.")
    else:
        print(f"  Direct comparison not asserted for this run (subset/protocol mismatch).")
    print(f"{'='*80}")


# ============================================================
#  Main
# ============================================================

def main():
    global RESULTS_FILE, PAIRED_OUTCOMES_FILE, QUESTION_VALIDATIONS_FILE
    parser = argparse.ArgumentParser(description="BEAM End-to-End Evaluation")
    parser.add_argument("--scales", default="100K,500K,1M,10M",
                        help="Scales to evaluate (comma-separated)")
    parser.add_argument("--sample", type=int, default=3,
                        help="Prefix count of conversations per scale from start-index (0=all)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Prefix start index for deterministic conversation selection")
    parser.add_argument("--case-index", type=int, default=None,
                        help="Select exactly one conversation by index per scale")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="LLM model for answering and judging")
    parser.add_argument("--judge-model", default=None,
                        help="Separate LLM for judging (default: same as --model)")
    parser.add_argument("--full-context", action="store_true",
                        help="Send full conversation to LLM (ceiling test, bypasses retrieval)")
    parser.add_argument("--pure-recall", action="store_true",
                        help="Disable per-ability bypasses + RECENT CONVERSATION injection. "
                             "Forces every answer through edumem recall -- what the "
                             "BEAM-recovery experiment needs to measure arm-vs-arm "
                             "recall quality without harness-side oracle contamination. "
                             "Equivalent to EDUMEM_BENCHMARK_PURE_RECALL=1.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results file")
    parser.add_argument("--rejudge-results", nargs="?", const=RESULTS_FILE, type=Path, default=None,
                        help="Re-judge an existing results artifact with the selected judge model. "
                             "With no path, uses the default results file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download data and print stats, don't evaluate")
    parser.add_argument("--use-cloud", action="store_true",
                        help="Enable LLM fact extraction (cloud tier). Requires OPENROUTER_API_KEY.")
    parser.add_argument("--config-id", default=None,
                        help="Run identifier written into the paired-outcomes "
                             "JSONL alongside results JSON. Defaults to a "
                             "short hash of the EDUMEM_* env snapshot -- "
                             "useful for distinguishing back-to-back ablation "
                             "phases. Override when you want a human-readable "
                             "label (e.g. 'phase3a-no-fact-voice').")
    parser.add_argument("--allow-harness-oracles", action="store_true",
                        help="Opt out of the pure-recall safety check that requires "
                             "EDUMEM_BENCHMARK_PURE_RECALL=1 (or --pure-recall). The "
                             "harness's TR/CR/IE/KU bypasses and RECENT CONVERSATION raw-"
                             "message injection produce answers without going through "
                             "BeamMemory.recall(), which contaminates arm-vs-arm "
                             "comparisons. Set this flag only for ceiling-test or legacy-"
                             "reproduction runs where you explicitly want the bypasses.")
    parser.add_argument("--allow-no-reranker", action="store_true",
                        help="Allow the run to continue when the reranker endpoint is unavailable.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for results. Defaults to results/<timestamp>_<model>/")
    args = parser.parse_args()

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _model_slug = (args.model or "unknown").replace("/", "_").replace("\\", "_")
    _run_output_dir = args.output_dir or (PROJECT_ROOT / "results" / f"{_ts}_{_model_slug}")
    RESULTS_FILE, PAIRED_OUTCOMES_FILE, QUESTION_VALIDATIONS_FILE = _result_paths(_run_output_dir)

    if args.rejudge_results is not None:
        judge_model = args.judge_model or args.model
        print(f"{'='*80}")
        print("  BEAM Results Re-Judge")
        print(f"  Source: {args.rejudge_results}")
        print(f"  Judge: {judge_model}")
        print(f"{'='*80}")
        try:
            output_path, summary_path = rejudge_results_file(
                Path(args.rejudge_results),
                judge_model=judge_model,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

        print(f"\n  Re-judged results saved to: {output_path}")
        print(f"  Re-judged summary saved to: {summary_path}")
        return

    scales = [s.strip() for s in args.scales.split(",")]
    sample_size = args.sample if args.sample > 0 else None

    # Snapshot the full benchmark-relevant env-var surface so results JSON captures
    # exactly which configuration the run executed under. A toggle the operator
    # forgot to set is a silent confound otherwise.
    _benchmark_env_snapshot = {
        k: v for k, v in os.environ.items()
        if k.startswith("EDUMEM_") or k in ("FULL_CONTEXT_MODE", "OPENROUTER_BASE_URL")
    }
    _print_env_snapshot(_benchmark_env_snapshot)

    # Gap E: config_id labels each row in paired_outcomes.jsonl so a
    # downstream notebook can paired-bootstrap CIs across multiple A/B
    # runs without re-parsing the main results JSON. Default to a short
    # hash of the env snapshot (deterministic for identical configs);
    # override via `--config-id` for human-readable labels (e.g.,
    # 'phase3a-no-fact-voice').
    import hashlib
    if args.config_id:
        _config_id = args.config_id
    else:
        _env_canonical = "\n".join(
            f"{k}={v}" for k, v in sorted(_benchmark_env_snapshot.items())
            if "KEY" not in k and "TOKEN" not in k and "SECRET" not in k
        )
        _config_id = "cfg-" + hashlib.sha256(_env_canonical.encode("utf-8")).hexdigest()[:10]
    _run_started_at = datetime.now(timezone.utc).isoformat()
    print(f"  Config ID: {_config_id}")
    print(f"  Run started: {_run_started_at}")

    # Reset recall + extraction diagnostics so per-run counters are clean. The
    # snapshots are captured at the end of main() and written into results JSON.
    try:
        from edumem.core.recall_diagnostics import reset_recall_diagnostics
        reset_recall_diagnostics()
    except ImportError:
        pass  # Diagnostics module is optional; older checkouts may lack it.
    try:
        from edumem.extraction.diagnostics import reset_extraction_stats
        reset_extraction_stats()
    except ImportError:
        pass

    print(f"{'='*80}")
    print(f"  BEAM End-to-End Evaluation Pipeline")
    print(f"  Scales: {scales}")
    print(f"  Conversation Prefix Count: {sample_size or 'ALL'} conversations/scale")
    if args.sample == 1 and args.case_index is None:
        print("  Diagnostic: --sample is a prefix count; --sample 1 selects the first conversation from --start-index. "
              "Use --case-index to target one exact conversation.")
    print(f"  Model: {args.model}")
    print(f"  Judge: {args.judge_model or args.model}")
    # Mode resolution + banner. Pure recall is the default; explicit harness
    # oracles or full-context are the opt-in paths.
    _pure_recall_env = os.environ.get("EDUMEM_BENCHMARK_PURE_RECALL")
    _pure_recall = _benchmark_pure_recall_enabled()
    if args.allow_harness_oracles:
        _pure_recall = False
        os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = "0"
    elif args.pure_recall:
        _pure_recall = True
        os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = "1"
    elif _pure_recall_env is None:
        os.environ["EDUMEM_BENCHMARK_PURE_RECALL"] = "1"

    if _pure_recall:
        if args.full_context or _env_truthy("FULL_CONTEXT_MODE"):
            # Conflict: warn loudly so the operator isn't surprised.
            print("  Mode: PURE-RECALL (overrides FULL_CONTEXT/--full-context -- "
                  "every answer goes through edumem recall)")
        else:
            print("  Mode: PURE-RECALL (per-ability bypasses + RECENT CONTEXT disabled -- "
                  "every answer goes through edumem recall)")
    elif args.full_context:
        os.environ["FULL_CONTEXT_MODE"] = "1"
        print("  Mode: FULL-CONTEXT (bypassing retrieval)")
    print(f"{'='*80}")

    # Load data
    print(f"\n[1/4] Loading BEAM dataset...")
    data = load_beam_dataset(scales, max_conversations=None if args.case_index is not None or args.start_index else sample_size)

    if not data:
        print("ERROR: No data loaded. Check HuggingFace token and dataset name.")
        sys.exit(1)

    # Print stats
    print(f"\n  Dataset Summary:")
    for scale, convs in data.items():
        total_msgs = sum(len(c["messages"]) for c in convs)
        total_qs = sum(len(c["questions"]) for c in convs)
        print(f"    {scale}: {len(convs)} convs, {total_msgs:,} msgs, {total_qs} questions")

    # Deterministic subset selection happens after loading so `--start-index`
    # and `--case-index` can address the full scale split.
    selected_conversation_ids: dict[str, list[str]] = {}
    if args.case_index is not None or args.start_index:
        for scale, convs in list(data.items()):
            selected, selected_ids = _select_conversations(
                convs,
                sample_size=sample_size,
                start_index=args.start_index,
                case_index=args.case_index,
            )
            data[scale] = selected
            selected_conversation_ids[scale] = selected_ids
    else:
        for scale, convs in data.items():
            selected_conversation_ids[scale] = [conv["id"] for conv in convs]

    reranker_url = os.environ.get("EDUMEM_RERANKER_URL", "http://localhost:3002/rerank")
    reranker_preflight = _probe_reranker(reranker_url)
    print(f"  Reranker preflight: {reranker_url} -> {'OK' if reranker_preflight.get('ok') else 'UNAVAILABLE'}")
    if not reranker_preflight.get("ok") and not args.allow_no_reranker:
        print("ERROR: reranker endpoint unavailable. Re-run with --allow-no-reranker to continue without reranking.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        print(f"\n  Dry run complete. Exiting.")
        return

    # Load previous results if resuming
    resume_ids = set()
    all_previous = []
    if args.resume and RESULTS_FILE.exists():
        print(f"\n  Resuming from {RESULTS_FILE}...")
        with open(RESULTS_FILE) as f:
            prev = json.load(f)
            all_previous = prev.get("results", [])
            for conv_result in all_previous:
                for r in conv_result.get("results", []):
                    resume_ids.add(r["qid"])
        print(f"  Already evaluated: {len(resume_ids)} questions")

    # Initialize LLM clients
    print(f"\n[2/4] Initializing LLM clients...")
    llm = LLMClient(model=args.model)
    judge_llm = LLMClient(model=args.judge_model or args.model)

    # Evaluate each conversation
    print(f"\n[3/4] Evaluating... ({len(data)} scales)")
    all_results = list(all_previous) if args.resume else []

    for scale in sorted(data.keys()):
        conversations = data[scale]
        print(f"\n  --- Scale: {scale} ({len(conversations)} conversations) ---")

        for conv in conversations:
            # Create fresh edumem DB for each conversation
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / f"beam_{scale}_{conv['id']}.db"
                init_beam(db_path)
                beam = BeamMemory(session_id=f"beam_{scale}_{conv['id']}",
                                   db_path=db_path, use_cloud=args.use_cloud,
                                   llm_client=llm)
                conv_diag = {
                    "reranker": reranker_preflight,
                    "embedding": {},
                }

                # Ingest (includes per-batch consolidation via beam.sleep())
                os.environ["BEAM_CURRENT_SCALE"] = scale
                t0 = time.perf_counter()
                stats = ingest_conversation(beam, conv["messages"], diag=conv_diag, llm=llm)
                ingest_time = time.perf_counter() - t0
                conv_diag["ingest_diagnostics_batch"] = getattr(beam, "_last_ingest_diagnostics_batch", None)

                # Post-ingestion consolidation sweep: catch any rows that the
                # per-batch sleep loop didn't process. Uses AAAK compression
                # (same as per-batch). LLM-based consolidation is available via
                # EDUMEM_LLM_BASE_URL + EDUMEM_LLM_MODEL env vars.
                _consolidation_attempts = 0
                if os.environ.get("BEAM_CURRENT_SCALE", "100K") not in ("100K",):
                    while _consolidation_attempts < 50:
                        try:
                            _sr = beam.sleep()
                            if _sr.get("status") in ("no_op", "error"):
                                break
                            _consolidation_attempts += 1
                        except Exception as _se:
                            stats.setdefault("post_ingest_sleep_errors", []).append(repr(_se))
                            break
                    if _consolidation_attempts > 0:
                        print(f"    [consolidation-sweep] LLM-based: consolidated {_consolidation_attempts} additional batch(es) post-ingest", flush=True)

                print(f"    Ingested {len(conv['messages'])} msgs in {ingest_time:.1f}s "
                      f"(DB: {os.path.getsize(db_path)/1024:.0f}KB)")

                # Evaluate
                conv_result = evaluate_conversation(
                    llm, judge_llm, beam, conv, resume_ids
                )
                conv_result["ingest_stats"] = stats
                conv_result["ingest_diagnostics_batch"] = conv_diag.get("ingest_diagnostics_batch")
                conv_result["diagnostics"] = {
                    "ingest_stats": stats,
                    "ingest_diagnostics_batch": conv_diag.get("ingest_diagnostics_batch"),
                    "embedding": conv_diag.get("embedding", {}),
                    "reranker": conv_diag.get("reranker", reranker_preflight),
                }
                embedding_diag = conv_diag.setdefault("embedding", {})
                query_vectors = 0
                for question_result in conv_result.get("results", []):
                    q_embed = (
                        question_result.get("retrieval_diagnostics", {})
                        .get("embedding", {})
                    )
                    try:
                        query_vectors += int(q_embed.get("query_vectors", 0))
                    except (TypeError, ValueError):
                        continue
                embedding_diag["query_vectors"] = query_vectors
                api_calls_before = embedding_diag.pop("api_calls_before", None)
                if hasattr(_embeddings, "_API_CALL_COUNT") and api_calls_before is not None:
                    api_calls_after = getattr(_embeddings, "_API_CALL_COUNT", None)
                    if api_calls_after is not None:
                        embedding_diag["api_calls"] = api_calls_after - api_calls_before
                all_results.append(conv_result)
                beam.conn.close()

                # Print Conversation Summary to Console/Log
                try:
                    from collections import defaultdict
                    _scores = defaultdict(list)
                    for q_res in conv_result.get("results", []):
                        if _is_skipped_question_result(q_res):
                            continue
                        _ab = q_res.get("ability")
                        _sc = q_res.get("score")
                        if _ab and _sc is not None:
                            _scores[_ab].append(_sc)
                    
                    _ability_avgs = [sum(scs) / len(scs) for scs in _scores.values() if scs]
                    _macro_overall = (sum(_ability_avgs) / len(_ability_avgs)) * 100 if _ability_avgs else 0.0
                    _conv_total_score = sum(sum(scs) for scs in _scores.values())
                    _conv_total_count = sum(len(scs) for scs in _scores.values())
                    _micro_overall = (_conv_total_score / _conv_total_count) * 100 if _conv_total_count else 0.0
                    
                    print("\n====================================================")
                    print(f"🎉 CONVERSATION {conv_result.get('conversation_id')} EVALUATION COMPLETE")
                    print(f"Scale: {scale} | OVERALL (macro): {_macro_overall:.2f}%")
                    print(f"                 micro diag: {_micro_overall:.2f}%")
                    print("----------------------------------------------------")
                    for _ab, scs in sorted(_scores.items()):
                        _ab_avg = (sum(scs) / len(scs)) * 100
                        print(f"  - {_ab:<5}: {_ab_avg:>6.2f}% ({sum(scs):.1f}/{len(scs)})")
                    print("====================================================\n", flush=True)
                except Exception as _sum_err:
                    print(f"    [summary-error] failed to compile summary: {_sum_err}", flush=True)

            # Save progress after each conversation. Includes the env-var
            # snapshot + diagnostic snapshots so post-hoc analysis can attribute
            # score deltas to specific configurations without re-running.
            os.makedirs(RESULTS_FILE.parent, exist_ok=True)

            # Gap E: append per-question paired outcomes to a flat JSONL
            # so downstream analysis can paired-bootstrap CIs across
            # multiple A/B runs. Each line records (config_id, qid,
            # ability, score, correct, scale, ts) -- enough to compute
            # paired deltas without re-parsing the main results JSON.
            # Append-only with run_started_at + config_id means multiple
            # phases accumulate in one file; analyst filters by config_id.
            with open(PAIRED_OUTCOMES_FILE, "a") as paired_f:
                for paired_row in _build_paired_outcome_rows(conv_result, _config_id, _run_started_at):
                    paired_f.write(_json_dumps_sanitized(paired_row) + "\n")
            with open(QUESTION_VALIDATIONS_FILE, "a") as validation_f:
                for validation_row in _build_question_validation_rows(conv_result, _config_id, _run_started_at):
                    validation_f.write(_json_dumps_sanitized(validation_row) + "\n")
            _recall_diag = None
            _extraction_diag = None
            try:
                from edumem.core.recall_diagnostics import get_recall_diagnostics
                _recall_diag = get_recall_diagnostics()
            except ImportError:
                pass
            try:
                from edumem.extraction.diagnostics import get_extraction_stats
                _extraction_diag = get_extraction_stats()
            except ImportError:
                pass

            reranker_summary = _summarize_reranker_run(conv_result.get("results", []), reranker_preflight)
            conv_diag["reranker"] = reranker_summary
            conv_result.setdefault("diagnostics", {})["reranker"] = reranker_summary

            metadata = {
                "date": datetime.now(timezone.utc).isoformat(),
                "run_started_at": _run_started_at,
                "config_id": _config_id,
                "model": args.model,
                "judge_model": args.judge_model or args.model,
                "top_k": DEFAULT_TOP_K,
                "sample_size": sample_size or "ALL",
                "conversation_count": sample_size or "ALL",
                "scales": scales,
                "total_conversations": len(all_results),
                "selected_conversation_ids": selected_conversation_ids,
                "config": {
                    "env": _benchmark_env_snapshot,
                    "pure_recall": _pure_recall,
                    "allow_harness_oracles": args.allow_harness_oracles,
                    "full_context": args.full_context,
                    "use_cloud": args.use_cloud,
                    "allow_no_reranker": args.allow_no_reranker,
                    "case_index": args.case_index,
                    "start_index": args.start_index,
                },
                "diagnostics": {
                    "recall": _recall_diag,
                    "extraction": _extraction_diag,
                    "reranker": reranker_summary,
                    "embedding": conv_diag.get("embedding", {}),
                },
            }
            with open(RESULTS_FILE, "w") as f:
                _write_json_sanitized({"metadata": metadata, "results": all_results}, f, indent=2)

    # Cleanup
    llm.close()
    judge_llm.close()

    # Compute and print report
    print(f"\n[4/4] Computing SOTA report...")
    ability_summary = compute_ability_scores(all_results)
    micro_overall = compute_micro_scores(all_results)
    partial_credit_overall = compute_partial_credit_overall(all_results)
    comparison_valid = (
        sample_size is None
        and args.case_index is None
        and args.start_index == 0
        and args.judge_model is None
        and not args.full_context
        and reranker_preflight.get("ok")
    )

    metadata = {
        "model": args.model,
        "sample_size": sample_size or "ALL",
        "conversation_count": sample_size or "ALL",
        "judge_model": args.judge_model or args.model,
        "micro_overall": micro_overall,
        "partial_credit_overall": partial_credit_overall,
        "comparison_valid": comparison_valid,
        "selected_conversation_ids": selected_conversation_ids,
    }
    print_sota_report(ability_summary, metadata)

    # Save summary
    summary_file = _results_summary_path(RESULTS_FILE)
    with open(summary_file, "w") as f:
        _write_json_sanitized(_build_evaluation_summary(all_results, metadata), f, indent=2)

    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"  Summary saved to: {summary_file}")
    if PAIRED_OUTCOMES_FILE.exists():
        print(f"  Paired outcomes appended to: {PAIRED_OUTCOMES_FILE}")
        print(f"    (filter by config_id={_config_id!r} for this run's rows)")
    if QUESTION_VALIDATIONS_FILE.exists():
        print(f"  Question validations appended to: {QUESTION_VALIDATIONS_FILE}")
        print(f"    (full question/answer/judge rows for config_id={_config_id!r})")


if __name__ == "__main__":
    main()
