"""Small, dependency-free text guards used by the inference server."""

from __future__ import annotations

import re


# Preserve ordinary spelling, numeric values, and identifiers. Extremely long
# repeated-character runs and ultra-long single tokens are pathological and
# should be collapsed/truncated before tokenization. Collapse 3+ repeats and
# truncate continuous non-space runs longer than 200 chars to a safe prefix.
_REPEATED_CHARACTER = re.compile(r"(.)\1{2,}")
_TRUNCATION_MARKER = " ... "


def _utf8_prefix(text: str, byte_limit: int) -> str:
    return text.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")


def _utf8_suffix(text: str, byte_limit: int) -> str:
    return text.encode("utf-8")[-byte_limit:].decode("utf-8", errors="ignore")


def sanitize_rerank_text(text: str, *, max_utf8_bytes: int = 352) -> str:
    """Return safe reranker input within a conservative tokenizer budget.

    The OpenVINO reranker is a 512-token MiniLM model. A UTF-8 byte is a
    conservative upper bound for a tokenizer token, so the server budgets 128
    bytes for the query and 352 for each document, leaving room for pair
    separators and special tokens. When truncation is needed, both ends are
    retained because conclusions and current values frequently occur at the
    end of a recalled memory.

    This sanitizer additionally guards against two pathological cases that
    can trigger out-of-bounds writes in compiled tokenizers:
      - extremely long single-token runs (base64, hashes) -> truncate to 40 chars
      - repeated-character grids (===== or ─────) -> collapse 3+ repeats to 2
    """
    minimum_budget = len(_TRUNCATION_MARKER.encode("utf-8")) + 2
    if max_utf8_bytes < minimum_budget:
        raise ValueError("max_utf8_bytes must be at least 7")

    cleaned = " ".join((text or "").split())

    # Truncate pathological ultra-long single-token sequences (e.g. base64,
    # long hashes, or blown-up logs with no whitespace). Keep a meaningful
    # 40-char prefix and an ellipsis so the reranker retains key tokens.
    cleaned = re.sub(r"\S{200,}", lambda m: m.group(0)[:40] + "...", cleaned)

    # Collapse consecutive identical characters (3 or more -> 2). This avoids
    # exponential subword generation inside compiled tokenizers.
    cleaned = _REPEATED_CHARACTER.sub(r"\1\1", cleaned)

    if not cleaned:
        return "empty"

    encoded = cleaned.encode("utf-8")
    if len(encoded) <= max_utf8_bytes:
        return cleaned

    marker_bytes = len(_TRUNCATION_MARKER.encode("utf-8"))
    content_budget = max_utf8_bytes - marker_bytes
    prefix_budget = content_budget // 2
    suffix_budget = content_budget - prefix_budget
    return (
        _utf8_prefix(cleaned, prefix_budget)
        + _TRUNCATION_MARKER
        + _utf8_suffix(cleaned, suffix_budget)
    )
