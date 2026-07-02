"""
Query intent adapter
====================

Adapts the shared typed question-intent artifact from ``query_mode.py`` into the
hybrid-scoring categories used by enhanced recall. This keeps query routing in
one place instead of maintaining a second regex bag here.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from edumem.core.query_mode import analyze_question_intent


@dataclass
class QueryIntent:
    """Classification result for a query."""

    category: str  # temporal, factual, entity, preference, procedural, general
    confidence: float  # 0.0 - 1.0
    signals: list = field(default_factory=list)  # which typed signals matched

    # Weight adjustments (multipliers)
    vec_bias: float = 1.0
    fts_bias: float = 1.0
    importance_bias: float = 1.0


INTENT_WEIGHTS = {
    "temporal": {"vec_bias": 0.6, "fts_bias": 1.5, "importance_bias": 0.8},
    "factual": {"vec_bias": 1.0, "fts_bias": 1.2, "importance_bias": 0.9},
    "entity": {"vec_bias": 1.1, "fts_bias": 1.0, "importance_bias": 1.3},
    "preference": {"vec_bias": 0.9, "fts_bias": 0.8, "importance_bias": 1.5},
    "procedural": {"vec_bias": 1.3, "fts_bias": 0.9, "importance_bias": 0.7},
    "general": {"vec_bias": 1.0, "fts_bias": 1.0, "importance_bias": 1.0},
}


def _intent_signals(query: str) -> tuple[str, list[str]]:
    profile = analyze_question_intent(query)
    signals: list[str] = []

    if profile.ordering:
        signals.append("ordering")
    if profile.duration or profile.temporal or profile.timeline_reference:
        signals.append("temporal")
    if profile.preference:
        signals.append("preference")
    if profile.guidance:
        signals.append("guidance")
    if profile.background:
        signals.append("background")
    if profile.knowledge_update:
        signals.append("knowledge_update")
    if profile.listing:
        signals.append("listing")
    if profile.aggregation:
        signals.append("aggregation")
    if profile.contradiction or profile.yesno_check:
        signals.append("verification")

    if profile.duration or profile.temporal or profile.timeline_reference:
        return "temporal", signals
    if profile.preference:
        return "preference", signals
    if profile.guidance:
        return "procedural", signals
    if profile.background:
        return "entity", signals
    if (
        profile.knowledge_update
        or profile.listing
        or profile.aggregation
        or profile.contradiction
        or profile.yesno_check
    ):
        return "factual", signals
    return "general", signals


def classify_intent(query: str) -> QueryIntent:
    """
    Classify the search intent of a query using the shared typed intent profile.
    """

    category, signals = _intent_signals(query)
    weights = INTENT_WEIGHTS.get(category, INTENT_WEIGHTS["general"])
    confidence = 0.0 if category == "general" else min(0.4 + 0.12 * len(signals), 0.95)

    return QueryIntent(
        category=category,
        confidence=confidence,
        signals=signals,
        vec_bias=weights["vec_bias"],
        fts_bias=weights["fts_bias"],
        importance_bias=weights["importance_bias"],
    )


def adjust_weights(
    base_vec: float = 0.5,
    base_fts: float = 0.3,
    base_importance: float = 0.2,
    intent: Optional[QueryIntent] = None,
) -> Tuple[float, float, float]:
    """
    Adjust hybrid scoring weights based on query intent.
    """

    if intent is None:
        intent = QueryIntent(category="general", confidence=0.0)

    vw = base_vec * intent.vec_bias
    fw = base_fts * intent.fts_bias
    iw = base_importance * intent.importance_bias

    total = vw + fw + iw
    if total > 0:
        vw, fw, iw = vw / total, fw / total, iw / total

    return (vw, fw, iw)
