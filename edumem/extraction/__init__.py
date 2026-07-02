"""edumem Cloud — LLM-driven fact extraction engine.

100% open source (MIT). Same engine runs on Free (self-hosted) and Cloud (managed).

Self-hosted users provide their own LLM API key via the EDUMEM_LLM_API_KEY env var
(OPENROUTER_API_KEY is still read as a deprecated fallback).
Cloud users get managed extraction through the edumem Cloud service.
"""

from dataclasses import dataclass

from .client import ExtractionClient
from .diagnostics import (
    ExtractionDiagnostics,
    get_diagnostics,
    get_extraction_stats,
    reset_extraction_stats,
)
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    CARD_UPDATE_SYSTEM_PROMPT,
    CARD_UPDATE_USER_TEMPLATE,
    SESSION_OVERVIEW_SYSTEM_PROMPT,
    SESSION_OVERVIEW_USER_TEMPLATE,
)


@dataclass
class ExtractionConfig:
    """Configuration for the LLM fact extraction engine."""

    enabled: bool = False
    model: str | None = None
    batch_size: int = 20
    min_confidence: float = 0.3


__all__ = [
    "ExtractionClient",
    "ExtractionConfig",
    "ExtractionDiagnostics",
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_USER_TEMPLATE",
    "CARD_UPDATE_SYSTEM_PROMPT",
    "CARD_UPDATE_USER_TEMPLATE",
    "SESSION_OVERVIEW_SYSTEM_PROMPT",
    "SESSION_OVERVIEW_USER_TEMPLATE",
    "get_diagnostics",
    "get_extraction_stats",
    "reset_extraction_stats",
]
