"""LLM client for fact extraction via OpenRouter.

Reuses the same patterns as tools/evaluate_beam_end_to_end.py LLMClient.
100% open source (MIT).
"""

import json as _json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
def _default_extraction_model() -> str:
    """Resolve the model used for fact + conclusion extraction.

    Defaults to the canonical chat model (EDUMEM_LLM_MODEL) so extraction runs on
    the SAME provider/endpoint as the answer path -- e.g. qwen3.6 on NAN.
    Otherwise the google/gemini default silently no-ops against a NAN base_url
    (NAN does not serve Gemini), producing zero facts/conclusions. Precedence:
    EDUMEM_EXTRACTION_MODEL (explicit override) > EDUMEM_LLM_MODEL (canonical
    chat model) > google/gemini-2.5-flash (self-hosted / OpenRouter fallback).
    """
    return (
        os.environ.get("EDUMEM_EXTRACTION_MODEL")
        or os.environ.get("EDUMEM_LLM_MODEL")
        or "google/gemini-2.5-flash"
    )


DEFAULT_EXTRACTION_MODEL = _default_extraction_model()
# The LLM API key for chat-based extraction. Single env var (EDUMEM_LLM_API_KEY)
# is the canonical key used across the LLM path (answer, judge, extraction,
# consolidation). OPENROUTER_API_KEY is read only as a deprecated fallback for
# callers that still set it.
EXTRACTION_API_KEY = (
    os.environ.get("EDUMEM_LLM_API_KEY", "")
    or os.environ.get("OPENROUTER_API_KEY", "")
)
OPENROUTER_BASE_URL = (
    os.environ.get("EDUMEM_LLM_BASE_URL", "")
    or os.environ.get("OPENROUTER_BASE_URL", "")
    or "https://openrouter.ai/api/v1"
).rstrip("/")
def _reasoning_payload_extra(model: str) -> dict:
    """Per-model reasoning control so reasoning models don't spend the whole
    max_tokens budget on hidden chain-of-thought (-> content=None, minutes per
    call, empty extractions). Mirrors the answer-path LLMClient: qwen/gemma use
    chat_template_kwargs.enable_thinking=False; deepseek uses reasoning_effort=low.
    """
    ml = (model or "").lower()
    if "qwen" in ml or "gemma" in ml:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "deepseek" in ml:
        return {"reasoning_effort": "low"}
    return {}


def _fallback_models() -> list:
    """Extraction fallback models, tried in order after the primary fails.

    Default EMPTY: a hardcoded cross-provider fallback (e.g. a Gemini model) is
    dead weight on a NAN endpoint -- if the primary qwen3.6 call fails, falling
    back to a model the endpoint can't serve just fails again. Opt in via
    EDUMEM_EXTRACTION_FALLBACK_MODELS (comma-separated) for OpenRouter setups.
    """
    return [m.strip() for m in
            os.environ.get("EDUMEM_EXTRACTION_FALLBACK_MODELS", "").split(",")
            if m.strip()]


def _reasoning_payload_extra(model: str) -> dict:
    """Return extra payload fields to disable/limit reasoning for models that
    otherwise spend max_tokens on hidden chain-of-thought, producing null
    content and taking minutes per call.

    Mirrors the answer-path LLMClient in tools/evaluate_beam_end_to_end.py.

    Per NAN docs (https://nan.builders/docs/api):
      - qwen3.x / gemma4: chat_template_kwargs.enable_thinking=false
      - deepseek-v4-flash: reasoning_effort low
    """
    _ml = model.lower()
    if "qwen" in _ml or "gemma" in _ml:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "deepseek" in _ml:
        return {"reasoning_effort": "low"}
    return {}


class ExtractionClient:
    """OpenAI-compatible API client for fact extraction via OpenRouter."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = model or _default_extraction_model()
        self.api_key = api_key or EXTRACTION_API_KEY
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.call_count = 0

    def chat(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send chat completion with fallback and retry.

        Returns the response text, or empty string on total failure.

        [C13.b] Records the API-TRANSPORT outcome of the chat call.
        Does NOT record cloud-tier "extraction success" — that's only
        known after the caller parses the response as facts (see
        `extract_facts` below, which records the final outcome).

        Transport outcomes recorded:
          - record_attempt("cloud") — chat() entered
          - record_no_output("cloud") — API returned empty content
            (post-retry-and-fallback) OR all retries failed without
            a usable response. The latter also records a failure.
          - record_failure("cloud") — every model + retry combination
            raised an exception. Captures `last_exc` for the sample.

        /review caught the pre-fix behavior of recording success
        on non-empty HTTP responses — that conflated "API returned
        text" with "extraction yielded facts," producing
        success-AND-failure double-counting when `extract_facts()`
        couldn't parse the response.
        """
        from .diagnostics import get_diagnostics, _safe_for_log
        diag = get_diagnostics()
        diag.record_attempt("cloud")

        models_to_try = [self.model] + [
            m for m in _fallback_models() if m != self.model
        ]
        last_exc = None

        for model in models_to_try:
            for attempt in range(3):
                try:
                    result = self._call_api(
                        model, messages, temperature, max_tokens
                    )
                    if not result:
                        # API returned empty content. Record on the
                        # cloud-tier no_output counter; extract_facts
                        # will record the outer call outcome.
                        diag.record_no_output("cloud")
                    # Note: don't record success here. extract_facts()
                    # decides based on parseable output.
                    return result
                except Exception as e:
                    last_exc = e
                    msg = str(e)
                    if "429" in msg or "rate" in msg.lower():
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    else:
                        break  # Non-retryable, try next model
            # Brief pause between models
            time.sleep(1)

        # All models failed
        diag.record_failure(
            "cloud", exc=last_exc, reason="all_models_failed"
        )
        if last_exc is not None:
            logger.warning(
                "ExtractionClient.chat: all models failed; last error: %s",
                _safe_for_log(last_exc),
            )
        return ""

    def _call_api(
        self,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Single API call.

        Uses `requests` (not urllib) because some providers (e.g. NAN via
        Cloudflare) block the default Python-urllib User-Agent with HTTP 403
        error 1010. Mirrors the LLMClient header set in evaluate_beam_end_to_end.
        """
        import requests as _requests
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(_reasoning_payload_extra(model))
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://edumem.site",
            "X-Title": "edumem Extraction",
        }
        timeout = int(os.environ.get("EDUMEM_EXTRACTION_TIMEOUT", "60"))
        resp = _requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        self.call_count += 1
        return data["choices"][0]["message"]["content"]

    def extract_facts(self, messages: list) -> list:
        """Extract structured facts from a list of conversation messages.

        Args:
            messages: List of dicts with 'role' and 'content' keys.

        Returns:
            List of fact dicts (subject, predicate, object, etc.), or empty list on failure.
        """
        from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE

        # Build conversation text from messages
        conversation_text = ""
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                conversation_text += f"[{i}] [{role}]: {content}\n"

        if not conversation_text.strip():
            return []

        user_prompt = EXTRACTION_USER_TEMPLATE.format(
            conversation_text=conversation_text,
        )

        chat_messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # [C13.b] Outer-call accounting for the cloud entry point.
        # /review caught the pre-fix behavior: chat() recorded
        # transport success but `extract_facts` never called
        # record_call, so totals.success_rate excluded the cloud path
        # entirely. Now extract_facts owns the outer-call signal AND
        # the cloud-tier success/failure based on whether the response
        # actually parses into facts.
        from .diagnostics import get_diagnostics
        diag = get_diagnostics()

        response = self.chat(chat_messages, temperature=0.0, max_tokens=4096)

        if not response:
            # chat() already recorded transport-level failure /
            # no_output. Record the outer-call outcome at the totals
            # level for bird's-eye success-rate accounting.
            diag.record_call(succeeded=False, all_empty=True)
            return []

        # Parse JSON from response
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                facts = _json.loads(response[json_start:json_end])
                if isinstance(facts, list):
                    # Successful extraction: record cloud-tier
                    # success AND outer-call success.
                    diag.record_success("cloud", fact_count=len(facts))
                    diag.record_call(succeeded=True)
                    return facts
            # Response had brackets but contents didn't parse OR no
            # bracket at all OR parsed to non-list. Treat as parse
            # failure on the cloud tier so operators can spot the
            # model returning unusable output.
            diag.record_failure(
                "cloud", reason="no_facts_in_response"
            )
            diag.record_call(succeeded=False, all_empty=True)
        except (_json.JSONDecodeError, ValueError) as e:
            # [C13.b] Operator-visible signal: model returned text
            # but couldn't be parsed as a fact list. Distinguishes
            # "model has nothing to say" (no brackets — handled
            # above) from "model returned malformed JSON" (this
            # branch).
            diag.record_failure(
                "cloud", exc=e, reason="json_parse_failed"
            )
            diag.record_call(succeeded=False)
            logger.warning(
                "ExtractionClient.extract_facts: JSON parse failed on "
                "model response; %d chars returned",
                len(response),
            )

        return []

    def extract_conclusions(self, messages: list) -> list:
        """Extract synthesized CONCLUSIONS from a batch of messages.

        Hindsight-style synthesis for SUM/narrative recall: full self-contained
        insight sentences spanning multiple messages, NOT atomic SPO triples.
        Returns a list of dicts: {text, theme, source, confidence}. Empty list
        on failure or when the batch has nothing synthesis-worthy.
        """
        from .prompts import CONCLUSION_SYSTEM_PROMPT, CONCLUSION_USER_TEMPLATE

        conversation_text = ""
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                conversation_text += f"[{i}] [{role}]: {content}\n"

        if not conversation_text.strip():
            return []

        chat_messages = [
            {"role": "system", "content": CONCLUSION_SYSTEM_PROMPT},
            {"role": "user", "content": CONCLUSION_USER_TEMPLATE.format(
                conversation_text=conversation_text)},
        ]

        response = self.chat(chat_messages, temperature=0.0, max_tokens=4096)
        if not response:
            return []

        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                conclusions = _json.loads(response[json_start:json_end])
                if isinstance(conclusions, list):
                    return conclusions
        except (_json.JSONDecodeError, ValueError):
            logger.warning(
                "ExtractionClient.extract_conclusions: JSON parse failed; "
                "%d chars returned", len(response),
            )
        return []
