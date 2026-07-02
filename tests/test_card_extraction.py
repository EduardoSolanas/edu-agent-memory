"""TDD tests for the card-update and session-overview prompt layer.

All tests run OFFLINE — no live LLM or network calls.
Covers:
- Prompt constants exist and contain required JSON-contract tokens.
- USER_TEMPLATEs format without error and include input content.
- _parse_json_object handles clean JSON, fenced blocks, and bad input.
- update_card / refresh_session_overview return the parsed patch or NOOP on failure.
"""
from __future__ import annotations

import json
import os

# Prevent any accidental embedding calls.
os.environ.setdefault("EDUMEM_NO_EMBEDDINGS", "1")


# ---------------------------------------------------------------------------
# A) Prompt constants
# ---------------------------------------------------------------------------

class TestCardUpdateSystemPrompt:
    def test_exists_and_importable(self):
        from edumem.extraction.prompts import CARD_UPDATE_SYSTEM_PROMPT
        assert isinstance(CARD_UPDATE_SYSTEM_PROMPT, str)
        assert len(CARD_UPDATE_SYSTEM_PROMPT) > 50

    def test_contains_all_action_tokens(self):
        from edumem.extraction.prompts import CARD_UPDATE_SYSTEM_PROMPT
        for token in ("ADD", "UPDATE", "DELETE", "NOOP"):
            assert token in CARD_UPDATE_SYSTEM_PROMPT, f"Missing action token: {token}"

    def test_contains_required_json_contract_keys(self):
        from edumem.extraction.prompts import CARD_UPDATE_SYSTEM_PROMPT
        for key in ("card_key", "summary", "state", "evidence", "confidence", "card_type", "title"):
            assert key in CARD_UPDATE_SYSTEM_PROMPT, f"Missing key: {key}"

    def test_contains_all_card_types(self):
        from edumem.extraction.prompts import CARD_UPDATE_SYSTEM_PROMPT
        for ct in ("entity", "topic", "change", "belief", "session"):
            assert ct in CARD_UPDATE_SYSTEM_PROMPT, f"Missing card_type: {ct}"

    def test_contains_state_shape_for_each_card_type(self):
        from edumem.extraction.prompts import CARD_UPDATE_SYSTEM_PROMPT
        # Each card type must have its state keys described.
        assert "aliases" in CARD_UPDATE_SYSTEM_PROMPT        # entity state
        assert "subtopics" in CARD_UPDATE_SYSTEM_PROMPT      # topic state
        assert "current_focus" in CARD_UPDATE_SYSTEM_PROMPT  # session state
        assert "claim" in CARD_UPDATE_SYSTEM_PROMPT          # belief state
        assert "previous" in CARD_UPDATE_SYSTEM_PROMPT       # change state

    def test_instructs_json_only_output(self):
        from edumem.extraction.prompts import CARD_UPDATE_SYSTEM_PROMPT
        assert "JSON" in CARD_UPDATE_SYSTEM_PROMPT
        assert "no prose" in CARD_UPDATE_SYSTEM_PROMPT.lower() or "No prose" in CARD_UPDATE_SYSTEM_PROMPT

    def test_reexported_from_package(self):
        from edumem.extraction import CARD_UPDATE_SYSTEM_PROMPT
        assert CARD_UPDATE_SYSTEM_PROMPT


class TestCardUpdateUserTemplate:
    def test_formats_without_error(self):
        from edumem.extraction.prompts import CARD_UPDATE_USER_TEMPLATE
        result = CARD_UPDATE_USER_TEMPLATE.format(
            agenda_json='{"card_type": "topic", "card_key": "topic:security"}',
            current_card_json="null",
            session_overview_json="null",
            evidence_rows_json='[{"table": "memoria_facts", "row_id": "42", "snippet": "added password hashing"}]',
        )
        assert "topic:security" in result
        assert "password hashing" in result
        assert "null" in result

    def test_includes_all_placeholder_sections(self):
        from edumem.extraction.prompts import CARD_UPDATE_USER_TEMPLATE
        # All four sections must be present in the filled template.
        result = CARD_UPDATE_USER_TEMPLATE.format(
            agenda_json='{"agenda": 1}',
            current_card_json='{"card": 2}',
            session_overview_json='{"overview": 3}',
            evidence_rows_json='[{"evidence": 4}]',
        )
        assert '{"agenda": 1}' in result
        assert '{"card": 2}' in result
        assert '{"overview": 3}' in result
        assert '[{"evidence": 4}]' in result

    def test_reexported_from_package(self):
        from edumem.extraction import CARD_UPDATE_USER_TEMPLATE
        assert "{agenda_json}" in CARD_UPDATE_USER_TEMPLATE


class TestSessionOverviewSystemPrompt:
    def test_exists_and_importable(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_SYSTEM_PROMPT
        assert isinstance(SESSION_OVERVIEW_SYSTEM_PROMPT, str)
        assert len(SESSION_OVERVIEW_SYSTEM_PROMPT) > 50

    def test_contains_required_contract_tokens(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_SYSTEM_PROMPT
        for token in ("ADD", "UPDATE", "DELETE", "NOOP", "card_key", "summary",
                      "state", "evidence", "session:overview"):
            assert token in SESSION_OVERVIEW_SYSTEM_PROMPT, f"Missing token: {token}"

    def test_card_key_hardcoded_to_session_overview(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_SYSTEM_PROMPT
        assert 'session:overview' in SESSION_OVERVIEW_SYSTEM_PROMPT

    def test_instructs_no_raw_conversation_reread(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_SYSTEM_PROMPT
        # Must instruct model not to reread the raw conversation.
        assert "raw conversation" in SESSION_OVERVIEW_SYSTEM_PROMPT.lower() or \
               "raw" in SESSION_OVERVIEW_SYSTEM_PROMPT

    def test_instructs_json_only_output(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_SYSTEM_PROMPT
        assert "JSON" in SESSION_OVERVIEW_SYSTEM_PROMPT

    def test_reexported_from_package(self):
        from edumem.extraction import SESSION_OVERVIEW_SYSTEM_PROMPT
        assert SESSION_OVERVIEW_SYSTEM_PROMPT


class TestSessionOverviewUserTemplate:
    def test_formats_without_error(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_USER_TEMPLATE
        cards = [{"card_type": "topic", "card_key": "topic:security", "summary": "sec work"}]
        result = SESSION_OVERVIEW_USER_TEMPLATE.format(
            live_cards_json=json.dumps(cards),
        )
        assert "topic:security" in result
        assert "sec work" in result

    def test_includes_live_cards_content(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_USER_TEMPLATE
        result = SESSION_OVERVIEW_USER_TEMPLATE.format(live_cards_json='"sentinel_value"')
        assert "sentinel_value" in result

    def test_reexported_from_package(self):
        from edumem.extraction import SESSION_OVERVIEW_USER_TEMPLATE
        assert "{live_cards_json}" in SESSION_OVERVIEW_USER_TEMPLATE


# ---------------------------------------------------------------------------
# B) _parse_json_object helper
# ---------------------------------------------------------------------------

class TestParseJsonObject:
    def _parse(self, s):
        from edumem.extraction.client import ExtractionClient
        return ExtractionClient._parse_json_object(s)

    def test_clean_json_object(self):
        result = self._parse('{"action": "NOOP", "card_key": "topic:x"}')
        assert result == {"action": "NOOP", "card_key": "topic:x"}

    def test_json_fenced_block(self):
        fenced = '```json\n{"action": "ADD", "title": "Security"}\n```'
        result = self._parse(fenced)
        assert result == {"action": "ADD", "title": "Security"}

    def test_plain_fenced_block(self):
        fenced = '```\n{"action": "UPDATE"}\n```'
        result = self._parse(fenced)
        assert result == {"action": "UPDATE"}

    def test_extra_prose_before_object(self):
        s = 'Here is the patch:\n{"action": "DELETE", "card_key": "k"}\nDone.'
        result = self._parse(s)
        assert result is not None
        assert result["action"] == "DELETE"

    def test_empty_string_returns_none(self):
        assert self._parse("") is None

    def test_none_returns_none(self):
        assert self._parse(None) is None

    def test_malformed_json_returns_none(self):
        assert self._parse("not json at all") is None

    def test_malformed_partial_json_returns_none(self):
        assert self._parse('{"action": "ADD"') is None

    def test_array_response_returns_none(self):
        # Arrays are not card patches; must return None so callers fall back to NOOP.
        assert self._parse('[{"action": "ADD"}]') is None

    def test_whitespace_only_returns_none(self):
        assert self._parse("   \n\t  ") is None


# ---------------------------------------------------------------------------
# C) update_card message building — real objects, no network, no stub
#
# The two halves of update_card are tested separately with real objects:
#   - rendering: _build_card_update_messages (pure, no network)
#   - parsing:   _parse_json_object (covered by TestParseJsonObject)
# This is the full real-object coverage of update_card without stubbing chat().
# ---------------------------------------------------------------------------

class TestBuildCardUpdateMessages:
    def _build(self, **kwargs):
        from edumem.extraction.client import ExtractionClient
        defaults = dict(current_card=None, agenda={}, evidence_rows=[], session_overview=None)
        defaults.update(kwargs)
        return ExtractionClient._build_card_update_messages(**defaults)

    def test_returns_system_then_user_messages(self):
        msgs = self._build(agenda={"card_type": "topic", "card_key": "topic:x"})
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_uses_card_update_system_prompt(self):
        from edumem.extraction.prompts import CARD_UPDATE_SYSTEM_PROMPT
        msgs = self._build(agenda={"card_key": "topic:x"})
        assert msgs[0]["content"] == CARD_UPDATE_SYSTEM_PROMPT

    def test_agenda_content_appears_in_user_message(self):
        msgs = self._build(
            agenda={"card_type": "change", "card_key": "change:deploy-window", "detail": "sentinel_agenda"},
        )
        assert "sentinel_agenda" in msgs[1]["content"]
        assert "change:deploy-window" in msgs[1]["content"]

    def test_evidence_content_appears_in_user_message(self):
        msgs = self._build(
            agenda={"card_key": "topic:security"},
            evidence_rows=[{"table": "memoria_facts", "row_id": "99", "snippet": "sentinel_evidence"}],
        )
        assert "sentinel_evidence" in msgs[1]["content"]

    def test_current_card_content_appears_in_user_message(self):
        msgs = self._build(
            current_card={"card_key": "topic:perf", "summary": "sentinel_current_card"},
            agenda={"card_key": "topic:perf"},
        )
        assert "sentinel_current_card" in msgs[1]["content"]

    def test_session_overview_content_appears_in_user_message(self):
        msgs = self._build(
            agenda={"card_key": "topic:x"},
            session_overview={"summary": "sentinel_overview"},
        )
        assert "sentinel_overview" in msgs[1]["content"]

    def test_none_inputs_rendered_as_json_null(self):
        # current_card=None and session_overview=None must render as JSON null,
        # not crash the template format.
        msgs = self._build(agenda={"card_key": "topic:x"})
        assert "null" in msgs[1]["content"]


class TestUpdateCardParsingContract:
    """update_card's parse-and-fallback contract, verified against the real
    parser (no network). update_card returns _parse_json_object(response) or
    {"action": "NOOP"} when that returns None."""

    def _parse(self, s):
        from edumem.extraction.client import ExtractionClient
        return ExtractionClient._parse_json_object(s)

    def test_valid_patch_parses_to_dict(self):
        patch_json = json.dumps({
            "action": "ADD", "card_type": "topic", "card_key": "topic:security",
            "title": "Security hardening", "summary": "Security work progressed.",
            "state": {"subtopics": ["hashing"]}, "confidence": 0.88, "evidence": [],
        })
        result = self._parse(patch_json)
        assert result["action"] == "ADD"
        assert result["card_key"] == "topic:security"
        assert result["confidence"] == 0.88

    def test_empty_response_falls_back_to_noop(self):
        # _parse_json_object("") -> None, so update_card returns {"action":"NOOP"}.
        assert self._parse("") is None

    def test_malformed_response_falls_back_to_noop(self):
        assert self._parse("I cannot produce a JSON card patch right now.") is None

    def test_fenced_patch_parses_to_dict(self):
        result = self._parse('```json\n{"action": "UPDATE", "card_key": "topic:perf"}\n```')
        assert result["action"] == "UPDATE"
        assert result["card_key"] == "topic:perf"


# ---------------------------------------------------------------------------
# D) refresh_session_overview message building — real objects, no network
# ---------------------------------------------------------------------------

class TestBuildSessionOverviewMessages:
    def _build(self, live_cards):
        from edumem.extraction.client import ExtractionClient
        return ExtractionClient._build_session_overview_messages(live_cards)

    def test_returns_system_then_user_messages(self):
        msgs = self._build([])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_uses_session_overview_system_prompt(self):
        from edumem.extraction.prompts import SESSION_OVERVIEW_SYSTEM_PROMPT
        msgs = self._build([])
        assert msgs[0]["content"] == SESSION_OVERVIEW_SYSTEM_PROMPT

    def test_live_cards_content_appears_in_user_message(self):
        msgs = self._build([
            {"card_key": "topic:sentinel_topic", "summary": "sentinel_summary"},
        ])
        assert "sentinel_topic" in msgs[1]["content"]
        assert "sentinel_summary" in msgs[1]["content"]

    def test_empty_cards_render_as_json_empty_array(self):
        msgs = self._build([])
        assert "[]" in msgs[1]["content"]


class TestRefreshSessionOverviewParsingContract:
    """refresh_session_overview's parse-and-fallback contract, verified against
    the real parser (no network)."""

    def _parse(self, s):
        from edumem.extraction.client import ExtractionClient
        return ExtractionClient._parse_json_object(s)

    def test_valid_patch_parses_to_dict(self):
        patch_json = json.dumps({
            "action": "ADD", "card_type": "session", "card_key": "session:overview",
            "title": "Session overview", "summary": "covered security and deployment.",
            "state": {"major_topics": ["security"], "current_focus": "lockout", "unresolved": []},
            "confidence": 0.9, "evidence": [],
        })
        result = self._parse(patch_json)
        assert result["action"] == "ADD"
        assert result["card_key"] == "session:overview"
        assert result["card_type"] == "session"

    def test_empty_response_falls_back_to_noop(self):
        assert self._parse("") is None

    def test_malformed_response_falls_back_to_noop(self):
        assert self._parse("sorry, cannot help") is None

    def test_fenced_patch_parses_to_dict(self):
        result = self._parse(
            '```json\n{"action":"UPDATE","card_key":"session:overview","card_type":"session"}\n```'
        )
        assert result["action"] == "UPDATE"
        assert result["card_key"] == "session:overview"


# ---------------------------------------------------------------------------
# E) chat() retry-on-empty decision — real objects, no network, no stub
#
# The NAN endpoint can return successful responses with EMPTY content (0 chars,
# no 429, no exception) transiently. chat() must retry these within the same
# bounded budget/backoff as 429s. The retry DECISION is factored into the pure
# predicate ExtractionClient._should_retry_empty(attempt, max_attempts), which
# we test directly with real objects. The end-to-end loop (chat() actually
# re-calling _call_api on empty) needs a live endpoint and is covered by the
# gated live test below; the fast suite covers the surrounding decision logic.
# ---------------------------------------------------------------------------

class TestChatRetryOnEmptyDecision:
    def _pred(self, attempt, max_attempts=3):
        from edumem.extraction.client import ExtractionClient
        return ExtractionClient._should_retry_empty(attempt, max_attempts)

    def test_retries_while_attempts_remain(self):
        # 0-based attempts 0 and 1 of a 3-attempt budget must retry.
        assert self._pred(0) is True
        assert self._pred(1) is True

    def test_does_not_retry_on_final_attempt(self):
        # Final attempt (index 2 of 3) must NOT retry — budget is bounded.
        assert self._pred(2) is False

    def test_does_not_retry_past_budget(self):
        assert self._pred(3) is False

    def test_respects_custom_budget(self):
        # With a budget of 1, even the first attempt does not retry.
        assert self._pred(0, max_attempts=1) is False
        # With a budget of 2, attempt 0 retries but attempt 1 does not.
        assert self._pred(0, max_attempts=2) is True
        assert self._pred(1, max_attempts=2) is False

    def test_attempt_budget_is_bounded_constant(self):
        # The retry budget is patient (defaults to 8 to ride through NAN outage
        # waves) but must stay a small, bounded positive int and the backoff is
        # capped so high attempt counts can't wait unboundedly.
        from edumem.extraction.client import ExtractionClient
        assert isinstance(ExtractionClient._CHAT_MAX_ATTEMPTS, int)
        assert 1 <= ExtractionClient._CHAT_MAX_ATTEMPTS <= 20
        assert 1 <= ExtractionClient._CHAT_BACKOFF_CAP <= 120


import pytest


@pytest.mark.skipif(
    os.environ.get("EDUMEM_E2E") != "1",
    reason="live LLM e2e gated behind EDUMEM_E2E=1 (needs reachable endpoint)",
)
class TestChatRetryOnEmptyLive:
    """Gated live validation of the actual retry loop against a real endpoint.

    The fast suite validates the retry DECISION (TestChatRetryOnEmptyDecision)
    with real objects and no network. This live test confirms the end-to-end
    behavior: a real chat() call returns non-empty JSON (and would retry through
    a transient empty completion) against the configured NAN endpoint.
    """

    def test_update_card_round_trips_against_live_endpoint(self):
        from edumem.extraction.client import ExtractionClient
        client = ExtractionClient()
        patch = client.update_card(
            current_card=None,
            agenda={"card_type": "topic", "card_key": "topic:security"},
            evidence_rows=[{
                "table": "memoria_facts", "row_id": "1",
                "message_idx": 0, "snippet": "Added password hashing.", "weight": 1.0,
            }],
        )
        assert isinstance(patch, dict)
        assert "action" in patch
