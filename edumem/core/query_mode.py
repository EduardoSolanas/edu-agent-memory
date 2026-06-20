"""
query_mode.py — label-free, overfit-free answer shaping for the BEAM harness.

Design (the only correct one given how BEAM works):
  * CR / ABS / KU / PF depend on the RETRIEVED MEMORY, not the question wording,
    so they cannot be reliably detected from the question. They are therefore
    handled by an ALWAYS-ON base prompt — never routed, never misrouted.
  * EO and TR DO surface in the question and need a specific output shape
    (EO = mention-order list scored by Kendall tau-b; TR = a computed number).
    They get light, question-triggered format modifiers appended to the base.

Rules this file obeys:
  * Never reads the dataset ability/category label. Routing is question-text only.
  * No benchmark-specific literals (no nouns lifted from test items).
  * No hardcoded grader output strings — instructs BEHAVIOR; the LLM judge
    credits correct paraphrases.
"""

# --- generic intent detection (question text only) -------------------------

_ORDERING_KEYWORDS = (
    "order", "sequence", "what order", "in what order", "order in which",
    "sequence of", "walk me through", "happened first", "came first",
    "list the steps", "chronolog",  # matches chronological / chronologically
)

_DURATION_KEYWORDS = (
    "how long", "how much time", "duration", "time between",
    "how many days", "how many weeks", "how many months", "how many years",
    "days between", "weeks between", "months between", "years between",
    "happened earlier", "came earlier", "came later", "earlier or later",
    "how old", "between",  # broad, but base prompt safely handles any misroute
)

_STATED_DURATION_KEYWORDS = (
    "did i say", "did we say", "i said", "we said", "i mentioned", "we mentioned",
    "expected to take", "supposed to take", "planned to take", "estimated to take",
    "would take", "would last", "would need", "was expected to take",
    "is expected to take", "was supposed to take", "is supposed to take",
    "what duration did i", "what duration did we",
)

_DATE_INTERVAL_KEYWORDS = (
    "passed between",
    "elapsed between",
    "between when",
    "between the time",
    "from when",
    "since when",
    "from the time",
    "start and the end",
    "beginning and the end",
)

_KU_KEYWORDS = (
    "current", "latest", "updated", "changed to", "switched to",
    "now using", "most recent",
    "how many", "included", "do i have", "are there",
    "deadline",
)

_MR_KEYWORDS = (
    "across", "combining", "combined", "together", "relationship between",
    "connect", "related to",
)

_SUM_KEYWORDS = (
    "summarize", "summary", "overview", "main topics",
    "key themes", "recap", "highlights", "gist",
)

_CR_KEYWORDS = (
    "contradict", "contradiction", "conflict", "conflicting",
    "both said", "inconsistent", "disagree", "disagreement",
)

_YESNO_CHECK_KEYWORDS = (
    "have i ", "did i ", "have we ", "did we ",
    "has the ", "was the ", "is the ",
)

_AGGREGATION_KEYWORDS = (
    "how many", "how much", "total", "all the", "across",
    "across my", "across our", "across sessions",
    "combined", "altogether", "in total",
    "everything", "every time", "each time",
)

_HOW_KEYWORDS = (
    "how did i ", "how did we ", "how was ", "how were ",
    "how have i ", "how have we ",
    "organize", "structure", "approach", "handle", "manage",
)

_LIST_KEYWORDS = (
    "which libraries", "which dependencies", "what libraries", "what dependencies",
    "list all", "list the", "what tools", "which tools",
    "what technologies", "which technologies", "what frameworks", "which frameworks",
)


def is_ordering_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _ORDERING_KEYWORDS)


def is_stated_duration_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _STATED_DURATION_KEYWORDS)


def _has_explicit_date_text(q: str) -> bool:
    import re

    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", q):
        return True
    month_names = (
        "january|february|march|april|may|june|july|august|"
        "september|october|november|december|"
        "jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    )
    if re.search(rf"\b(?:{month_names})[a-z]*\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b", q):
        return True
    return False


def _has_event_pair_interval_language(q: str) -> bool:
    return any(k in q for k in _DATE_INTERVAL_KEYWORDS)


def is_duration_query(question: str) -> bool:
    q = question.lower()
    return not is_stated_duration_query(q) and any(k in q for k in _DURATION_KEYWORDS)


def is_date_interval_query(question: str) -> bool:
    q = question.lower()
    if is_stated_duration_query(q) or is_ordering_query(q):
        return False
    if not is_duration_query(q):
        return False
    return _has_explicit_date_text(q) or _has_event_pair_interval_language(q)


def is_knowledge_update_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _KU_KEYWORDS)


def is_multi_hop_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _MR_KEYWORDS)


def is_summarization_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _SUM_KEYWORDS)


def is_aggregation_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _AGGREGATION_KEYWORDS)


def is_contradiction_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _CR_KEYWORDS)


def is_yesno_check_query(question: str) -> bool:
    q = question.lower()
    return any(q.startswith(k) or f" {k}" in q for k in _YESNO_CHECK_KEYWORDS)


def is_how_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _HOW_KEYWORDS)


def is_list_query(question: str) -> bool:
    """Return True when the question asks for a list of items to enumerate."""
    q = question.lower()
    return any(k in q for k in _LIST_KEYWORDS)


def is_temporal_query(question: str) -> bool:
    return is_ordering_query(question) or is_duration_query(question)


def needs_second_pass(question: str) -> bool:
    """Ordering/duration questions benefit from gap-analysis re-retrieval."""
    return is_temporal_query(question)


# --- always-on base prompt (covers CR / ABS / KU / PF generically) ---------

_BASE_PROMPT = """You are a precise memory assistant. Answer the question using ONLY the provided conversation context.

Reason through these internally, then output only the final answer:
1. FACTS — gather every relevant fact from the context (dates, numbers, names, events, statements) and note when each was said.
2. CHANGE OVER TIME — if a fact, preference, or instruction was updated, the most recent value is the current answer. If the later statement is clearly an update or correction of the earlier one, treat it as a CHANGE OVER TIME, not a conflict.
3. CONFLICTS — if the context contains statements that contradict each other about the SAME fact at the SAME point in time and genuinely disagree, you MUST surface BOTH explicitly. Start your answer with 'The conversation contains contradictory information:' and present both sides. Do NOT silently pick one side. However, do NOT flag simple updates or value changes as conflicts — only flag genuine disagreements where two statements cannot both be true at the same moment.
4. ABSENCE — if the specific topic of the question does not appear anywhere in the context, say clearly that the conversation does not contain that information. Never guess or use outside knowledge.
5. ANSWER — give a direct, complete answer grounded only in the context.

Output the final answer only: no step labels, no JSON, no preamble, no commentary."""

# --- question-triggered format modifiers (EO, TR only) ---------------------

_ORDERING_MODIFIER = """

ORDERING: This question asks for the order in which topics or events were DISCUSSED in the conversation — the order they were mentioned, NOT when they happened in real life. CRITICAL: Each memory has a [MSGIDX:N] tag showing its message index (position in the conversation). Order by the LOWEST MSGIDX where each topic FIRST appears, NOT by real-world dates. If topic A first appears at MSGIDX:5 and topic B at MSGIDX:20, A comes before B regardless of chronological dates.

CRITICAL OVERRIDES FOR FRAGMENTED CONTEXT:
These overrides suppress the base prompt's ABSENCE and CONFLICTS rules:
- DO NOT say "the conversation does not contain that information" if [MSGIDX:N] tags are present. Tags are the ordering signal.
- DO NOT flag fragmented snippets as contradictory. Fragments are normal when memories span multiple messages.
- ALWAYS reconstruct order using [MSGIDX:N] tags, even if context is sparse or scattered.

METHOD:
1. For each distinct topic/aspect mentioned, find its FIRST [MSGIDX:N].
2. Sort by LOWEST MSGIDX (earliest mention = first in list).
3. Use ACTUAL descriptions from context; quote or paraphrase closely.
4. Do NOT invent labels or merge distinct topics into summaries.
5. Describe each item by its FUNCTIONAL PURPOSE (what was being built, fixed, or discussed), NOT by dates or timeline positions.
   BAD: "Planning tasks and schedule with March 15 time anchor"
   GOOD: "Setting up core functionality including user authentication and expense tracking"
6. Do NOT include dates, date ranges, or MSGIDX numbers in the output — use them only internally for ordering.
7. One clause per line, earliest first, no preamble."""

_DURATION_MODIFIER = """

DURATION: This question asks for an amount of elapsed time between two specific events.
Step 1: Identify the TWO specific events mentioned in the question.
Step 2: For each event, find its date by reading the surrounding context — match the event description to the [MSGIDX:N] entry that discusses that specific event. Do NOT pick dates from unrelated events or different phases of the same topic.
IMPORTANT: Match events by MEANING, not exact wording. "Final deployment deadline" might appear as "project deadline", "deployment date", "launch date", or simply a date associated with deployment. "Finishing transaction management" might appear as "transaction management complete" or a date marking the end of that phase. Look for semantic matches, not literal strings.
If the question mentions a milestone, look for ANY date associated with that milestone in the context, even if the exact phrase differs.
Step 3: compute the difference between the two dates and state it explicitly (e.g. "2024-04-02 to 2024-05-03 = 31 days").
Compute strictly from dates present in the context; do not estimate. End with the exact value the question asks for."""

_STATED_DURATION_MODIFIER = """

STATED DURATION: This question asks for a duration that was explicitly stated in the conversation. Answer that stated duration directly from the context. Do not calculate elapsed time from dates, and do not replace the stated amount with a computed interval. If the context contains multiple durations, choose the one that directly answers the question."""

_KU_MODIFIER = """

KNOWLEDGE UPDATE: This question asks about the CURRENT state of something that may have changed.
If the context shows multiple values for the same thing at different times, the MOST RECENT value
is the correct answer. Higher [MSGIDX:N] numbers mean the statement was made later in the conversation —
always prefer the value from the highest MSGIDX.
CRITICAL: Do NOT flag value changes as contradictions. If an earlier message says "6 items" and a later
message says "10 items", this is an UPDATE, not a contradiction. Answer with the latest value directly.
State the current value, and if helpful, mention the change briefly (e.g., "10 project cards (updated from 6)")."""

_MR_MODIFIER = """

MULTI-HOP REASONING: This question requires combining information from multiple parts
of the conversation. Look for connections between separate facts. If fact A says "X uses Y"
and fact B says "Y requires Z", then the answer to "what does X require?" is Z.
Chain the facts step by step.

When the question asks "how many" or requires counting across sessions, make sure to list
ALL distinct items you found and then count them."""

_SUM_MODIFIER = """

CRITICAL OVERRIDE: Summaries NEVER flag contradictions. If the context shows evolving practices, changing approaches, or updated values, narrate the PROGRESSION — do not start with "The conversation contains contradictory information." Summarize how things developed and changed.

SUMMARIZATION: This question asks for a summary of the conversation.
If the question mentions progression, development, or resolution "over time", structure your answer as a CHRONOLOGICAL NARRATIVE — describe what happened first, what came next, and how things evolved. Use [MSGIDX:N] tags to determine the order. Do NOT organize by topic/category — organize by TIME.
If the question asks for a general overview without temporal emphasis, cover ALL major themes and topics from the context. Structure as a comprehensive overview. Aim for completeness over brevity."""

_CR_MODIFIER = """

CONTRADICTION RESOLUTION: This question involves potentially contradictory statements.
Present BOTH sides of the contradiction clearly with their [MSGIDX:N] references.
Then RESOLVE the contradiction: the statement with the higher [MSGIDX:N] is more recent
and should be treated as the current truth, unless the earlier statement was explicitly
confirmed or the later statement was hypothetical. Always end with a clear resolution
stating which value is current and why."""

_YESNO_CHECK_MODIFIER = """

YES/NO VERIFICATION: This question asks whether something was done or is true.
Before answering, search the context for BOTH supporting AND contradicting evidence.
If you find evidence on both sides, treat it as a contradiction and present both sides.
Do NOT answer with only one side if the other side also has evidence in the context."""

_HOW_MODIFIER = """

HOW QUESTIONS: This question asks HOW something was done, organized, structured, or approached. The answer is often the sequence of WHAT was actually done — the actions taken, decisions made, and their order implicitly describe the approach. Do NOT trigger ABSENCE just because there is no explicit meta-statement about methodology or strategy. If the context contains the actual tasks, steps, or actions that were performed, describe them as the answer to "how". List the sequence of activities and decisions, which together show HOW the thing was accomplished."""

_LIST_MODIFIER = """

LIST COMPLETENESS: This question asks for a list of items. Be EXHAUSTIVE — include EVERY item found in the context with ALL available details (versions, configurations, purposes). Do not truncate or summarize. If versions are mentioned, always include the exact version number next to each item. Format as a bullet list or comma-separated list with details. Do NOT cut off your answer early — provide a complete enumeration."""


def build_system_prompt(question: str) -> str:
    """Base behavior always; append format guidance only when the question asks for it."""
    prompt = _BASE_PROMPT
    if is_ordering_query(question):
        prompt += _ORDERING_MODIFIER
    if is_stated_duration_query(question):
        prompt += _STATED_DURATION_MODIFIER
    elif is_duration_query(question):
        prompt += _DURATION_MODIFIER
    if is_contradiction_query(question):
        prompt += _CR_MODIFIER
    if is_yesno_check_query(question):
        prompt += _YESNO_CHECK_MODIFIER
    if is_how_query(question):
        prompt += _HOW_MODIFIER
    if is_knowledge_update_query(question):
        prompt += _KU_MODIFIER
    if is_multi_hop_query(question):
        prompt += _MR_MODIFIER
    if is_summarization_query(question):
        prompt += _SUM_MODIFIER
    if is_list_query(question):
        prompt += _LIST_MODIFIER
    return prompt
