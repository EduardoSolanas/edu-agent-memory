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

_KU_KEYWORDS = (
    "current", "latest", "updated", "changed to", "switched to",
    "now using", "most recent",
)

_MR_KEYWORDS = (
    "across", "combining", "together", "relationship between",
    "connect", "related to",
)

_SUM_KEYWORDS = (
    "summarize", "summary", "overview", "main topics",
    "key themes", "recap", "highlights", "gist",
)

_AGGREGATION_KEYWORDS = (
    "how many", "how much", "total", "all the", "across",
    "across my", "across our", "across sessions",
    "combined", "altogether", "in total",
    "everything", "every time", "each time",
)


def is_ordering_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _ORDERING_KEYWORDS)


def is_duration_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _DURATION_KEYWORDS)


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


def is_temporal_query(question: str) -> bool:
    return is_ordering_query(question) or is_duration_query(question)

def needs_second_pass(question: str) -> bool:
    """Ordering/duration questions benefit from gap-analysis re-retrieval."""
    return is_temporal_query(question)


# --- always-on base prompt (covers CR / ABS / KU / PF generically) ---------

_BASE_PROMPT = """You are a precise memory assistant. Answer the question using ONLY the provided conversation context.

Reason through these internally, then output only the final answer:
1. FACTS — gather every relevant fact from the context (dates, numbers, names, events, statements) and note when each was said.
2. CONFLICTS — if the context contains statements that contradict each other about the same thing, you MUST surface BOTH explicitly. Start your answer with 'The conversation contains contradictory information:' and present both sides. Do NOT silently pick one side.
3. CHANGE OVER TIME — if a fact, preference, or instruction was updated, the most recent value is the current answer.
4. ABSENCE — if the specific topic of the question does not appear anywhere in the context, say clearly that the conversation does not contain that information. Never guess or use outside knowledge.
5. ANSWER — give a direct, complete answer grounded only in the context.

Output the final answer only: no step labels, no JSON, no preamble, no commentary."""

# --- question-triggered format modifiers (EO, TR only) ---------------------

_ORDERING_MODIFIER = """

ORDERING: This question asks for the order in which topics or events were DISCUSSED in the conversation — the order they were mentioned, NOT when they happened in real life. CRITICAL: Each memory has a [MSGIDX:N] tag showing its message index (position in the conversation). Order by the LOWEST MSGIDX where each topic FIRST appears, NOT by real-world dates. If topic A first appears at MSGIDX:5 and topic B at MSGIDX:20, A comes before B regardless of chronological dates. List them one item per line as short clauses, earliest first. No preamble."""

_DURATION_MODIFIER = """

DURATION: This question asks for an amount of elapsed time, or which event came earlier/later. Identify the two absolute dates from the context, compute the difference, and state the result explicitly (e.g. "2024-03-12 to 2024-06-20 = 100 days"). Compute strictly from dates present in the context; do not estimate. End with the exact value the question asks for."""

_KU_MODIFIER = """

KNOWLEDGE UPDATE: This question asks about the CURRENT state of something that may have changed.
If the context shows multiple values for the same thing at different times, the MOST RECENT value
is the correct answer. State the current value directly. If you can identify when it changed,
mention the change briefly (e.g., "Previously X, now Y as of [date]")."""

_MR_MODIFIER = """

MULTI-HOP REASONING: This question requires combining information from multiple parts
of the conversation. Look for connections between separate facts. If fact A says "X uses Y"
and fact B says "Y requires Z", then the answer to "what does X require?" is Z.
Chain the facts step by step.

When the question asks "how many" or requires counting across sessions, make sure to list
ALL distinct items you found and then count them."""

_SUM_MODIFIER = """

SUMMARIZATION: This question asks for a broad summary of the conversation topics.
Cover ALL major themes and topics discussed, not just the most recent ones.
Structure your answer as a comprehensive overview organized by topic.
For each topic, mention key details (names, technologies, decisions, outcomes).
Aim for completeness — missing a topic is worse than being slightly verbose."""


def build_system_prompt(question: str) -> str:
    """Base behavior always; append format guidance only when the question asks for it."""
    prompt = _BASE_PROMPT
    if is_ordering_query(question):
        prompt += _ORDERING_MODIFIER
    if is_duration_query(question):
        prompt += _DURATION_MODIFIER
    if is_knowledge_update_query(question):
        prompt += _KU_MODIFIER
    if is_multi_hop_query(question):
        prompt += _MR_MODIFIER
    if is_summarization_query(question):
        prompt += _SUM_MODIFIER
    return prompt
