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

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata


@dataclass(frozen=True)
class _CueGroup:
    """Reusable lexical cue family for one abstract concept."""

    prefixes: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionIntentProfile:
    """Typed intent artifact derived from question text only."""

    normalized: str
    tokens: tuple[str, ...]
    ordering: bool
    stated_duration: bool
    duration: bool
    date_interval: bool
    timeline_reference: bool
    knowledge_update: bool
    state_transition: bool
    multi_hop: bool
    summarization: bool
    aggregation: bool
    broad_aggregation: bool
    contradiction: bool
    yesno_check: bool
    how: bool
    guidance: bool
    preference: bool
    listing: bool
    background: bool
    temporal: bool
    second_pass: bool


_WORD_RE = re.compile(r"[a-z0-9]+")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_DATE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+"
    r"\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b"
)

_WH_CUES = _CueGroup(
    prefixes=(
        "what", "which", "when", "how", "cuanto", "cuanta", "cuantos", "cuantas",
        "cual", "cuales", "que", "como", "quando", "quel", "quelle", "quels",
        "quelles", "combien", "wie", "welch", "wann",
    ),
)
_ORDER_CUES = _CueGroup(
    prefixes=("order", "sequen", "chronolog", "orden", "ordem", "folge", "reihe", "paso", "etap"),
    phrases=(
        "what order", "in what order", "order in which", "walk me through",
        "happened first", "came first", "list the steps", "en que orden",
        "em que ordem", "dans quel ordre", "in welcher reihenfolge",
    ),
)
_DURATION_CUES = _CueGroup(
    prefixes=("durat", "between", "tiempo", "duracion", "dura", "zwischen", "tempo"),
    phrases=(
        "how long", "how much time", "time between", "how old",
        "cuanto tiempo", "cuantos dias", "cuantas semanas", "cuantos meses",
        "cuantos anos", "combien de temps", "wie lange",
    ),
)
_TEMPORAL_UNIT_CUES = _CueGroup(
    prefixes=(
        "day", "week", "month", "year", "dias", "dia", "seman", "mes", "ano",
        "jour", "jours", "semaine", "mois", "an", "ans", "tag", "tage",
        "woche", "monat", "jahr",
    ),
)
_EVENT_PAIR_CUES = _CueGroup(
    prefixes=("between", "from", "since", "entre", "desde", "zwischen", "depuis"),
    phrases=(
        "passed between", "elapsed between", "between when", "between the time",
        "from when", "since when", "from the time", "start and the end",
        "beginning and the end", "desde cuando",
    ),
)
_REPORT_CUES = _CueGroup(
    prefixes=(
        "say", "said", "mention", "mentioned", "dije", "dicho", "mencion",
        "mencione", "decir", "dis", "mentionn", "sagte",
    ),
    phrases=(
        "did i say", "did we say", "i said", "we said", "i mentioned",
        "we mentioned", "what duration did i", "what duration did we",
        "dije que", "mencione que",
    ),
)
_EXPECTATION_CUES = _CueGroup(
    prefixes=("expect", "suppos", "plan", "estim", "would", "take", "last", "need", "durar", "iba", "previst", "dauer"),
    phrases=(
        "expected to take", "supposed to take", "planned to take",
        "estimated to take", "would take", "would last", "would need",
        "was expected to take", "is expected to take", "was supposed to take",
        "is supposed to take", "iba a durar",
    ),
)
_CURRENT_CUES = _CueGroup(
    prefixes=("current", "latest", "recent", "actual", "ahora", "now", "ultima", "ultimo", "ultim", "recient", "newest"),
    phrases=("most recent", "now using", "a dia de hoy"),
)
_CHANGE_CUES = _CueGroup(
    prefixes=("updat", "chang", "switch", "migrat", "cambi", "actualiz", "modific", "nuevo", "nueva"),
    phrases=("changed to", "switched to", "moved to", "paso a", "cambio a"),
)
_STATE_VALUE_CUES = _CueGroup(
    prefixes=("version", "status", "deadline", "fecha", "count", "total", "included", "inclu", "uso", "using", "state"),
)
_TIME_REFERENCE_CUES = _CueGroup(
    prefixes=("date", "deadline", "timeline", "schedule", "when", "fecha", "cuando", "quand", "wann"),
)
_POSSESSION_CUES = _CueGroup(
    prefixes=("have", "has", "using", "use", "tengo", "tenemos", "uso", "usamos", "hay"),
    phrases=("do i have", "do we have", "are there", "what do i have", "what do we have"),
)
_RELATION_CUES = _CueGroup(
    prefixes=("across", "combin", "together", "relat", "connect", "link", "depend", "asoci", "relacion", "conect", "vincul"),
    phrases=("relationship between", "related to", "connected to", "como se relaciona", "en conjunto"),
)
_SUMMARY_CUES = _CueGroup(
    prefixes=("summar", "overview", "recap", "highlight", "gist", "resum", "resumen"),
    phrases=("main topics", "key themes", "vision general", "temas principales"),
)
_CONFLICT_CUES = _CueGroup(
    prefixes=("contradict", "conflict", "inconsist", "disagree", "contradic", "conflic", "desacuer"),
    phrases=("both said",),
)
_COUNT_CUES = _CueGroup(
    prefixes=("many", "much", "total", "count", "cuanto", "cuanta", "cuantos", "cuantas", "combien"),
    phrases=("how many", "how much", "in total", "altogether", "en total"),
)
_BROAD_SCOPE_CUES = _CueGroup(
    prefixes=("across", "combined", "together", "overall", "global"),
    phrases=("all the", "across sessions", "every time", "each time", "in total", "en conjunto"),
)
_VERSUS_CUES = _CueGroup(
    prefixes=("versus", "vs", "previous", "old", "new", "anterior", "nuevo"),
    phrases=("compared to", "previous versus current", "current versus previous", "from"),
)
_YESNO_START_CUES = _CueGroup(prefixes=("have", "did", "has", "was", "is", "do", "am", "he", "ha", "esta"))
_HOW_CUES = _CueGroup(
    prefixes=("how", "como", "comment", "wie", "organize", "structure", "approach", "handle", "manage"),
    phrases=("how did i", "how did we", "how was", "how were", "how have i", "how have we"),
)
_GUIDANCE_CUES = _CueGroup(
    prefixes=(
        "instruction", "format", "template", "style", "layout", "follow", "guid",
        "constraint", "rule", "must", "should", "always", "never", "help",
        "setup", "configur", "install", "build", "create", "organ", "structur",
        "workflow", "deploy", "implement", "formato", "plantill", "estilo",
        "ayud", "configur", "despleg", "paso",
    ),
    phrases=(
        "help me", "set up", "what steps", "how can i", "how should i",
        "walk me through", "que pasos", "como puedo", "como debo",
        "ayudame", "en que formato",
    ),
)
_PREFERENCE_CUES = _CueGroup(
    prefixes=(
        "prefer", "like", "dislike", "hate", "love", "enjoy", "favorit", "favourit",
        "want", "need", "avoid", "prefier", "gust", "odia", "encant", "quier",
        "gost", "bevorzug", "lieb",
    ),
    phrases=(
        "what do i like", "what do i prefer", "what do i hate", "what do i dislike",
        "what do i love", "what format do i prefer", "que formato prefiero",
        "que prefiero", "what is my preference",
    ),
)
_LIST_CUES = _CueGroup(
    prefixes=("list", "bibliotec", "dependenc", "librar", "framework", "tool", "technolog", "herramient", "bibliothe"),
    phrases=(
        "list all", "list the", "what libraries", "which libraries",
        "what dependencies", "which dependencies", "what tools", "which tools",
        "what technologies", "which technologies", "what frameworks",
        "which frameworks", "que bibliotecas", "que dependencias",
    ),
)
_BACKGROUND_CUES = _CueGroup(
    phrases=(
        "my background", "personal background", "work experience",
        "previous development", "previous projects", "prior projects",
        "experiencia laboral", "proyectos anteriores", "antecedentes personales",
    ),
)


def _normalize_question(question: str) -> str:
    folded = unicodedata.normalize("NFKD", question or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    lowered = folded.lower()
    return " ".join(lowered.split())


def _tokenize(question: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(question))


def _contains_phrase(normalized: str, phrases: tuple[str, ...]) -> bool:
    if not normalized or not phrases:
        return False
    padded = f" {normalized} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _contains_prefix(tokens: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
    for token in tokens:
        for prefix in prefixes:
            if len(prefix) < 3:
                if token == prefix:
                    return True
            elif token.startswith(prefix):
                return True
    return False


def _matches(tokens: tuple[str, ...], normalized: str, cue: _CueGroup) -> bool:
    return _contains_phrase(normalized, cue.phrases) or _contains_prefix(tokens, cue.prefixes)


def _starts_with(tokens: tuple[str, ...], cue: _CueGroup) -> bool:
    if not tokens:
        return False
    first = tokens[0]
    return any(first.startswith(prefix) for prefix in cue.prefixes)


def _has_explicit_date_text(normalized: str) -> bool:
    return bool(_ISO_DATE_RE.search(normalized) or _MONTH_DATE_RE.search(normalized))


@lru_cache(maxsize=4096)
def analyze_question_intent(question: str) -> QuestionIntentProfile:
    normalized = _normalize_question(question)
    tokens = _tokenize(normalized)

    order_signal = _matches(tokens, normalized, _ORDER_CUES)
    duration_signal = _matches(tokens, normalized, _DURATION_CUES)
    temporal_unit_signal = _matches(tokens, normalized, _TEMPORAL_UNIT_CUES)
    event_pair_phrase_signal = _contains_phrase(normalized, _EVENT_PAIR_CUES.phrases)
    event_pair_connector_signal = event_pair_phrase_signal or _contains_prefix(tokens, _EVENT_PAIR_CUES.prefixes)
    report_signal = _matches(tokens, normalized, _REPORT_CUES)
    expectation_signal = _matches(tokens, normalized, _EXPECTATION_CUES)
    current_signal = _matches(tokens, normalized, _CURRENT_CUES)
    change_signal = _matches(tokens, normalized, _CHANGE_CUES)
    state_value_signal = _matches(tokens, normalized, _STATE_VALUE_CUES)
    timeline_reference_signal = _matches(tokens, normalized, _TIME_REFERENCE_CUES)
    possession_signal = _matches(tokens, normalized, _POSSESSION_CUES)
    relation_signal = _matches(tokens, normalized, _RELATION_CUES)
    summary_signal = _matches(tokens, normalized, _SUMMARY_CUES)
    conflict_signal = _matches(tokens, normalized, _CONFLICT_CUES)
    count_signal = _matches(tokens, normalized, _COUNT_CUES)
    broad_scope_signal = _matches(tokens, normalized, _BROAD_SCOPE_CUES)
    versus_signal = _matches(tokens, normalized, _VERSUS_CUES)
    guidance_signal = _matches(tokens, normalized, _GUIDANCE_CUES)
    preference_signal = _matches(tokens, normalized, _PREFERENCE_CUES)
    list_signal = _matches(tokens, normalized, _LIST_CUES)
    background_signal = _matches(tokens, normalized, _BACKGROUND_CUES)
    how_signal = _matches(tokens, normalized, _HOW_CUES)
    wh_signal = _starts_with(tokens, _WH_CUES)

    stated_duration = report_signal and (expectation_signal or duration_signal or temporal_unit_signal)
    duration = not stated_duration and (
        duration_signal
        or (count_signal and temporal_unit_signal)
        or (event_pair_connector_signal and temporal_unit_signal)
    )
    date_interval = (
        not stated_duration
        and not order_signal
        and duration
        and (
            _has_explicit_date_text(normalized)
            or event_pair_phrase_signal
            or (event_pair_connector_signal and temporal_unit_signal)
        )
    )
    aggregation = count_signal or broad_scope_signal
    broad_aggregation = aggregation and broad_scope_signal
    state_transition = change_signal or conflict_signal or (
        versus_signal
        and (
            ("current" in tokens)
            or ("new" in tokens)
            or ("now" in tokens)
            or ("actual" in tokens)
            or ("ultimo" in tokens)
        )
    )
    knowledge_update = (
        current_signal
        or change_signal
        or (state_value_signal and (wh_signal or possession_signal or count_signal))
        or (count_signal and possession_signal)
    )
    multi_hop = relation_signal or (aggregation and broad_scope_signal)
    contradiction = conflict_signal
    yesno_check = _starts_with(tokens, _YESNO_START_CUES)
    guidance = how_signal or guidance_signal
    preference = preference_signal
    listing = list_signal
    background = background_signal
    ordering = order_signal
    summarization = summary_signal
    how = how_signal and not yesno_check
    temporal = ordering or duration
    strict_duration = duration and (
        temporal_unit_signal
        or _contains_phrase(
            normalized,
            (
                "time between",
                "duration",
                "how many days",
                "how many weeks",
                "how many months",
                "how many years",
                "cuantos dias",
                "cuantas semanas",
                "cuantos meses",
                "cuantos anos",
            ),
        )
    )
    second_pass = ordering or date_interval or strict_duration

    return QuestionIntentProfile(
        normalized=normalized,
        tokens=tokens,
        ordering=ordering,
        stated_duration=stated_duration,
        duration=duration,
        date_interval=date_interval,
        timeline_reference=timeline_reference_signal,
        knowledge_update=knowledge_update,
        state_transition=state_transition,
        multi_hop=multi_hop,
        summarization=summarization,
        aggregation=aggregation,
        broad_aggregation=broad_aggregation,
        contradiction=contradiction,
        yesno_check=yesno_check,
        how=how,
        guidance=guidance,
        preference=preference,
        listing=listing,
        background=background,
        temporal=temporal,
        second_pass=second_pass,
    )


def is_ordering_query(question: str) -> bool:
    return analyze_question_intent(question).ordering


def is_stated_duration_query(question: str) -> bool:
    return analyze_question_intent(question).stated_duration


def is_duration_query(question: str) -> bool:
    return analyze_question_intent(question).duration


def is_date_interval_query(question: str) -> bool:
    return analyze_question_intent(question).date_interval


def is_knowledge_update_query(question: str) -> bool:
    return analyze_question_intent(question).knowledge_update


def is_multi_hop_query(question: str) -> bool:
    return analyze_question_intent(question).multi_hop


def is_summarization_query(question: str) -> bool:
    return analyze_question_intent(question).summarization


def is_aggregation_query(question: str) -> bool:
    return analyze_question_intent(question).aggregation


def is_contradiction_query(question: str) -> bool:
    return analyze_question_intent(question).contradiction


def is_yesno_check_query(question: str) -> bool:
    return analyze_question_intent(question).yesno_check


def is_how_query(question: str) -> bool:
    return analyze_question_intent(question).how


def is_guidance_query(question: str) -> bool:
    return analyze_question_intent(question).guidance


def is_preference_query(question: str) -> bool:
    return analyze_question_intent(question).preference


def is_list_query(question: str) -> bool:
    """Return True when the question asks for a list of items to enumerate."""
    return analyze_question_intent(question).listing


def is_background_query(question: str) -> bool:
    """Return True for biographical or prior-project questions."""
    return analyze_question_intent(question).background


def is_temporal_query(question: str) -> bool:
    return analyze_question_intent(question).temporal


def needs_second_pass(question: str) -> bool:
    """Ordering/duration questions benefit from gap-analysis re-retrieval."""
    return analyze_question_intent(question).second_pass


def wants_instruction_preference_context(question: str) -> bool:
    """Shared gate for guidance/policy/preference retrieval helpers."""
    profile = analyze_question_intent(question)
    return profile.guidance or profile.preference


def structured_recall_intent(question: str) -> str:
    """Map question text to the shared structured-recall intent vocabulary."""
    profile = analyze_question_intent(question)

    if profile.ordering:
        return "ordered"
    if profile.duration or profile.temporal:
        return "timeline"
    if profile.contradiction or profile.yesno_check or profile.state_transition:
        return "change"
    if profile.summarization or profile.broad_aggregation or profile.multi_hop:
        return "summary"
    if profile.knowledge_update or profile.aggregation:
        return "current"
    return ""


# --- always-on base prompt (covers CR / ABS / KU / PF generically) ---------

_BASE_PROMPT = """You are a precise memory assistant. Answer the question using ONLY the provided conversation context.

Reason through these internally, then output only the final answer:
1. FACTS — gather every relevant fact from the context (dates, numbers, names, events, statements) and note when each was said.
2. CHANGE OVER TIME — if a fact, preference, or instruction was updated, the most recent value is the current answer. If the later statement is clearly an update or correction of the earlier one, treat it as a CHANGE OVER TIME, not a conflict. A `[Fact CURRENT ...] key: current (was: previous)` entry is one pre-resolved update chain: answer with `current`; `was` is history. Never treat the current and was values as contradictory. This rule does not suppress genuinely competing statements or a question that explicitly asks about a contradiction.
3. CONFLICTS — if the context contains statements that contradict each other about the SAME fact at the SAME point in time and genuinely disagree, you MUST surface BOTH explicitly. Start your answer with 'The conversation contains contradictory information:' and present both sides. Do NOT silently pick one side. However, do NOT flag simple updates or value changes as conflicts — only flag genuine disagreements where two statements cannot both be true at the same moment.
4. ABSENCE — if the specific topic of the question does not appear anywhere in the context, say clearly that the conversation does not contain that information. Never guess or use outside knowledge. Tangential or loosely-related facts do NOT make a question answerable: if the EXACT thing asked (the specific feature, value, event, or detail) is not DIRECTLY stated in the context, abstain — do NOT synthesize or infer an answer from related-but-different facts.
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
   BAD: "Tasks and planning for the week of Jan 10"
   GOOD: "Implementing the login flow and session management"
6. Do NOT include dates, date ranges, or MSGIDX numbers in the output — use them only internally for ordering.
7. One clause per line, earliest first, no preamble.
8. If the same feature appears multiple times at different stages (building it, then later optimizing, refactoring, or debugging it), keep only the FIRST occurrence where that feature was introduced. Later refinements of an already-listed feature are NOT new topics. Treat the entire lifecycle of a feature as one item in the order.

If the context includes [Fact ... MSGIDX:N] entries, use their MSGIDX values as ordering anchors. These are pre-extracted fact appearances that supplement the MSGIDX tags on full memories.

TIMELINE COVERAGE:
- Spread your items across the ENTIRE conversation timeline — from the LOWEST MSGIDX to the HIGHEST MSGIDX present in the context. Do NOT cluster all items on the earliest setup/feature-building topics.
- Each item should be a distinct PHASE or milestone, distributed from the start of the conversation to the end. Ensure later phases are represented when present — e.g., testing, deployment, optimization, security hardening, error handling, and final refinements — not only the initial setup.
- Pick the right granularity: each item is one high-level phase/aspect (how someone would summarize a milestone), NOT several sub-steps of the same early phase. Merge fine-grained early sub-steps that belong to one phase into a single item.
- When the question says "ONLY N" items, output EXACTLY N items — no more, no fewer — and make those N span the full range of MSGIDX values.
- Avoid GENERIC planning/setup labels when the context includes more CONCRETE implementation phases. Prefer milestone labels such as core functionality, testing, deployment, security, debugging, documentation, or error handling over generic labels like project scope, planning, timeline setup, or environment setup.
- If the earliest messages are mostly planning/setup fragments, merge them into one opening phase and spend the remaining slots on later concrete milestones."""

_DURATION_MODIFIER = """

DURATION: This question asks for an amount of elapsed time between two specific events.
Step 1: Identify the TWO specific events mentioned in the question.
Step 2: For each event, find its date by reading the surrounding context — match the event description to the [MSGIDX:N] entry that discusses that specific event. Do NOT pick dates from unrelated events or different phases of the same topic.
IMPORTANT: Match events by MEANING, not exact wording. The question's phrasing may differ from how events appear in context — e.g., a "launch deadline" might be stored as "go-live date" or "release target." Look for semantic matches, not literal strings.
If the question mentions a milestone, look for ANY date associated with that milestone in the context, even if the exact phrase differs.
IMPORTANT: If the conversation contains multiple different dates for the SAME event or milestone (e.g., an early plan and a later update), prefer the MOST RECENTLY STATED date for that SAME milestone. A later message supersedes an earlier planned date only when it clearly updates or restates the same event. Do NOT replace one event with a newer date from a different phase just because it is later.
Step 3: compute the difference between the two dates and state it explicitly (e.g. "2024-04-02 to 2024-05-03 = 31 days").
Compute strictly from dates present in the context; do not estimate. End with the exact value the question asks for.

If the context includes [Fact TIMELINE ...] entries with pre-computed deltas (e.g., "= 17 days"), use these pre-computed values directly. They are authoritative."""

_STATED_DURATION_MODIFIER = """

STATED DURATION: This question asks for a duration that was explicitly stated in the conversation. Answer that stated duration directly from the context. Do not calculate elapsed time from dates, and do not replace the stated amount with a computed interval. If the context contains multiple durations, choose the one that directly answers the question."""

_KU_MODIFIER = """

KNOWLEDGE UPDATE: This question asks about the CURRENT state of something that may have changed.
If the context shows multiple values for the same thing at different times, the MOST RECENT value
is the correct answer. Higher [MSGIDX:N] numbers mean the statement was made later in the conversation —
always prefer the value from the highest MSGIDX.
CRITICAL: Do NOT flag value changes as contradictions. If an earlier message says "6 items" and a later
message says "10 items", this is an UPDATE, not a contradiction. Answer with the latest value directly.
State the current value, and if helpful, mention the change briefly (e.g., "10 project cards (updated from 6)").

If the context includes [Fact CURRENT ...] entries, these are pre-resolved value changes. The value shown is the current one; the "(was: ...)" part shows the superseded value. Use the current value as your answer directly."""

_MR_MODIFIER = """

MULTI-HOP REASONING: This question requires combining information from multiple parts
of the conversation. Look for connections between separate facts. If fact A says "X uses Y"
and fact B says "Y requires Z", then the answer to "what does X require?" is Z.
Chain the facts step by step.

CRITICAL OVERRIDE — ABSENCE rule does NOT apply here. Multi-hop questions ask you to
SYNTHESIZE an answer from scattered facts. If you find the individual pieces (e.g. two
column names mentioned in separate messages), combine them into a direct answer (e.g.
"Two columns: category and notes"). Never say "the total count is not explicitly stated"
when the individual items ARE in the context — count them yourself.

When the question asks "how many" or requires counting across sessions:
1. List every distinct item you found in the context.
2. State the count.
3. Name each item.

Items introduced during design, schema definition, or initial planning count the same as
items added later. If a field, column, or component was named in a schema or plan, it is
part of what was introduced — do not exclude it because it was defined early.

Features the user describes with concrete detail (versions, implementation steps,
configuration settings) are things the user is doing. Only treat a feature as "not done"
or "not implemented" if the user explicitly said they were NOT going to do it."""

_SUM_MODIFIER = """

CRITICAL OVERRIDE: Summaries NEVER flag contradictions. If the context shows evolving practices, changing approaches, or updated values, narrate the PROGRESSION — do not start with "The conversation contains contradictory information." Summarize how things developed and changed.

SUMMARIZATION: This question asks for a summary of the conversation.
If the question mentions progression, development, or resolution "over time", structure your answer as a CHRONOLOGICAL NARRATIVE — describe what happened first, what came next, and how things evolved. Use [MSGIDX:N] tags to determine the order. Do NOT organize by topic/category — organize by TIME.
If the question asks for a general overview without temporal emphasis, cover ALL major themes and topics from the context. Structure as a comprehensive overview. Aim for completeness over brevity.

If the question asks about a SPECIFIC domain, topic, or aspect (e.g., "security challenges", "database issues", "performance problems"), constrain your summary to ONLY that domain. Do NOT write a general project overview — cover only the aspects relevant to what was asked. Structure as a chronological narrative of how that specific domain evolved."""

_CR_MODIFIER = """

CONTRADICTION RESOLUTION: This question involves potentially contradictory statements.
Present BOTH sides of the contradiction clearly with their [MSGIDX:N] references.
Then RESOLVE the contradiction: the statement with the higher [MSGIDX:N] is more recent
and should be treated as the current truth, unless the earlier statement was explicitly
confirmed or the later statement was hypothetical. Always end with a clear resolution
stating which value is current and why.

If the context includes [Fact CHANGED ...] entries, these are pre-resolved contradictions showing both the old and new values with their MSGIDX timestamps. Present BOTH values and resolve using temporal ordering (higher MSGIDX = more recent)."""

_YESNO_CHECK_MODIFIER = """

YES/NO VERIFICATION: This question asks whether something was done or is true.
Before answering, search the context for BOTH supporting AND contradicting evidence.
If you find evidence on both sides, treat it as a contradiction and present both sides.
Do NOT answer with only one side if the other side also has evidence in the context."""

_HOW_MODIFIER = """

HOW QUESTIONS: This question asks HOW something was done, organized, structured, or approached. The answer is often the sequence of WHAT was actually done — the actions taken, decisions made, and their order implicitly describe the approach. Do NOT trigger ABSENCE just because there is no explicit meta-statement about methodology or strategy. If the context contains the actual tasks, steps, or actions that were performed, describe them as the answer to "how". List the sequence of activities and decisions, which together show HOW the thing was accomplished.

PROCEDURAL CONFLICT OVERRIDE: This overrides base rule 3. Unless the question explicitly asks about a contradiction or conflict, IGNORE the CONFLICTS rule entirely and do not start with 'The conversation contains contradictory information:'. Negative or contradictory snippets about different sprints, features, or time periods are not conflicts about the procedure being asked about. Answer from the relevant actions, tasks, and decisions; ignore unrelated negative snippets."""

_LIST_MODIFIER = """

LIST COMPLETENESS: This question asks for a list of items. Be EXHAUSTIVE — include EVERY item found in the context with ALL available details (versions, configurations, purposes). Do not truncate or summarize. If versions are mentioned, always include the exact version number next to each item. For dependency/library/framework questions, OMIT any item whose version is not explicitly stated in the context rather than listing it without a version. Format as a bullet list or comma-separated list with details. Do NOT cut off your answer early — provide a complete enumeration."""

_BACKGROUND_MODIFIER = """

BACKGROUND / PRIOR-PROJECT QUESTIONS: These questions ask about the user's personal background, prior work experience, or previous development projects. Only answer them when the context directly contains that biographical or prior-project information. Details about the CURRENT project, current codebase, or current implementation work do NOT count as evidence about personal background or previous projects. If that direct evidence is missing, give an absence answer and STOP there — do NOT append tangential current-project examples after abstaining."""


def _procedural_base_prompt() -> str:
    """Replace conflict-first behavior with relevance-first behavior for HOW queries."""
    start = _BASE_PROMPT.index("3. CONFLICTS")
    end = _BASE_PROMPT.index("4. ABSENCE")
    relevance_rule = (
        "3. RELEVANCE — use facts about the specific procedure, task, or approach "
        "asked about. Ignore unrelated statements about other work.\n"
    )
    return _BASE_PROMPT[:start] + relevance_rule + _BASE_PROMPT[end:]


def build_system_prompt(question: str) -> str:
    """Base behavior always; append format guidance only when the question asks for it."""
    profile = analyze_question_intent(question)
    yesno_query = profile.yesno_check
    how_query = profile.how and not yesno_query
    contradiction_query = profile.contradiction
    prompt = _procedural_base_prompt() if how_query and not contradiction_query else _BASE_PROMPT
    if profile.ordering:
        prompt += _ORDERING_MODIFIER
    if profile.stated_duration:
        prompt += _STATED_DURATION_MODIFIER
    elif profile.duration:
        prompt += _DURATION_MODIFIER
    if contradiction_query:
        prompt += _CR_MODIFIER
    if yesno_query:
        prompt += _YESNO_CHECK_MODIFIER
    if how_query:
        prompt += _HOW_MODIFIER
    if profile.knowledge_update:
        prompt += _KU_MODIFIER
    if profile.multi_hop:
        prompt += _MR_MODIFIER
    if profile.summarization:
        prompt += _SUM_MODIFIER
    if profile.listing:
        prompt += _LIST_MODIFIER
    if profile.background:
        prompt += _BACKGROUND_MODIFIER
    return prompt
