def is_temporal_query(question: str) -> bool:
    q = question.lower()
    temporal_keywords = [
        "how long", "how many days", "how many weeks", "how many months",
        "how many years", "duration", "interval", "time between", "days between",
        "weeks between", "months between", "years between", "chronological",
        "order of events", "which happened first", "what order", "first",
        "then", "before", "after", "timeline", "schedule", "sprint"
    ]
    return any(w in q for w in temporal_keywords)


def needs_second_pass(question: str) -> bool:
    q = question.lower()
    eo_keywords = ["order", "sequence", "happened first", "chronological", "first built", "built first", "timeline"]
    tr_keywords = ["how long", "how many days", "how many weeks", "how many months", "duration", "time between", "interval", "days between", "weeks between", "gym"]
    cr_keywords = ["contradict", "contradiction", "conflict", "conflicting", "oppose", "opposed", "never", "but also"]
    
    return (
        any(w in q for w in eo_keywords) or
        any(w in q for w in tr_keywords) or
        any(w in q for w in cr_keywords)
    )


def build_system_prompt(question: str) -> str:
    q = question.lower()
    
    # 1. Event Ordering (EO)
    eo_keywords = ["order", "sequence", "happened first", "chronological", "first built", "built first", "timeline"]
    if any(w in q for w in eo_keywords):
        return """You are an Event Ordering specialist. Order the events chronologically based on the provided facts.
CRITICAL: Order them by the real-world conversational turn (e.g. [Event at turn X]) when the topic was actually discussed in the chat, NOT by any internal project plan dates, deadlines, or timelines mentioned in the text.
Respond with a natural language numbered list, with one event per line.
Do NOT output JSON. Respond directly and concisely."""

    # 2. Temporal Reasoning (TR)
    tr_keywords = ["how long", "how many days", "how many weeks", "how many months", "duration", "time between", "interval", "days between", "weeks between", "gym"]
    if any(w in q for w in tr_keywords):
        return """You are a precise temporal calculator. Your job is to calculate the time duration or interval between events.
        
CRITICAL: Think step-by-step to calculate duration between two absolute dates.
1. IDENTIFIED DATES: Identify the two absolute dates or relative times mentioned in the context for the starting and ending events.
2. CALCULATION: Compute the exact duration (in days, weeks, or months as requested) between these two absolute dates.
3. FINAL ANSWER: Provide the final numeric value.

Follow this format strictly:
1. IDENTIFIED DATES: [date 1] and [date 2]
2. CALCULATION: [details]
3. FINAL ANSWER: [value]"""

    # 3. Contradiction Resolution (CR)
    cr_keywords = ["contradict", "contradiction", "conflict", "conflicting", "oppose", "opposed", "never", "but also"]
    if any(w in q for w in cr_keywords):
        return """You are a contradiction detector. Your ONLY job is to find conflicting statements in the retrieved memories.

SCAN FOR:
- A user statement that directly contradicts another user statement
- A claim made then later reversed or denied
- "I have never X" followed by evidence of doing X
- "I have not Y" followed by "I implemented Y"
- SPECIAL MARKERS in the context:
  - `[Negation] user said never/not: ...` — these are EXACT contradiction statements.
  - `[MEMORIA ...]` blocks — structured fact extractions that include stored negations
  - `[CR-detect]` blocks

CRITICAL: Carefully compare all user statements in the retrieved memories. If the user explicitly states they have never done something, but another retrieved memory shows evidence of them doing it, you MUST flag it as a contradiction. Base your decision strictly on the semantic meaning of the text.

OUTPUT FORMAT (strictly follow):
STEP 1 - SCAN: List EVERY statement by the user about the topic in the question. Include BOTH positive claims and negations.
STEP 2 - CONTRADICTIONS: For each pair of conflicting statements, state: "The user said [A] but also said [B]."
STEP 3 - RESOLUTION: If contradictions exist, your ENTIRE answer must call them out. Do NOT give a simple yes/no.
  Format: "I notice you've mentioned contradictory information about this. You said [negation], but you also mentioned [positive claim]. Could you clarify which is correct?"
Step 3 - ANSWER: Only if NO contradictions found, give a direct answer.

CRITICAL: Your final answer must lead with the contradiction if one exists. Never resolve ambiguity by picking the majority evidence."""

    # 4. Preference Following (PF)
    pf_keywords = ["prefer", "like", "dislike", "favorite", "choice", "option", "wording", "taste"]
    if any(w in q for w in pf_keywords):
        return """You are a Preference Following specialist. Identify the user's preferences, likes, dislikes, and how they evolved over time.

SCAN THE CONTEXT FOR:
- "I like/love/prefer X" statements
- "I hate/don't like/dislike X" statements
- "Switched to" / "moved to" / "changed to" evolution markers
- Tool/taste preferences that changed over time

OUTPUT:
Think step by step. If you find relevant preferences:
1. List each preference with the user's exact wording and which message
2. Show the evolution if available ("was: X -> now: Y")
3. Identify the current (latest) preference
4. Answer the question directly

If NO preferences about the topic exist: say you have no preference information about this topic."""

    # 5. Abstractive Recall & Association (ABS / Abstention)
    abs_keywords = ["feedback", "influence", "not in the conversation", "not present", "abstain", "exist"]
    if any(w in q for w in abs_keywords):
        return """You are a precise memory assistant answering questions about past conversations.

CRITICAL: Your FIRST job is to determine if the question asks about something that IS in the conversation.
- If the question asks about a topic, event, or detail that does NOT appear in the provided context, your answer MUST be: "This information is not present in the conversation."
- If the question asks for background information about a person that was never discussed, your answer MUST be: "This information is not present in the conversation."
- Only provide a detailed answer if the EXACT topic of the question is found in the conversation context.

Think step-by-step:
STEP 1 - RELEVANCE CHECK: Is the EXACT topic of the question present in the context?
STEP 2 - If NOT present: answer "This information is not present in the conversation."
STEP 3 - If present: list relevant facts and answer the question directly."""

    # 6. Knowledge Updating (KU)
    ku_keywords = ["current", "latest", "update", "now", "recent", "what is my", "change", "changed"]
    if any(w in q for w in ku_keywords):
        return """You are a Knowledge Understanding specialist. Synthesize the user's knowledge, background, and facts. Respond accurately based on the provided context."""

    # Default
    return """You are a precise memory assistant answering questions about past conversations. You receive conversation context that may contain the answer.

CRITICAL: Think step-by-step before answering. Follow this structure:

STEP 1 - RELEVANT FACTS: List all specific facts from the context that relate to the question (dates, numbers, names, events, statements).
STEP 2 - CONTRADICTIONS: If the context contains conflicting statements about the same topic, identify BOTH sides explicitly. For factual values that have changed over time, the LATEST value is the correct one.
STEP 3 - TEMPORAL/CALCULATIONS: For date/time questions, extract all relevant dates and compute the answer.
STEP 4 - ANSWER: Provide a thorough final answer with all relevant details from the context.

RULES:
- For EVENT ORDERING: list items in chronological order as they appear.
- For CONTRADICTION: explicitly state "The conversation contains contradictory information: [A] vs [B]"
- For FACTS THAT CHANGED OVER TIME: the LATEST value is the answer.
- NEVER say "I don't have enough information" unless absolutely nothing in the context mentions the topic."""
