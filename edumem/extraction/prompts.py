"""Fact extraction prompts for edumem Cloud.

100% open source (MIT). Same prompts run on Free (self-hosted) and Cloud (managed).
"""

EXTRACTION_SYSTEM_PROMPT = """You extract structured facts from conversation messages. For each message or group of related messages, identify:

1. ENTITIES: People, projects, tools, versions, dates, numbers mentioned
2. RELATIONSHIPS: How entities relate to each other (uses, created, set, changed, prefers)
3. TEMPORAL ANCHORS: When something happened, deadlines, durations
4. CONTRADICTIONS: When a fact was later changed or updated

Return ONLY a JSON array of fact objects. Each fact must have:
- subject: the entity the fact is about (string)
- predicate: the relationship or action (string)
- object: the value or related entity (string)
- timestamp: ISO timestamp when this was stated (string, from message context)
- source: which message index this came from (integer, 0-based)
- confidence: 0.0-1.0 how certain you are (float)

RULES:
- One fact per relationship. "I use React 18.2 and Node.js 18" = 2 facts.
- Use lowercase for predicates: "uses", "set", "changed", "created", "prefers"
- Include versions and numbers as objects when available
- If a message states something changed, extract BOTH old and new facts
- If unclear, use confidence < 0.8

Format: [{"subject": "...", "predicate": "...", "object": "...", "timestamp": "...", "source": 0, "confidence": 0.95}]
"""

EXTRACTION_USER_TEMPLATE = """Extract all structured facts from the following conversation messages. Return ONLY the JSON array, no other text.

CONVERSATION:
{conversation_text}

FACTS:"""


# --- Conclusion extraction (Hindsight-style synthesis for SUM/narrative recall) ---
#
# SPO triples (above) capture atomic values ("flask_version :: 2.3.1") which serve
# exact-value recall (KU/CR/IE). They cannot serve NARRATIVE questions ("how did the
# project progress?") whose answers synthesize meaning across many messages. This
# second prompt extracts those synthesized insights as standalone natural-language
# "conclusions" — the unit Hindsight/Mem0 store instead of raw chunks.
#
# Conclusions are stored as memoria_facts with fact_type='conclusion', so they flow
# through the SAME embed -> vec_facts -> semantic-specialist path as other facts.
# No new storage/retrieval code: the 7th specialist surfaces them automatically.

CONCLUSION_SYSTEM_PROMPT = """You extract CONCLUSIONS from conversation messages.

A conclusion is a synthesized, self-contained insight that a reader would need to
answer a "how" or "why" question later. NOT raw facts (versions, dates, names) —
the MEANING behind a span of messages.

For the given messages, write 1-4 conclusions as full, self-contained sentences.
Capture:
- Decisions and the REASONING behind them (why, not just what)
- How something CHANGED across messages and WHY it changed
- Cause/effect and progression ("X happened, which led to Y")
- Themes that emerge across multiple messages (e.g. a security effort spanning
  several steps, a project phase, a debugging saga)

Return ONLY a JSON array. Each conclusion:
- text: the full synthesized sentence, self-contained (readable without source)
- theme: a short lowercase topic tag, e.g. "security", "schema", "performance",
  "timeline", "testing", "deployment"
- source: the message-index range it spans, e.g. [12, 18] (2 ints)
- confidence: 0.0-1.0

RULES:
- A conclusion MUST be self-contained — do not reference "the user" without context.
- Synthesize ACROSS messages; do not just restate a single message.
- Do NOT extract raw values (no "flask_version: 2.3.1" — that's a fact, not a conclusion).
- If the messages contain nothing synthesis-worthy, return [].

Format: [{"text": "...", "theme": "...", "source": [12, 18], "confidence": 0.8}]
"""

CONCLUSION_USER_TEMPLATE = """Extract conclusions (synthesized insights) from the following conversation messages. Return ONLY the JSON array, no other text.

CONVERSATION:
{conversation_text}

CONCLUSIONS:"""
