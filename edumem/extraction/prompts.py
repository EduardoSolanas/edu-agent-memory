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

For the given messages, write up to 12 conclusions as full, self-contained
sentences. Prefer finer per-subtopic insights over broad theme summaries: one
sentence per distinct sub-effort, step, or outcome.
Capture:
- Decisions and the REASONING behind them (why, not just what)
- How something CHANGED across messages and WHY it changed
- Cause/effect and progression ("X happened, which led to Y")
- Distinct subtopics that emerge across multiple messages (e.g. password hashing,
  UNIQUE constraints, CSRF, and account lockout are separate security conclusions,
  not one merged security summary)

Return ONLY a JSON array. Each conclusion:
- text: the full synthesized sentence, self-contained (readable without source)
- theme: a short lowercase topic tag, e.g. "security", "schema", "performance",
  "timeline", "testing", "deployment"
- source: the message-index range it spans, e.g. [12, 18] (2 ints)
- confidence: 0.0-1.0

RULES:
- A conclusion MUST be self-contained — do not reference "the user" without context.
- Synthesize ACROSS messages; do not just restate a single message.
- Split broad themes into multiple conclusions when the messages cover multiple
  distinct subtopics or steps.
- Do NOT extract raw values (no "flask_version: 2.3.1" — that's a fact, not a conclusion).
- If the messages contain nothing synthesis-worthy, return [].

Format: [{"text": "...", "theme": "...", "source": [12, 18], "confidence": 0.8}]
"""

CONCLUSION_USER_TEMPLATE = """Extract conclusions (synthesized insights) from the following conversation messages. Return ONLY the JSON array, no other text.

CONVERSATION:
{conversation_text}

CONCLUSIONS:"""


# --- Card update prompt (dream/sleep worker, Phase E) ---
#
# Given a live card (or None for ADD), the agenda item that triggered the update,
# top supporting raw evidence rows, and the current session overview, the model
# emits ONE card patch following the mem0 ADD/UPDATE/DELETE/NOOP contract.
# The patch is stored as a versioned memory_card row (layer 1 representation store).

CARD_UPDATE_SYSTEM_PROMPT = """You are a memory-card synthesizer. Given the current live card (if any), an agenda item describing what changed, top supporting raw evidence rows, and the current session overview, emit ONE card patch as STRICT JSON.

Output EXACTLY this shape (no prose, no markdown, no code fences):
{"action":"ADD|UPDATE|DELETE|NOOP","card_type":"entity|topic|change|belief|session","card_key":"...","title":"...","summary":"...","state":{...},"confidence":0.0-1.0,"evidence":[{"table":"...","row_id":"...","message_idx":0,"snippet":"...","weight":1.0}]}

RULES:
- Use NOOP when the live card already fully captures the evidence. Do not add noise.
- UPDATE must keep the SAME card_key as the live card. Never change card_key on UPDATE.
- DELETE only when the card contradicts new evidence and should be retired.
- ADD when no live card exists and the evidence is worth synthesizing.
- summary must be self-contained and answer-shaped — a reader could cite it directly without the raw evidence.
- state must be a compact JSON object matching the card_type:
  entity: {"aliases":[],"attributes":{},"current_state":"..."}
  topic:  {"subtopics":[],"counts":{},"open_items":[],"key_dates":[]}
  change: {"current":"...","previous":"...","changed_at_msg_idx":0}
  belief: {"claim":"...","support_level":"...","evidence_count":0}
  session:{"major_topics":[],"current_focus":"...","unresolved":[]}
- evidence array: each entry links back to a layer-0 raw row. Include only entries actually supporting the patch.
- confidence: 0.0-1.0 float reflecting certainty of the synthesized card.
- Output JSON ONLY. No prose before or after.
"""

CARD_UPDATE_USER_TEMPLATE = """Synthesize a card patch from the inputs below. Return ONLY the JSON object, no other text.

AGENDA ITEM:
{agenda_json}

LIVE CARD (null if none):
{current_card_json}

SESSION OVERVIEW (null if none):
{session_overview_json}

SUPPORTING EVIDENCE ROWS:
{evidence_rows_json}

CARD PATCH:"""


# --- Session overview refresh prompt (Phase F) ---
#
# Given the current live topic/entity/change/belief cards, produce one
# session:overview card patch. Does NOT reread the raw conversation.

SESSION_OVERVIEW_SYSTEM_PROMPT = """You are a memory-card synthesizer. Given the current live topic, entity, change, and belief cards for a session, synthesize ONE session:overview card patch as STRICT JSON.

Output EXACTLY this shape (no prose, no markdown, no code fences):
{"action":"ADD|UPDATE|DELETE|NOOP","card_type":"session","card_key":"session:overview","title":"...","summary":"...","state":{"major_topics":[],"current_focus":"...","unresolved":[]},"confidence":0.0-1.0,"evidence":[{"table":"...","row_id":"...","message_idx":0,"snippet":"...","weight":1.0}]}

RULES:
- card_key MUST always be "session:overview".
- card_type MUST always be "session".
- Do NOT reread or reference the raw conversation — summarize only what the provided cards already capture.
- summary must be self-contained and answer-shaped.
- state.major_topics: top 3-7 topics active this session.
- state.current_focus: the most recently active subtopic or goal.
- state.unresolved: open questions or pending items visible in the cards.
- Use NOOP when the provided cards match the existing overview with no material change.
- evidence entries should link to the card_key of the source cards (use table="memory_cards").
- Output JSON ONLY. No prose before or after.
"""

SESSION_OVERVIEW_USER_TEMPLATE = """Synthesize a session:overview card patch from the live cards below. Return ONLY the JSON object, no other text.

LIVE CARDS:
{live_cards_json}

SESSION OVERVIEW PATCH:"""
