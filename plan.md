# BEAM Benchmark Recovery Plan: 43% → 90%

## Constraints

1. **No cloud LLM calls during recall.** Recall must stay under 200-300ms. Only local embedding (GTE ModernBERT) and local reranker (Ettin-17m) calls are allowed. Cloud LLM is used ONLY for the final answer generation and judging.
2. **Sleep/consolidation: skip for ≤100K, enable for ≥500K.** At 100K, all messages fit in working_memory — consolidation destroys detail the questions ask about. At 500K+, compression is needed to fit context.
3. **All retrieval improvements must be algorithmic** — regex, SQL, scoring math, reranker. No LLM-in-the-loop for retrieval or gap analysis.

## Current State (Beam 1-4 average on 100K scale)

| Ability | Best Score | Avg Score | Root Cause |
|---------|-----------|-----------|------------|
| **ABS** | 100% | 100% | **SOLVED** — abstention works perfectly |
| **IE** | 83% | 45% | Recall finds wrong messages; context→value index unreliable |
| **KU** | 50% | 37% | Fails to retrieve BOTH old + new values for updates |
| **TR** | 50% | 31% | Date math wrong; timeline extraction mismatches events |
| **SUM** | 50% | 36% | Retrieved context too fragmented for coherent summaries |
| **CR** | 25% | 12% | Negation statements missed by FTS5 (stop-word "never") |
| **EO** | 27% | 19% | MSGIDX ordering lost after consolidation; tau-b scoring harsh |
| **MR** | 12% | 3% | Multi-hop requires 2+ facts from different messages; recall finds 0-1 |
| **Overall** | 43.58% | 35.7% | Retrieval precision is the bottleneck for 6/8 abilities |

## Architecture Overview

```
BEAM Dataset → ingest_conversation() → BeamMemory (SQLite)
                                          ├── working_memory (FTS5 + sqlite-vec)
                                          ├── episodic_memory (consolidated summaries)
                                          └── memoria_* tables (facts, timelines, kg)

Question → _multi_strategy_recall() → [reranker] → memories[] → build_system_prompt() → LLM → answer
              ↑ local only (< 300ms)      ↑ local                                        ↑ cloud (OK)
              no cloud LLM calls          ettin-17m                                       answer + judge
```

Key files:
- `edumem/core/beam.py` — BeamMemory class, recall(), remember_batch(), sleep()
- `edumem/core/query_mode.py` — system prompt builder (base + EO/TR modifiers)
- `tools/evaluate_beam_end_to_end.py` — evaluation harness, answer_with_memory(), judge
- `edumem/core/polyphonic_recall.py` — multi-voice recall (4 voices: vector/graph/fact/temporal)
- `edumem/core/extraction.py` — LLM-driven structured fact extraction
- `edumem/core/embeddings.py` — embedding model selection (default: bge-small-en-v1.5 via fastembed)
- `server.py` — OpenVINO inference server (GTE ModernBERT embeddings + Ettin-17m reranker)

Embedding model: `Alibaba-NLP/gte-modernbert-base` served via OpenVINO on Intel iGPU/CPU.
Reranker: `ettin-17m-ov` (cross-encoder reranker, also OpenVINO).
Models are baked into the Docker image at `/app/models/`.

---

## Phase 0: Stop Destroying Evidence (target: +15-20pp immediately)

### 0.1 — Disable Sleep for ≤100K Scale (ALL: +15-20pp)

**Problem:** `ingest_conversation()` (line ~606) backdates all message timestamps by TTL+1 hours, then calls `beam.sleep()` up to 50 times per batch. Sleep runs AAAK compression that destroys exact dates, numbers, negation sentences, and MSGIDX ordering tags — the very details that CR/TR/EO/IE/MR questions ask about.

At 100K scale, all messages fit in `working_memory` (~500-2000 messages). There is no need to consolidate.

**File:** `tools/evaluate_beam_end_to_end.py`

**Fix:**
1. Add a `--no-sleep` flag (or gate on scale size)
2. In `ingest_conversation()`, skip the backdate+sleep block entirely when scale ≤ 100K
3. In the main eval loop (line ~2431), skip the post-ingestion `beam.sleep()` loop for ≤ 100K

**Implementation:**
```python
# In ingest_conversation(), replace the sleep block (lines ~596-645) with:
_scale = os.environ.get("BEAM_CURRENT_SCALE", "100K")
_skip_sleep = _scale in ("100K",)  # Only consolidate at 500K+

if not _skip_sleep:
    # existing backdate + sleep logic stays for 500K/1M/10M
    try:
        cursor = beam.conn.cursor()
        # ... existing backdate code ...
        while max_iters > 0:
            result = beam.sleep()
            # ...
    except Exception as e:
        stats.setdefault("sleep_errors", []).append(repr(e))
# else: messages stay in working_memory with original timestamps
```

And in the main loop (line ~2431):
```python
if scale not in ("100K",):
    _consolidation_attempts = 0
    while _consolidation_attempts < 50:
        _sr = beam.sleep()
        if _sr.get("status") in ("no_op", "error"):
            break
        _consolidation_attempts += 1
```

**Why this is P0:** This alone could explain most of the lost accuracy. The LLM never sees the actual message content because AAAK compression replaced it with lossy summaries before questions were asked.

### 0.2 — Remove Cloud LLM Calls from Recall Path

**Problem:** `answer_with_memory()` makes cloud LLM calls mid-retrieval:
- **Line 1692:** Gap analysis call (`llm.chat()`) to extract search terms between Pass 1 and Pass 2
- **Line 1787:** Second LLM answer call after gap retrieval

These add 2-5 seconds per question and violate the 200-300ms recall budget.

**File:** `tools/evaluate_beam_end_to_end.py` — the recursive retrieval loop (lines ~1544-1791).

**Fix — Replace LLM gap analysis with regex extraction:**
```python
# Replace the LLM gap analysis call (lines 1665-1696) with:
import re as _re_gap
gap_queries = []

# Extract dates from Pass 1 context
gap_queries.extend(_re_gap.findall(r'\b\d{4}-\d{2}-\d{2}\b', pass1_ctx))

# Extract month+day patterns
gap_queries.extend(_re_gap.findall(
    r'(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?',
    pass1_ctx, _re_gap.IGNORECASE
))

# Extract named entities from the question that aren't in pass1 results
q_entities = _re_gap.findall(r'\b[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*\b', question)
for ent in q_entities[:3]:
    if ent.lower() not in pass1_ctx.lower():
        gap_queries.append(ent)

# Extract key technical terms from the question
q_terms = _extract_search_terms(question)
for term in q_terms[:3]:
    if term.lower() not in pass1_ctx.lower()[:2000]:
        gap_queries.append(term)

# Deduplicate
gap_queries = list(dict.fromkeys(gap_queries))
```

This replaces the cloud LLM call with pure regex — faster, free, and deterministic. The existing regex fallback (line 1713) already does part of this but only for dates.

**Also remove the second LLM answer call (Pass 2, line 1787).** Instead:
- Do the gap retrieval (local DB + embedding + reranker — all under 300ms)
- Merge the gap results with Pass 1 results
- Send the merged context to the LLM for a SINGLE answer call

---

## Phase 1: Fix Retrieval Precision (target: 60% overall)

Retrieval is the bottleneck for 6/8 abilities. The LLM is competent when given the right context (ABS = 100% proves this). All fixes in this phase are algorithmic — no cloud LLM calls.

### 1.1 — Fix FTS5 Tokenization for Negations (CR: 12% → 60%)

**Problem:** FTS5 treats "never", "not", "haven't" as stop-words. CR questions ask "Have I worked with X?" and the rubric requires surfacing "I have never worked with X" — but FTS5 silently drops the negation token.

**File:** `edumem/core/beam.py`, around the FTS5 index creation. `tools/evaluate_beam_end_to_end.py` for ingestion tagging.

**Fix:**
1. At ingestion time, detect negation sentences and prepend a `[NEG]` tag so FTS5 can match them:
   ```python
   # In ingest_conversation(), before storing content:
   import re
   _NEG_RE = re.compile(
       r"((?:^|[.!?]\s+)[^.!?]*\b(?:never|not|haven't|didn't|wasn't|don't|can't|won't|no longer|stopped)\b[^.!?]*)",
       re.IGNORECASE | re.MULTILINE
   )
   for match in _NEG_RE.finditer(content):
       neg_sentence = match.group(1).strip()
       content += f"\n[NEG] {neg_sentence}"
   ```

2. In `_multi_strategy_recall()`, when question contains "have I", "did I", "do I", add a parallel search:
   ```python
   if any(w in question.lower() for w in ["have i", "did i", "do i", "am i"]):
       _add_unique(_recall_safe(beam, f"NEG {' '.join(terms[:3])}", top_k))
   ```

3. Move the existing negation SQL search (line ~1587, inside recursive loop) into `_multi_strategy_recall()` so ALL CR questions benefit, not just those routed to two-pass.

### 1.2 — Preserve Message Ordering Metadata (EO: 19% → 60%)

**Problem:** `[MSGIDX:N]` tags are baked into content at ingestion, but after sleep/consolidation they're lost. Even without sleep (Phase 0 fix), the recall sorts by score, not by conversation order.

**File:** `edumem/core/beam.py` — schema + recall. `tools/evaluate_beam_end_to_end.py` — context building.

**Fix:**
1. Add `message_index INTEGER` column to `working_memory` table schema in `init_beam()`
2. Populate it during `ingest_conversation()` from the message's position:
   ```python
   batch_items.append({
       "content": content,
       "source": f"beam_{msg.get('role', 'unknown')}",
       "importance": 0.3 + (0.1 * ((batch_start + i) % 5)),
       "timestamp": "2024-03-15T12:00:00",
       "message_index": batch_start + i,  # NEW
   })
   ```
3. In `recall()`, include `message_index` in the returned dict
4. In `answer_with_memory()`, when `is_ordering_query(question)`, sort retrieved memories by `message_index` ASC before building context:
   ```python
   if is_ordering_query(question):
       memories.sort(key=lambda m: m.get("message_index", float('inf')))
   ```
5. In `query_mode.py` `_ORDERING_MODIFIER`, strengthen:
   ```
   "CRITICAL: Order by FIRST MENTION in the conversation (message index), NOT by real-world dates.
    If topic A was first discussed in message 5 and topic B in message 20, A comes before B
    regardless of when A and B happened in real life."
   ```

### 1.3 — Two-Hop Retrieval Chain (MR: 3% → 50%)

**Problem:** MR questions require combining facts from 2+ different messages (e.g., "What framework does my API use?" requires finding "API" in msg 10 and "Flask" in msg 30). Single-query recall finds one hop but not both.

**File:** `tools/evaluate_beam_end_to_end.py` — `_multi_strategy_recall()`.

**Fix — Entity extraction + second recall pass (all local, no LLM):**
```python
# After the initial recall strategies in _multi_strategy_recall():
import re

# Extract entities from first-pass results for a second hop
if len(all_memories) > 0:
    entities = set()
    for mem in all_memories[:10]:
        content = mem.get("content", "")
        # Named entities (capitalized phrases)
        entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content))
        # Technical terms (camelCase, snake_case)
        entities.update(re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', content))
        entities.update(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', content))
    # Remove entities already in the question
    q_lower = question.lower()
    entities = {e for e in entities if e.lower() not in q_lower}
    # Second-hop retrieval for top entities
    for entity in list(entities)[:5]:
        _add_unique(_recall_safe(beam, entity, max(5, top_k // 3)))
```

This is similar to the existing gap analysis loop but purely local — regex entity extraction + DB recall. No LLM call.

### 1.4 — Integrate Reranker in Eval Harness (ALL: +5-10pp)

**Problem:** The Docker image ships `ettin-17m-ov` (cross-encoder reranker) via `server.py` but the BEAM eval pipeline never calls it. After `_multi_strategy_recall()` returns candidates, they're scored by the bi-encoder — the reranker would be much more precise for nuanced fact matching.

**File:** `tools/evaluate_beam_end_to_end.py` — after `_multi_strategy_recall()` returns.

**Fix — Add reranker pass (local, sub-50ms for 60 candidates):**
```python
import requests

RERANKER_URL = os.environ.get("EDUMEM_RERANKER_URL", "http://localhost:8000/rerank")

def _rerank(question: str, memories: list, top_n: int = 30) -> list:
    """Re-score candidates with cross-encoder reranker. Local call, ~20-50ms."""
    texts = [m.get("content", "")[:500] for m in memories[:top_n * 2]]
    if not texts:
        return memories
    try:
        resp = requests.post(RERANKER_URL, json={"query": question, "texts": texts}, timeout=5)
        resp.raise_for_status()
        scores = resp.json()
        for item in scores:
            idx = item["index"]
            if idx < len(memories):
                memories[idx]["rerank_score"] = item["score"]
        memories.sort(key=lambda m: m.get("rerank_score", m.get("score", 0)), reverse=True)
    except Exception:
        pass  # Best-effort; fall back to original scoring
    return memories[:top_n]

# Usage in answer_with_memory(), after multi_strategy_recall:
memories = _multi_strategy_recall(beam, question, top_k * 3)
memories = _rerank(question, memories, top_n=top_k)
```

**Latency budget:** GTE ModernBERT embed ~10ms, FTS5 ~5ms, Ettin-17m rerank ~30ms → total recall ~50-80ms, well under 300ms.

---

## Phase 2: Fix Answer Generation (target: 75% overall)

Once retrieval delivers the right context, the LLM must produce correctly-formatted answers. These are prompt-only changes — no recall impact.

### 2.1 — Knowledge Update Answer Format (KU: 37% → 70%)

**Problem:** KU questions ask "What is the current value of X?" The rubric expects the LATEST value, but the LLM often hedges or gives the old value.

**File:** `edumem/core/query_mode.py`

**Fix — Add KU modifier:**
```python
_KU_KEYWORDS = (
    "current", "latest", "updated", "changed to", "switched to",
    "now using", "most recent",
)

def is_knowledge_update_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _KU_KEYWORDS)

_KU_MODIFIER = """

KNOWLEDGE UPDATE: This question asks about the CURRENT state of something that may have changed.
If the context shows multiple values for the same thing at different times, the MOST RECENT value
is the correct answer. State the current value directly. If you can identify when it changed,
mention the change briefly (e.g., "Previously X, now Y as of [date]")."""
```

Add to `build_system_prompt()`:
```python
if is_knowledge_update_query(question):
    prompt += _KU_MODIFIER
```

### 2.2 — Summarization Context Budget (SUM: 36% → 65%)

**Problem:** SUM questions need a broad overview. The current `MAX_MEMORY_CONTEXT_CHARS = 16000` truncates too aggressively. Also, retrieved memories are score-sorted (detail-heavy), not topic-diverse.

**File:** `tools/evaluate_beam_end_to_end.py` — `answer_with_memory()`, context building section.

**Fix:**
1. For SUM-type questions, increase the context budget to 24000 chars
2. Apply MMR diversity reranking before context assembly — `edumem/core/mmr.py` already exists but isn't used in the eval harness
3. Detect SUM questions by keyword: "summarize", "summary", "overview", "main topics", "key themes"

```python
# In answer_with_memory(), after reranking:
_is_sum = any(w in question.lower() for w in ["summarize", "summary", "overview", "main topics"])
if _is_sum:
    from edumem.core.mmr import mmr_rerank
    if mmr_rerank:
        memories = mmr_rerank(memories, query_embedding, lambda_param=0.5, top_k=top_k * 2)
    _local_max_chars = 24000
else:
    _local_max_chars = MAX_MEMORY_CONTEXT_CHARS
```

### 2.3 — Temporal Reasoning with Date Anchoring (TR: 31% → 65%)

**Problem:** TR questions ask "How many days between X and Y?" The LLM gets dates in the context but miscalculates. The existing TR oracle (Python date math) is disabled (`if False:` at line 1309).

**File:** `tools/evaluate_beam_end_to_end.py` — around line 1309.

**Fix — Re-enable the Python date math oracle (zero LLM, zero latency):**
```python
# Line 1309 change:
if not _pure_recall and ability == "TR" and conversation_messages:
    timeline = _extract_timeline_from_conversation(conversation_messages)
    if timeline and len(timeline) >= 2:
        py_answer = _compute_tr_python(question, timeline)
        if py_answer and "0 days" not in py_answer.lower():
            return _ret(py_answer)  # Direct Python answer, zero LLM cost
    # If Python fails, inject timeline into context for the LLM
    if timeline and len(timeline) >= 2:
        tl_lines = ["TIMELINE (dates from conversation):"]
        for t in timeline:
            tl_lines.append(f"  {t['date_str']}: {t['event_text'][:200]}")
        memories.insert(0, {"content": "\n".join(tl_lines), "score": 1.0, "source": "tr_timeline"})
```

When Python date math succeeds → instant answer, no LLM error possible.
When Python fails → timeline is injected as structured context for the LLM's single answer call.

---

## Phase 3: System-Level Improvements (target: 85-90% overall)

### 3.1 — Use a Stronger Answering Model

**Problem:** The answering model quality directly caps accuracy. Small/cheap models make factual errors even with perfect context.

**Recommendation:** Use `gpt-4o` or `claude-sonnet-4-20250514` as the answering model.

**Implementation:** Change `--model` flag in `run_beam_official.py`.

### 3.2 — Separate Judge Model from Answer Model

**Problem:** Same model for answering and judging creates correlated errors.

**Fix:** Already supported via `--judge-model` flag:
```bash
python tools/evaluate_beam_end_to_end.py \
  --model gpt-4o \
  --judge-model claude-sonnet-4-20250514 \
  --scales 100K --sample 3
```

### 3.3 — Polyphonic Recall Tuning (ALL: +5-10pp)

**Problem:** Polyphonic recall (4-voice: vector/graph/fact/temporal) is gated by `EDUMEM_POLYPHONIC_RECALL=1` and may not be enabled during eval. All 4 voices are local — no cloud calls.

**File:** `edumem/core/polyphonic_recall.py`

**Fix:**
1. Enable during BEAM: `export EDUMEM_POLYPHONIC_RECALL=1`
2. Tune voice weights for BEAM question types:
   - IE/KU: boost fact voice weight (facts have exact values)
   - TR/EO: boost temporal voice weight (ordering matters)
   - CR: boost vector voice (semantic similarity catches paraphrased contradictions)
   - MR: equal weights (need multiple signal types)
3. Measure per-voice attribution using `recall_provenance` in results

---

## Phase 4: Per-Ability Surgical Fixes (target: 90%)

### 4.1 — IE: Information Extraction (45% → 85%)

**What's broken:** The context→value index (`beam._context_facts`) uses word overlap scoring that's too coarse. "port number" matches "port authority" because "port" overlaps.

**File:** `tools/evaluate_beam_end_to_end.py`, the `context_answer` matching block (~line 1456).

**Fix:**
1. Use bigram overlap instead of unigram
2. Require ≥3 word overlap for matches (currently ≥2)
3. Add the question's key entity as a required match

### 4.2 — CR: Contradiction Resolution (12% → 75%)

**Beyond 1.1 (negation fix):** The system prompt must explicitly instruct the LLM to surface contradictions.

**File:** `edumem/core/query_mode.py`

**Fix — Strengthen step 2 of `_BASE_PROMPT`:**
```python
"2. CONFLICTS — if the context contains statements that contradict each other about the same thing,
   you MUST surface BOTH explicitly. Start your answer with 'The conversation contains contradictory
   information:' and present both sides. Do NOT silently pick one side."
```

### 4.3 — MR: Multi-Session Reasoning (3% → 55%)

**Beyond 1.3 (two-hop retrieval):** The answering LLM needs to CHAIN facts.

**File:** `edumem/core/query_mode.py`

**Fix — Add MR modifier:**
```python
_MR_KEYWORDS = (
    "across", "combining", "together", "relationship between",
    "how does X relate to Y", "connect",
)

_MR_MODIFIER = """

MULTI-HOP REASONING: This question requires combining information from multiple parts
of the conversation. Look for connections between separate facts. If fact A says "X uses Y"
and fact B says "Y requires Z", then the answer to "what does X require?" is Z.
Chain the facts step by step."""
```

### 4.4 — EO: Event Ordering Scoring Fix (19% → 70%)

**Problem:** Kendall tau-b is harsh — one swap tanks the score. The LLM reorders by real-world chronology instead of conversation-mention order.

**Fix (prompt-side):** In `_ORDERING_MODIFIER` in `query_mode.py`:
```
"CRITICAL: Order by FIRST MENTION in the conversation (message index), NOT by real-world dates.
 If topic A was first discussed in message 5 and topic B in message 20, A comes before B
 regardless of when A and B happened in real life."
```

Combined with 1.2 (sort memories by `message_index` before sending to LLM).

---

## Implementation Priority (ordered by impact per effort)

| Priority | Task | Expected Δ | Effort | Files |
|----------|------|-----------|--------|-------|
| **P0** | 0.1 Disable sleep for ≤100K | ALL +15-20pp | 1h | evaluate_beam_e2e.py |
| **P0** | 0.2 Remove cloud LLM from recall | recall <300ms | 2h | evaluate_beam_e2e.py |
| **P0** | 3.1 Stronger answering model | ALL +10pp | 0h (config) | run_beam_official.py |
| **P1** | 1.1 Negation tagging for CR | CR +48pp | 2h | beam.py, evaluate_beam_e2e.py |
| **P1** | 1.4 Reranker integration | ALL +5-10pp | 2h | evaluate_beam_e2e.py |
| **P1** | 2.3 Re-enable TR oracle | TR +34pp | 1h | evaluate_beam_e2e.py |
| **P1** | 1.3 Two-hop retrieval for MR | MR +47pp | 3h | evaluate_beam_e2e.py |
| **P1** | 1.2 Message index for EO | EO +41pp | 3h | beam.py, evaluate_beam_e2e.py |
| **P2** | 2.1 KU modifier in prompt | KU +33pp | 1h | query_mode.py |
| **P2** | 4.2 CR prompt strengthening | CR +15pp | 1h | query_mode.py |
| **P2** | 2.2 SUM context + MMR | SUM +29pp | 2h | evaluate_beam_e2e.py |
| **P2** | 4.1 IE bigram matching | IE +20pp | 2h | evaluate_beam_e2e.py |
| **P3** | 4.3 MR prompt modifier | MR +10pp | 1h | query_mode.py |
| **P3** | 4.4 EO prompt fix | EO +10pp | 1h | query_mode.py |
| **P3** | 3.3 Polyphonic recall tuning | ALL +5pp | 2h | polyphonic_recall.py |
| **P3** | 3.2 Separate judge model | score accuracy | 0h (config) | config only |

## Expected Outcome After All Phases

| Ability | Current Best | After Phase 0 | After Phase 1 | After Phase 2-4 |
|---------|-------------|---------------|---------------|-----------------|
| ABS | 100% | 100% | 100% | 100% |
| IE | 83% | 85% | 88% | 92% |
| KU | 50% | 65% | 70% | 80% |
| TR | 50% | 60% | 70% | 80% |
| SUM | 50% | 60% | 65% | 80% |
| CR | 25% | 40% | 70% | 85% |
| EO | 27% | 50% | 65% | 75% |
| MR | 12% | 25% | 55% | 70% |
| **Overall** | **43.58%** | **60%** | **73%** | **~83%** |

## Testing Protocol

After each change:
1. Run: `python tools/evaluate_beam_end_to_end.py --scales 100K --sample 1 --pure-recall`
2. Compare per-ability scores against this baseline
3. If any ability regresses >5pp, revert that change
4. After all Phase 1 changes, run `--sample 3` for statistical stability
5. Final validation: `--sample 5 --scales 100K,500K` to confirm scaling

## Environment Variables Reference

```bash
# Core eval settings
EDUMEM_BENCHMARK_PURE_RECALL=1      # Force all answers through recall (no bypasses)
EDUMEM_BEAM_OPTIMIZATIONS=1          # Broader FTS5 OR semantics
EDUMEM_POLYPHONIC_RECALL=1           # Enable 4-voice recall engine (all local)
BEAM_CURRENT_SCALE=100K              # Used to gate sleep on/off

# Recall tuning
EDUMEM_VEC_WEIGHT=0.5                # Vector similarity weight
EDUMEM_FTS_WEIGHT=0.3                # FTS5 text weight
EDUMEM_IMPORTANCE_WEIGHT=0.2         # Importance score weight
EDUMEM_TEMPORAL_HALFLIFE_HOURS=24    # Temporal decay halflife

# Reranker (local)
EDUMEM_RERANKER_URL=http://localhost:8000/rerank

# Model selection (via run_beam_official.py --provider/--model)
# Recommended: --provider openai --model gpt-4o
```
