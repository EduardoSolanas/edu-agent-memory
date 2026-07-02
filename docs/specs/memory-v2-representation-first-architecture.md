# Memory V2 - Representation-First Architecture Spec

**Audience:** implementation agents working in this repo.

**Status:** canonical v2 architecture and rollout spec for a big-jump quality move. This file subsumes the old memory-consolidation redesign note. Phase 1 raw consolidation is already implemented, but only as partial coverage on the cloud write path, not universal coverage across every write route.

**Primary goal:** materially improve BEAM `SUM`, `MR`, `ABS`, and `PF` by changing the unit of memory we retrieve, while keeping recall simple and local.

**Hard constraints**
- Write path can spend LLM budget.
- Recall path should stay local: embed + rerank + bounded evidence hydration, target p95 under 300ms before the final answer-model call.
- The benchmark still ends with an answer LLM, but recall itself should not depend on an extra online synthesis model.

**Architecture sources to copy**
- mem0: semantic write-time `ADD` / `UPDATE` / `DELETE` / `NOOP`
- Hindsight: separated memory types, especially synthesized summaries and beliefs
- Honcho: ingest extraction plus background dream/sleep consolidation

---

## 0. Measurement loop

Use the judge-free recall gate and budget sweep to validate write-path changes.

```bash
export EDUMEM_RETRIEVAL_E2E=1
export EDUMEM_EMBEDDING_API_URL="http://127.0.0.1:3002"
export EDUMEM_RERANKER_URL="http://127.0.0.1:3002/rerank"

python -m pytest tests/test_beam_retrieval_recall.py -q -s
python tools/retrieval_budget_sweep.py
```

- Rebuild `tests/.beam_recall_cache/beam_100K_x3.db*` after any write/extraction/consolidation change.
- The recall metric is literal nugget-token presence in assembled context. It under-credits IF formatting and computed MR counts; confirm those on the full judged runner.
- Baseline on the 3-conv cache is still about `0.456` overall, with the lowest abilities concentrated in `SUM`, `MR`, `ABS`, and `PF`.

---

## 1. Current reality

Read this section before implementing anything new.

### Already landed
- The raw consolidation gate exists in `_store_memoria_fact()`, but it only applies when `use_cloud=True`, after exact live/source dedup, and it skips `date` facts. That is useful, but it is not universal coverage.
- The current gate already covers cloud conclusions and parsed LLM fact candidates, but direct regex and derived writers such as some metric, date, and duration inserts still bypass it through `_insert_fact()`.
- Conclusions are already uniquely keyed in code. The live key uses normalized theme plus source span plus a content hash, so same-theme conclusions can coexist.
- Every themed conclusion batch already refreshes a live aggregate count fact with `fact_type='aggregate'`.
- Duration derivation already keys on the ISO date pair, not the shared topic words.
- All reranker and embedding URLs should default to `127.0.0.1`, not `localhost`.

### Existing storage
These are the layer-0 raw evidence tables:
- `working_memory`
- `episodic_memory`
- `memoria_facts`
- `memoria_timelines`
- `memoria_kg`
- `memoria_summaries`
- `facts`

The `facts` table matters because it is the raw SPO evidence store. It should be treated as layer-0 evidence, and it should be available for card evidence backfill where appropriate.

### Existing retrieval shape
- The current read path still uses specialist fusion as the default fallback path.
- `query_mode.py` already classifies intent from question text.
- EO already has message-index-based ordering behavior in the current code path and should stay anchored to `MSGIDX` rather than rewritten into a topical summary.
- `_assemble_memory_context()` lives in `edumem/core/context_assembly.py`. Any card-context formatting or final memory-context shape work should extend that module instead of re-inlining formatting logic back into `beam.py`.

---

## 2. Confirmed defects and why v2 exists

These were the redesign note's root-cause findings. Some have already been partially addressed, but they still define the architecture target.

### D1 - Deterministic-key dedup is the wrong primitive
Content-addressed or theme-keyed identity alone is too brittle. It can explode duplicates when keys vary or clobber valid distinct facts when keys collide.

### D2 - SUM coverage is too coarse
Theme-level conclusions are not enough. SUM needs finer, subtopic-level synthesized claims.

### D3 - MR needs aggregation, not just retrieval
Some MR answers are computed counts or comparisons. The retrieval path should expose components plus derived aggregate facts so the answer model can synthesize cleanly.

### D4 - Silent failure swallowing hides consolidation bugs
Consolidation and extraction failures must be loud. A degraded cache should not fail quietly.

---

## 3. Design decision

### One-line summary

Keep raw evidence as layer 0, but add a layer 1 representation store made of durable memory cards. Cards are the primary retrieval unit; raw evidence is the fallback and provenance layer.

### What we are copying

#### From mem0
- Candidate facts are compared against semantically similar live memories.
- The LLM chooses one action: `ADD`, `UPDATE`, `DELETE`, or `NOOP`.
- The write path applies that decision before retrieval sees the candidate.

#### From Hindsight
- Raw evidence and synthesized memory are different things.
- We store separated synthesized units such as entity, topic, change, belief, and session cards.

#### From Honcho
- Heavy reasoning happens on ingest and in background dream/sleep consolidation.
- Recall reads from compact representations instead of reconstructing meaning from many specialists.

---

## 4. V2 storage model

## 4.1 Layer 0 - raw evidence

Keep the raw tables and treat them as evidence/backstop:
- `memoria_facts`
- `memoria_timelines`
- `memoria_kg`
- `memoria_summaries`
- `working_memory`
- `episodic_memory`
- `facts`

Layer 0 is for provenance, fallback retrieval, and card backfill.

## 4.2 Layer 1 - representation store

Add these tables.

### `memory_cards`

Primary answer-shaped memory unit.

```sql
CREATE TABLE IF NOT EXISTS memory_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    card_type TEXT NOT NULL,
    card_key TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.7,
    source_start_msg_idx INTEGER,
    source_end_msg_idx INTEGER,
    version_id INTEGER NOT NULL DEFAULT 0,
    previous_card_id INTEGER,
    valid_to_msg_idx INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (card_type IN ('entity', 'topic', 'change', 'belief', 'session'))
);
```

### `memory_card_evidence`

Evidence links from cards back to layer 0.

```sql
CREATE TABLE IF NOT EXISTS memory_card_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL,
    evidence_table TEXT NOT NULL,
    evidence_row_id TEXT NOT NULL,
    message_idx INTEGER,
    snippet TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (evidence_table IN (
        'memoria_facts',
        'memoria_timelines',
        'memoria_kg',
        'memoria_summaries',
        'working_memory',
        'episodic_memory',
        'facts'
    ))
);
```

### `memory_card_queue`

Agenda for background dream/sleep updates.

Important rule: queue rows should coalesce by live card key. Do not create one
pending row per supporting raw fact for the same card.

### `vec_cards`

Embeddings for live cards only.

### `fts_cards`

Lexical lookup over card titles and summaries.

---

## 5. Write path

## 5.1 Phase A - ingest raw message

Keep the current message insert path and metadata preservation.

## 5.2 Phase B - atomic extraction

Run the existing cloud extraction path and keep it atomic:
- SPO facts
- dates and timelines
- relations and entities
- conclusions
- derived aggregates when appropriate

## 5.3 Phase C - mem0-style raw consolidation gate

This phase already exists in shipped code, but it is partial coverage, not a universal gate over every write route.

Still missing from full coverage (writers that reach `_insert_fact()` directly,
skipping the gate):
- regex metrics
- direct date inserts
- duration derivation (`_derive_durations` -> `_insert_fact`)
- any other direct `_insert_fact()` producer that should be semantically consolidated

Clarification:
- duration derivation already has the important landed identity fix: it keys on
  the ISO date pair rather than shared topic words
- do not redesign duration key identity again here; the remaining question is
  only whether any duration writes should also pass through the broader semantic
  consolidation gate

Rules:
1. Exact dedup by `source_memory_id` and exact live `(fact_type, key, value)`.
2. Fetch top-k semantically similar live rows from the same session.
3. Ask the LLM for `ADD`, `UPDATE`, `DELETE`, or `NOOP`.
4. Persist the result into layer 0 raw stores.
5. Only embed surviving live rows.

Important:
- `UPDATE` must version-chain the matched live row.
- `NOOP` must skip both write and embed.
- `DELETE` must invalidate the matched row.
- `ADD` must allow distinct same-theme items to coexist.

## 5.4 Phase D - agenda build for cards

Every successful raw write can emit one or more card agenda items.

Mapping rules:
- concrete subject or entity mention -> `entity`
- conclusion or related fact cluster -> `topic`
- versioned update or contradiction -> `change`
- stable preference or inferred stance -> `belief`
- every N successful raw writes or each sleep cycle -> `session`

## 5.5 Phase E - dream worker / sleep worker

Background worker drains `memory_card_queue`.

For each agenda item:
1. Load the live card if it exists.
2. Load the best supporting raw evidence rows from layer 0.
3. Load the current `session:overview` card if available.
4. Ask the LLM to emit a card patch.
5. Apply the patch as versioned `ADD` / `UPDATE` / `DELETE` / `NOOP`.
6. Rewrite `memory_card_evidence`.
7. Embed the live card into `vec_cards`.

The dream worker writes compact cards, not more raw facts.

## 5.6 Phase F - session synopsis refresh

Refresh a live `session:overview` card from live topic/entity/change/belief cards. Do not reread the whole raw conversation by default.

---

## 6. Read path

## 6.1 Cards first

Default retrieval should:
1. detect intent locally
2. search live cards
3. rerank live cards locally
4. hydrate evidence from linked raw rows
5. fall back to current raw retrievers only if needed

## 6.2 Intent routing

Use local heuristics only:
- `current`
- `change`
- `timeline`
- `ordered`
- `summary`
- `aggregate`
- `fact`

Mapping guidance:
- `SUM` queries -> prefer `session` and `topic`
- `MR` queries -> prefer `topic`, `belief`, and `change`
- `KU` queries -> prefer `change`
- `CR` queries -> prefer `change` and `belief`
- `ABS` and `PF` queries -> prefer `topic`, `belief`, and `entity`

## 6.3 Candidate retrieval

Search `vec_cards` and `fts_cards`, merge hits with simple RRF, rerank locally, keep only the top cards, then hydrate evidence from layer 0.

### EO / ordered-query safeguard

Cards-first retrieval must not turn EO into a topical summary problem.

- If intent is `ordered`, preserve `MSGIDX` anchors in the final context.
- Hydrate raw evidence that carries `MSGIDX` or `[Fact ... MSGIDX:N]` markers before reranking final candidates.
- If cards do not provide enough ordering evidence, fall back to the existing raw ordered retriever path.
- The current msg-index sort remains the authority for EO.

## 6.4 Fallback path

Use existing raw retrieval when:
- no live cards are found
- card confidence is too low
- the intent needs exact raw data that no card covers

Fallback sources:
- `memoria_retrieve()`
- `fact_recall()`
- current timeline/date specialists

## 6.5 Reflect stays answer-path only

SUM and MR can still use a narrow reflect-style synthesis helper on the final
answer path during rollout, but recall itself should stay cards-first plus local
evidence hydration rather than depending on a second online synthesis step.

## 6.6 Context shape

Cards first, evidence second.

Implementation seam:
- extend `edumem/core/context_assembly.py` for card-context rendering and final
  assembled memory shape
- do not rebuild `_assemble_memory_context()` ad hoc inside `beam.py`

Example:

```text
[Card TOPIC] Security hardening
Security work progressed from password hashing to RBAC and then account lockout.

[Card CHANGE] Deployment window
Current: February 5 through February 12 (was: February 1 through February 10)

[Evidence]
- MSGIDX:40 password hashing was added...
- MSGIDX:72 RBAC was introduced...
- MSGIDX:96 account lockout slowed repeated login attempts...
```

---

## 7. Rollout guidance

### Phase 1 - raw consolidation only
- Already shipped, but only partial coverage.
- Keep it on while cards are introduced.

### Phase 2 - card schema + queue
- Add card tables and queue.
- Populate cards during ingest and sleep.
- Do not change benchmark routing yet.

### Phase 3 - cards first for SUM/MR/ABS/PF
- These abilities benefit most from synthesized representation.
- Keep raw fallback.

### Phase 4 - cards first for everything except exact raw edge cases
- Prefer `change` cards for KU/CR.
- Keep timeline fallback for exact interval/date math and EO ordering.

### Phase 5 - simplify old specialist fusion
- Retire specialists only after cards-first proves itself.

Suggested flags:
- `EDUMEM_CARD_LAYER=1`
- `EDUMEM_CARD_RECALL=1`
- `EDUMEM_CARD_QUEUE_WORKER=1`
- `EDUMEM_CARD_RAW_FALLBACK=1`
- `EDUMEM_CARD_RERANK_TOPK=15`
- `EDUMEM_CARD_RETRIEVE_TOPK=20`

---

## 8. Pitfalls and confirmed defects

- Reasoning-model kwargs still matter. New qwen/gemma calls must disable hidden thinking, and deepseek should use low reasoning effort.
- Do not run both extraction modes at once.
- Rebuild the recall cache after any write-path change.
- The recall gate under-credits IF and computed MR counts.
- Use `127.0.0.1`, not `localhost`, for local embedding and reranker services.
- Never swallow consolidation or extraction errors with `except: pass`.

---

## 9. Success metrics

Track:
- mean live `memoria_facts` rows per conversation
- mean live `memory_cards` rows per conversation
- evidence links per card
- queue backlog and card refresh lag
- duplicate live raw rows by semantic cluster
- card retrieval hit rate
- raw fallback rate
- rerank p50 and p95
- retrieval p95 latency

Reasonable targets:
- `memory_cards`: roughly 20-60 live cards per 200-message conversation
- `memory_card_evidence`: 3-10 links per live card
- retrieval p95: under 300ms
- raw fallback on fewer than 20% of SUM/MR queries

BEAM target:
- lift the flat low abilities, especially `SUM`, `MR`, `ABS`, and `PF`, without making EO worse.

---

## 10. Final decision

The recommended big-jump path for this repo is:

1. mem0-style raw consolidation gate
2. Hindsight-style synthesized card layer
3. Honcho-style background dream/sleep updater
4. cards-first local recall
5. raw evidence as fallback

That is the canonical spec now.
