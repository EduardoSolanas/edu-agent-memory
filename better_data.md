# better_data.md — Mem0/Hindsight-style write-time extraction

**Audience:** a cheap implementation agent (haiku). Follow this exactly. It encodes
the design AND the pitfalls we already paid for. Do not improvise the schema or
skip the bounds — the bounds are the whole point.

---

## 1. Goal
Store **cleaner, structured, deduplicated** memory at write time (ingest), the way
Mem0 / Hindsight / Honcho do, instead of regex-derived noisy keys. This lifts the
**fact abilities** (KU, MR, IE, TR, SUM) — which is where our measured gains already
come from.

## 2. What we learned (read before coding)
- The regex extractor produces noisy, fragmented keys (e.g. `sure_it's_optimized_ms`),
  so the same concept never version-chains.
- A first LLM-extraction attempt **failed**: the prompt was **unbounded**, so the model
  emitted 3–4K-token essays that **truncated** (`finish_reason=length`) → invalid JSON
  → fell back to regex, and one call **hung the whole run**. Bounded output is mandatory.
- **Honest ceiling:** CR (contradiction resolution) and answer latency are
  **answer-model-bound (qwen3.6)** — better storage will NOT fix them. Do not expect
  this work to fix CR or speed. Target the fact abilities only.

## 3. Design principles (copy the leaders)
1. **Bounded:** at most **5 facts per message**. Compact JSON. No prose.
2. **Structured (Hindsight):** extract typed `facts`, named `entities`, and
   `relations` (subject–predicate–object), plus `dates`. Not just `key:value`.
3. **Canonical keys (Mem0):** the same real-world concept across messages must get the
   **same snake_case key** so `_insert_fact` version-chains it. Keep stated *targets/goals*
   under distinct keys from *actual measurements*.
4. **Robust:** `max_tokens` cap, a **per-call timeout**, and graceful fallback to the
   regex path on any error/empty/timeout — never lose extraction, never hang.
5. **Flag-gated, default OFF** (`EDUMEM_LLM_EXTRACTION`) so prod/default behavior and
   the fast test suite are unchanged; we A/B it explicitly.

## 4. Implementation (file: `edumem/core/beam.py`)
There is already a flag-gated path with `_build_llm_extraction_prompt`,
`_parse_llm_extraction`, `_store_llm_extraction` — **rework them per below**, do not add
a parallel path.

### 4a. `_build_llm_extraction_prompt(self, content, existing_keys=None) -> str`
Prompt the LLM to return **ONLY** this JSON (no prose, no markdown), and **cap each list at 5**:
```json
{"facts":[{"key":"snake_case","value":"...","type":"metric|state|preference|instruction"}],
 "entities":[{"name":"...","kind":"service|project|person|tool|component"}],
 "relations":[{"subject":"...","predicate":"snake_case","object":"..."}],
 "dates":[{"key":"snake_case","date":"YYYY-MM-DD","context":"..."}]}
```
Rules to embed in the prompt:
- Extract only persistent, high-signal items. If nothing qualifies, return all-empty lists.
- **Max 5 items per list.** Be terse — values are short strings, not sentences.
- Canonical snake_case keys; same concept across messages → same key. Actual
  measurements vs stated targets get distinct keys.
- Language-agnostic (detect language; do not rely on English keywords).
- If `existing_keys` is provided, REUSE a matching key rather than inventing a variant.

### 4b. `_parse_llm_extraction(self, raw) -> dict`
Strip ```json fences / surrounding prose, isolate the outer `{...}`, `json.loads`,
validate shape, truncate each list to 5 defensively. Any failure → all-empty dict.

### 4c. `_store_llm_extraction(self, session, msg_idx, parsed, ctx, source_memory_id=None) -> dict`
- `facts` → `_insert_fact(session, msg_idx, fact["type"] mapped to a stored fact_type, fact["key"], fact["value"], ctx, importance, source_memory_id=...)` (version-chains on repeated key).
- `entities` → `memoria_kg` as `(name, "is_a", kind)` (or the existing entity insert helper — read the schema).
- `relations` → `memoria_kg` triples; map negation predicates (never_used/not/no/did_not…) to the canonical `negation` predicate the negation specialist reads.
- `dates` → `memoria_timelines`.
- Return per-category counts. Pure given `parsed` (no LLM call) — main TDD target.

### 4d. Call config + wiring in `extract_and_store_facts`
When `self._llm_client is not None and os.environ.get("EDUMEM_LLM_EXTRACTION","0")=="1"`:
- Load existing canonical keys for the session (pass to the prompt for reuse).
- `self._llm_client.chat([...], temperature=0.0, max_tokens=512)` — **cap at 512** so it
  cannot run away into an essay. Wrap with a **timeout** (e.g. `EDUMEM_EXTRACTION_TIMEOUT`,
  default ~20s) so one stall can't hang ingest.
- Parse → if non-empty, `_store_llm_extraction(...)` and return counts.
- On ANY exception / empty / `finish_reason==length` (treat truncation as failure) →
  **fall through to the existing regex path.** Never lose extraction.
When the flag is off or no client → behavior EXACTLY as today.

## 5. TDD (real objects, no mocks; `EDUMEM_NO_EMBEDDINGS=1` for unit speed)
Write FIRST, then implement:
- `test_extraction_prompt_is_bounded`: prompt states max 5, asks for the 4-key JSON, mentions canonical keys.
- `test_parse_truncates_to_five_and_handles_garbage`: a 7-item fenced JSON → 5 kept; garbage → all-empty.
- `test_store_extraction_chains_and_routes`: hand-built parsed dict → facts chain in `memoria_facts` (previous_value set on repeated key), a relation/negation lands in `memoria_kg` retrievable by `_memoria_negation_retrieve`, a date in `memoria_timelines`.
- `test_extraction_truncation_falls_back_to_regex`: simulate an empty/failed parse → regex path still runs (no LLM client → unchanged behavior).
Fast suite must stay green except the 3 known pre-existing `test_server_runtime` sanitizer failures.

## 6. A/B verification (the parent runs this; agents do NOT)
Measure `EDUMEM_LLM_EXTRACTION=1` vs `0`, **on the real path**:
- **Dense embeddings REQUIRED** (or numbers are invalid): set
  `EDUMEM_EMBEDDING_API_URL=http://localhost:3002`,
  `EDUMEM_EMBEDDING_MODEL=Alibaba-NLP/gte-modernbert-base`, `EDUMEM_EMBEDDINGS_VIA_API=1`.
- **Per-conversation isolation:** run each case as its own process
  (`--case-index 0`, `1`, `2`) — a single long process dies on a Windows temp-dir cleanup
  (`WinError 32`) around conv 2–3.
- Use `--model qwen3.6 --judge-model deepseek-v4-flash`, context `EDUMEM_MAX_CONTEXT_CHARS≈16000`.
- Compare per-ability vs the dense baseline `Y:/beam_results_20260620_114259`. Success =
  fact abilities (KU/MR/IE/TR/SUM) up, no big regression elsewhere.

## 7. Constraints / non-goals
- **Bounded output is non-negotiable** (max 5/list, max_tokens 512, timeout). The prior
  failure was unbounded output.
- Default OFF; do not change the regex/default path.
- Do NOT expect this to fix **CR** or **latency** — those are answer-model-bound.
- Edit only `edumem/core/beam.py` + its tests. No commit; the parent commits + A/Bs.

## 8. Known-good infra notes (so you don't rediscover them)
- LLM client: `self._llm_client.chat(messages, temperature=, max_tokens=)`.
- `_insert_fact` chains versions on `(session_id, key, fact_type)`.
- MEMORIA tables: `memoria_facts`, `memoria_timelines`, `memoria_kg` (read schemas before inserting).
- Negation predicate the specialist reads: `predicate='negation'` in `memoria_kg`.

---

## 9. Knowledge graph: structured storage, SIMPLE recall (Hindsight-style, no traversal)
We want Hindsight's clean graph **storage** WITHOUT its graph-traversal **recall** complexity.
The two are separable: store the graph, don't traverse it at query time.

### 9a. Write side = the graph (already in §3/§4)
The `entities` and `relations` extracted in §4a are the graph: nodes = named entities
(`service|project|person|tool|component`), edges = `(subject, predicate, object)` triples.
Store them in `memoria_kg`. Dedup by reusing canonical entity names/keys (pass
`existing_keys`/known entities into the prompt so the same entity isn't re-created under
variant spellings). That is the entire "graph" — clean, deduplicated structured rows.

### 9b. Read side = KG as ONE MORE fusion source (recall stays uniform RRF)
Do NOT add graph traversal, recursion, or a graph engine. Add a specialist
`_memoria_kg_retrieve(self, query, top_k)` that returns `memoria_kg` rows matching the
query terms (simple SQL `LIKE`/FTS on subject/predicate/object/entity-name), rendered as
readable text with `[MSGIDX:N]`. Then include it in `_memoria_fused_retrieve`'s specialist
list so RRF fuses it alongside fact/timeline/negation/chrono. **Recall logic does not get
more complex** — it is still "run all specialists → RRF fuse → rerank," just with one more
source. The graph is retrieved as flat rows, never walked.

### 9c. Optional, ONLY if measurement shows MR still gaps: 1-hop expansion
If multi-hop questions still fail, add a single 1-hop expansion in `_memoria_kg_retrieve`:
when the query matches an entity, also pull that entity's **direct** neighbor triples (one
SQL join on `memoria_kg`, no recursion) into the candidate set before fusing. One hop only.
Do NOT build recursive traversal. MR already improved via RRF (it is mostly aggregation),
so treat 1-hop as a contingency, not a default — add it only if the A/B shows MR lagging.

### 9d. TDD for the KG recall source
- `test_memoria_kg_retrieve_returns_matching_triples`: insert a few `memoria_kg` triples,
  assert `_memoria_kg_retrieve("...query terms...")` returns the matching ones as text with MSGIDX.
- `test_fused_recall_includes_kg_source`: with KG triples present, `memoria_retrieve(query)`
  fused context contains a KG-derived row (proves KG is in the fusion). No traversal asserted.
- (If 1-hop implemented) `test_kg_one_hop_expansion`: query matches entity A; A-(rel)->B in KG;
  assert B's triple appears even when the query didn't name B. Keep it ONE hop.

### 9e. Scope note
KG-as-fusion-source is a small extension: write side already builds `memoria_kg` (§4c);
read side is one new specialist added to the existing fusion. No change to recall's shape.
Same honest ceiling: this lifts fact/multi-hop abilities, not CR/latency (answer-model-bound).
