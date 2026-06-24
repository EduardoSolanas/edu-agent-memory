# AGENTS.md — guidance for AI agents working on this repo

This project is a memory system benchmarked against **BEAM** (10 memory abilities:
ABS, CR, EO, IE, IF, KU, MR, PF, SUM, TR). This file explains how the BEAM
evaluation and tests fit together, how to run them, and what is currently
known/unknown — so you don't rediscover it or draw wrong conclusions.

## Architecture essentials

- **Versioned facts** live in the `memoria_facts` table (`edumem/core/beam.py`).
  Columns `version_id`, `previous_value`, `message_idx`, `valid_from_msg_idx`
  chain successive values of the same `(session_id, key, fact_type)`.
  `_insert_fact()` chains versions automatically; `_insert_change_fact()` creates
  an explicit two-row chain from "switched/changed from X to Y" statements.
- **Intent-based rendering, NOT ability codes.** `_format_versioned_fact(fact, intent)`
  and `_memoria_fact_retrieve(query, top_k, intent)` branch on GENERIC intents:
  `current`, `change`, `timeline`, `ordered`, or `''` (flat). BEAM ability codes
  (KU/CR/TR/EO) are mapped to intents ONLY at the public boundary
  `memoria_retrieve()` via `_ABILITY_TO_INTENT`. Keep benchmark labels out of core
  memory logic — do not reintroduce `ability == 'KU'` checks inside the rendering
  layer.
- **Retrieval merge:** for CR/TR/EO, `memoria_retrieve()` merges a specialist
  retriever (negation/timeline/chrono) with versioned-fact retrieval via
  `_merge_memoria_results()`.
- **Prompt shaping:** `edumem/core/query_mode.py` builds the answer system prompt.
  An always-on base prompt handles CR/ABS/KU/PF generically; question-triggered
  modifiers add EO/TR/etc. formatting. The base prompt's CONFLICTS rule is
  sensitive — over-triggering it causes false "contradictory information" answers
  (see Known gaps).

## Tests

- **Fast suite** (`tests/`, runs offline in ~10s): `python -m pytest tests/ -q`.
  Must stay green WITHOUT any Docker stack or network. Currently ~229 passed.
- **Static rendering/integration tests** (`tests/test_beam_evaluator.py`): assert
  the versioned formats unconditionally (`[Fact CURRENT]`, `[Fact CHANGED]`,
  `[Fact TIMELINE]`, `MSGIDX:`). These are deterministic, no LLM.
- **Live e2e** (`tests/test_beam_e2e_full.py`): gated by `EDUMEM_E2E=1`, skips by
  default. Runs the REAL pipeline ingest→recall→answer (qwen3.6) on one BEAM 100K
  conversation and grades the answer against **static expectations** mined from
  prior judge results by `tools/generate_beam_fixture.py`. There is **no live
  judge** — grading is deterministic nugget/absence/order checks.

### Running the live e2e
Requires the reranker up on `http://localhost:3002/rerank` and the NAN LLM
endpoint. The API key lives in repo `.env` as `NAN_APY_KEY`. Each shell call is
fresh, so set env + run pytest in ONE command (Git Bash):
```
export OPENROUTER_API_KEY="$(grep -E '^NAN_APY_KEY=' .env | head -1 | cut -d= -f2- | tr -d '\r\"')"
export OPENROUTER_BASE_URL="https://api.nan.builders/v1"
export EDUMEM_E2E=1
export EDUMEM_E2E_ANSWER_MODEL="qwen3.6"
python -m pytest tests/test_beam_e2e_full.py -v -s
```
A full run is ~30–40 min. The fixture marks each question `hard` (passed in the
baseline → regression guard) or `xfail` (known-failing → a target to flip). When
your change fixes a known failure it shows as **xpass** — that is the success
signal.

## Discipline

- **Always add/extend BEAM tests when you change retrieval, prompt shaping, fact
  extraction, or MEMORIA rendering.** Build cases from REAL benchmark failures,
  not synthetic examples. A test must FAIL if the code it covers is reverted —
  avoid `if`-guards that make assertions vacuous.
- Static nuggets must be **atomic** (numbers, dates, quoted terms, distinctive
  keywords), never whole rubric sentences — the answer LLM paraphrases, so literal
  multi-word phrase matching is brittle.

## Known gaps / root causes (as of 2026-06-20, evidence-backed)

These are diagnosed from the captured real run
(`results/beam_results_20260620_104723/`) plus source-conversation verification —
deterministic, not n=1 speculation.

1. **THE big one — the benchmark bypasses the ability-keyed versioned-fact path.**
   `answer_with_memory` sets `routing_ability = None if _pure_recall else ability`
   (evaluate_beam_end_to_end.py ~line 2700), and `_benchmark_pure_recall_enabled()`
   defaults TRUE. So every benchmark question calls
   `memoria_retrieve(question, ability=None)` — our `CR`/`TR`/`EO` versioned-fact
   merge branches require a concrete ability and NEVER fire; intent maps to `''`
   (flat rendering), so `[Fact CURRENT/CHANGED/TIMELINE]` never reaches the prompt.
   The per-ability bypasses (~lines 2722/2776/2859) are gated `if not _pure_recall`
   → off. Result: Phases 1–3 are effectively dead code in the benchmark. This is
   why storage shows 63 version chains (ingest is ability-independent) yet answers
   don't improve (surfacing is ability-gated and the gate is shut).
   - **Why by design:** pure-recall is deliberately *label-free* — `query_mode.py`
     routes on question text only, never the dataset ability label (using the gold
     label = overfitting).
   - **The fix:** derive intent from the QUESTION TEXT (`query_mode` already has
     `is_knowledge_update_query` / `is_contradiction_query` / `is_duration_query` /
     `is_ordering_query`) and pass that intent into the recall path. Legitimate and
     it activates the versioned-fact surfacing.

2. **Temporal answers: right facts retrieved, wrong pair anchored.** q18 (TR, 0%):
   the answer plan (msg 2/3: "Dec 16–Jan 15 transaction management", "Feb 16–Mar 15
   deployment") IS retrieved (msg 2 at rank 7), but the model isn't anchored to it,
   drowns in sprint dates (msg 29/53/86), and bails to "contradictory information".
   Fix is disambiguation/anchoring + suppressing the false-contradiction fallback,
   NOT just recall.

3. **Aggregation under-counts across instances.** q12 (MR, 0%): two column-add
   requests exist — `category` (msg 106) and `notes` (msg 162) — both retrieved
   (ranks 20/22) but the answer counted only 1. Aggregation questions need all
   instances surfaced into the final context together.

4. **EO orders by latest-state, not first-mention.** q4/q5 (~13%): the model labels
   topics by their optimized form and the first-appearance ordering drifts.

CAVEATS for whoever continues:
- `recall_provenance.memories[].final_context_included` is UNRELIABLE — it showed
  `False` for messages the model demonstrably used. Do not trust it to decide what
  reached the prompt; instrument the actual assembled context instead.
- Do not claim versioning "fixed" CR/EO/TR from storage metrics alone — and note
  it can't help at all until gap #1 is addressed.

## Retrieval-recall benchmark analysis (2026-06-24)

### What was done

Added **cached DB** (`tests/.beam_recall_cache/beam_100K_conv0.db`) to avoid
re-ingesting the BEAM 100K conversation (~11 min). The test fixture
`get_cached_beam_and_conv()` builds once, reuses on subsequent runs. Also
monkey-patched `_embed_api` with `requests.Session` connection pooling. The
Hindsight-style cross-encoder reranker was then **wired into the real
pipeline** — `_fusion_rerank` in `_memoria_fused_retrieve` reorders fused
facts by query relevance after RRF (default ON, gated
`EDUMEM_FUSION_RERANK`; the EO msg_idx sort still wins for ordering queries;
None-on-failure keeps RRF order). The redundant test monkey-patch and the
harness `_rerank` (which re-ranked the wrong object) were removed.

### Performance

| Change | Time | Improvement |
|--------|------|-------------|
| Original (no cache, no pooling) | ~11 min ingest + 436s | — |
| + Cached DB (skip ingest) | 436s | skip 11-min ingest |
| + Connection pooling (`requests.Session`) | **103s** | 4.2× faster |
| + Container down (embedding timeout) | ~33s/question | 30s timeout each |

Container warm: **103s** for 20 questions (~5s/question). Full benchmark:
**0.347 overall recall** (passes 0.30 gate).

### The 67% waste

~65% of rubric nuggets are NOT found (0.347 recall). The 30 facts retrieved per
question are **dominated by generic timeline/milestone scaffolding** regardless
of question topic. Example: ABS question "How did user feedback influence
UI/UX?" retrieved only date-range milestones — zero user feedback or UI/UX
content. The MEMORIA specialists all return similar date-heavy context; RRF
fusion doesn't distinguish content-facts from temporal-noise.

**Abilities sorted by recall:**

| Ability | Recall | Problem |
|---------|--------|---------|
| IE/KU/MR/CR/IF | 0.500-0.750 | Works when facts contain exact values |
| EO | 0.300-0.367 | Retrieves generic timeline, not topic ordering |
| TR | 0.250 | Wrong date pair anchored |
| SUM | 0.100 | Retrieved snippets lack narrative structure |
| ABS/PF | **0.000** | No relevant facts retrieved at all |

### Architecture comparison with Hindsight / Mem0 / Honcho

| System | Signals | Fusion | Reranker | Context trim |
|--------|---------|--------|----------|-------------|
| **Hindsight** | 4 (sem, BM25, graph, temporal) | RRF | Cross-encoder | Token-limit |
| **Memo** | 3 (sem, BM25, entity) | Additive score | Optional | Top-k cap |
| **Honcho** | Reasoning-first (distill→conclusions) | Intent-driven | N/A | Token-budget |
| **edumem MEMORIA** | 7 specialists (fact, timeline, negation, chrono, KG, summary, **semantic-KNN-over-facts**) | RRF (k=60) | Cross-encoder (`_fusion_rerank` in `_memoria_fused_retrieve`: RRF → rerank → EO-sort; gated `EDUMEM_FUSION_RERANK`, default ON; None-on-failure keeps RRF order) | Score threshold + dedup |
| **edumem polyphonic** (gated) | 4 voices (vector, graph, fact, temporal) | RRF (k=60) | **None** (diversity rerank is disabled: returns 0.0) | Token budget |

~~Key gap vs Hindsight: no cross-encoder reranker after RRF fusion.~~ **Now
closed:** the reranker at `localhost:3002/rerank` is wired into
`_memoria_fused_retrieve` via `_fusion_rerank`. Caveat remains: it only
re-orders existing fused facts — it cannot retrieve NEW ones, so it is an
upper-bound-only improvement signal on the live recall benchmark.

### What to try next

1. **Enable polyphonic recall** (`EDUMEM_POLYPHONIC_RECALL=1`) — 4 voices with
   RRF fusion + context budget. May improve diversity over MEMORIA's 6
   specialists (which overlap heavily on temporal data).
2. ~~**Add cross-encoder reranker after RRF**~~ — **done** (wired into
   `_memoria_fused_retrieve` as `_fusion_rerank`, see above).
3. **Specialist deduplication** — MEMORIA's 6 specialists all return similar
   date/milestone facts. Deduplicate across specialist outputs before RRF
   (content hash, not just memory_id).
4. **Ability-aware intent routing** — Gap #1 in AGENTS.md: `routing_ability =
   None` in pure-recall mode disables versioned-fact surfacing. Fixing this
   would activate CR/TR/EO specialist merge branches.

## Semantic recall over versioned facts (2026-06-24)

The 7th fusion specialist (`_memoria_semantic_retrieve`) adds embedding-based
recall over `memoria_facts` via the repurposed `vec_facts` table, targeting the
ABS/PF **0.000 recall** case where a paraphrased query shares no literal tokens
with any fact's key/value.

- **Write path:** live facts (plain + new-version + change-new branches; date
  branch skipped for content quality; dead/superseded rows never enqueued) are
  embedded (`context_snippet` text, else `key: value`) and queued, then
  `_flush_fact_embeddings()` does ONE batched `embed([...])` at the
  `remember_batch` boundary (no N per-fact commits). Supersession adds a
  `DELETE FROM vec_facts` cleanup hook so stale values never surface.
- **Read path:** the specialist runs KNN over `vec_facts`, joins back to
  `memoria_facts` filtered to `valid_to_msg_idx IS NULL`, assigns a synthetic
  `source_memory_id: semantic:fact:{rid}` fusion key, and RRF-combines with the
  6 lexical specialists.
- **Infra:** the 4 vec helpers (`_vec_available`/`_effective_vec_type`/
  `_vec_insert`/`_vec_search`) gained a `table` param (threaded into internal
  calls) so the facts table reuses the proven vec_episodes contract.
- **No new flag:** gated by `EDUMEM_NO_EMBEDDINGS` + `_vec_available(table=)`.
- **Backfill is out of scope (write-time-only):** the cached recall DB
  (`tests/.beam_recall_cache/beam_100K_conv0.db`) must be rebuilt to populate
  `vec_facts` before the live benchmark can show the ABS/PF lift.
- Tests: `tests/test_semantic_fact_recall.py` (9 cases, all non-vacuous — each
  fails if its covered wiring is reverted).
