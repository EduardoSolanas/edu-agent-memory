# AGENTS.md — guidance for AI agents working on this repo

This project is a memory system benchmarked against **BEAM** (10 memory abilities:
ABS, CR, EO, IE, IF, KU, MR, PF, SUM, TR). This file explains how the BEAM
evaluation and tests fit together, how to run them, and what is currently
known/unknown — so you don't rediscover it or draw wrong conclusions.

## Environment & invocation reference (canonical — 2026-06-25)

### LLM provider (answer / judge / extraction / consolidation)
The whole LLM path keys off ONE canonical pair. `OPENROUTER_*` is read only as a
**deprecated fallback** — prefer the `EDUMEM_LLM_*` names. On the NAN stack the key
is `NAN_APY_KEY` in repo `.env`; base_url is `https://api.nan.builders/v1`.

| Var | Purpose | Default |
|---|---|---|
| `EDUMEM_LLM_API_KEY` | LLM key (answer/judge/extraction). Fallback: `OPENROUTER_API_KEY`. | — |
| `EDUMEM_LLM_BASE_URL` | OpenAI-compatible base URL. Fallback: `OPENROUTER_BASE_URL` → `https://openrouter.ai/api/v1`. | openrouter.ai |
| `EDUMEM_LLM_MODEL` | Canonical chat model (answer). NAN = `qwen3.6`. | — |
| `EDUMEM_EXTRACTION_MODEL` | Fact+conclusion extraction model. Resolves `EDUMEM_EXTRACTION_MODEL` → `EDUMEM_LLM_MODEL` → `google/gemini-2.5-flash`. So on NAN it follows qwen3.6 automatically (the gemini default 401s on NAN → silent no-op). | `EDUMEM_LLM_MODEL` |
| `EDUMEM_EXTRACTION_FALLBACK_MODELS` | Comma-sep extraction fallbacks. | empty |
| `EDUMEM_JUDGE_MODEL` | Judge model (full runner uses `--judge-model`). NAN = `deepseek-v4-flash`. | — |
| `BEAM_LLM_TIMEOUT` | Answer-LLM request timeout (s). | 300 |
| `EDUMEM_EXTRACTION_TIMEOUT` | ExtractionClient/summary call timeout (s). | 20 |

**Reasoning models are handled automatically (no env var):** both the answer
`LLMClient` and `ExtractionClient` send `chat_template_kwargs={"enable_thinking":
false}` for `qwen*`/`gemma*` and `reasoning_effort="low"` for `deepseek*`. Without
it, qwen3.6 spends the whole `max_tokens` budget on hidden chain-of-thought →
`content=None`, minutes/call, empty output. Any new qwen client MUST do this.

### Write-time extraction modes (there are TWO — do not run both)
| Mechanism | Enabled by | Produces |
|---|---|---|
| `ExtractionClient` (`self._extraction_client`) | `BeamMemory(use_cloud=True)` / full runner `--use-cloud` | SPO facts → `facts` table; **conclusions** → `memoria_facts` (`fact_type='conclusion'`); via `EDUMEM_EXTRACTION_MODEL` |
| `self._llm_client` paths | pass `llm_client=` into `BeamMemory` **and** a flag | `EDUMEM_LLM_EXTRACTION=1`: MEMORIA entities/relations/dates JSON schema (OVERLAPS with ExtractionClient — running both double-extracts facts); `EDUMEM_LLM_SUMMARY=1`: rolling-summary track; `EDUMEM_LLM_FACT_CONSOLIDATION` (default 1): consolidation |

The cache/benchmark uses `use_cloud=True` (ExtractionClient). Offline
(`use_cloud=False`, no `llm_client`) = regex only (`episodic_graph` SPO + regex
MEMORIA), no LLM. Passing `llm_client` to *also* turn on `EDUMEM_LLM_EXTRACTION`
re-introduces the duplication that was just removed — don't, unless replacing
ExtractionClient outright.

### Dedup (Mem0-style)
- `memoria_facts`: `_insert_fact` dedups by `(fact_type, key, value)` over **live**
  rows (skips the dup AND its embedding); same key + new value still
  version-chains; distinct dates coexist.
- `facts` (SPO): `_spo_fact_id(session, s, p, o)` is content-addressed so
  `INSERT OR IGNORE` collapses identical triples across batches.

### Embeddings / reranker
| Var | Default |
|---|---|
| `EDUMEM_EMBEDDING_API_URL` | http://localhost:3002 |
| `EDUMEM_EMBEDDING_MODEL` | Alibaba-NLP/gte-modernbert-base (768-dim) |
| `EDUMEM_EMBEDDINGS_VIA_API` | set `1` to use the API embedder |
| `EDUMEM_RERANKER_URL` | http://localhost:3002/rerank |
| `EDUMEM_NO_EMBEDDINGS` | set `1` for offline (FTS/keyword only; no dense, no `vec_facts`) |

### Invocations
- **Full runner (MEASUREMENT, live judge):**
  ```bash
  export EDUMEM_LLM_API_KEY="$(grep -E '^NAN_APY_KEY=' .env | cut -d= -f2- | tr -d '\r\"')"
  export EDUMEM_LLM_BASE_URL="https://api.nan.builders/v1"
  python tools/evaluate_beam_end_to_end.py --scales 100K --case-index 0 \
    --model qwen3.6 --judge-model deepseek-v4-flash --pure-recall [--use-cloud] \
    --output-dir results/<tag>
  ```
  Knobs: `BEAM_QUESTION_WORKERS` (4), `EDUMEM_MAX_CONTEXT_CHARS` (16000),
  `EDUMEM_BENCHMARK_PURE_RECALL` (on; `--pure-recall`).
- **Pre-LLM retrieval-recall test (deterministic, NO answer/judge LLM):**
  ```bash
  EDUMEM_RETRIEVAL_E2E=1 python -m pytest tests/test_beam_retrieval_recall.py -q -s
  ```
  Builds `tests/.beam_recall_cache/beam_100K_conv0.db` ONCE with `use_cloud=True`
  (qwen3.6 facts+conclusions; key auto-loaded from `.env`), then measures nugget
  recall in the assembled context. Reruns reuse the cache → seconds, offline.
  **Delete the cache dir to rebuild after any extraction/prompt change.**
- **Context-budget sweep:** `EDUMEM_RETRIEVAL_E2E=1 python tools/retrieval_budget_sweep.py`
  (recall vs `EDUMEM_MAX_CONTEXT_CHARS`).

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

### Measurement vs regression-guard: TWO DIFFERENT TOOLS — do not confuse them

There are two pipelines that run the real system on BEAM 100K, and they answer
different questions. Picking the wrong one wastes a run.

| | **Full runner** (the MEASUREMENT) | **Live e2e test** (the REGRESSION GUARD) |
|---|---|---|
| Invoke | `python -m tools.evaluate_beam_end_to_end ...` (CLI, not pytest) | `EDUMEM_E2E=1 python -m pytest tests/test_beam_e2e_full.py` |
| Grading | **Live LLM judge** (`deepseek-v4-flash`), 0-to-1 partial credit | **Static expectations** (deterministic nugget/absence/order) |
| Output | Writes timestamped `results/<YYYYMMDD_HHMMSS>_<model>/` with `beam_e2e_results.json`, `beam_e2e_summary.json` (per-ability 0-1 scores), `paired_outcomes.jsonl`, `beam_question_validations.jsonl` | Prints pass/fail + a JSON artifact to a temp path |
| Use for | **Measuring whether a change moved recall/quality** (the 0.654 baseline in this file came from here). Diff a new run's `beam_e2e_summary.json` against a prior baseline. | Catching regressions fast (pass/fail, no judge latency, ~11-17 min) |
| Cost | Live judge round-trips; slower | No judge; faster |

**The e2e test's pass/fail (e.g. 10/20) is NOT the project's quality score.**
The quality score is the full runner's per-ability `avg_score` (0-1) and
`micro_overall`. Base fix-prioritization and "did the change help" decisions on
the **full runner**, and use the **e2e test** only as a fast regression guard
between changes. Mixing them up leads to optimizing a static guard instead of
real recall — a trap hit once already.

### Running the live e2e
Requires the reranker up on `http://localhost:3002/rerank` and the NAN LLM
endpoint. The API key lives in repo `.env` as `NAN_APY_KEY`. Each shell call is
fresh, so set env + run pytest in ONE command (Git Bash):
```
export EDUMEM_LLM_API_KEY="$(grep -E '^NAN_APY_KEY=' .env | head -1 | cut -d= -f2- | tr -d '\r\"')"
export EDUMEM_LLM_BASE_URL="https://api.nan.builders/v1"
export EDUMEM_E2E=1
export EDUMEM_E2E_ANSWER_MODEL="qwen3.6"
python -m pytest tests/test_beam_e2e_full.py -v -s
# (OPENROUTER_API_KEY / OPENROUTER_BASE_URL still work as deprecated fallbacks.)
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

## Known gaps / root causes (as of 2026-06-26, evidence-backed)

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

5. **TR context is anomalously short (6014 chars vs 16K budget).** In the
   3-conversation recall test, TR's mean context length was only 6014 chars —
   roughly 1/3 of the 16K sub-budget. The timeline/chrono specialists gate
   on `is_temporal`, which should fire for "how many weeks" / "between which
   dates" queries, but the actual populated rows in `memoria_timelines` may
   be filtering too aggressively. Possibly the anchor-date matching logic
   (`date LIKE ?` with month prefix) returns too few rows, or the question
   terms don't overlap with stored timeline descriptions. The short context
   suggests timeline/chrono are hitting empty fallback more often than expected.

6. **Multi-conversation cross-talk despite isolated session_ids.** With 3
   conversations (582 messages, 60 questions), overall recall dropped from
   0.415 (1-conv) to 0.334. Some of this is harder questions (each
   conversation has different difficulty), but PF/EO/MR/SUM all regressed
   meaningfully. The ABS/PF floor (0.000) persists even with the semantic
   specialist enabled — embeddings are populated in vec_facts but KNN doesn't
   bridge the paraphrase gap for these abilities.

CAVEATS for whoever continues:
- `recall_provenance.memories[].final_context_included` is UNRELIABLE — it showed
  `False` for messages the model demonstrably used. Do not trust it to decide what
  reached the prompt; instrument the actual assembled context instead.
- Do not claim versioning "fixed" CR/EO/TR from storage metrics alone — and note
  it can't help at all until gap #1 is addressed.
- The `server_nvidia.py` GPU memory limit (`NVIDIA_GPU_MEM_LIMIT`, default 4GB
  per session with `arena_extend_strategy: kSameAsRequested`) is critical — the
  ONNX Runtime CUDA EP pre-allocates a huge arena by default, consuming 23.7/24.5GB
  for two small models. Restart the container if GPU memory is exhausted.

## Retrieval-recall benchmark analysis (2026-06-24)

### What was done

Added **cached DB** (`tests/.beam_recall_cache/beam_100K_conv0.db`, later
`beam_100K_x3.db` for 3 conversations with isolated session_ids) to avoid
re-ingesting. The test fixture builds once **with `use_cloud=True`** (real
qwen3.6 facts+conclusions), reuses on subsequent runs. Delete
`tests/.beam_recall_cache/` to force a rebuild. Each conversation gets its own
session_id (e.g. `retrieval-recall-cache-conv0`) to prevent fact cross-contamination.
Also monkey-patched `_embed_api` with `requests.Session` connection pooling. The
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
**0.347 overall recall** (passes 0.30 gate). The 3-conversation cache (vs 1)
produced **0.334 overall** across 60 questions. Per-ability: KU 0.833, TR 0.750,
IE 0.500, EO 0.289, SUM 0.261, CR 0.250, IF 0.250, MR 0.125, PF 0.083, ABS 0.000.

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
