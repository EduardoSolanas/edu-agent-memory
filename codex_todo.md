# BEAM 100K Low-Score Fix Plan

## Implementation status (2026-06-19)

- [x] FIX-01 through FIX-18 are implemented with regression coverage.
- [x] Pure recall is label-free and does not use raw-conversation answer bypasses.
- [x] Official-compatible, partial-credit, macro, and micro scores are separate.
- [x] Stored answers can be re-judged without re-ingestion or re-answering.
- [x] The full Python suite passes: `136 passed, 2 skipped` when live services are not configured.
- [x] The Node integration suite passes: `4 passed`.
- [x] The live OpenVINO/container suite passes: `10 passed` against image `41f066997619`.
- [x] The image uses CPU-only Torch and contains no NVIDIA/CUDA packages.
- [ ] Run the five-conversation phase gate and compare fixed QIDs before/after.
- [ ] Run the full-scale release gate and preserve its final benchmark artifacts.

The unchecked phase/release gates require paid live-model benchmark runs; they
are validation work, not missing implementation.

> Scope: fix confirmed benchmark, retrieval, ingestion, and judging defects that
> affect the BEAM 100K run. Do not add benchmark-answer oracles, answer-key
> matching, or unrelated architecture.

## Assumptions

- "One case" means `--sample 1`: one conversation, with up to 50 probing
  questions.
- The answering model and judge are both `qwen3.6` through the NAN provider.
- Pure-recall mode remains the required default.
- Published-score compatibility and corrected experimental scoring must be
  reported separately.

## Current evidence

- The repository's recent runs are approximately 26.8%-43.6% OVERALL.
- `--sample 1` always selects dataset row 0; it is not random sampling.
- The local test suite currently passes: `106 passed`.
- No local `results/beam_e2e_results.json` is available, so per-question failure
  attribution still requires a retained benchmark artifact.
- A local reproduction confirmed duplicate structured extraction. A fact from
  message index 1 produced structured rows with indexes `[0, 0, 1]`.

## Definition of done

- [ ] Every implementation change follows Red -> Green -> Refactor.
- [ ] Tests use real functions, databases, and services; no mocks, stubs, or
  test doubles.
- [ ] `python -m pytest -q` passes in full.
- [ ] A fixed set of BEAM conversations is evaluated before and after each
  phase.
- [ ] Results retain answers, judge responses, recall provenance, component
  health, configuration, and per-ability scores.
- [ ] Official-compatible and corrected scoring are never mixed in one number.
- [ ] No ground-truth ability label or raw-conversation bypass influences a
  pure-recall answer.

---

## P0 - Make the benchmark configuration real and observable

### FIX-01 - Connect the shipped reranker

**Problem**

- `tools/evaluate_beam_end_to_end.py:1784` defaults to
  `http://localhost:8000/rerank`.
- `server.py:460` starts the shipped inference server on port `3002`.
- `benchmarks/run_beam_official.py:93-106` does not set
  `EDUMEM_RERANKER_URL`.
- `_rerank()` catches every error and silently returns the raw top-N list.

**Red**

- [ ] Add a test proving the official runner resolves the reranker URL to the
  configured inference server.
- [ ] Add a test proving reranker failure is recorded in run diagnostics rather
  than silently treated as success.
- [ ] Add a real HTTP integration smoke test against the shipped `/rerank`
  service in the environment that runs BEAM.

**Green**

- [ ] Set `EDUMEM_RERANKER_URL=http://localhost:3002/rerank` in the official
  runner, unless explicitly overridden.
- [ ] Record reranker calls, successes, failures, and fallback count in result
  metadata.
- [ ] Print one preflight line showing the endpoint and health status.

**Acceptance**

- [ ] Preflight fails clearly, or requires an explicit `--allow-no-reranker`,
  when reranking is expected but unavailable.
- [ ] At least one benchmark query records reranker scores.
- [ ] No reranker exception disappears without a diagnostic counter.

### FIX-02 - Enable and verify dense embeddings

**Problem**

- `edumem/core/embeddings.py:130-131` falls back to no embedding model when
  `fastembed` is unavailable.
- `setup.md:60-63` does not install `fastembed`.
- The official runner does not configure the shipped embedding endpoint.
- A successful run can therefore be keyword-only without saying so.

**Red**

- [ ] Add a real preflight test that fails when neither local nor API embeddings
  are available for a BEAM run.
- [ ] Add an ingestion test proving `memory_embeddings` receives one vector per
  working-memory message.
- [ ] Add a semantic recall test where question and evidence are paraphrases
  with no important exact-token overlap.

**Green**

- [ ] Configure the shipped embedding endpoint in
  `benchmarks/run_beam_official.py`, including model name and dimension.
- [ ] Update setup documentation with the single supported benchmark embedding
  path.
- [ ] Persist embedding backend, model, dimension, inserted-vector count, and
  query-vector count in results metadata.

**Acceptance**

- [ ] `memory_embeddings` count equals eligible working-memory row count after
  ingest.
- [ ] The semantic paraphrase test retrieves the correct evidence in top 10.
- [ ] Results identify the run as dense, keyword-only, or failed; never
  "unknown".

### FIX-03 - Admit vector-only working-memory hits

**Problem**

- `edumem/core/beam.py:4931-4936` admits a working-memory candidate using
  lexical relevance even when vector search found it.
- A strong semantic candidate with zero lexical overlap is discarded before
  `vec_sim` is blended at lines 4943-4946.
- This makes dense working-memory retrieval behave mainly as lexical reranking.

**Red**

- [ ] Add a deterministic scoring test: lexical relevance `0`, strong vector
  similarity, candidate must be admitted.
- [ ] Add the inverse test: weak vector similarity and no lexical overlap must
  still be rejected.
- [ ] Add a real SQLite recall test using stored vectors and paraphrased text.

**Green**

- [ ] Give working-memory vector-only candidates an explicit similarity floor,
  matching the simple episodic rule at `beam.py:5381-5386`.
- [ ] Keep the existing lexical threshold for FTS-only candidates.
- [ ] Do not change unrelated score weights in the same patch.

**Acceptance**

- [ ] Strong vector-only evidence survives filtering and reaches the final
  top-K.
- [ ] Exact-token regression tests retain their previous ranking.

---

## P0 - Repair structured-memory data integrity

### FIX-04 - Remove duplicate structured extraction

**Problem**

- `BeamMemory.remember_batch()` extracts structured facts at
  `edumem/core/beam.py:2944`.
- The harness extracts the same raw messages again at
  `tools/evaluate_beam_end_to_end.py:653-660`.
- ISO dates are also duplicated inside tagged content because the original date
  and `[DATES: ...]` copy are both parsed.
- Tables have no uniqueness protection for these repeated writes.

**Red**

- [ ] Add an ingest test with one metric, date, timeline event, instruction, and
  preference.
- [ ] Assert exactly one canonical structured row per extracted item.
- [ ] Assert repeated ingestion follows the intended idempotency policy.

**Green**

- [ ] Choose one extraction owner. Prefer `remember_batch()` so all callers get
  consistent behavior.
- [ ] Remove the harness's second extraction pass.
- [ ] Derive benchmark extraction counts from database deltas or shared return
  data without re-running extraction.
- [ ] Ignore synthetic `[DATES]`, `[DURATIONS]`, `datetokens`, and `[MSGIDX]`
  metadata during extraction, or deduplicate within the extraction call.

**Acceptance**

- [ ] The reproduction that currently yields three date rows yields one.
- [ ] No structured table contains duplicate `(source_memory_id, semantic
  payload)` rows after one ingest.

### FIX-05 - Correct message-index propagation

**Problem**

- `edumem/core/beam.py:3903` uses
  `r'\[MSGIDX:(\\d+)\]'`, which matches a literal `\d`, not digits.
- Internal batch extraction consequently records structured rows at
  `message_idx=0`.
- Incorrect indexes corrupt EO ordering, KU evolution, and preference history.

**Red**

- [ ] Add a test that `[MSGIDX:42]` produces `message_idx == 42` in every
  structured table.
- [ ] Add a two-message test proving indexes remain `[0, 1]`, without duplicate
  zero-index rows.

**Green**

- [ ] Correct the regex to `r'\[MSGIDX:(\d+)\]'`.
- [ ] Prefer passing the existing `message_index` field through batch metadata
  rather than reparsing content where this can be done surgically.

**Acceptance**

- [ ] Facts, timelines, instructions, preferences, and KG rows preserve their
  source message index.
- [ ] EO structured recall returns monotonically ordered, non-duplicated
  message indexes.

### FIX-06 - Keep assistant text out of user preference/instruction memory

**Problem**

- `tools/evaluate_beam_end_to_end.py:603-607` tags instructions and preferences
  without checking message role.
- Core structured extraction also processes all roles and may treat assistant
  recommendations as user intent.
- Strategy 6 later retrieves these rows as authoritative user constraints.

**Red**

- [ ] Add a conversation containing an assistant recommendation and a conflicting
  user preference.
- [ ] Assert only the user's statement is stored as a user preference or
  instruction.
- [ ] Preserve factual extraction from assistant content where appropriate.

**Green**

- [ ] Pass role/source into structured extraction.
- [ ] Limit user preference and user instruction extraction to user-authored
  messages.
- [ ] Keep role-neutral metrics, dates, and entities unchanged.

**Acceptance**

- [ ] IF/PF context contains no assistant-authored preference presented as the
  user's preference.

### FIX-17 - Audit consolidation coverage before changing its call path

**Problem**

- The consolidation call path exists, but low graph-fact extraction coverage
  can look like a consolidation failure, especially for first-person statements
  such as `I use PostgreSQL`.
- Adding another consolidation call would duplicate work instead of identifying
  which stage lost the fact.

**Red**

- [ ] Add real-ingest cases covering equivalent first-person and third-person
  facts, plus one fact that supersedes an earlier value.
- [ ] Assert diagnostics expose graph-fact, consolidated-fact, and supersession
  row deltas for each case.
- [ ] Assert one ingest does not create duplicate consolidated or supersession
  rows.

**Green**

- [ ] Record extraction-to-graph, graph-to-consolidated, and supersession counts
  in benchmark diagnostics, including source message role and index.
- [ ] Use the existing canonical consolidation path only; do not add a second
  `consolidate_fact()` call in the harness.

**Acceptance**

- [ ] Every missing consolidated fact is attributable to extraction, graph
  write, consolidation, or supersession.
- [ ] First-person coverage is reported explicitly, and diagnostic collection
  creates no additional memory rows.

---

## P1 - Reduce retrieval noise and benchmark leakage

### FIX-18 - Add per-strategy recall telemetry before trigger tuning

**Problem**

- MR, IF, PF, and TR retrieval branches can fail to activate or contribute no
  candidates, but aggregate recall does not distinguish those cases.
- Changing trigger regexes without branch-level evidence risks broadening noisy
  retrieval.

**Red**

- [ ] Add real retrieval cases that should activate MR, IF, PF, and TR, plus a
  neutral factual query that should not activate IF/PF/TR.
- [ ] Assert provenance records branch activation and candidate contribution
  counts for every strategy.

**Green**

- [ ] Record, per query and strategy, whether the branch activated and how many
  candidate IDs it contributed before and after merge/deduplication.
- [ ] Retain the telemetry in diagnostic output without changing triggers,
  regexes, ranking weights, or final context.

**Acceptance**

- [ ] Every query explains which of MR/IF/PF/TR ran and which contributed final
  candidates.
- [ ] Trigger or regex changes are proposed only from fixed-case telemetry, and
  telemetry collection does not change retrieved results.

### FIX-07 - Retrieve instructions/preferences only when relevant

**Problem**

- `tools/evaluate_beam_end_to_end.py:1116-1126` injects up to 10 arbitrary
  instructions and 10 arbitrary preferences for every question.
- These entries receive score `0.8`, so they can crowd out relevant factual
  evidence.

**Red**

- [ ] Add a factual IE query and assert unrelated IF/PF memories are absent.
- [ ] Add genuine IF and PF queries and assert relevant tagged evidence is
  present.

**Green**

- [ ] Gate tag retrieval using label-free query semantics.
- [ ] Apply query terms to instruction/preference retrieval; do not select the
  first ten rows globally.
- [ ] Preserve pure-recall constraints.

**Acceptance**

- [ ] Non-IF/PF questions contain zero arbitrary tag-only injections.
- [ ] IF/PF recall improves without reducing IE/KU recall coverage.

### FIX-08 - Remove ground-truth ability routing from pure recall

**Problem**

- The main recall call intentionally passes `ability=None` at
  `tools/evaluate_beam_end_to_end.py:1933`.
- The harness then passes the dataset ability label into `memoria_retrieve()` at
  lines 1941 and 2203.
- This is an ability-label oracle, even though it may increase rather than lower
  the score.

**Red**

- [ ] Add a pure-recall test that fails if dataset ability reaches retrieval or
  answer construction.
- [ ] Add label-invariance coverage: changing the dataset label must not change
  retrieved memories for the same question.

**Green**

- [ ] Pass `ability=None` to MEMORIA during pure-recall runs.
- [ ] Use the existing label-free classifier inside `BeamMemory` when specialist
  routing is needed.
- [ ] Keep the ability label only for selecting the official evaluation metric
  after the answer is produced.

**Acceptance**

- [ ] Pure-recall answers and provenance are identical when only the stored
  ability label changes.

### FIX-09 - Replace arbitrary ranking metadata

**Problem**

- Every message receives the same timestamp at
  `tools/evaluate_beam_end_to_end.py:613`.
- Importance cycles from `0.3` to `0.7` based solely on index modulo five at
  line 612.
- Importance contributes directly to recall score, creating periodic ranking
  noise unrelated to relevance.

**Red**

- [ ] Add a test proving changing a message's index modulo five does not change
  semantic ranking.
- [ ] Add a test proving message order remains available through
  `message_index` without fake timestamp or importance variation.

**Green**

- [ ] Use a constant neutral importance for benchmark messages unless BEAM
  provides real importance metadata.
- [ ] Preserve mention order via `message_index`.
- [ ] Use actual dataset timestamps only if present; otherwise document the
  neutral timestamp policy.

**Acceptance**

- [ ] Reindexing the same conversation does not alter non-ordering recall scores.

### FIX-10 - Support natural-language dates consistently

**Problem**

- Ingest datetokens at `tools/evaluate_beam_end_to_end.py:590-595` support only
  ISO dates.
- The pure-recall temporal cheatsheet at lines 1671-1725 also supports only ISO
  dates.
- BEAM commonly uses named and relative dates.

**Red**

- [ ] Add cases for `March 15`, `Mar 15`, `15 March`, ordinals, and ISO dates.
- [ ] Add relative-date tests only where an explicit reference date exists.
- [ ] Assert ingest tokens and cheatsheet extraction use the same parser.

**Green**

- [ ] Introduce one small shared date-extraction function.
- [ ] Use it for ingest tagging and recalled-memory cheatsheets.
- [ ] Do not infer a year or resolve a relative date without an explicit anchor.

**Acceptance**

- [ ] More than 80% of sampled TR questions with date evidence produce a
  non-empty recalled temporal reference.
- [ ] TR improves on the fixed case set without raw-conversation access.

---

## P1 - Make judging and score reporting honest

### FIX-11 - Add a separate judge model to the official runner

**Problem**

- `tools/evaluate_beam_end_to_end.py` supports `--judge-model`.
- `benchmarks/run_beam_official.py` exposes only `--model` and therefore forces
  answer and judge to use the same Qwen model.

**Red**

- [ ] Add command-construction tests for default same-model behavior and an
  explicit separate judge.
- [ ] Assert result metadata contains both exact model identifiers.

**Green**

- [ ] Add `--judge-model` to the runner and forward it to the evaluator.
- [ ] Add separate judge provider/base URL/key only if required by the actual
  selected provider; avoid a general provider abstraction.

**Acceptance**

- [ ] A run can use Qwen for answers and a separately specified judge.
- [ ] Re-running stored answers with another judge does not require re-ingest or
  re-answering.

### FIX-12 - Separate official and corrected partial-credit scoring

**Problem**

- The official BEAM grader invites scores `1.0`, `0.5`, or `0.0`.
- Its non-EO evaluators use `int(response['score'])`, converting `0.5` to `0`.
- Silently fixing this would break comparability; silently accepting it hides a
  harsh, model-dependent scoring artifact.

**Red**

- [ ] Add a fixture with rubric scores `[1.0, 0.5, 0.0]`.
- [ ] Assert official-compatible score is `1/3`.
- [ ] Assert corrected partial-credit score is `0.5`.
- [ ] Assert outputs carry an explicit scoring-mode identifier.

**Green**

- [ ] Preserve the upstream official score unchanged as `official_score`.
- [ ] Compute a separate float mean from retained judge responses as
  `partial_credit_score`.
- [ ] Never overwrite or relabel one as the other.

**Acceptance**

- [ ] Reports display both scores with clear labels.
- [ ] Published comparisons use only the official-compatible field.

### FIX-13 - Make judge failures inspectable

**Problem**

- Qwen JSON formatting, empty content, reasoning-only responses, and API errors
  can become zeros or fatal grader failures without enough retained evidence.
- The current result truncates `ai_answer` to 500 characters and does not retain
  the complete raw judge response in a uniform field.

**Red**

- [ ] Add real parser tests for pristine JSON, fenced JSON, prose-wrapped JSON,
  empty content, and invalid JSON.
- [ ] Assert parse/API failures are classified separately from valid score zero.

**Green**

- [ ] Retain full answers and raw judge payloads in a diagnostic artifact.
- [ ] Record finish reason, response-content presence, parse status, retry count,
  and API error class without storing secrets.
- [ ] Keep the compact main results file if desired, but link it to diagnostics
  by QID.

**Acceptance**

- [ ] Every zero score can be classified as valid judgment, retrieval/answer
  failure, judge API failure, or judge parse failure.

### FIX-14 - Use one OVERALL definition in console and report

**Problem**

- Conversation output at `tools/evaluate_beam_end_to_end.py:3035-3041` reports a
  micro-average over questions.
- Final output at lines 2681-2715 reports a macro-average over abilities.
- Both are labelled as overall accuracy.

**Red**

- [ ] Add unequal-question-count coverage showing different micro and macro
  values.
- [ ] Assert the console's primary OVERALL equals the final report's OVERALL.

**Green**

- [ ] Use macro-average as the primary BEAM OVERALL everywhere.
- [ ] Optionally show micro-average as a separately labelled diagnostic.

**Acceptance**

- [ ] Console, summary JSON, and final report agree on `OVERALL`.

---

## P2 - Make sampling and comparisons reproducible

### FIX-15 - Replace misleading `--sample 1` behavior

**Problem**

- `tools/evaluate_beam_end_to_end.py:360-363` takes the first N dataset rows.
- There is no seed, explicit case selector, or confidence interval.
- One conversation is not a stable estimate of the scale score.

**Red**

- [ ] Add tests for deterministic `--case-index`, ordered `--start-index`, and
  seeded sampling if seeded sampling is retained.
- [ ] Assert selected conversation IDs are saved in metadata.

**Green**

- [ ] Add the simplest explicit selector needed for debugging one case.
- [ ] Keep full-scale evaluation as the release criterion.
- [ ] Rename console text from "Sample Size" to "Conversation Count".

**Acceptance**

- [ ] The same selector always evaluates the same conversation IDs.
- [ ] Release claims use all conversations at the target scale, or clearly state
  the subset and uncertainty.

### FIX-16 - Remove invalid direct-SOTA claim

**Problem**

- `tools/evaluate_beam_end_to_end.py:2797` says direct comparison is valid.
- A one-conversation subset, different answering model, different judge, and
  ambiguous scoring mode are not directly comparable.

**Red**

- [ ] Add report tests that suppress the direct-comparison claim unless dataset
  coverage, scoring mode, answer model, and judge protocol match the declared
  baseline.

**Green**

- [ ] Report comparison prerequisites explicitly.
- [ ] Label partial/subset runs as diagnostics, not SOTA evaluations.

**Acceptance**

- [ ] A `--sample 1` report cannot print "Direct comparison valid".

---

## Corrections to the existing `TODO.md`

- [ ] Re-check RC1 before implementing its proposed FIX-1.
  `remember_batch()` currently calls `_ingest_graph_and_veracity()` at
  `edumem/core/beam.py:2935`, which calls `consolidate_fact()` at line 3002 when
  graph facts exist. The accurate question is whether extraction produced facts
  and whether consolidation succeeded, not whether the call path exists.
- [ ] Replace the proposed EO prefix-dedup diagnosis with the confirmed duplicate
  extraction and broken MSGIDX propagation defects.
- [ ] Keep the verified ISO-only temporal diagnosis.
- [ ] Do not treat `ABS=100%` alone as proof that the answer model is healthy;
  judge behavior and missing evidence can both inflate abstention.

---

## Implementation order

1. [ ] FIX-13: retain enough diagnostics to explain every failure.
2. [ ] FIX-17 and FIX-18: establish consolidation and recall-strategy baselines.
3. [ ] FIX-04 and FIX-05: stop duplicate/corrupt structured writes.
4. [ ] FIX-01, FIX-02, and FIX-03: activate real reranking and semantic recall.
5. [ ] FIX-06 through FIX-10: remove retrieval noise, leakage, and temporal gaps.
6. [ ] FIX-11, FIX-12, and FIX-14: make judge selection and reporting explicit.
7. [ ] FIX-15 and FIX-16: run reproducible samples and honest comparisons.

Each numbered fix should be a separate commit or separately revertible patch.
Do not combine score-weight tuning with correctness fixes.

## Validation matrix

### Fast gate after every fix

```powershell
python -m pytest -q
```

- [ ] Run one pinned conversation.
- [ ] Compare the same QIDs before/after.
- [ ] Inspect per-ability deltas, not only OVERALL.
- [ ] Confirm no new API, parse, reranker, or embedding failures.

### Phase gate

- [ ] Run at least five fixed 100K conversations.
- [ ] Report macro OVERALL, micro diagnostic, per-ability means, and question
  counts.
- [ ] Report official-compatible and partial-credit scores separately.
- [ ] Compare recall hit rate and judge success rate independently.

### Release gate

- [ ] Evaluate the full target-scale split.
- [ ] Repeat judging or use a fixed deterministic judge configuration.
- [ ] Preserve result JSON, diagnostic JSONL, environment snapshot, commit SHA,
  selected conversation IDs, answer model, judge model, and endpoint health.
- [ ] Do not claim improvement unless the fixed QID set and scoring mode match.

## Required result diagnostics

For every question, retain:

- [ ] QID, conversation ID, scale, and ability used only for evaluation.
- [ ] Full question, ideal answer, rubric, and full generated answer.
- [ ] Retrieved memory IDs, content hashes, sources, message indexes, scores,
  dense/FTS/reranker components, and final context inclusion.
- [ ] Answer model and judge model identifiers.
- [ ] Raw judge response, parsed rubric scores, official score, corrected
  partial-credit score, and parse status.
- [ ] Embedding and reranker health/call counters.
- [ ] Timing and API error classification.

These diagnostics are required to distinguish retrieval failure, answer failure,
and judge failure instead of guessing from the aggregate score.
