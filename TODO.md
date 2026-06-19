# BEAM Benchmark — Score Improvement TODO

> Verified against tree **`010d111`** (2026-06-18). All proposed fixes are
> **generic and pure-recall-legal** — no ability-label oracles, no
> `conversation_messages` bypasses, no per-question special-casing. Each root
> cause cites the exact file:line that was inspected.

## Context

The new commits since the prior baseline (`010d111` IF/PF/MR/CR/EO,
`c38d525` conflict resolver + TR cheatsheet, `d540d7d` pure-recall TR/EO/CR,
`340914b` macro-average OVERALL) are **architecturally sound** — unlike the
old harness oracles, they all run inside pure-recall mode. They patch real
symptoms but **do not fix the underlying recall surface**, so scores stay low.

### The diagnostic signature

```
ABS=100% across all 4 runs; everything else 15–83%.
```

**This is the textbook signature of a retrieval problem, not an answering
problem.** ABS=100 means the LLM correctly abstains when it finds no evidence —
but it also abstains (or hallucinates) when evidence *was* in the corpus but was
not retrieved. High ABS + low everything-else ⇒ the reasoner is fine; the
memories handed to it are wrong or missing.

## Current scores (latest 4 runs)

| Run | ABS | CR | EO | IE | KU | MR | SUM | TR | OVERALL |
|-----|-----|-----|------|-----|-----|-----|-----|-----|---------|
| 1 | 100 | 25.0 | 21.2 | 50.0 | 50.0 | 12.5 | 40.0 | 50.0 | 43.6 |
| 2 | 100 | 0.0 | 15.2 | 83.3 | 50.0 | 0.0 | 41.7 | 0.0 | 36.3 |
| 3 | 100 | 12.5 | 14.6 | 50.0 | 0.0 | 0.0 | 12.5 | 25.0 | 26.8 |
| 4 | 100 | 12.5 | 27.0 | 0.0 | 50.0 | 0.0 | 50.0 | 50.0 | 36.2 |

**EO (15–27%) is near the Kendall tau-b chance floor (~20–30%)** ⇒ ordering
reconstruction is effectively broken. **TR swings 0–50% run-to-run** ⇒ no stable
date anchor. **CR/MR/MR/SUM vary wildly** ⇒ retrieval instability.

---

## Verified root causes

### RC1 — Write-time conflict resolver never runs at ingest  → CR, KU, MR
- Harness ingest calls `beam.remember_batch()` then
  `beam.extract_and_store_facts()` (`tools/evaluate_beam_end_to_end.py:650,660`).
  `extract_and_store_facts` writes to `memoria_*` / `facts` tables — **not** to
  `consolidated_facts`.
- The new supersession logic (`_llm_resolve_conflict` + `superseded_by`,
  `veracity_consolidation.py:599`) only fires inside `consolidate_fact()`,
  reached via `_ingest_graph_and_veracity()` — **never called during BEAM
  ingest**.
- **Consequence:** `consolidated_facts.superseded_by` is empty for every fact,
  so the new CR "retrieve non-superseded facts" strategy (`010d111`) finds
  nothing. CR stuck at 0–25%.

### RC2 — Temporal cheatsheet + datetok are ISO-only  → TR
- `_inject_temporal_cheatsheet` matches only `r'\d{4}-\d{2}-\d{2}'`
  (`tools/evaluate_beam_end_to_end.py:1692`). Datetok ingest tokens
  (`tools/evaluate_beam_end_to_end.py:594`) are also built from ISO dates only.
- BEAM conversations date events mostly in natural language ("March 15",
  "next Tuesday", "two weeks from the sprint start"). The cheatsheet therefore
  stays empty for most TR questions even though the dates exist in the corpus.
- **Consequence:** TR swings 0–50% with no stable anchor.

### RC3 — EO dedup is prefix-aware, not MSGIDX-aware  → EO
- `_add_unique` dedups on `mem.get("content", "")[:80]`
  (`tools/evaluate_beam_end_to_end.py:958`). Every message is prefixed
  `[MSGIDX:N]`, and the same speaker's messages share opening words, so the
  prefix crowds out distinguishing content.
- Dedup runs *before* the `message_index` sort (line 1962), so a later mention
  of a topic can be dropped as a "duplicate" → corrupted ordering → tau-b
  collapses to chance.

### RC4 — Episodic memory is populated by the lossy AAAK fallback
- When no LLM consolidation runs (the BEAM case), `sleep()`/episodic writes fall
  back to AAAK shorthand, which destroys entity names, numbers, and dates.
- Most CR/MR/KU/TR questions depend on episodic content; once entities/dates are
  mangled at consolidation time, no recall depth recovers them.

### RC5 (informational) — OVERALL methodology changed at `340914b`
- OVERALL is now a macro-average of abilities. Correct for leaderboard parity,
  but means CR=0 / MR=0 / TR=0 now count full weight. **No code change** — just
  don't compare new OVERALL to old OVERALL.

---

## Proposed fixes (priority order)

### FIX-1 (P0) — Wire consolidation into the BEAM ingest path
**Addresses:** RC1, RC4 → CR, KU, MR

- [ ] In the harness ingest loop (`tools/evaluate_beam_end_to_end.py:660`),
      after `extract_and_store_facts`, feed the same extracted facts into
      `beam.veracity_consolidator.consolidate_fact(...)` (the consolidator
      already accepts `llm_client`; `BeamMemory` stores `_llm_client` from
      `c38d525`).
- [ ] Gate behind a flag (`EDUMEM_BEAM_CONSOLIDATE=1`) so it can be A/B tested.
- [ ] Add a one-line probe print confirming `consolidated_facts.superseded_by`
      is non-empty after ingest.
- [ ] **Acceptance:** CR > 40% on a pinned seed; supersession rows > 0.

### FIX-2 (P0) — Broaden date extraction beyond ISO
**Addresses:** RC2 → TR

- [ ] Add `_extract_dates(text) -> list[str]` supporting ISO, long/short month
      ("March 15", "Mar 15", "15 March"), ordinals ("March 15th"), and relative
      anchors ("next Tuesday", "two weeks from…") resolved against the
      message's own timestamp where available.
- [ ] Use it in **both** places: datetok ingest tokens
      (`tools/evaluate_beam_end_to_end.py:594`) and the cheatsheet
      (`tools/evaluate_beam_end_to_end.py:1692`).
- [ ] **Acceptance:** TR > 50% stable across seeds; cheatsheet non-empty for
      >80% of TR questions in a probe run.

### FIX-3 (P0) — Make EO dedup MSGIDX-aware
**Addresses:** RC3 → EO

- [ ] In `_add_unique` (`tools/evaluate_beam_end_to_end.py:956`), use dedup key
      `(message_index, normalized_content[:40])` when `message_index` is
      present; fall back to current behavior otherwise.
- [ ] Ensure the `message_index` sort (line 1962) is stable with recall score as
      a secondary key so ties don't reshuffle.
- [ ] **Acceptance:** EO > 45% (above the ~30% chance floor) on a pinned seed.

### FIX-4 (P1) — Probe, then broaden IF/PF/MR triggers
**Addresses:** validate `010d111` gains (MR=0 in runs 2/3)

- [ ] Add a probe: print whether `_multi_strategy_recall` expanded queries
      (MR branch hit) and whether tagged `[INSTRUCTION]`/`[PREFERENCE]` memories
      were returned, for a sample of MR/IF/PF questions.
- [ ] If the branch isn't hit, broaden the detection regex (generic, not
      answer-key-based).
- [ ] **Acceptance:** MR > 30% on a pinned seed; MR expansion fires for >70% of
      MR questions.

### FIX-5 (low) — Document the OVERALL methodology boundary
**Addresses:** RC5 — doc only

- [ ] Note in the run report header that OVERALL changed methodology at
      `340914b`. Compare per-ability averages across runs, not OVERALL, when one
      run predates the change.

---

## Validation plan

1. [ ] Pin a fixed `--seed` so per-ability deltas are attributable to code, not
       sampling.
2. [ ] Apply FIX-1 and FIX-2 first (highest leverage), re-run, compare
       per-ability.
3. [ ] Add FIX-3, re-run.
4. [ ] Expected direction: CR↑, TR↑, EO↑; ABS may drop from 100% toward 85–95%
       (healthy — evidence now found and answered rather than abstained on).

## Investigation tasks (carry-forward)

- [ ] Get a local `beam_e2e_results.json` (results dir is gitignored; benchmark
      runs in Docker). Inspect actual per-question answers + recall provenance to
      confirm the diagnoses with data.
- [ ] Check the metadata env snapshot for any silent `EDUMEM_*` toggle that's off
      (reranker URL, LLM consolidation) that could explain a category gap.

## Notes on methodology

- Pure-recall is the enforced default (preflight at `evaluate_beam_end_to_end.py`
  refuses to run otherwise). Any fix must work **through `BeamMemory.recall()`**
  or the ingest layer — not as a harness-side oracle.
- OVERALL is a **macro-average across abilities** (`compute_ability_scores`),
  so a category at 0% drags the whole number disproportionately.
- The new commits' fixes all survive pure-recall mode (verified via the
  `_pure_recall` gate map at `evaluate_beam_end_to_end.py:1833,1840,1894,1975,
  2042,2141`).

## Non-goals (explicitly excluded)

- No ability-label oracles in the harness.
- No reading `conversation_messages` to answer questions in pure-recall mode.
- No per-question special-casing / answer-key matching.
