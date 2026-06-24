# sum_memory.md — write-time rolling-summary track (for SUM)

**Audience:** a cheap implementation agent (haiku). Follow this exactly. It encodes
the design AND the pitfalls we already paid for (bounded output, flag-gated,
graceful fallback). Do not improvise the schema or skip the bounds.

---

## 1. Goal
Lift **summarization (SUM)** recall. Today a SUM query retrieves ~10 atomic facts
and the answer model must reconstruct a narrative from fragments — it scores
~0.3-0.4 in the case-0 judged A/B. The leaders (Mem0 rolling summary, Honcho
synthesized representation) DO NOT reconstruct at query time: they **build
summaries at write time** and read a coherent object back. We add the same: a
flag-gated write-time **segment-summary track**, fused as one more recall source.

## 2. What we already learned (read before coding)
- **Bounded output is non-negotiable.** A prior unbounded LLM prompt emitted
  multi-thousand-token essays that truncated (`finish_reason=length`) → invalid
  output → one call hung the whole run. Every LLM call here MUST set `max_tokens`,
  a per-call timeout, and fall back gracefully (see `extract_and_store_facts` in
  `edumem/core/beam.py` for the existing pattern: ThreadPoolExecutor timeout +
  `EDUMEM_EXTRACTION_TIMEOUT`, try/except → regex path).
- **The graph is NOT the answer for SUM.** It lifts fact abilities (KU/MR) but a
  graph gives connected facts, not a progression narrative. SUM needs its own track.
- **Default OFF.** Like `EDUMEM_LLM_EXTRACTION`, gate this behind
  `EDUMEM_LLM_SUMMARY` (default `"0"`) so prod/default behavior and the fast
  offline test suite stay unchanged. The LLM call is only exercised by the gated
  live e2e.

## 3. Design (KISS; copy the existing MEMORIA patterns)
1. **Storage:** new table `memoria_summaries`. One row per summarized segment.
2. **Write side (segment summaries):** every `EDUMEM_SUMMARY_SEGMENT` messages
   (default 20), LLM-summarize that segment of the conversation into a compact
   narrative and store it. Bounded: `max_tokens=256`, timeout, fallback to
   no-op (never lose ingestion, never hang).
3. **Read side:** a specialist `_memoria_summary_retrieve(query, top_k)` returns
   summary rows; added to `_memoria_fused_retrieve`'s specialist list (flag-gated)
   so RRF fuses summaries alongside fact/timeline/negation/chrono/kg. **Recall
   shape does not change** — one more source.
4. **Pure store/retrieve split:** the LLM call and the store are separate. The
   store (`_store_summary`) and retrieve are PURE (no LLM) — those are the TDD
   targets; the LLM call is gated + live-e2e only.

## 4. Implementation (file: `edumem/core/beam.py`)

### 4a. Schema — add to `init_beam(...)`
```sql
CREATE TABLE IF NOT EXISTS memoria_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT 'default',
    seg_start INTEGER,          -- first message_idx in the segment
    seg_end INTEGER,            -- last message_idx in the segment
    summary TEXT NOT NULL,
    source TEXT DEFAULT 'llm_summary',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON memoria_summaries(session_id, seg_end);
```
Guard any new write-only DDL the same way the `wm_au` trigger is now guarded
(re-running init_beam on an existing DB must issue no write — use
`CREATE ... IF NOT EXISTS`, which is a no-op when present).

### 4b. `_should_summarize(self, message_idx, segment_size) -> bool` (PURE)
Return `True` exactly at segment boundaries: `(message_idx + 1) % segment_size == 0`.
No I/O. **TDD target.**

### 4c. `_segment_text(self, session, seg_start, seg_end) -> str` (PURE given DB)
Pull the raw messages for `[seg_start, seg_end]` from existing message storage
(`working_memory` / `episodic_memory` by `message_index`), concatenated as
`role: content` lines. Read the actual column names before querying.

### 4d. `_build_summary_prompt(self, segment_text, prior_summary=None) -> str`
Bounded prompt. MUST instruct: return ONLY a compact narrative summary, **<= 150
words**, no prose preamble, no markdown. If `prior_summary` is given, instruct to
**update** it with the new segment (hierarchical/rolling), not restate. **TDD
target:** assert the prompt embeds the word cap and asks for narrative-only.

### 4e. `_store_summary(self, session, seg_start, seg_end, summary, source='llm_summary') -> bool` (PURE)
Insert one `memoria_summaries` row; return True on insert. No LLM call.
**Primary TDD target.**

### 4f. `_maybe_summarize_segment(self, session, message_idx)` — write-time hook
Called from the ingest/write path. When `self._llm_client is not None and
os.environ.get("EDUMEM_LLM_SUMMARY","0")=="1"` and `_should_summarize(...)`:
- `seg_end = message_idx`, `seg_start = message_idx - segment_size + 1` (clamp >=0).
- `text = _segment_text(...)`; if empty, return.
- Optionally load the most recent prior summary for rolling update.
- Call `self._llm_client.chat([{user: prompt}], temperature=0.0, max_tokens=256)`
  wrapped in the SAME ThreadPoolExecutor timeout pattern as
  `extract_and_store_facts` (`EDUMEM_EXTRACTION_TIMEOUT`).
- On any exception / empty / `finish_reason==length` → return (no-op). Never raise.
- Else `_store_summary(...)`.
Wire the call site wherever per-message ingestion already runs
`extract_and_store_facts` (right after it, same `message_idx`).

### 4g. `_memoria_summary_retrieve(self, query, top_k) -> dict`
- Term-match `summary` against query terms (same stop-word + `[a-zA-Z]{3,}`
  tokenizing used by the other specialists).
- **SUM-specific breadth is OK here** (unlike KG): if no term matches, return the
  most recent `top_k` summaries `ORDER BY seg_end DESC`. Summaries ARE the broad
  narrative, so a breadth fallback is correct (this is the one place "return
  recent" helps rather than adds noise).
- Render: `ctx_lines = [f"[MSGIDX:{seg_end}] {summary}"]`; return
  `{"context": ..., "facts": [...], "source": "memoria_summaries"}`, else the
  empty `{"context":"","facts":[],"source":"fallback"}`.

### 4h. Wire into `_memoria_fused_retrieve`
Add "Specialist 6: summaries", flag-gated like the KG one:
```python
if os.environ.get("EDUMEM_LLM_SUMMARY", "0") == "1":
    try:
        ...
        specialists.append(('summary', self._memoria_summary_retrieve(query, top_k=top_k)))
    except Exception:
        specialists.append(('summary', {"context":"","facts":[],"source":"fallback"}))
```
Default OFF → fusion is byte-for-byte unchanged when the flag is off.

## 5. TDD (real objects, no mocks; `EDUMEM_NO_EMBEDDINGS=1`)
Write FIRST, then implement. Mirror the style in `tests/test_beam_evaluator.py`
(`_make_beam(tmp_path)`):
- `test_should_summarize_only_on_segment_boundary`: pure predicate, segment=20 →
  True at 19/39/59, False elsewhere.
- `test_store_summary_inserts_and_is_retrievable`: `_store_summary(...)` then
  `_memoria_summary_retrieve("...matching term...")` returns the row text with
  `[MSGIDX:seg_end]`.
- `test_summary_retrieve_breadth_fallback_returns_recent`: insert 3 summaries,
  query with no matching term → returns recent summaries (ordered by seg_end desc).
- `test_summary_prompt_is_bounded`: `_build_summary_prompt(...)` mentions the
  <=150 word cap and asks for narrative-only.
- `test_fused_recall_includes_summary_source_when_flag_on`: with flag on and a
  summary row present, `memoria_retrieve(query)` fused context contains the
  summary row; with flag OFF, it does NOT (default unchanged).
- `test_summary_flag_off_writes_nothing`: ingest with `EDUMEM_LLM_SUMMARY` unset
  and no client → `memoria_summaries` stays empty; behavior == today.
Full fast suite must stay green.

## 6. e2e validation (parent runs; agent does NOT)
Per the BEAM integration-test rule, the parent measures SUM with the live judged
A/B (`EDUMEM_LLM_SUMMARY=1` vs `0`) on the real path, reusing the lean reuse-DB
driver. Success = **SUM up**, no regression on KU/MR/IE/TR and the others. SUM is
the only target; do not expect this to move CR/latency (answer-model-bound).

## 7. Constraints / non-goals
- **Bounded output** (max_tokens 256, timeout, fallback) — non-negotiable.
- **Default OFF**; do not change the regex/default or the flag-off fusion path.
- One level of rolling update is enough; do NOT build a multi-level summary tree.
- Edit only `edumem/core/beam.py` + its tests. No commit; the parent commits + A/Bs.

## 8. Known-good infra notes
- LLM client: `self._llm_client.chat(messages, temperature=, max_tokens=)`.
- Timeout pattern + flag gating: copy from `extract_and_store_facts`.
- MEMORIA tables live in `init_beam`; read existing column names before querying.
- Specialist + RRF fusion wiring: copy the KG specialist block in
  `_memoria_fused_retrieve` (`EDUMEM_KG_FUSION`-gated) as the template.
