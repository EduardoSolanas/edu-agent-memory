# BEAM Benchmark TODO

Current: **62.6% overall** (100K, case 1, qwen3.6 + deepseek-v4-flash judge)

## Scores (2026-06-20 11:00 run)

| Ability | Score | Status |
|---------|-------|--------|
| KU | 100% | FIXED |
| ABS | 100% | FIXED |
| MR | 87.5% | OK |
| PF | 75% | OK |
| IE | 50% | NEEDS FIX |
| TR | 50% | NEEDS FIX |
| IF | 50% | NEEDS FIX |
| CR | 50% | NEEDS FIX |
| SUM | 40% | NEEDS FIX |
| EO | 23.4% | NEEDS FIX |

## Failures to fix

### EO — Event Ordering (23.4%)
- Model uses abstract labels ("Planning tasks and schedule") instead of actual topic descriptions ("Core functionality", "Transaction error handling")
- Rubric expects short functional descriptions matching what was actually discussed
- Fix: strengthen ORDERING modifier to force functional topic names, ban timeline/date-based labels

### SUM — Summarization (40%)
- q1 (security/database summary) starts with "contradictory information" instead of summarizing
- CONFLICTS rule hijacks the answer when security practices evolved over time
- Fix: SUM modifier should suppress CONFLICTS — summaries should narrate evolution, never flag contradictions

### CR — Contradiction Resolution (50%)
- q0: "Have I worked with Flask routes?" — AI only reports one side, misses the contradiction entirely
- q1: works (0.75) — correctly identifies contradiction about Flask-Login
- Fix: improve retrieval for CR questions so both contradicting statements are surfaced

### IE — Information Extraction (50%)
- q1: "How did I organize tasks over the sprint?" — AI falsely triggers ABSENCE
- Sprint scheduling details exist in context but retrieval didn't surface them
- Fix: retrieval issue — IE questions about organization/planning need broader recall

### TR — Temporal Reasoning (50%)
- q0: "Weeks between transaction management and final deployment deadline?" — AI can't find the dates
- Rubric expects Jan 15 → Mar 15 = 8 weeks, but model can't locate "final deployment deadline"
- Fix: retrieval issue — TR questions need the specific date-bearing messages surfaced

### IF — Instruction Following (50%)
- q1: "Which libraries are used?" — AI lists libraries with versions but rubric wants "explicit version details for each dependency"
- AI does include versions but may be missing some or formatting differently than expected
- Fix: check if all dependencies are listed; may be a retrieval completeness issue

## Done
- [x] Fix grader `<question>` placeholder binding
- [x] LLM-based message classification (replace regex)
- [x] KU modifier with MSGIDX + anti-contradiction override
- [x] EO modifier with fragmented context overrides
- [x] KU keywords expanded (how many, deadline, etc.)
- [x] Timestamped output dirs for benchmark runs
