import json

with open('results/beam_question_validations.jsonl') as f:
    rows = [json.loads(line) for line in f if line.strip()]

failed = [r for r in rows if r.get('validation_status') == 'evaluated' and not r.get('validation_passed')]

print(f'Total failed: {len(failed)}/{len([r for r in rows if r.get("validation_status") == "evaluated"])}')
print()

for r in failed:
    ai = (r.get('ai_answer_full') or r.get('ai_answer') or '')
    ability = r.get('ability')
    q = r.get('question') or ''
    ideal = r.get('ideal_answer_full') or r.get('ideal_answer') or ''
    score = r.get('score', '?')

    # Classify which fix should address this
    fixes_that_apply = []
    if 'does not contain' in ai.lower():
        fixes_that_apply.append('RETRIEVAL_MISS')
    if 'contradictory' in ai.lower() and ability != 'CR':
        fixes_that_apply.append('FALSE_CONTRADICTION (fix #2 should address)')
    if 'contradictory' in ai.lower() and ability == 'CR':
        fixes_that_apply.append('CR_PARTIAL (fix #9 should address)')
    if 'IDENTIFIED DATES' in ai:
        fixes_that_apply.append('CALCULATOR_MISROUTE (fix #5 should address)')
    if ability == 'KU' and score == 0.0:
        fixes_that_apply.append('KU_WRONG_VALUE (fix #8 should address)')
    if ability == 'SUM':
        fixes_that_apply.append('SUM_INCOMPLETE (fixes #6,#7 should address)')
    if ability == 'TR' and score == 0.0:
        fixes_that_apply.append('TR_WRONG_DATES (fix #10 should address)')
    if ability in ('IF', 'PF') and 'does not contain' in ai.lower():
        fixes_that_apply.append('IF_PF_RETRIEVAL (fix #4 should address)')

    expected_fixed = len(fixes_that_apply) > 0
    status = 'EXPECTED FIXED' if expected_fixed else 'STILL OPEN'

    print(f'=== {ability} | score={score} | {status} ===')
    print(f'  Fixes: {fixes_that_apply if fixes_that_apply else "NONE - NEW ISSUE"}')
    print(f'  Q: {q[:150]}')
    print(f'  Ideal: {ideal[:200]}')
    print(f'  AI: {ai[:300]}')
    print()

# Count expected fixed vs still open
expected_fixed_count = 0
still_open_count = 0
for r in failed:
    ai = (r.get('ai_answer_full') or r.get('ai_answer') or '')
    ability = r.get('ability')
    has_fix = False
    if 'does not contain' in ai.lower(): has_fix = True
    if 'contradictory' in ai.lower(): has_fix = True
    if 'IDENTIFIED DATES' in ai: has_fix = True
    if ability in ('KU', 'SUM', 'TR') and r.get('score', 1) == 0.0: has_fix = True
    if has_fix:
        expected_fixed_count += 1
    else:
        still_open_count += 1

print()
print('=== SUMMARY ===')
print(f'Expected fixed by our changes: {expected_fixed_count}')
print(f'Still open (need new fixes): {still_open_count}')
