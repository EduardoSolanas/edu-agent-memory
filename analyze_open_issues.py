import json

with open('results/beam_question_validations.jsonl') as f:
    rows = [json.loads(line) for line in f if line.strip()]

failed = [r for r in rows if r.get('validation_status') == 'evaluated' and not r.get('validation_passed')]

# Find the truly unaddressed failures (EO #4 and others without diagnostic signals)
print("=== DETAILED LOOK AT TRULY UNADDRESSED ISSUES ===\n")

for r in failed:
    ai = (r.get('ai_answer_full') or r.get('ai_answer') or '')
    ability = r.get('ability')
    q = r.get('question') or ''
    ideal = r.get('ideal_answer_full') or r.get('ideal_answer') or ''
    score = r.get('score', '?')

    # Check if this truly has no diagnostic signals
    has_diagnostic = (
        'does not contain' in ai.lower() or
        'contradictory' in ai.lower() or
        'IDENTIFIED DATES' in ai or
        (ability in ('KU', 'SUM', 'TR') and score == 0.0)
    )

    if not has_diagnostic:
        print(f"ABILITY: {ability}")
        print(f"SCORE: {score}")
        print(f"QUESTION: {q}")
        print(f"IDEAL ANSWER: {ideal}")
        print(f"AI ANSWER: {ai}")
        print(f"QID: {r.get('qid')}")
        print()
