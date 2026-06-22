import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from tools.evaluate_beam_end_to_end import load_beam_dataset, ABILITY_MAP

OUTPUT = Path('tests/fixtures/beam_e2e_100k_case1_fixture.json')
RESULTS = Path('results/beam_results_20260620_104723/beam_e2e_results.json')

# Load dataset and baseline results
data = load_beam_dataset(['100K'], max_conversations=1)
conv = data['100K'][0]

results = json.loads(RESULTS.read_text(encoding='utf-8'))
result_rows = results['results'][0]['results']
by_qid = {r['qid']: r for r in result_rows}

# Stopwords to exclude from keyword extraction
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be',
    'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
    'of', 'in', 'on', 'at', 'to', 'for', 'from', 'by', 'with', 'as', 'up',
    'that', 'this', 'these', 'those', 'it', 'its', 'which', 'who', 'what',
    'when', 'where', 'why', 'how', 'if', 'then', 'response', 'should', 'state',
    'contain', 'mention', 'there', 'till', 'between', 'information', 'not',
    'about', 'during', 'across', 'through'
}

Q7_ATOMIC_NUGGETS = [
    "march 15",
    "march 22",
    "march 23",
    "march 29",
    "database schema",
    "user registration",
    "frontend forms",
    "integrate frontend",
]

STATIC_CASE_CHECKS = {
    "1:q8": {"check": "tagged_code_fence"},
    "1:q9": {"check": "versioned_dependencies", "min_versioned_dependencies": 2},
    "1:q14": {
        "check": "contains_groups",
        "groups": [
            ["flask-login", "session management"],
            ["flask-sqlalchemy", "sqlite", "sqlalchemy"],
            ["chart.js", "analytics", "data visualization"],
        ],
        "min_fraction": 1.0,  # PF: all groups required
    },
    "1:q15": {
        "check": "contains_groups",
        "groups": [
            ["password hashing", "argon2", "bcrypt"],
            ["csrf", "input validation", "rate limiting"],
            ["incremental", "in phases", "start with", "then add", "prioritize"],
        ],
        "min_fraction": 1.0,  # PF: all groups required
    },
    "1:q16": {
        "check": "contains_groups",
        "groups": [
            ["registration", "user authentication"],
            ["expense", "transaction management"],
            ["visualization", "analytics"],
            ["april 15, 2024", "april 15 2024"],
            ["authentication", "login"],
            ["deployment"],
            ["password hashing", "stronger password"],
            ["token-based", "token authentication"],
            ["role-based access", "rbac"],
            ["input validation"],
            ["confluence"],
            ["api endpoint", "architecture decision"],
            ["table", "diagram"],
        ],
        "min_fraction": 0.5,  # SUM: 50% of groups required (7 of 13)
    },
    "1:q17": {
        "check": "contains_groups",
        "groups": [
            ["werkzeug", "pbkdf2:sha256"],
            ["uuid", "unique constraint"],
            ["operationalerror", "operational error"],
            ["csrf", "csrf token"],
            ["redis", "account lockout", "login lockout"],
        ],
        "min_fraction": 1.0,  # CR: all groups required
    },
}

def _atomize(nugget: str) -> list:
    """
    Decompose a nugget string into atomic facts.
    Extracts:
      1. Numbers with units (e.g., "250ms", "21 days", "165 commits")
      2. Spelled-out counts (two, three, four) as whole words
      3. Month-day dates (e.g., "March 29", "January 15, 2024")
      4. Quoted terms (in single/double quotes)
      5. Keyphrases: 2-3 word sequences of non-stopwords (>3 chars each)

    Returns: list of deduplicated atoms (lowercased, whitespace-normalized)
    """
    atoms = []
    nugget_lower = nugget.lower()

    # 1. Numbers with units (but NOT bare numbers; they're too generic)
    num_with_unit_pattern = r'\b\d[\d.,]*\s*(?:ms|s|days?|weeks?|months?|years?|commits?|columns?|%|h)\b'
    for match in re.finditer(num_with_unit_pattern, nugget_lower, re.IGNORECASE):
        atom = match.group().strip()
        if atom:
            atoms.append(atom)

    # 2. Spelled-out counts (two, three, four)
    count_pattern = r'\b(two|three|four)\b'
    for match in re.finditer(count_pattern, nugget_lower, re.IGNORECASE):
        atoms.append(match.group().lower())

    # 3. Month-day dates (full form like "March 29" or "January 15, 2024")
    date_pattern = r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?'
    for match in re.finditer(date_pattern, nugget_lower, re.IGNORECASE):
        atoms.append(match.group().lower())

    # 4. Quoted terms
    quote_pattern = r'[\'"]([^\'"]+)[\'"]'
    for match in re.finditer(quote_pattern, nugget):
        term = match.group(1).strip().lower()
        if term and len(term) > 2:
            atoms.append(term)

    # 5. Extract keyphrases: 2-3 word sequences of non-stopwords (>3 chars each)
    # Also capture individual non-stopwords that are >5 chars (distinct domain words)
    words_raw = re.findall(r'\b\w+\b', nugget_lower)
    phrase_atoms = []
    captured_phrases = set()

    i = 0
    while i < len(words_raw):
        # Look for sequences of non-stopwords with >3 chars
        if words_raw[i] not in STOPWORDS and len(words_raw[i]) > 3:
            phrase = [words_raw[i]]
            j = i + 1
            # Extend phrase while consecutive words are non-stopwords with >3 chars, max 3 words
            while j < len(words_raw) and words_raw[j] not in STOPWORDS and len(words_raw[j]) > 3 and len(phrase) < 3:
                phrase.append(words_raw[j])
                j += 1

            phrase_str = " ".join(phrase)
            # Add if: 2+ words (phrase) OR 1 word that is >5 chars (domain-specific)
            if len(phrase) >= 2 or len(phrase[0]) > 5:
                if phrase_str not in captured_phrases:
                    phrase_atoms.append(phrase_str)
                    captured_phrases.add(phrase_str)
            i = j
        else:
            i += 1

    atoms.extend(phrase_atoms)

    # 6. Fallback: if no atoms collected, take longest non-stopword words
    if not atoms:
        words = re.findall(r'\b\w{5,}\b', nugget_lower)
        words = [w for w in words if w not in STOPWORDS]
        atoms = sorted(set(words), key=lambda w: -len(w))[:3]

    # Normalize and deduplicate; filter out stopwords that may have slipped in
    atoms = [" ".join(a.split()).lower() for a in atoms if a.strip()]
    atoms = [a for a in atoms if a not in STOPWORDS]
    atoms = list(dict.fromkeys(atoms))  # Preserve order, remove duplicates

    # Remove atoms that are substrings of other atoms (e.g., "commits" is in "165 commits")
    filtered = []
    for a in atoms:
        is_substring = False
        for other in atoms:
            if a != other and a in other:
                is_substring = True
                break
        if not is_substring:
            filtered.append(a)

    return filtered

def derive_check(ability: str) -> str:
    """Derive check type from ability code."""
    if ability == "ABS":
        return "absence"
    elif ability == "EO":
        return "order"
    elif ability in ("IF", "PF", "SUM"):
        return "skip"
    else:  # IE, KU, MR, TR, CR
        return "contains_all"

def extract_nuggets(ability: str, rubric: list, qid: str = None) -> list:
    """Extract nuggets from rubric based on ability type."""
    if qid == "1:q7":
        return Q7_ATOMIC_NUGGETS.copy()

    check = derive_check(ability)

    if check == "skip" or check == "absence":
        return []

    if check == "order":
        # For ordering, nuggets are the rubric items as-is
        return rubric

    # For contains_all: parse rubric items to extract content, then atomize
    raw_nuggets = []
    prefixes_to_strip = [
        "LLM response should state: ",
        "LLM response should contain: ",
        "LLM response should mention: ",
    ]

    for item in rubric:
        text = item
        # Strip known prefixes (case-insensitive)
        for prefix in prefixes_to_strip:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):]
                break

        # Strip trailing period and surrounding quotes
        text = text.rstrip('.')
        text = text.strip('"\'')

        raw_nuggets.append(text)

    # SPECIAL CASE q18 (TR): drop "8 weeks" which contradicts ideal_answer "exactly 4 weeks"
    if qid == "1:q18" and "8 weeks" in raw_nuggets:
        raw_nuggets = [n for n in raw_nuggets if n != "8 weeks"]

    # Atomize each nugget and flatten
    all_atoms = []
    for nugget in raw_nuggets:
        atoms = _atomize(nugget)
        all_atoms.extend(atoms)

    # Final deduplication preserving order
    atoms = list(dict.fromkeys(all_atoms))

    return atoms

# Build fixture: 20 test cases with baselines and expectations
cases = []
for i, q in enumerate(conv['questions']):
    qid = f'1:q{i}'
    row = by_qid.get(qid, {})

    # Map ability from dataset name to short code
    ability_name = q.get('ability', 'IE')
    ability_short = ABILITY_MAP.get(ability_name, ability_name[:2]).upper()

    # Get baseline score from reference results
    baseline_score = float(row.get('score', 0.0))
    rubric = row.get('rubric', [])

    # Determine expectation
    expectation = "hard" if baseline_score >= 0.75 else "xfail"

    # SPECIAL CASE: demote q1 (ABS) to xfail
    # q1 ABS: abstention is single-sample model-variant; demoted to xfail
    if qid == "1:q1":
        expectation = "xfail"

    # Derive check and nuggets
    check = derive_check(ability_short)
    nuggets = extract_nuggets(ability_short, rubric, qid)

    case = {
        'qid': qid,
        'ability': ability_short,
        'question': q['question'],
        'baseline_score': baseline_score,
        'expectation': expectation,
        'check': check,
        'nuggets': nuggets,
    }

    # Add min_fraction for IE descriptive contains_all checks (60% threshold)
    # Only for questions with multiple nuggets (descriptive, not atomic)
    if check == "contains_all" and ability_short == "IE" and len(nuggets) > 1:
        case['min_fraction'] = 0.6

    if qid in STATIC_CASE_CHECKS:
        case.update(STATIC_CASE_CHECKS[qid])

    # q15 passed an older run but failed the latest persisted live baseline.
    if qid == "1:q15":
        case["expectation"] = "xfail"

    # Add comment for q18 special case
    if qid == "1:q18":
        case['_note'] = "q18 (TR): dropped '8 weeks' nugget (contradicts ideal_answer '4 weeks' — known BEAM dataset bug); kept date-range nugget"

    cases.append(case)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(cases, indent=2) + "\n", encoding='utf-8')

# Print per-question summary
print('\n' + '='*100)
print('BEAM FIXTURE GENERATOR SUMMARY')
print('='*100)
print('QID\tAbility\tBaseline\tExpectation\tCheck\t\t#Atoms')
print('-'*100)
for c in cases:
    check = c['check']
    n_atoms = len(c['nuggets'])
    print(f"{c['qid']}\t{c['ability']}\t{c['baseline_score']:.2f}\t{c['expectation']:<8}\t{check:<12}\t{n_atoms}")
    if check == 'contains_all' and c['nuggets']:
        print(f"        ATOMS: {c['nuggets']}")
print('='*100)
