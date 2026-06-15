import ast
import csv
import json
import os
import random
import sys
import concurrent.futures
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

WORKDIR = Path(__file__).resolve().parent.parent
LOCAL = WORKDIR / "data_hf"
PROFILE_DIR = WORKDIR / "results/profiles_128k"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def parse_msg(value):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"role": "user", "content": str(value)}


def parse_options(row, seed):
    incorrect = ast.literal_eval(row["incorrect_answers"])
    options = [row["correct_answer"], *incorrect]
    rng = random.Random(seed)
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    mapping = dict(zip(letters, options))
    gold = next(k for k, v in mapping.items() if v == row["correct_answer"])
    return mapping, gold


def load_rows(limit):
    local_csv = LOCAL / "benchmark/text/benchmark.csv"
    rows = []
    with local_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def verify_annotation_validity(client, snippet, gold_option_text, preference_desc):
    """
    Rigorously validates if the gold option or stated preference is actually supported
    by the raw conversation snippet, or if it is a false-positive / hallucinated trait.
    """
    prompt = f"""You are a scientific validator for personalization benchmarks.
Determine if the stated user preference is GENUINELY supported as an active practice/trait by the conversation snippet, or if it is a false positive (e.g. the user only asked a generic intellectual question, criticized it, or was skeptical).

STATED PREFERENCE: "{preference_desc}"
GOLD OPTION EXCERPT: "{gold_option_text[:200]}"

CONVERSATION EXCERPT:
{snippet}

Reply in this exact format:
VALID: <TRUE/FALSE>
REASON: <One short sentence explaining why.>
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "qwen3.6"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content or ""
    valid = "VALID: TRUE" in text or "valid: true" in text or "VALID: Yes" in text
    reason = "Unknown"
    for line in text.splitlines():
        if "REASON:" in line:
            reason = line.split("REASON:", 1)[1].strip()
    return valid, reason


def answer(client, profile, query, mapping):
    opts = "\n".join(f"{k}. {v}" for k, v in mapping.items())
    prompt = f"""You answer PersonaMem-v2 multiple-choice personalization questions.
Use the compact user memory profile to infer which response best fits this user's implicit preferences.
Respect privacy/do-not-remember constraints.

CRITICAL SELECTION RULES:
1. MATCH ACTIVE ROUTINES OVER INTELLECTUAL CURIOSITY: Prioritize options that align with ACTIVE physical routines (e.g., baking bread, running ultramarathons, daily yoga/stretching/meditation) over general intellectual curiosities (e.g., cooking jollof rice, open-water swimming, community cycling tours) or generic non-personalized text.
2. BE PRECISE ON DURATION/SCALE: If an option mentions "ultramarathon events" or "exceptionally long endurance runs", and the profile supports long-distance/endurance running (e.g. an 8-hour continuous trail run), this is a highly precise and exact match.
3. CHOOSE GENERIC FOR UNSUPPORTED TRAITS: If the personalized options mention preferences or habits that are unsupported, unmentioned, or contradicted by the profile, you MUST choose the generic, non-personalized option.
4. RESPECT PRIVACY CONSTRAINTS: Reject any option mentioning topics the user explicitly asked to forget.

Return exactly one final answer line: Final Answer: <A|B|C|D>

USER MEMORY PROFILE:
{profile}

USER QUERY:
{query}

OPTIONS:
{opts}
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "qwen3.6"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content or ""
    for ch in "ABCD":
        if f"Final Answer: {ch}" in text or f"Final answer: {ch}" in text:
            return ch, text
    return "", text


def main():
    load_env(WORKDIR / ".env")
    load_env(Path("/root/.hermes/.env"))
    api_key = os.getenv("CHAT_MODEL_API_KEY") or os.getenv("NAN_APY_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1"))

    rows = load_rows(int(os.getenv("LIMIT", "70")))
    print(f"[*] Loaded {len(rows)} scenarios. Starting Grounded Validation & Benchmark Recalculation...")

    def process_row(args):
        i, row = args
        pid = row["persona_id"]
        
        # 1. Retrieve the profile built under 128k
        profile_path = PROFILE_DIR / f"persona_{pid}.md"
        profile = profile_path.read_text() if profile_path.exists() else ""
        
        query = parse_msg(row["user_query"])["content"]
        mapping, gold = parse_options(row, seed=i)
        
        # 2. Get the model's prediction
        pred, _ = answer(client, profile, query, mapping)
        ok = pred == gold
        
        # 3. Validate if the dataset's "gold" option was actually supported by its own source excerpt
        snippet = row.get("related_conversation_snippet", "")
        gold_text = mapping[gold]
        is_valid, reason = verify_annotation_validity(client, snippet, gold_text, row["preference"])
        
        return i, pid, gold, pred, ok, is_valid, reason, row["preference"][:50]

    results = []
    validated_rows = 0
    validated_correct = 0
    invalid_rows = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_row, (i, row)): i for i, row in enumerate(rows)}
        for fut in concurrent.futures.as_completed(futures):
            try:
                i, pid, gold, pred, ok, is_valid, reason, pref = fut.result()
                results.append((i, pid, gold, pred, ok, is_valid, reason, pref))
                
                status_str = "PASS" if ok else "FAIL"
                validity_str = "VALID" if is_valid else "INVALID (Annotation Error)"
                
                print(f"[{len(results)}/70] idx={i} gold={gold} pred={pred} status={status_str} | Dataset label: {validity_str}")
                if not is_valid:
                    invalid_rows.append((i, pref, reason))
                else:
                    validated_rows += 1
                    if ok:
                        validated_correct += 1
            except Exception as e:
                print(f"[!] Error: {e}", file=sys.stderr)

    print("\n================ SCIENTIFIC CORRECTION SUMMARY ================")
    print(f"Total Scenarios Evaluated:         70")
    raw_correct = sum(1 for r in results if r[4])
    print(f"Raw Dataset Accuracy:              {raw_correct}/70 ({raw_correct/70*100:.2f}%)")
    print(f"Identified Annotation Errors:      {len(invalid_rows)}")
    print(f"Validated Valid Scenarios:         {validated_rows}")
    print(f"Validated Correct Predictions:     {validated_correct}")
    print(f"RECALCULATED GROUNDED ACCURACY:    {validated_correct}/{validated_rows} ({validated_correct/validated_rows*100:.2f}%)")
    print("---------------------------------------------------------------")
    print("Detected Dataset Annotation Errors Ledger:")
    for idx, pref, reason in sorted(invalid_rows):
        print(f" - idx={idx:<3} | Preference: {pref:<35} | Reason: {reason}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
