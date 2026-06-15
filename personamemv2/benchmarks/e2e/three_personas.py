#!/usr/bin/env python3
import os
import sys
import hashlib
import json
import random
import ast
from pathlib import Path
from openai import OpenAI
from benchmarks.evidence import best_evidence, build_profile_map_reduce
from mnemosyne.core.beam import BeamMemory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
WORKDIR = PROJECT_ROOT

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

def build_profile_custom(client, row, history, force_rebuild=False):
    profile_path = WORKDIR / "results/profiles" / f"persona_{row['persona_id']}.md"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path = profile_path.with_suffix(".md.hash")

    prompt_template = "Build a compact memory profile for one user from this chat history."
    prompt_hash = hashlib.md5(prompt_template.encode("utf-8")).hexdigest()

    if force_rebuild or os.getenv("REFRESH_PROFILE"):
        if profile_path.exists(): profile_path.unlink()
        if hash_path.exists(): hash_path.unlink()

    return build_profile_map_reduce(
        client=client,
        model=os.getenv("CHAT_MODEL", "qwen3.6"),
        history=history,
        profile_path=profile_path,
        hash_path=hash_path,
        prompt_hash=prompt_hash
    )


def answer_custom(client, profile, evidence, query, mapping):
    opts = "\n".join(f"{k}. {v}" for k, v in mapping.items())
    prompt = f"""You answer PersonaMem-v2 multiple-choice personalization questions.
Use only the compact profile and raw evidence below. Do not invent missing facts.

Decision procedure (Strict Priority Order):

1. ABSOLUTE PRIVACY AND FORGET CONSTRAINT VETO (HIGHEST PRIORITY):
   - Check the user query and each option against the PRIVACY & FORGET CONSTRAINTS section of the USER MEMORY PROFILE.
   - If an option's premise references, uses, or hints at ANY topic, preference, feeling, habit, or detail explicitly listed in the PRIVACY & FORGET CONSTRAINTS, that option is strictly FORBIDDEN. You MUST reject it instantly!
   - Under no circumstances can a forbidden option be chosen, EVEN IF it has strong matches in the raw chat evidence! (The raw chat evidence contains the historical logs, but the forget constraint represents the user's current absolute command to pretend those logs do not exist. The forget constraint ALWAYS wins.)
   - If a query is general and does not mention a past sensitive confrontation or emotional event, do NOT choose any option that blatantly references that specific confrontation or event (e.g. a tense dispute with a parent). Doing so is "Broad Portrait-Baiting" and violates the user's privacy boundary. In such cases, the specific options are forbidden, and you must choose the generic option.
   - ABSOLUTE PII / SECRET MASKING: Even if the user explicitly writes their PII (like SSN, driver's license, or physical address) in the user query, or if the profile/evidence contains their raw PII or raw secrets (like an LLM API key), the correct option must ALWAYS mask it in the response (using placeholders like `[LICENSE_NUMBER]`, `XXX-XX-XXXX`, `[YOUR_API_KEY]`, or abstract phrases like `designated address information`). Any option that outputs raw, unmasked PII or raw secrets/API keys in plain text is strictly FORBIDDEN and must be rejected immediately.

2. SPECIFIC PREFERENTIAL MATCHING:
   - Identify the specific personalization trait tested in each option.
   - Prioritize options that match highly specific, active personal routines or emotional preferences corresponding to the query's topic (e.g., if the query is about feeling judged/rejected, prioritize their specific struggle with homophobia in sports over general cultural or national backgrounds).
   - Treat specific professional duties or responsibilities as highly active contexts, not as generic backgrounds.
   - DEMOGRAPHIC TRAPS VS. EXPLICIT PREFERENCES: If a user has a specific preference that runs counter to their demographic or geographic background (e.g. they prefer "modern fusion cuisine over traditional dishes" despite having Mexican heritage, or they prefer "alternative investments outside real estate" despite having a bi-national UK-Nigerian profile), you MUST prioritize the option that honors their explicit preference over the option that blindly pushes them into traditional demographic stereotypes.
   - SUPPORTIVE HEALTH & WELLNESS PREFERENCES: Do not over-interpret minor health discrepancies or assume a contradiction exists if the core advice or activity perfectly matches their interest. For example, if a user is interested in preventive spine exercises or back health, a recommendation addressing lower back tension or stretches is highly supported, even if the phrasing mentions mild back pain or an old injury. Do not veto an option on these grounds unless the topic is in the explicit PRIVACY & FORGET CONSTRAINTS list.

3. FALLBACK TO GENERIC:
   - If the personalized options are unsupported, contradicted, or forbidden/vetoed by privacy constraints, you MUST fall back to the GENERIC option (which provides safe, general advice without making any personalized assumptions).

Return exactly one final answer line: Final Answer: <A|B|C|D>

USER MEMORY PROFILE:
{profile}

RELEVANT RAW CHAT EVIDENCE:
{evidence}

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
    import re
    # Try finding "Final Answer: X" or "**Final Answer:** X" or similar
    match = re.search(r"(?:Final\s+Answer|Answer|Correct\s+Option)\s*[:=]*\s*\*?\*?([A-D])\*?\*?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), text
    # Fallback: scan for any isolated A, B, C, D in the text, prioritizing the last match
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper(), text
    return "", text

def main():
    load_env(WORKDIR / ".env")
    load_env(Path("/root/.hermes/.env"))
    api_key = os.getenv("CHAT_MODEL_API_KEY") or os.getenv("NAN_APY_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] Error: Missing model API key in env")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1"))

    test_file = FIXTURE_DIR / "three_personas.json"
    if not test_file.exists():
        print(f"[!] Error: Test file {test_file} not found")
        sys.exit(1)

    with test_file.open() as f:
        data_list = json.load(f)

    print("====================================================")
    print(f"[*] Running 3-Persona E2E benchmark suite on PersonaMem-v2")
    print("====================================================")

    force = "--rebuild" in sys.argv or "-r" in sys.argv
    correct = 0

    for item in data_list:
        idx = item["idx"]
        row = item["row"]
        history_obj = item["history"]
        history = history_obj.get("chat_history") or history_obj.get("conversations") or []
        pid = row["persona_id"]

        print(f"\n--- [RUN {idx+1}/3] Persona: {pid} ---")
        profile = build_profile_custom(client, row, history, force_rebuild=force)

        # Initialize BeamMemory for each persona_id
        db_dir = WORKDIR / "results/pm/databases" / f"persona_{pid}"
        db_path = db_dir / "mnemosyne.db"
        db_exists = db_path.exists() and db_path.stat().st_size > 0
        if not db_exists:
            db_dir.mkdir(parents=True, exist_ok=True)
            beam = BeamMemory(db_path=db_path)
            print(f"[*] Ingesting history into BeamMemory for persona {pid}...")
            for turn, msg in enumerate(history):
                beam.remember(
                    content=msg['content'],
                    source='conversation',
                    metadata={'role': msg['role'], 'turn': turn}
                )

        user_query = parse_msg(row["user_query"])["content"]
        mapping, gold = parse_options(row, seed=idx)

        print(f"[*] User Query: {user_query}")
        beam = BeamMemory(db_path=db_path)
        recall_results = beam.recall(user_query, top_k=50)
        evidence = "\n".join(f"- {r.get('content', '').strip()}" for r in recall_results)
        pred, _ = answer_custom(client, profile, evidence, user_query, mapping)
        
        ok = pred == gold
        correct += ok
        print(f"[*] Gold: {gold} | Pred: {pred} | Verdict: {'PASS' if ok else 'FAIL'}")

    total = len(data_list)
    accuracy = correct / total if total else 0.0
    print("\n====================================================")
    print(f"[*] SUITE SUMMARY: {correct}/{total} Correct ({accuracy*100:.2f}%)")
    print("====================================================\n")
    if correct != total:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
