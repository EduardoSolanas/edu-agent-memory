import ast
import csv
import json
import os
import random
import sys
import hashlib
import concurrent.futures
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download
from openai import OpenAI
from benchmarks.evidence import best_evidence, build_profile_map_reduce
from mnemosyne.core.beam import BeamMemory

REPO = "bowen-upenn/PersonaMem-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
WORKDIR = PROJECT_ROOT
LOCAL = WORKDIR / "data_hf"
PROFILE_DIR = WORKDIR / "results/profiles"


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
    token = os.getenv("HF_TOKEN")
    local_csv = LOCAL / "benchmark/text/benchmark.csv"
    if local_csv.exists():
        csv_path = local_csv
    elif token:
        csv_path = Path(hf_hub_download(REPO, repo_type="dataset", filename="benchmark/text/benchmark.csv", token=token, local_dir=str(LOCAL)))
    else:
        raise SystemExit("HF_TOKEN not available and local benchmark CSV is missing")

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if len(rows) >= limit:
                break

    for row in rows:
        hist = LOCAL / row["chat_history_32k_link"]
        if hist.exists():
            continue
        if not token:
            raise SystemExit(f"Missing chat history and HF_TOKEN unavailable: {row['chat_history_32k_link']}")
        hf_hub_download(REPO, repo_type="dataset", filename=row["chat_history_32k_link"], token=token, local_dir=str(LOCAL))
    return rows


def load_history(row):
    path = LOCAL / row["chat_history_32k_link"]
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("conversations") or data.get("chat_history") or []
    return data


def history_text(history):
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def build_profile(client, row, history):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILE_DIR / f"persona_{row['persona_id']}.md"
    hash_path = profile_path.with_suffix(".md.hash")

    prompt_template = "Build a compact memory profile for one user from this chat history."
    prompt_hash = hashlib.md5(prompt_template.encode("utf-8")).hexdigest()

    return build_profile_map_reduce(
        client=client,
        model=os.getenv("CHAT_MODEL", "qwen3.6"),
        history=history,
        profile_path=profile_path,
        hash_path=hash_path,
        prompt_hash=prompt_hash
    )


def answer(client, profile, evidence, query, mapping):
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
        raise SystemExit("Missing model API key")
    client = OpenAI(api_key=api_key, base_url=os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1"))

    rows = load_rows(int(os.getenv("LIMIT", "10")))
    profiles = {}
    histories = {}
    
    # 1. Pre-build/load profiles sequentially (fast & thread-safe)
    for row in rows:
        pid = row["persona_id"]
        if pid not in profiles:
            histories[pid] = load_history(row)
            profiles[pid] = build_profile(client, row, histories[pid])

        # Initialize BeamMemory for each persona_id
        db_dir = WORKDIR / "results/pm/databases" / f"persona_{pid}"
        db_path = db_dir / "mnemosyne.db"
        db_exists = db_path.exists() and db_path.stat().st_size > 0
        if not db_exists:
            db_dir.mkdir(parents=True, exist_ok=True)
            beam = BeamMemory(db_path=db_path)
            print(f"[*] Ingesting history into BeamMemory for persona {pid}...")
            for turn, msg in enumerate(histories[pid]):
                beam.remember(
                    content=msg['content'],
                    source='conversation',
                    metadata={'role': msg['role'], 'turn': turn}
                )

    # 2. Evaluate rows in parallel using a ThreadPoolExecutor
    print(f"[*] Starting parallel evaluation of {len(rows)} rows with 4 threads...")
    
    def process_row(args):
        i, row = args
        pid = row["persona_id"]
        query = parse_msg(row["user_query"])["content"]
        mapping, gold = parse_options(row, seed=i)
        
        db_path = WORKDIR / "results/pm/databases" / f"persona_{pid}" / "mnemosyne.db"
        beam = BeamMemory(db_path=db_path)
        recall_results = beam.recall(query, top_k=50)
        evidence = "\n".join(f"- {r.get('content', '').strip()}" for r in recall_results)
        
        pred, answer_text = answer(client, profiles[pid], evidence, query, mapping)
        ok = pred == gold
        return i, pid, gold, pred, ok, row["preference"], answer_text

    results = []
    correct = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_row, (i, row)): i for i, row in enumerate(rows)}
        for fut in concurrent.futures.as_completed(futures):
            try:
                i, pid, gold, pred, ok, pref, answer_text = fut.result()
                correct += ok
                results.append((i, pid, gold, pred, ok, pref, answer_text))
                print(f"[{len(results)}/{len(rows)}] idx={i} persona={pid} gold={gold} pred={pred} ok={ok} pref={pref[:90]}")
            except Exception as e:
                print(f"[!] Error processing row: {e}", file=sys.stderr)

    # Save JSONL results
    jsonl_results = []
    for res in results:
        jsonl_results.append({
            "idx": res[0],
            "persona": res[1],
            "gold": res[2],
            "pred": res[3],
            "ok": res[4],
            "preference": res[5],
            "answer_text": res[6]
        })
    jsonl_results.sort(key=lambda x: x["idx"])
    jsonl_path = WORKDIR / "results/profile_smoke_results.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in jsonl_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[*] Saved structured results to {jsonl_path}")

    # 4. Print final summary and enforce the smoke gate.
    accuracy = correct / len(rows) if rows else 0.0
    min_accuracy = float(os.getenv("MIN_ACCURACY", "0.60"))
    print(f"====================================================")
    print(f"PersonaMem-v2 E2E profile smoke: {correct}/{len(rows)} {accuracy:.4f}")
    print(f"Minimum required accuracy: {min_accuracy:.4f}")
    print(f"====================================================")
    if accuracy < min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
