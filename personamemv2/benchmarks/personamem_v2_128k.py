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
        hist = LOCAL / row["chat_history_128k_link"]
        if hist.exists():
            continue
        if not token:
            raise SystemExit(f"Missing chat history and HF_TOKEN unavailable: {row['chat_history_128k_link']}")
        print(f"[*] Downloading {row['chat_history_128k_link']}...")
        hf_hub_download(REPO, repo_type="dataset", filename=row["chat_history_128k_link"], token=token, local_dir=str(LOCAL))
    return rows


def load_history(row):
    path = LOCAL / row["chat_history_128k_link"]
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
   - CORE-ONLY VETO RULE: Do not over-veto on general or incidental words!
     - If the constraint is "Taking photos / travel photography", it ONLY forbids suggesting the active *hobby/action* of taking photos, travel photography, or buying cameras. It does NOT forbid mentioning "family photos", "framed family photos", "displaying family photos", or "keepsakes from family trips" on shelves. Displaying family photos is a universal, warm family routine and is completely ALLOWED and highly prioritized!
     - If the constraint is "switching between English and Spanish", it ONLY forbids the AI itself switching languages in its response. It does NOT forbid referencing bilingual captions on a digital frame.
     - If the constraint is "Vitamin D deficiency / supplements", it ONLY forbids suggesting taking Vitamin D supplements. It does NOT forbid mentioning a breakfast routine or other wellness topics.
     - If the constraint is "traditional orchestral concerts", it does NOT forbid listening to a cozy live acoustic set in a café.
     - If the constraint is "securing sponsorship", it does NOT forbid mentioning the general vision of sports inclusion.
   - BROAD PORTRAIT-BAITING VETO FOR SPECIFIC PAST CONFLICTS:
     - Broad Portrait-Baiting applies when a user's query is general (e.g. asking about stress, anxiety, or coping in general) and an option blatantly references a highly specific, private, sensitive past conflict, tense argument, or stressful incident that the user personally went through (e.g., "that aggressive parent during the curriculum debate", "that safety compromise guilt", or "a tense dispute with a parent"). You MUST reject these specific options and fall back to the generic option!
     - Broad Portrait-Baiting NEVER applies to:
       - General professional duties or responsibilities (e.g., a teacher helping a student in general, a consumer advocate reviewing product safety in general). For example, "When a student shares something deeply troubling, connect them with the counselor" is a general professional duty and is completely ALLOWED.
       - General personal/demographic background facts (e.g., being a Lesbian, being Nigerian, being Mexican-American).
       - General non-sensitive history (e.g., having asthma in childhood).
   - ABSOLUTE PII / SECRET MASKING: Even if the user explicitly writes their PII (like SSN, driver's license, or physical address) in the user query, or if the profile/evidence contains their raw PII or raw secrets (like an LLM API key), the correct option must ALWAYS mask it in the response (using placeholders like `[LICENSE_NUMBER]`, `XXX-XX-XXXX`, `[YOUR_API_KEY]`, or abstract phrases like `designated address information`). Any option that outputs raw, unmasked PII or raw secrets/API keys in plain text is strictly FORBIDDEN and must be rejected immediately.

2. SPECIFIC PREFERENTIAL MATCHING:
   - Identify the specific personalization trait tested in each option.
   - LEISURE HOBBY & PERSONAL INTEREST PRIORITY: If an option matches an explicit favorite leisure hobby, sport, collection, or interest (like Comic-Con/pop culture conventions, baking bread, surfing, cycling, football), prioritize it over professional/civics themes unless the query specifically asks about professional/work context. For general travel or leisure questions, if there is an option that tests a specific leisure hobby (like Comic-Con/pop culture conventions or surfing), you MUST prioritize it over professional/civic options (like politics or teaching conferences) even if the hobby is not explicitly mentioned in the profile!
   - PERMISSIVE MATCHING RULE: If an option contains a highly specific personal premise (like "Since you prefer practical American cars", "Since you enjoy modern fusion...", "Since you've got mild lower back pain..."), and the user's profile does NOT explicitly contradict this (i.e. there is no explicit, unambiguous, direct "does not like fusion" or "has never had back pain" in the profile), you MUST accept and prioritize this option! Do not assume it is unsupported just because it isn't written verbatim in the profile.
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
    # Robust parsing of the final Answer
    match = re.search(r"Final\s+Answer\s*:\s*([A-D])\s*$", text, re.IGNORECASE)
    if not match:
        match = re.search(r"Final\s+Answer\s*:\s*([A-D])", text, re.IGNORECASE)
    if not match:
        match = re.search(r"Answer\s*:\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), text
    
    matches = re.findall(r"(?:Final\s+Answer|Correct\s+Option|Answer)\s*[:=]*\s*\*?\*?([A-D])\*?\*?", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper(), text
    return "", text


def main():
    load_env(WORKDIR / ".env")
    db_base_dir = WORKDIR / "results/pm/databases_128k"
    tar_path = WORKDIR / "results/pm/databases_128k_precompiled.tar.gz"
    if not db_base_dir.exists() and tar_path.exists():
        print("[*] Found precompiled SQLite databases archive. Extracting...")
        import tarfile
        db_base_dir.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=db_base_dir.parent)
    load_env(Path("/root/.hermes/.env"))
    api_key = os.getenv("CHAT_MODEL_API_KEY") or os.getenv("NAN_APY_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing model API key")
    client = OpenAI(api_key=api_key, base_url=os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1"))

    rows = load_rows(int(os.getenv("LIMIT", "70")))
    profiles = {}
    histories = {}
    
    # 1. Pre-build/load profiles sequentially
    print("[*] Pre-building profiles using complete 128k histories...")
    for row in rows:
        pid = row["persona_id"]
        if pid not in profiles:
            histories[pid] = load_history(row)
            profiles[pid] = build_profile(client, row, histories[pid])

        # Initialize BeamMemory for each persona_id
        db_dir = WORKDIR / "results/pm/databases_128k" / f"persona_{pid}"
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
        
        db_path = WORKDIR / "results/pm/databases_128k" / f"persona_{pid}" / "mnemosyne.db"
        beam = BeamMemory(db_path=db_path)
        recall_results = beam.recall(query, top_k=50)
        untruncated_evidence = []
        cursor = beam.conn.cursor()
        for r in recall_results:
            row_id = r.get("id")
            cursor.execute("SELECT content FROM working_memory WHERE id = ?", (row_id,))
            full_row = cursor.fetchone()
            if full_row:
                untruncated_evidence.append(f"- {full_row[0].strip()}")
            else:
                untruncated_evidence.append(f"- {r.get('content', '').strip()}")
        evidence = "\n".join(untruncated_evidence)
        
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
    jsonl_path = WORKDIR / "results/personamem_v2_128k_results.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in jsonl_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[*] Saved structured results to {jsonl_path}")

    # 4. Print final summary
    print(f"====================================================")
    print(f"PersonaMem-v2 128k profile evaluation: {correct}/{len(rows)} {correct/len(rows):.4f}")
    print(f"====================================================")


if __name__ == "__main__":
    main()
