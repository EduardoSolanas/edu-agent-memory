#!/usr/bin/env python3
"""Custom LoCoMo evaluation runner for edumem.

Loads the raw snap-research locomo10.json dataset, structures its multi-session
conversations into chronological timelines, builds profiles, and runs factual QA evaluations.
"""
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from openai import OpenAI

WORKDIR = Path("/opt/edumem")

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

def format_locomo_timeline(conversation):
    # Find all session indices present in the conversation dict
    sessions = []
    for k in conversation.keys():
        if k.startswith("session_") and not k.endswith("_date_time"):
            try:
                idx = int(k.split("_")[1])
                sessions.append(idx)
            except ValueError:
                pass
    sessions.sort()
    
    timeline_lines = []
    for idx in sessions:
        date_key = f"session_{idx}_date_time"
        date_val = conversation.get(date_key, "Unknown Date")
        timeline_lines.append(f"\n--- SESSION {idx} | Date/Time: {date_val} ---")
        
        session_turns = conversation.get(f"session_{idx}", [])
        for turn in session_turns:
            speaker = turn.get("speaker", "Unknown").upper()
            text = turn.get("text", "").strip()
            if text:
                timeline_lines.append(f"{speaker}: {text}")
                
    return "\n".join(timeline_lines)

def build_memory_profile(client, timeline_text):
    prompt = f"""Build a highly accurate, structured chronological memory profile from these chat sessions.
RULES:
1. FOCUS ON TIMELINE & RECENT FACTS: Track when events or preferences happened. Explicitly record the latest/current state and what was updated.
2. EXTRACT EXACT DETAIL: Record specific personal facts, names, dates, plans, and decisions.
3. BE CONCISE AND NEAT: Organize into logical categories.

CHAT SESSIONS TIMELINE:
{timeline_text}
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "qwen3.6"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""

def answer_question(client, profile, question):
    prompt = f"""You answer factual questions about the user based ONLY on their chronological memory profile.

USER MEMORY PROFILE:
{profile}

QUESTION: {question}

Return a concise, direct answer based strictly on the memory profile.
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "qwen3.6"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""

def grade_answer(client, question, gold_answer, pred_answer):
    prompt = f"""You are an expert AI judge evaluating whether a predicted answer is semantically equivalent or correct relative to the gold answer for a given question.

QUESTION: {question}
GOLD ANSWER: {gold_answer}
PREDICTED ANSWER: {pred_answer}

If the predicted answer is correct (contains the correct key information even if phrased slightly differently), reply exactly: CORRECT
Otherwise, reply exactly: INCORRECT
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "qwen3.6"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    res = (resp.choices[0].message.content or "").strip().upper()
    return "CORRECT" in res

def process_item(client, idx, qa_pair):
    question = qa_pair["question"]
    gold = qa_pair["answer"]
    category = qa_pair.get("category", 1)
    
    # Get prediction
    pred = answer_question(client, os.environ["CURRENT_PROFILE"], question)
    
    # Evaluate
    ok = grade_answer(client, question, gold, pred)
    
    return idx, category, gold, pred, ok

def main():
    load_env(WORKDIR / ".env")
    load_env(Path("/root/.hermes/.env"))
    
    api_key = os.getenv("CHAT_MODEL_API_KEY") or os.getenv("NAN_APY_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] Error: Missing model API key in env")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1"))

    data_file = WORKDIR / "data/locomo10.json"
    if not data_file.exists():
        print(f"[!] Error: Dataset file {data_file} not found")
        sys.exit(1)

    with data_file.open() as f:
        data = json.load(f)

    # We evaluate conversation 0 for the benchmark
    conversation_sample = data[0]
    qa_list = conversation_sample["qa"]
    
    limit = int(os.getenv("LIMIT", "50"))
    eval_qa_pairs = qa_list[:limit]
    total = len(eval_qa_pairs)

    print("====================================================")
    print(f"[*] Starting edumem LoCoMo Evaluation of {total} questions with 4 threads...")
    print("====================================================")

    # Step 1: Format chronological timeline of sample conversation
    timeline = format_locomo_timeline(conversation_sample["conversation"])
    
    # Step 2: Build chronological profile
    print("[*] Compiling conversation timeline into a chronological memory profile...")
    profile = build_memory_profile(client, timeline)
    os.environ["CURRENT_PROFILE"] = profile
    
    results = []
    correct_count = 0
    cat_stats = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_item, client, i, item): i for i, item in enumerate(eval_qa_pairs)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                idx, category, gold, pred, ok = fut.result()
                correct_count += ok
                results.append((idx, category, gold, pred, ok))
                
                if category not in cat_stats:
                    cat_stats[category] = {"correct": 0, "total": 0}
                cat_stats[category]["total"] += 1
                if ok:
                    cat_stats[category]["correct"] += 1
                
                print(f"[{len(results)}/{total}] idx={idx} category={category} ok={ok} gold='{gold}' pred='{pred[:80]}...'")
            except Exception as e:
                print(f"[!] Error processing item idx={idx}: {e}", file=sys.stderr)

    # Print final summary
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Evaluated: {len(results)}")
    print(f"Overall Accuracy: {correct_count}/{len(results)} ({correct_count/len(results)*100:.2f}%)")
    print("----------------------------------------------------")
    
    cat_names = {
        1: "Single-hop reasoning",
        2: "Temporal reasoning",
        3: "Multi-hop reasoning",
        4: "Open domain knowledge"
    }
    
    for cat, stats in sorted(cat_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        cat_name = cat_names.get(cat, f"Category {cat}")
        print(f"Category: {cat_name:<30} | {stats['correct']}/{stats['total']} ({acc:.2f}%)")
    print("====================================================\n")

if __name__ == "__main__":
    main()