#!/usr/bin/env python3
import os
import sys
import json
import csv
import ast
import random
from pathlib import Path
from openai import OpenAI

WORKDIR = Path(__file__).resolve().parent
LOCAL = WORKDIR / "data_hf"

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

def get_row_by_idx(idx):
    local_csv = LOCAL / "benchmark/text/benchmark.csv"
    with local_csv.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        return reader[idx]

def main():
    load_env(WORKDIR / ".env")
    load_env(Path("/root/.hermes/.env"))
    client = OpenAI(api_key=os.getenv("CHAT_MODEL_API_KEY") or os.getenv("NAN_APY_KEY"), base_url=os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1"))

    # Let's inspect idx=3, idx=10, idx=28, idx=58
    target_idxs = [3, 10, 28, 58]
    for idx in target_idxs:
        row = get_row_by_idx(idx)
        pid = row["persona_id"]
        profile_path = WORKDIR / "results/profiles" / f"persona_{pid}.md"
        profile = profile_path.read_text() if profile_path.exists() else "No profile"

        user_query = parse_msg(row["user_query"])["content"]
        mapping, gold = parse_options(row, seed=idx)

        print(f"\n==============================================")
        print(f"[*] DIAGNOSING idx={idx} (persona={pid})")
        print(f"[*] Preference: {row['preference']}")
        print(f"[*] User Query: {user_query}")
        print(f"[*] Gold Answer ({gold}): {mapping[gold]}")
        print(f"[*] Options:")
        for k, v in mapping.items():
            print(f"  {k}. {v}")

        prompt = f"""You answer PersonaMem-v2 multiple-choice personalization questions.
Use the compact user memory profile to infer which response best fits this user's implicit preferences.
Respect privacy/do-not-remember constraints.

USER MEMORY PROFILE:
{profile}

USER QUERY:
{user_query}

OPTIONS:
""" + "\n".join(f"{k}. {v}" for k, v in mapping.items())

        resp = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "qwen3.6"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        print(f"\n[*] MODEL REASONING:")
        print(resp.choices[0].message.content)
        print(f"==============================================\n")

if __name__ == "__main__":
    main()
