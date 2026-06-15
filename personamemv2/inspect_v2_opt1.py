import sys
import os
from pathlib import Path

sys.path.insert(0, "/opt/mnemosyne/personamemv2")
from run_mnemosyne_v2_profile_smoke import load_env, load_rows, load_history, parse_options, parse_msg

load_env(Path("/opt/mnemosyne/personamemv2/.env"))
rows = load_rows(10)

for i, row in enumerate(rows):
    mapping, gold = parse_options(row, seed=i)
    print(f"\n--- SCENARIO {i} ---")
    print("Gold:", gold)
    print("Preference:", row["preference"])
    print("Query:", parse_msg(row["user_query"])["content"])
    print("Options:")
    for k,v in mapping.items():
        print(f"  {k}: {v}")
