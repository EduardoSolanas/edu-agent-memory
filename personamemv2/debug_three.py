import json
import os
from pathlib import Path
from openai import OpenAI
from benchmarks.evidence import best_evidence
from benchmarks.e2e.three_personas import load_env, parse_msg, parse_options, answer_custom, PROJECT_ROOT

load_env(PROJECT_ROOT / '.env')
load_env(Path('/root/.hermes/.env'))
client = OpenAI(
    api_key=os.getenv('CHAT_MODEL_API_KEY') or os.getenv('NAN_APY_KEY') or os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('CHAT_MODEL_BASE_URL', 'https://api.nan.builders/v1'),
)
data = json.load(open('benchmarks/e2e/fixtures/three_personas.json'))
for n in [1, 2]:
    item = data[n]
    row = item['row']
    history_obj = item['history']
    history = history_obj.get('chat_history') or history_obj.get('conversations') or []
    profile = open(f"results/profiles/persona_{row['persona_id']}.md").read()
    query = parse_msg(row['user_query'])['content']
    mapping, gold = parse_options(row, seed=item['idx'])
    evidence = best_evidence(history, query, mapping)
    pred, text = answer_custom(client, profile, evidence, query, mapping)
    print('\n====', n, 'pid', row['persona_id'], 'gold', gold, 'pred', pred, 'pref', row['preference'])
    print('OPTIONS')
    for k, v in mapping.items():
        print(k, v[:180].replace('\n', ' '))
    print('EVIDENCE')
    print(evidence[:2500])
    print('REASON TAIL')
    print(text[-3000:])
