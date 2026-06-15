import csv, json
bench = list(csv.DictReader(open('data_hf/benchmark/text/benchmark.csv')))
terms_by_idx = {
    26: ['run','trail','endurance','ultra','marathon','pike'],
    3: ['surf','coast','wave'],
    5: ['back','sports injury','desk','stretch'],
    31: ['herbal','coffee','tea'],
    68: ['documentar','science','technology'],
    57: ['homophobia','inclusive','sports'],
    59: ['acoustic','music','performance'],
    67: ['premier','football','score'],
}
for idx, terms in terms_by_idx.items():
    row = bench[idx]
    data = json.load(open('data_hf/' + row['chat_history_32k_link']))
    hist = data.get('chat_history') or data.get('conversations') or []
    print('\nIDX', idx, row['preference'])
    for n, m in enumerate(hist):
        c = str(m.get('content', ''))
        if any(t.lower() in c.lower() for t in terms):
            print(n, m.get('role'), c[:500].replace('\n', ' '))
