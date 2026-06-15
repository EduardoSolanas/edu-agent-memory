#!/usr/bin/env python3
import csv, hashlib, json, time
from pathlib import Path
import requests

QDRANT='http://127.0.0.1:6333'
EMBED='http://127.0.0.1:3002/embed'
COLLECTION='pm_shared'
DATA=Path('/opt/mnemosyne/data/personamem')
QUESTIONS=DATA/'questions_32k.csv'
CONTEXTS=DATA/'shared_contexts_32k.jsonl'
VECTOR_SIZE=768
TARGET_IDXS={0,1,3,6,8,9,10,11,12,15,16,17,19,20,22,25,26,27,34,35,36,37,38,39,40,41,42,43,45,47,48,50,51,53,55,56,57,58,59,60,61,62,63,64,65,68,71,74,75,77,78,79,80,81,82,83,84,85,86,99,114,182,205,220,238,254,275,332,359,374}

def req(method,path,**kw):
    r=requests.request(method,QDRANT+path,timeout=180,**kw); r.raise_for_status(); return r.json() if r.content else None

def ensure_collection():
    r=requests.get(f'{QDRANT}/collections/{COLLECTION}',timeout=30)
    if r.status_code==404:
        req('PUT',f'/collections/{COLLECTION}',json={'vectors':{'size':VECTOR_SIZE,'distance':'Cosine'},'on_disk_payload':True})
    elif r.status_code!=200:
        r.raise_for_status()

def clear_targets():
    ids=[]
    for idx in TARGET_IDXS:
        user=f'pm_exper_user_{idx}_default'
        # Delete by filter so reruns are clean.
        req('POST',f'/collections/{COLLECTION}/points/delete?wait=true',json={'filter':{'must':[{'key':'user_id','match':{'value':user}}]}})

def build_index():
    out={}
    with CONTEXTS.open(encoding='utf-8') as f:
        while True:
            off=f.tell(); line=f.readline()
            if not line: break
            item=json.loads(line); out[next(iter(item.keys()))]=off
    return out

def load_context(index,sid):
    with CONTEXTS.open(encoding='utf-8') as f:
        f.seek(index[sid]); item=json.loads(f.readline()); return next(iter(item.values()))

def embed(texts):
    r=requests.post(EMBED,json={'inputs':texts},timeout=240); r.raise_for_status(); return r.json()

def pid(user,turn):
    h=hashlib.sha256(f'{user}:{turn}'.encode()).hexdigest()
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}'

def upsert(points):
    req('PUT',f'/collections/{COLLECTION}/points?wait=true',json={'points':points})

def main():
    ensure_collection(); clear_targets(); index=build_index()
    pending_texts=[]; pending_payloads=[]; total=0; users=0
    with QUESTIONS.open(newline='',encoding='utf-8') as f:
        for conv_idx,row in enumerate(csv.DictReader(f)):
            if conv_idx not in TARGET_IDXS: continue
            users+=1
            user=f'pm_exper_user_{conv_idx}_default'
            ctx=load_context(index,row['shared_context_id'])[:int(row['end_index_in_shared_context'])]
            for turn,msg in enumerate(ctx):
                content=(msg.get('content') or '').strip()
                if not content: continue
                role=msg.get('role') or ''
                text=f'{role}: {content}' if role else content
                pending_texts.append(content)  # embed raw content, store role-prefixed text
                pending_payloads.append({
                    'id':pid(user,turn),'text':text,'agent_id':'agent-memory-personamem-quick-import',
                    'user_id':user,'memory_type':'personamem_turn','scope':'public','classification':'public',
                    'importance':1.0,'source':f'turn_{turn}','conversation_index':conv_idx,'turn_index':turn,
                    'shared_context_id':row['shared_context_id'],'question_type':row.get('question_type'),'topic':row.get('topic'),
                    'event_time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'deleted':False,
                    'metadata':{'persona_id':row.get('persona_id'),'question_id':row.get('question_id')}
                })
                if len(pending_texts)>=64:
                    vecs=embed(pending_texts); upsert([{'id':p.pop('id'),'vector':v,'payload':p} for p,v in zip(pending_payloads,vecs)])
                    total+=len(vecs); print(f'imported_messages={total}',flush=True)
                    pending_texts=[]; pending_payloads=[]
    if pending_texts:
        vecs=embed(pending_texts); upsert([{'id':p.pop('id'),'vector':v,'payload':p} for p,v in zip(pending_payloads,vecs)])
        total+=len(vecs); print(f'imported_messages={total}',flush=True)
    print(json.dumps({'collection':COLLECTION,'users':users,'messages':total},indent=2))

if __name__=='__main__': main()
