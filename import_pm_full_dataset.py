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
BATCH=128

def req(method,path,**kw):
    r=requests.request(method,QDRANT+path,timeout=300,**kw)
    r.raise_for_status()
    return r.json() if r.content else None

def recreate_collection():
    r=requests.get(f'{QDRANT}/collections/{COLLECTION}',timeout=30)
    if r.status_code==200:
        req('DELETE',f'/collections/{COLLECTION}')
    elif r.status_code!=404:
        r.raise_for_status()
    req('PUT',f'/collections/{COLLECTION}',json={'vectors':{'size':VECTOR_SIZE,'distance':'Cosine'},'on_disk_payload':True})

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
    r=requests.post(EMBED,json={'inputs':texts},timeout=300)
    r.raise_for_status()
    return r.json()

def pid(user,turn):
    h=hashlib.sha256(f'{user}:{turn}'.encode()).hexdigest()
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}'

def upsert(points):
    req('PUT',f'/collections/{COLLECTION}/points?wait=true',json={'points':points})

def flush(texts,payloads,total):
    if not texts: return total
    vecs=embed(texts)
    points=[]
    for payload,vector in zip(payloads,vecs):
        item_id=payload.pop('id')
        points.append({'id':item_id,'vector':vector,'payload':payload})
    upsert(points)
    total+=len(points)
    print(f'imported_messages={total}',flush=True)
    return total

def main():
    recreate_collection()
    index=build_index()
    texts=[]; payloads=[]; total=0; users=0
    with QUESTIONS.open(newline='',encoding='utf-8') as f:
        for conv_idx,row in enumerate(csv.DictReader(f)):
            users+=1
            user=f'pm_exper_user_{conv_idx}_default'
            ctx=load_context(index,row['shared_context_id'])[:int(row['end_index_in_shared_context'])]
            for turn,msg in enumerate(ctx):
                content=(msg.get('content') or '').strip()
                if not content: continue
                role=msg.get('role') or ''
                text=f'{role}: {content}' if role else content
                texts.append(content)
                payloads.append({
                    'id':pid(user,turn),'text':text,'agent_id':'agent-memory-personamem-full-import',
                    'user_id':user,'memory_type':'personamem_turn','scope':'public','classification':'public',
                    'importance':1.0,'source':f'turn_{turn}','conversation_index':conv_idx,'turn_index':turn,
                    'shared_context_id':row['shared_context_id'],'question_type':row.get('question_type'),'topic':row.get('topic'),
                    'event_time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'deleted':False,
                    'metadata':{'persona_id':row.get('persona_id'),'question_id':row.get('question_id')}
                })
                if len(texts)>=BATCH:
                    total=flush(texts,payloads,total); texts=[]; payloads=[]
    total=flush(texts,payloads,total)
    print(json.dumps({'collection':COLLECTION,'users':users,'messages':total},indent=2),flush=True)

if __name__=='__main__': main()
