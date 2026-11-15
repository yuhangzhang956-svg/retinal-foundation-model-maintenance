"""Dependency-free synthetic contract check; never reads real patient files."""
import hashlib, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
roles=['SOURCE_TRAIN','SOURCE_VALIDATION','SOURCE_TEST_LOCKED','ADAPT_TRAIN',
       'MODEL_VALIDATION','CALIBRATION_THRESHOLD','TARGET_TEST_LOCKED']
people=[]
for ri,role in enumerate(roles):
    for i in range(20):
        pid=f'SYN_{ri:02d}_{i:03d}'
        y=int(i%4==0)
        images=[(pid+'_0', (i*7+ri*3)%23/23), (pid+'_1', (i*11+ri*5)%29/29)]
        people.append({'id':pid,'role':role,'y':y,'images':images})
assert len(people)==140 and len({p['id'] for p in people})==140
assert all(len(p['images'])==2 for p in people)

def sigmoid(x):return 1/(1+math.exp(-x))
def forward(p):
    # Toy image forward, then the frozen patient maximum used by the real pipeline.
    scores=[sigmoid((1.4 if p['y'] else -1.1)+z-.5) for _,z in p['images']]
    return max(scores)

def policy(p):
    raw=forward(p)
    # Deterministic stand-in for a development-fitted monotone calibration map.
    logit=math.log(raw/(1-raw))
    return sigmoid(.2+.9*logit)

dev=[p for p in people if p['role']=='CALIBRATION_THRESHOLD']
test=[p for p in people if p['role']=='TARGET_TEST_LOCKED']
assert len(dev)==len(test)==20 and set(p['id'] for p in dev).isdisjoint(p['id'] for p in test)
candidates=sorted({policy(p) for p in dev},reverse=True)
valid=[]
for t in candidates:
    tp=sum(p['y'] and policy(p)>=t for p in dev)
    fp=sum((not p['y']) and policy(p)>=t for p in dev)
    if tp/sum(p['y'] for p in dev)>=.8:valid.append((fp,t))
assert valid
threshold=min(valid)[1]  # threshold selected using development labels only
tp=sum(p['y'] and policy(p)>=threshold for p in test)
fn=sum(p['y'] and policy(p)<threshold for p in test)
fp=sum((not p['y']) and policy(p)>=threshold for p in test)
tn=sum((not p['y']) and policy(p)<threshold for p in test)
assert tp+fn+fp+tn==20 and tp+fn==5
result={'status':'PASS_CLEAN_STDLIB_SYNTHETIC_CONTRACT','python_dependency':'stdlib only',
        'synthetic_patients':140,'synthetic_images':280,'role_count':len(roles),
        'dev_n':20,'test_n':20,'locked_threshold':threshold,
        'test_confusion':{'tp':tp,'fn':fn,'fp':fp,'tn':tn},
        'patient_id_sha256':hashlib.sha256('|'.join(p['id'] for p in people).encode()).hexdigest(),
        'real_patient_or_foundation_model_reproduction':False}
(ROOT/'clean_smoke_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
