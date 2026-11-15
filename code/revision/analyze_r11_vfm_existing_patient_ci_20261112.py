"""Derive patient CIs from existing, locked VisionFM source predictions only."""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

base=Path(__file__).resolve().parent
private=base/'private_predictions'
out=base/'unified_experiment/r11_matched_source_20261112/visionfm_existing_patient_auroc_ci.csv'
rows=[]
for seed in (20267001,20267002,20267003):
    for domain,role in (('BRSET','SOURCE_TEST_LOCKED'),('mBRSET','TARGET_TEST_LOCKED')):
        p=private/f'UNIFIED_R20261109_B3_seed{seed}_source_{domain}_{role}_patients_PRIVATE.csv'
        d=pd.read_csv(p)
        assert len(d)==(202 if domain=='BRSET' else 400)
        assert d.patient_id.nunique()==len(d)
        y=d.y.to_numpy(dtype=int);score=d.p_float32.to_numpy(dtype=float)
        assert int(y.sum())==(53 if domain=='BRSET' else 93)
        auc=float(roc_auc_score(y,score))
        rng=np.random.default_rng(seed+(1000 if domain=='BRSET' else 2000))
        pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
        boots=[]
        for _ in range(2000):
            ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            boots.append(roc_auc_score(y[ix],score[ix]))
        lo,hi=np.quantile(boots,[.025,.975])
        rows.append({'model':'VisionFM','seed':seed,'cohort':domain,'patients':len(d),
                     'positive':int(y.sum()),'auroc':auc,'ci_low':float(lo),
                     'ci_high':float(hi),'bootstrap':2000,
                     'source':'existing unified float32 patient predictions; no model rerun'})
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(rows)
