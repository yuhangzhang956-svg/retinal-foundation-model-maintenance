"""Recover Platt maps from frozen pairs using executed input dtypes.

VisionFM's patient probabilities were emitted as float32. The executed
``logit_np`` first clipped and transformed those float32 values. Fitting a
float64 transform instead creates spurious errors at extreme probabilities.
"""
from __future__ import annotations
import csv,math,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
P=BASE/"private_predictions";OUT=BASE/"analyses"/"recovered_calibration_maps.csv"
def read(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def logit(v):
    p=max(1e-7,min(1-1e-7,float(v)));return math.log(p/(1-p))
def output_logit(v):
    p=max(1e-15,min(1-1e-15,float(v)));return math.log(p/(1-p))
def sigmoid(z):
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
def fit(rr,xfield,yfield):
    x=[logit(r[xfield]) for r in rr];y=[output_logit(r[yfield]) for r in rr]
    mx=statistics.mean(x);my=statistics.mean(y)
    a=sum((u-mx)*(v-my) for u,v in zip(x,y))/sum((u-mx)**2 for u in x)
    b=my-a*mx
    residual=max(abs(sigmoid(a*u+b)-float(r[yfield])) for u,r in zip(x,rr))
    return a,b,residual
def fit_float32_visionfm(rr,xfield,yfield):
    # Mirrors the executed phase2a1_visionfm_cpu.py logit_np path.
    p=np.asarray([float(r[xfield]) for r in rr],dtype=np.float32)
    clipped=np.clip(p,1e-7,1-1e-7)
    x=np.log(clipped/(1-clipped)).astype(np.float64)
    q=np.asarray([float(r[yfield]) for r in rr],dtype=np.float64)
    y=np.log(q/(1-q))
    slope,intercept=np.polyfit(x,y,1)
    pred=1/(1+np.exp(-(intercept+slope*x)))
    return float(slope),float(intercept),float(np.max(np.abs(pred-q)))
rows=[]
ret=defaultdict(list)
for r in read(P/"02_eval_predictions_DeepDRiD.csv"):ret[r["seed"]].append(r)
for seed,rr in sorted(ret.items()):
    for label,target in (("sourcecal","p_sourcecal"),("localcal","p_localcal")):
        a,b,res=fit(rr,"p_raw",target)
        rows.append({"backbone":"RETFound","domain":"DeepDRiD","seed":seed,
                     "map_from":"p_raw","map_to":label,"slope":a,"intercept":b,
                     "max_probability_residual":res,"pairs":len(rr),
                     "verification_status":"PAIR_RECONSTRUCTION_PASS" if res < 1e-8 else "UNVERIFIED_PAIR_MISMATCH",
                     "input_numeric_path":"float64_probability_logit",
                     "provenance_type":"recovered from frozen evaluation probability pairs; not executed fit coefficient file"})
vision=defaultdict(list)
for r in read(P/"02_patient_predictions_VisionFM_source.csv"):vision[(r["seed"],r["domain"])].append(r)
for (seed,domain),rr in sorted(vision.items()):
    for label,target in (("sourcecal","p_sourcecal"),("localcal","p_localcal")):
        if not rr[0][target]:continue
        a,b,res=fit_float32_visionfm(rr,"p",target)
        rows.append({"backbone":"VisionFM","domain":domain,"seed":seed,
                     "map_from":"p","map_to":label,"slope":a,"intercept":b,
                     "max_probability_residual":res,"pairs":len(rr),
                     "verification_status":"PAIR_RECONSTRUCTION_PASS" if res < 1e-8 else "UNVERIFIED_PAIR_MISMATCH",
                     "input_numeric_path":"float32_numpy_clip_and_logit_as_executed",
                     "provenance_type":"recovered from frozen evaluation probability pairs; not executed fit coefficient file"})
with OUT.open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("maps",len(rows),"max_residual",max(r["max_probability_residual"] for r in rows))
