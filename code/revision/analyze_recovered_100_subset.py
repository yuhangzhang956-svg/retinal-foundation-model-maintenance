"""Verify archived 100-patient mBRSET development subset against locked manifest."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
BASE=Path(__file__).resolve().parent
PRIV=BASE/"private_predictions"
MAN=BASE.parent/"outputs"/"audit"/"mBRSET_master_manifest_v2.csv"

def read(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))

subset=read(PRIV/"01_locked_100_patient_development_split.csv")
trajectories=read(PRIV/"01_sampling_trajectories.csv")
manifest=read(MAN)
pats=defaultdict(list)
for r in manifest:
    if r["exclusion_status"]=="INCLUDED":pats[r["patient_id"]].append(r)
ids=[r["patient_id"] for r in subset]
assert len(ids)==len(set(ids))==100
assert all(pid in pats and all(r["split"]=="ADAPTATION_POOL" for r in pats[pid]) for pid in ids)
assert sum(len(pats[pid]) for pid in ids)==375
labels={pid:int(float(pats[pid][0]["referable_dr_patient"])) for pid in ids}
seed100=[r for r in trajectories if r["trajectory_seed"]=="20261001" and int(r["budget"])==100]
trajectory_ids={r["patient_id"] for r in seed100}
same=(trajectory_ids==set(ids))
assert len(seed100)==100 and same
assert all(int(float(r["referable_dr_patient"]))==labels[r["patient_id"]] for r in seed100)
result={"development_subset_patients":100,"eligible_images":375,
        "rdr_positive_patients":sum(labels.values()),"rdr_negative_patients":100-sum(labels.values()),
        "contained_in_adaptation_pool":True,"overlap_with_frozen_target_test":0,
        "matches_trajectory_seed_20261001_budget_100":same,
        "patient_identifiers_public_output":False}
(BASE/"data_audit"/"r21_100_subset_verified.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
