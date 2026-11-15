"""Independent R2-1 audit of locked manifests. Never writes patient identifiers."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
AUDIT = ROOT / "outputs" / "audit"

def read(name):
    with (AUDIT / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def grade(value):
    return int(float(value))

def audit(dataset):
    all_rows=read(f"{dataset}_master_manifest_v2.csv")
    by_patient=defaultdict(list)
    for r in all_rows: by_patient[r["patient_id"]].append(r)
    included=[r for r in all_rows if r["exclusion_status"]=="INCLUDED"]
    included_pids={r["patient_id"] for r in included}
    assert all(r["exclusion_status"] in {"INCLUDED","EXCLUDED"} for r in all_rows)
    reason_rows=defaultdict(list)
    for r in all_rows:
        reason_rows[r["exclusion_reason"] if r["exclusion_status"]=="EXCLUDED" else "INCLUDED"].append(r)
    contradiction_pids={r["patient_id"] for r in reason_rows.get("contradictory_patient_level_diabetes_status", [])}
    flow=[]
    for reason, rr in sorted(reason_rows.items()):
        pids={r["patient_id"] for r in rr}
        exclusive=(len(pids-contradiction_pids) if dataset=="BRSET" and reason=="not_documented_diabetic"
                   else len(pids) if reason in ("INCLUDED","contradictory_patient_level_diabetes_status")
                   else len(pids-included_pids))
        flow.append({"dataset":dataset,"reason_or_status":reason,"patients_with_rows":len(pids),
                     "images":len(rr),"exclusive_stage_patients":exclusive,
                     "patients_not_in_final":len(pids-included_pids),
                     "patients_partly_retained":len(pids & included_pids),
                     "overlap_with_contradictory_diabetes_patients":len(pids & contradiction_pids) if reason=="not_documented_diabetic" else 0})
    groups=defaultdict(list)
    for r in included: groups[r["split"]].append(r)
    split_rows=[]; split_pids={}
    for split, rr in sorted(groups.items()):
        pats=defaultdict(list)
        for r in rr: pats[r["patient_id"]].append(r)
        split_pids[split]=set(pats)
        pos=sum(grade(pr[0]["referable_dr_patient"]) for pr in pats.values())
        split_rows.append({"dataset":dataset,"split":split,"patients":len(pats),"images":len(rr),
                           "rdr_positive_patients":pos,"rdr_negative_patients":len(pats)-pos,
                           "rdr_prevalence":pos/len(pats)})
    assert sum(r["patients"] for r in split_rows)==len(included_pids)
    assert sum(r["images"] for r in split_rows)==len(included)
    assert all(not (split_pids[a]&split_pids[b]) for i,a in enumerate(split_pids) for b in list(split_pids)[i+1:])
    image_bad=[]; eye_bad=[]; patient_bad=[]
    eyes=defaultdict(list); patients=defaultdict(list)
    for r in included:
        g=grade(r["original_icdr"])
        if grade(r["referable_dr_image"]) != int(g>=2): image_bad.append(r["image_id"])
        eyes[(r["patient_id"],r["eye_laterality"])].append(r)
        patients[r["patient_id"]].append(r)
    for rr in eyes.values():
        expected=max(grade(r["original_icdr"]) for r in rr)
        if any(grade(r["icdr_eye"])!=expected or grade(r["referable_dr_eye"])!=int(expected>=2) for r in rr):
            eye_bad.append(rr[0]["patient_id"])
    for rr in patients.values():
        expected=max(grade(r["original_icdr"]) for r in rr)
        if any(grade(r["icdr_patient"])!=expected or grade(r["referable_dr_patient"])!=int(expected>=2) for r in rr):
            patient_bad.append(rr[0]["patient_id"])
    assert not image_bad and not eye_bad and not patient_bad, (len(image_bad),len(eye_bad),len(patient_bad))
    k_rows=[]
    for split,rr in sorted(groups.items()):
        pats=defaultdict(list)
        for r in rr: pats[r["patient_id"]].append(r)
        for k in (1,2,3,4):
            eligible=[pr for pr in pats.values() if len(pr)>=k]
            pos=sum(grade(pr[0]["referable_dr_patient"]) for pr in eligible)
            k_rows.append({"dataset":dataset,"split":split,"fixed_images_per_patient_k":k,
                           "eligible_patients":len(eligible),"eligible_rdr_positive":pos,
                           "excluded_for_fewer_than_k":len(pats)-len(eligible),
                           "original_images_in_eligible_patients":sum(map(len,eligible)),
                           "images_after_fixed_k":k*len(eligible)})
    audit={"dataset":dataset,"original_rows":len(all_rows),"original_patients":len(by_patient),
           "final_images":len(included),"final_patients":len(included_pids),
           "final_rdr_positive":sum(grade(pr[0]["referable_dr_patient"]) for pr in patients.values()),
           "final_eyes":len(eyes),"label_mismatches":{"image":len(image_bad),"eye":len(eye_bad),"patient":len(patient_bad)},
           "split_patient_overlap":0}
    hashes=defaultdict(set)
    for r in included:
        h=r.get("source_image_sha256","")
        if h: hashes[h].add((r["split"],r["patient_id"]))
    audit["duplicate_image_hashes_across_splits"]=sum(len({x[0] for x in owners})>1 for owners in hashes.values())
    assert audit["duplicate_image_hashes_across_splits"]==0
    return flow,split_rows,k_rows,audit

if __name__=="__main__":
    flow=[]; splits=[]; k=[]; reports=[]
    for ds in ("BRSET","mBRSET"):
        a,b,c,d=audit(ds);flow+=a;splits+=b;k+=c;reports.append(d)
    write(OUT/"data_audit"/"r21_flow_reconstructed.csv",flow)
    write(OUT/"data_audit"/"r21_split_prevalence.csv",splits)
    write(OUT/"data_audit"/"r21_fixed_k_eligibility.csv",k)
    (OUT/"data_audit"/"r21_label_integrity.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"flow":flow,"label_integrity":reports},ensure_ascii=False,indent=2))
