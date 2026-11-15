"""Aggregate only the archived mBRSET local-100 patients; no IDs leave this process."""
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent

def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

subset = rows(BASE / "private_predictions" / "01_locked_100_patient_development_split.csv")
ids = {r["patient_id"] for r in subset}
assert len(ids) == len(subset) == 100
manifest = rows(BASE.parent / "outputs" / "audit" / "mBRSET_master_manifest_v2.csv")
images = [r for r in manifest if r["patient_id"] in ids and r["exclusion_status"] == "INCLUDED"]
patients = defaultdict(list)
for image in images:
    patients[image["patient_id"]].append(image)
assert set(patients) == ids and len(images) == 375
assert all(all(r["split"] == "ADAPTATION_POOL" for r in rr) for rr in patients.values())
first = [rr[0] for rr in patients.values()]
assert sum(int(float(r["referable_dr_patient"])) for r in first) == 23
out=[]
def add(unit, variable, category, count, denom, value=""):
    out.append({"cohort":"mBRSET_HISTORICAL_LOCAL100", "unit":unit,"variable":variable,
                "category":category,"count":count,"denominator":denom,"value":value})
ages = sorted(float(r["age_resolved"]) for r in first if r["age_resolved"] and r["age_resolved"].replace(".","",1).isdigit())
add("patient","age_resolved","median [Q1,Q3]",len(ages),100,
    f"{statistics.median(ages)} [{ages[int((len(ages)-1)*.25)]},{ages[int((len(ages)-1)*.75)]}]" if ages else "NA")
for category,count in sorted(Counter(r["age_resolved"] or "<missing>" for r in first
                                     if not (r["age_resolved"] and r["age_resolved"].replace(".","",1).isdigit())).items()):
    add("patient","age_non_numeric_or_censored",category,count,100)
for variable, sample, unit in (("sex_resolved",first,"patient"),
                               ("source_quality_standardized",images,"image"),
                               ("source_device_standardized",images,"image")):
    for category,count in sorted(Counter(r.get(variable) or "<missing>" for r in sample).items()):
        add(unit,variable,category,count,len(sample))
for category,count in sorted(Counter(len(rr) for rr in patients.values()).items()):
    add("patient","eligible_images_per_patient",category,count,100)
dest=BASE/"data_audit"/"r21_local100_metadata_20261111.csv"
with dest.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(f"local100_patients={len(first)} images={len(images)} age_numeric={len(ages)} aggregate_rows={len(out)}")
