"""Lock patient roles for *future* unified experiments; no model fitting or test selection.

The new 200/200 target role split is a revision-period design, not a historical
or prospective preregistration. Existing frozen test partitions remain intact.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
OUT = BASE / "private_manifests"
OUT.mkdir(exist_ok=True)
SEED = 20261109


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_patients(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["exclusion_status"] == "INCLUDED"].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    assert df.groupby("patient_id")["split"].nunique().max() == 1
    assert df.groupby("patient_id")["referable_dr_patient"].nunique().max() == 1
    p = df.groupby("patient_id").agg(split=("split", "first"),
                                      y=("referable_dr_patient", "first"),
                                      n_images=("image_id", "size"))
    assert p["y"].isin([0, 1]).all()
    return p


br_path = PROJECT / "outputs" / "audit" / "BRSET_master_manifest_v2.csv"
mb_path = PROJECT / "outputs" / "audit" / "mBRSET_master_manifest_v2.csv"
local_path = BASE / "private_predictions" / "01_locked_100_patient_development_split.csv"
dd_val_path = BASE / "private_predictions" / "00_validation_patient_predictions_DeepDRiD.csv"
dd_test_path = BASE / "private_predictions" / "deepdrid_frozen_confirmatory_manifest.csv"
br = manifest_patients(br_path)
mb = manifest_patients(mb_path)
local = pd.read_csv(local_path)
local_ids = set(local["patient_id"].astype(str))
assert len(local_ids) == 100
assert local["split"].value_counts().to_dict() == {"LOCAL_TRAIN": 70, "LOCAL_VALIDATION": 30}
assert set(local_ids).issubset(set(mb[mb.split == "ADAPTATION_POOL"].index))
assert (mb.loc[list(local_ids), "y"].sum(), mb.loc[list(local_ids), "n_images"].sum()) == (23, 375)
assert len(br) == 1343 and len(mb) == 1287

remaining = mb[(mb.split == "ADAPTATION_POOL") & (~mb.index.isin(local_ids))]
assert len(remaining) == 787
assert int(remaining.y.sum()) == 183
rng = np.random.default_rng(SEED)
selection = {}
for label in (0, 1):
    ids = np.asarray(sorted(remaining[remaining.y == label].index), dtype=object)
    rng.shuffle(ids)
    count = 46 if label == 1 else 154
    selection[("MODEL_VALIDATION", label)] = set(ids[:count])
    selection[("CALIBRATION_THRESHOLD", label)] = set(ids[count:2 * count])
val_ids = selection[("MODEL_VALIDATION", 0)] | selection[("MODEL_VALIDATION", 1)]
cal_ids = selection[("CALIBRATION_THRESHOLD", 0)] | selection[("CALIBRATION_THRESHOLD", 1)]
assert len(val_ids) == len(cal_ids) == 200
assert not (val_ids & cal_ids or val_ids & local_ids or cal_ids & local_ids)

rows = []
for pid, row in br.iterrows():
    role = {"TRAIN": "SOURCE_TRAIN", "VALIDATION": "SOURCE_VALIDATION",
            "FROZEN_SOURCE_TEST": "SOURCE_TEST_LOCKED"}[row["split"]]
    rows.append({"dataset": "BRSET", "patient_id": pid, "role": role,
                 "y": int(row.y), "n_images": int(row.n_images)})
for pid, row in mb.iterrows():
    if row["split"] == "FROZEN_TARGET_TEST":
        role = "TARGET_TEST_LOCKED"
    elif pid in local_ids:
        role = "ADAPT_TRAIN"
    elif pid in val_ids:
        role = "MODEL_VALIDATION"
    elif pid in cal_ids:
        role = "CALIBRATION_THRESHOLD"
    else:
        role = "ADAPT_POOL_UNUSED"
    rows.append({"dataset": "mBRSET", "patient_id": pid, "role": role,
                 "y": int(row.y), "n_images": int(row.n_images)})

dd_val = pd.read_csv(dd_val_path)
assert dd_val.groupby("patient_id")["y"].nunique().max() == 1
dd_val = dd_val.groupby("patient_id").y.first()
dd_test = pd.read_csv(dd_test_path)
assert dd_test.groupby("patient_id")["referable_dr_patient"].nunique().max() == 1
dd_test_pt = dd_test.groupby("patient_id").agg(y=("referable_dr_patient", "first"),
                                                n_images=("image_id", "size"))
assert len(dd_val) == len(dd_test_pt) == 100
assert set(dd_val.index).isdisjoint(set(dd_test_pt.index))
for pid, y in dd_val.items():
    rows.append({"dataset": "DeepDRiD", "patient_id": str(pid),
                 "role": "EXTERNAL_CALIBRATION_THRESHOLD", "y": int(y), "n_images": 4})
for pid, row in dd_test_pt.iterrows():
    rows.append({"dataset": "DeepDRiD", "patient_id": str(pid),
                 "role": "EXTERNAL_TEST_LOCKED", "y": int(row.y), "n_images": int(row.n_images)})

out = OUT / "unified_experiment_patient_roles_PRIVATE.csv"
buffer = io.StringIO(newline="")
writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
proposed_bytes = b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")
if out.exists():
    assert out.read_bytes() == proposed_bytes, "Locked patient roles changed; investigate without overwriting"
else:
    out.write_bytes(proposed_bytes)
assert len({(r["dataset"], r["patient_id"]) for r in rows}) == len(rows)
counts = {}
for row in rows:
    key = (row["dataset"], row["role"])
    c = counts.setdefault(key, {"dataset": key[0], "role": key[1], "patients": 0,
                                "positive": 0, "images": 0})
    c["patients"] += 1
    c["positive"] += row["y"]
    c["images"] += row["n_images"]
count_rows = sorted(counts.values(), key=lambda r: (r["dataset"], r["role"]))
counts_out = BASE / "analyses" / "unified_experiment_role_counts_20261109.csv"
with counts_out.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(count_rows[0]))
    writer.writeheader()
    writer.writerows(count_rows)
lock = {"status": "INTERNAL_REVISION_PERIOD_ROLE_LOCK_NOT_HISTORICAL_PREREGISTRATION",
        "random_seed": SEED, "source_files_sha256": {str(p): sha(p) for p in
          (br_path, mb_path, local_path, dd_val_path, dd_test_path)},
        "private_role_manifest": str(out.relative_to(BASE)), "private_role_manifest_sha256": sha(out),
        "role_counts": count_rows,
        "target_label_budget_note": "100 adaptation-train + 200 model-validation + 200 calibration/threshold = 500 distinct mBRSET labelled patients; not a 100-label end-to-end workflow",
        "original_local100_note": "The historical RETFound experiment divided these 100 into 70 train and 30 validation; new unified design uses all 100 for adaptation training and distinct validation patients.",
        "frozen_target_test_used_for_selection": False}
lock_out = BASE / "reproducibility" / "unified_experiment_role_lock_20261109.json"
proposed_lock = json.dumps(lock, ensure_ascii=False, indent=2)
if lock_out.exists():
    assert lock_out.read_text(encoding="utf-8") == proposed_lock, "Role lock metadata changed; investigate"
else:
    lock_out.write_text(proposed_lock, encoding="utf-8")
print("ROLE_LOCK", len(rows), sha(out))
for r in count_rows:
    print(r)
