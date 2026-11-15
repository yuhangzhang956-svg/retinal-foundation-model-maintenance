"""Audit historical recalibration composites under two shared post hoc rules.

No criterion is selected as primary. Both Brier cutoffs are shown because
RETFound and VisionFM executed scripts used different cutoffs. The remaining
four components follow their shared executed definitions.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(__file__).resolve().parent

def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def as_bool(s):
    assert s in ("True", "False")
    return s == "True"

clinical = {}
for row in read(ROOT / "source_snapshots" / "clinical_seed.csv"):
    clinical[(int(row["seed"]), row["domain"], row["strategy"])] = row

rows = []
ret = read(BASE / "private_predictions" / "01_seed_level_recalibration_metrics_DeepDRiD.csv")
for r in ret:
    rows.append({
        "backbone": "RETFound", "domain": "DeepDRiD", "seed": int(r["seed"]),
        "brier_gain": float(r["Brier_gain"]),
        "brier_gain_ci_low": float(r["brier_gain_CI_low"]),
        "nll_gain": float(r["NLL_gain"]),
        "sourcecal_slope": float(r["sourcecal_slope"]),
        "localcal_slope": float(r["localcal_slope"]),
        "source_balanced_accuracy": float(r["source_threshold_balanced_accuracy"]),
        "local_balanced_accuracy": float(r["local_threshold_balanced_accuracy"]),
        "historical_success": as_bool(r["recalibration_success"]),
        "historical_brier_cutoff": ">=0.01_and_CI_lower>0",
    })

vision = read(ROOT / "source_snapshots" / "VisionFM_transport.csv")
for r in vision:
    seed = int(r["seed"])
    for domain, prefix in (("mBRSET", "MB"), ("DeepDRiD", "DD")):
        source = clinical[(seed, domain, "Source calibration + source threshold")]
        local = clinical[(seed, domain, "Local recalibration + local threshold")]
        source_ba = (float(source["sensitivity"]) + float(source["specificity"]))/2
        local_ba = (float(local["sensitivity"]) + float(local["specificity"]))/2
        rows.append({
            "backbone": "VisionFM", "domain": domain, "seed": seed,
            "brier_gain": float(r[f"{prefix}_Brier_gain"]),
            "brier_gain_ci_low": float(r[f"{prefix}_brier_gain_CI_low"]),
            "nll_gain": float(r[f"{prefix}_NLL_gain"]),
            "sourcecal_slope": float(r[f"{prefix}_sourcecal_slope"]),
            "localcal_slope": float(r[f"{prefix}_localcal_slope"]),
            "source_balanced_accuracy": source_ba,
            "local_balanced_accuracy": local_ba,
            "historical_success": as_bool(r[f"{prefix}_recalibration_success"]),
            "historical_brier_cutoff": ">0_and_CI_lower>0",
        })

for r in rows:
    brier_pos = r["brier_gain"] > 0
    brier_abs = r["brier_gain"] >= 0.01
    ci_pos = r["brier_gain_ci_low"] > 0
    nll_pos = r["nll_gain"] > 0
    slope_better = abs(r["localcal_slope"] - 1) < abs(r["sourcecal_slope"] - 1)
    ba_safe = r["local_balanced_accuracy"] >= r["source_balanced_accuracy"] - 0.05
    r.update({
        "brier_gain_positive": brier_pos,
        "brier_gain_ge_0_01": brier_abs,
        "brier_ci_low_positive": ci_pos,
        "nll_gain_positive": nll_pos,
        "slope_closer_to_one": slope_better,
        "balanced_accuracy_within_0_05": ba_safe,
        "shared_composite_positive_brier": brier_pos and ci_pos and nll_pos and slope_better and ba_safe,
        "shared_composite_brier_ge_0_01": brier_abs and ci_pos and nll_pos and slope_better and ba_safe,
        "interpretation": "post_hoc_shared_rule_sensitivity_not_historically_prespecified",
    })

assert len(rows) == 9
out = BASE / "analyses" / "r25_shared_full_composite_rule_audit.csv"
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for domain in ("DeepDRiD", "mBRSET"):
    for backbone in ("RETFound", "VisionFM"):
        rr = [r for r in rows if r["backbone"] == backbone and r["domain"] == domain]
        if rr:
            print(backbone, domain, "n", len(rr), "historical", sum(r["historical_success"] for r in rr),
                  "shared_positive", sum(r["shared_composite_positive_brier"] for r in rr),
                  "shared_ge001", sum(r["shared_composite_brier_ge_0_01"] for r in rr))
