"""Independent development-set threshold comparison using archived RETFound scores.

This analyzes frozen, original patient probabilities. It does not infer image-level
inference equivalence or recover the model checkpoint.
"""
from pathlib import Path
import csv
import math
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analyses" / "stats_vendor"))
import numpy as np
from sklearn.linear_model import LogisticRegression


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def logit(values):
    p = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def threshold_for_sensitivity(y, p, target=0.90):
    """Copy the executed Phase1C2 strict-specificity-update selection semantics."""
    best = None
    for th in np.unique(p):
        pred = p >= th
        tp = int(np.count_nonzero(pred & (y == 1)))
        fn = int(np.count_nonzero(~pred & (y == 1)))
        tn = int(np.count_nonzero(~pred & (y == 0)))
        fp = int(np.count_nonzero(pred & (y == 0)))
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        if sens >= target and (best is None or spec > best[0]):
            best = (spec, float(th))
    return best[1] if best is not None else 0.0


def counts(y, pred):
    return {
        "tp": int(np.count_nonzero(pred & (y == 1))),
        "fp": int(np.count_nonzero(pred & (y == 0))),
        "tn": int(np.count_nonzero(~pred & (y == 0))),
        "fn": int(np.count_nonzero(~pred & (y == 1))),
    }


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - margin, center + margin


val = read(ROOT / "private_predictions" / "00_validation_patient_predictions_DeepDRiD.csv")
ev = read(ROOT / "private_predictions" / "02_eval_predictions_DeepDRiD.csv")
summary = []
paired = []
for seed in (20260906, 20260907, 20260908):
    vv = [r for r in val if int(r["seed"]) == seed]
    ee = [r for r in ev if int(r["seed"]) == seed]
    assert len(vv) == len(ee) == 100
    assert len({r["patient_id"] for r in vv}) == len({r["patient_id"] for r in ee}) == 100
    assert {r["patient_id"] for r in vv}.isdisjoint({r["patient_id"] for r in ee})
    yv = np.asarray([int(r["y"]) for r in vv])
    ye = np.asarray([int(r["y"]) for r in ee])
    pv = np.asarray([float(r["p_raw"]) for r in vv])
    pe = np.asarray([float(r["p_raw"]) for r in ee])
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    fit.fit(logit(pv).reshape(-1, 1), yv)
    cv = fit.predict_proba(logit(pv).reshape(-1, 1))[:, 1]
    ce = fit.predict_proba(logit(pe).reshape(-1, 1))[:, 1]
    archive_cal = np.asarray([float(r["p_localcal"]) for r in ee])
    residual = float(np.max(np.abs(ce - archive_cal)))
    th_raw = threshold_for_sensitivity(yv, pv)
    th_cal = threshold_for_sensitivity(yv, cv)
    dr = pe >= th_raw
    dc = ce >= th_cal
    discord = int(np.count_nonzero(dr != dc))
    raw_counts = counts(ye, dr)
    cal_counts = counts(ye, dc)
    sensitivity_ci = wilson(raw_counts["tp"], raw_counts["tp"] + raw_counts["fn"])
    specificity_ci = wilson(raw_counts["tn"], raw_counts["tn"] + raw_counts["fp"])
    summary.append({
        "seed": seed, "validation_patients": len(vv),
        "validation_positive": int(yv.sum()), "evaluation_patients": len(ee),
        "evaluation_positive": int(ye.sum()),
        "raw_threshold_dev_selected": th_raw,
        "platt_threshold_dev_selected": th_cal,
        "platt_slope_dev_fitted": float(fit.coef_[0, 0]),
        "platt_intercept_dev_fitted": float(fit.intercept_[0]),
        "archived_platt_probability_max_residual": residual,
        "discordant_evaluation_decisions": discord,
        **{"raw_" + k: v for k, v in raw_counts.items()},
        **{"platt_" + k: v for k, v in cal_counts.items()},
        "sensitivity_95ci_wilson_low": sensitivity_ci[0],
        "sensitivity_95ci_wilson_high": sensitivity_ci[1],
        "specificity_95ci_wilson_low": specificity_ci[0],
        "specificity_95ci_wilson_high": specificity_ci[1],
        "status": "PASS" if discord == 0 and raw_counts == cal_counts and residual < 1e-10 else "REVIEW",
    })
    for r, raw_p, cal_p, a, b in zip(ee, pe, ce, dr, dc):
        paired.append({"seed": seed, "patient_id": r["patient_id"], "y": r["y"],
                       "p_raw": raw_p, "p_localcal_replay": cal_p,
                       "raw_threshold_only_decision": int(a), "platt_threshold_decision": int(b)})

out = ROOT / "analyses" / "r23_retfound_deepdrid_independent_threshold_comparison.csv"
with out.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
private_out = ROOT / "private_predictions" / "r23_retfound_deepdrid_threshold_comparison_PRIVATE.csv"
with private_out.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)
print("SUMMARY", [(r["seed"], r["discordant_evaluation_decisions"], r["status"]) for r in summary])
print("PAIRED_DECISIONS", len(paired), "DISCORDANT", sum(r["discordant_evaluation_decisions"] for r in summary))
assert all(r["status"] == "PASS" for r in summary)
