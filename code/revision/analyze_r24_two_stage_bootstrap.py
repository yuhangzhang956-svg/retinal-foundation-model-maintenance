"""Exploratory R2-4 fit-plus-evaluation uncertainty, using archived patient scores.

Fixed before this run: 500 stratified replicates per source seed and sample size;
sample sizes 10/25/50/100; historical C=1e6 Platt model; development and
evaluation patients resampled independently within outcome strata.
"""
from pathlib import Path
import csv
import sys
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analyses" / "stats_vendor"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def logit(values):
    p = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def nll(y, p):
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


val = read(ROOT / "private_predictions" / "00_validation_patient_predictions_DeepDRiD.csv")
ev = read(ROOT / "private_predictions" / "02_eval_predictions_DeepDRiD.csv")
historical = read(ROOT / "private_predictions" / "01_recalibration_dose_runs_DeepDRiD.csv")
rng = np.random.default_rng(20261108)
reps = 500
rows = []
summary = []
for seed in (20260906, 20260907, 20260908):
    vv = [r for r in val if int(r["seed"]) == seed]
    ee = [r for r in ev if int(r["seed"]) == seed]
    assert len(vv) == len(ee) == 100
    assert {r["patient_id"] for r in vv}.isdisjoint({r["patient_id"] for r in ee})
    yv = np.asarray([int(r["y"]) for r in vv])
    xv = logit([r["p_raw"] for r in vv])
    ye = np.asarray([int(r["y"]) for r in ee])
    xe = logit([r["p_raw"] for r in ee])
    source = np.asarray([float(r["p_sourcecal"]) for r in ee])
    pos_v, neg_v = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    pos_e, neg_e = np.flatnonzero(ye == 1), np.flatnonzero(ye == 0)
    assert (len(pos_v), len(neg_v), len(pos_e), len(neg_e)) == (50, 50, 48, 52)
    for n in (10, 25, 50, 100):
        hp = [r for r in historical if int(r["seed"]) == seed and int(r["budget"]) == n]
        assert len(hp) == 50
        for rep in range(reps):
            npos = n // 2
            nneg = n - npos
            dev_idx = np.concatenate((rng.choice(pos_v, npos, replace=True),
                                      rng.choice(neg_v, nneg, replace=True)))
            eval_idx = np.concatenate((rng.choice(pos_e, len(pos_e), replace=True),
                                       rng.choice(neg_e, len(neg_e), replace=True)))
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
            model.fit(xv[dev_idx, None], yv[dev_idx])
            local = model.predict_proba(xe[eval_idx, None])[:, 1]
            y = ye[eval_idx]
            psource = source[eval_idx]
            rows.append({"seed": seed, "budget": n, "replicate": rep,
                         "dev_unique_patients": int(len(np.unique(dev_idx))),
                         "eval_unique_patients": int(len(np.unique(eval_idx))),
                         "brier_gain_source_minus_local": brier(y, psource) - brier(y, local),
                         "nll_gain_source_minus_local": nll(y, psource) - nll(y, local),
                         "platt_slope": float(model.coef_[0, 0])})
        group = [r for r in rows if r["seed"] == seed and r["budget"] == n]
        result = {"seed": seed, "budget": n, "bootstrap_replicates": reps,
                  "historical_rows": len(hp),
                  "historical_distinct_nll_fits": len({r["nll"] for r in hp}),
                  "historical_median_absolute_brier_gain": float(np.median([float(r["brier_gain"]) for r in hp])),
                  "historical_median_absolute_nll_gain": float(np.median([float(r["nll_gain"]) for r in hp])),
                  "bootstrap_design": "stratified_dev_with_replacement_and_stratified_eval_with_replacement"}
        for key in ("brier_gain_source_minus_local", "nll_gain_source_minus_local"):
            values = np.asarray([r[key] for r in group], dtype=float)
            result[key + "_median"] = float(np.median(values))
            result[key + "_q025"] = float(np.quantile(values, 0.025))
            result[key + "_q975"] = float(np.quantile(values, 0.975))
            result[key + "_fraction_positive"] = float(np.mean(values > 0))
        summary.append(result)


def write(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


write(ROOT / "analyses" / "r24_two_stage_bootstrap_replicates.csv", rows)
write(ROOT / "analyses" / "r24_two_stage_bootstrap_summary.csv", summary)
print("REPLICATES", len(rows), "SUMMARY", len(summary))
for row in summary:
    print(row["seed"], row["budget"], row["historical_distinct_nll_fits"],
          round(row["brier_gain_source_minus_local_median"], 4),
          round(row["nll_gain_source_minus_local_median"], 4))
