"""Development-only resampling of local calibration and threshold policy.

The fixed target test is scored only after each resampled development policy is
fit. This is sensitivity to the 200-patient calibration sample, conditional on
one saved model and the already-used test cohort; it is not a new test estimate.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent
RUNS = [("UNIFIED_R20261109_B3", s) for s in (20267001, 20267002, 20267003)] + [
    ("UNIFIED_R20261109_RTF", 20260906)]


def fit_policy(y, z):
    fit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    fit.fit(z.reshape(-1, 1), y)
    p = fit.predict_proba(z.reshape(-1, 1))[:, 1]
    thresholds = np.unique(p)
    possible = []
    for threshold in thresholds:
        pred = p >= threshold
        sensitivity = np.sum(pred & (y == 1)) / np.sum(y == 1)
        specificity = np.sum(~pred & (y == 0)) / np.sum(y == 0)
        if sensitivity >= 0.90:
            possible.append((specificity, threshold))
    specificity, threshold = max(possible)
    return fit, float(threshold), float(specificity)


def load(tag, seed, method, role):
    path = BASE / "private_predictions" / (
        f"{tag}_seed{seed}_{method}_mBRSET_{role}_patients_PRIVATE.csv")
    f = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id").sort_index()
    assert f.index.is_unique and len(f) == (200 if role == "CALIBRATION_THRESHOLD" else 400)
    return f


def q(x, pct):
    return float(np.percentile(x, pct))


def main(reps):
    summaries = []
    out = BASE / "unified_experiment" / "UNIFIED_R20261109_local_policy_development_bootstrap.csv"
    for tag, seed in RUNS:
        for method in ("source", "head_only", "lora"):
            dev = load(tag, seed, method, "CALIBRATION_THRESHOLD")
            y = dev.y.to_numpy(dtype=int)
            score_column = "logit_margin_float32" if tag.endswith("RTF") else "logit_float32"
            z = dev[score_column].to_numpy(dtype=float)
            assert len(y) == 200 and y.sum() == 46
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            rng = np.random.default_rng(20261109 + seed + sum(map(ord, method)))
            fits = []
            for _ in range(reps):
                ix = np.r_[rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)]
                model, threshold, specificity = fit_policy(y[ix], z[ix])
                fits.append((model, threshold, specificity))
            # All policy fits above use development data only. Fixed test is
            # evaluated below; its outcome never selects a resample or policy.
            test = load(tag, seed, method, "TARGET_TEST_LOCKED")
            ty = test.y.to_numpy(dtype=int)
            tz = test[score_column].to_numpy(dtype=float)
            assert len(ty) == 400 and ty.sum() == 93
            thresholds, fn, fp = [], [], []
            for model, threshold, _ in fits:
                mapped = model.predict_proba(tz.reshape(-1, 1))[:, 1]
                pred = mapped >= threshold
                thresholds.append(threshold)
                fn.append(int(np.sum(~pred & (ty == 1))))
                fp.append(int(np.sum(pred & (ty == 0))))
            rows = pd.read_csv(BASE / "unified_experiment" /
                               f"{tag}_seed{seed}_independent_calibration_test_summary.csv")
            original = rows[(rows.method == method) & (rows.domain == "mBRSET")].iloc[0]
            summaries.append({"tag": tag, "seed": seed, "method": method,
                              "dev_patients": 200, "dev_positive": 46,
                              "fixed_test_patients": 400, "fixed_test_positive": 93,
                              "development_bootstrap_reps": reps,
                              "resampling": "stratified_patient_with_replacement_calibration_only",
                              "original_locked_threshold": float(original.selected_threshold),
                              "threshold_median": q(thresholds, 50),
                              "threshold_p025": q(thresholds, 2.5),
                              "threshold_p975": q(thresholds, 97.5),
                              "original_locked_test_fn": int(original.fn),
                              "test_fn_median_conditional": q(fn, 50),
                              "test_fn_p025_conditional": q(fn, 2.5),
                              "test_fn_p975_conditional": q(fn, 97.5),
                              "original_locked_test_fp": int(original.fp),
                              "test_fp_median_conditional": q(fp, 50),
                              "test_fp_p025_conditional": q(fp, 2.5),
                              "test_fp_p975_conditional": q(fp, 97.5),
                              "test_used_to_select_policy": False,
                              "interpretation": "development sample sensitivity conditional on fixed model and previously analysed test"})
            with out.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
                writer.writeheader(); writer.writerows(summaries)
            print(tag, seed, method, "DONE", flush=True)
    print("SUMMARY", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    assert args.reps >= 500
    main(args.reps)
