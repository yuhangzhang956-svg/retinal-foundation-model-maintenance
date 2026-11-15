"""Post hoc, development-only R2-4 N=10 regularization sensitivity.

The C grid was already evaluated as a descriptive sensitivity analysis; this
follow-up is therefore exploratory, not a prospectively blinded selection.
For each frozen 10-patient calibration subset, leave-one-out NLL chooses C
using only those ten development patients. The saved independent evaluation
metrics are joined *after* C is selected, without refitting them.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "analyses" / "stats_vendor"))
import numpy as np
from sklearn.linear_model import LogisticRegression

CS = (0.01, 0.1, 1.0, 1e6)  # fixed existing sensitivity grid


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def logit(p):
    p = max(1e-7, min(1 - 1e-7, float(p)))
    return math.log(p / (1 - p))


def loo_nll(x, y, C):
    losses = []
    for held in range(len(y)):
        keep = np.arange(len(y)) != held
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=5000)
        m.fit(x[keep].reshape(-1, 1), y[keep])
        p = float(m.predict_proba(x[held:held+1].reshape(-1, 1))[0, 1])
        p = max(1e-15, min(1 - 1e-15, p))
        losses.append(-math.log(p) if y[held] else -math.log1p(-p))
    return sum(losses) / len(losses)


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    val = {}
    for r in read(BASE / "private_predictions" / "00_validation_patient_predictions_DeepDRiD.csv"):
        val[(r["seed"], r["patient_id"])] = (int(r["y"]), logit(r["p_raw"]))
    subsets = defaultdict(list)
    for r in read(BASE / "private_predictions" / "02_nested_sampling_manifests_DeepDRiD.csv"):
        if r["budget"] == "10":
            subsets[(r["seed"], r["trajectory_seed"])].append(r["patient_id"])
    prior = {}
    for r in read(BASE / "analyses" / "r24_n10_regularization_runs.csv"):
        prior[(r["seed"], r["trajectory_seed"], float(r["C"]))] = r
    assert len(subsets) == 150 and len(prior) == 600
    out = []
    for (seed, traj), ids in sorted(subsets.items()):
        assert len(ids) == len(set(ids)) == 10
        xy = [val[(seed, pid)] for pid in ids]
        y = np.array([item[0] for item in xy], dtype=int)
        x = np.array([item[1] for item in xy], dtype=float)
        assert int(y.sum()) == 5
        losses = {C: loo_nll(x, y, C) for C in CS}
        # On exact ties, prefer smaller C (more regularization).
        chosen = min(CS, key=lambda C: (losses[C], C))
        selected = prior[(seed, traj, chosen)]
        historic = prior[(seed, traj, 1e6)]
        out.append({
            "seed": seed, "trajectory_seed": traj, "budget_patients": 10,
            "positive": 5, "negative": 5,
            "loo_nll_C0_01": losses[0.01], "loo_nll_C0_1": losses[0.1],
            "loo_nll_C1": losses[1.0], "loo_nll_C1e6": losses[1e6],
            "chosen_C_dev_only": chosen, "chosen_loo_nll": losses[chosen],
            "eval_nll_chosen": float(selected["eval_nll"]),
            "eval_brier_chosen": float(selected["eval_brier"]),
            "eval_nll_gain_vs_sourcecal": float(selected["eval_nll_gain_vs_source_cal"]),
            "eval_nll_historical_C1e6": float(historic["eval_nll"]),
            "eval_brier_historical_C1e6": float(historic["eval_brier"]),
            "eval_nll_change_chosen_minus_historical": float(selected["eval_nll"]) - float(historic["eval_nll"]),
            "analysis_status": "post_hoc_dev_only_LOOCV_exploratory",
        })
    dest = BASE / "analyses" / "r24_n10_dev_loocv_selection.csv"
    write(dest, out)
    freq = Counter(float(r["chosen_C_dev_only"]) for r in out)
    s = [{
        "runs": len(out), "C_0_01_selected": freq[0.01], "C_0_1_selected": freq[0.1],
        "C_1_selected": freq[1.0], "C_1e6_selected": freq[1e6],
        "median_eval_nll_chosen": float(np.median([r["eval_nll_chosen"] for r in out])),
        "median_eval_nll_historical": float(np.median([r["eval_nll_historical_C1e6"] for r in out])),
        "negative_sourcecal_nll_gain_chosen": sum(r["eval_nll_gain_vs_sourcecal"] < 0 for r in out),
        "chosen_worse_than_historical_nll_runs": sum(r["eval_nll_change_chosen_minus_historical"] > 0 for r in out),
        "chosen_better_than_historical_nll_runs": sum(r["eval_nll_change_chosen_minus_historical"] < 0 for r in out),
        "analysis_status": "post_hoc_dev_only_LOOCV_exploratory_not_test_selected",
    }]
    summary = BASE / "analyses" / "r24_n10_dev_loocv_summary.csv"
    write(summary, s)
    print(s[0])


if __name__ == "__main__":
    main()
