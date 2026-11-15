"""Paired patient bootstrap and stable-logit audit of one new RETFound seed.

Reads locked one-time test predictions. Makes no model, threshold or calibration
changes. This is a revision-period exploratory comparison, not historical data.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent
TAG = "UNIFIED_R20261109_RTF"
SEED = 20260906
RNG = np.random.default_rng(20261109)
REPS = 5000


def load(domain, role, method):
    path = BASE / "private_predictions" / (
        f"{TAG}_seed{SEED}_{method}_{domain}_{role}_patients_PRIVATE.csv")
    frame = pd.read_csv(path, dtype={"patient_id": str})
    assert frame.patient_id.is_unique
    assert frame.y.nunique() == 2
    return frame.set_index("patient_id").sort_index()


def nll_from_logit(y, z):
    return np.mean(np.logaddexp(0, z) - y * z)


def pair(domain, role, contrast):
    left, right = contrast
    a, b = load(domain, role, left), load(domain, role, right)
    assert a.index.equals(b.index)
    assert np.array_equal(a.y, b.y)
    y = a.y.to_numpy(dtype=int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    pa, pb = a.p_float32.to_numpy(), b.p_float32.to_numpy()
    qa, qb = a.p_independently_calibrated.to_numpy(), b.p_independently_calibrated.to_numpy()
    qa = np.clip(qa, 1e-12, 1 - 1e-12)
    qb = np.clip(qb, 1e-12, 1 - 1e-12)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    deltas = np.empty(REPS)
    brier = np.empty(REPS)
    cal_nll = np.empty(REPS)
    for k in range(REPS):
        ix = np.r_[RNG.choice(pos, len(pos), replace=True),
                   RNG.choice(neg, len(neg), replace=True)]
        yy = y[ix]
        deltas[k] = roc_auc_score(yy, pa[ix]) - roc_auc_score(yy, pb[ix])
        brier[k] = np.mean((qa[ix] - yy)**2 - (qb[ix] - yy)**2)
        cal_nll[k] = np.mean(-yy * np.log(np.clip(qa[ix], 1e-12, 1)) -
                             (1 - yy) * np.log(np.clip(1 - qa[ix], 1e-12, 1)) +
                             yy * np.log(np.clip(qb[ix], 1e-12, 1)) +
                             (1 - yy) * np.log(np.clip(1 - qb[ix], 1e-12, 1)))
    za, zb = a.logit_margin_float32.to_numpy(), b.logit_margin_float32.to_numpy()
    return {"domain": domain, "seed": SEED, "contrast": f"{left}-minus-{right}",
            "patients": len(y), "positive": len(pos), "bootstrap_reps": REPS,
            "bootstrap_type": "paired_patient_stratified_percentile_95pct",
            "delta_auroc": obs,
            "delta_auroc_ci_low": np.percentile(deltas, 2.5),
            "delta_auroc_ci_high": np.percentile(deltas, 97.5),
            "delta_calibrated_brier": np.mean((qa - y)**2 - (qb - y)**2),
            "delta_calibrated_brier_ci_low": np.percentile(brier, 2.5),
            "delta_calibrated_brier_ci_high": np.percentile(brier, 97.5),
            "delta_calibrated_nll": np.mean(-y*np.log(qa)-(1-y)*np.log1p(-qa) +
                                            y*np.log(qb)+(1-y)*np.log1p(-qb)),
            "delta_calibrated_nll_ci_low": np.percentile(cal_nll, 2.5),
            "delta_calibrated_nll_ci_high": np.percentile(cal_nll, 97.5),
            "left_raw_nll_stable_logit": nll_from_logit(y, za),
            "right_raw_nll_stable_logit": nll_from_logit(y, zb),
            "left_float32_p_one": int(np.sum(pa == 1)),
            "right_float32_p_one": int(np.sum(pb == 1)),
            "left_float16_p_one": int(np.sum(a.p_float16_diagnostic == 1)),
            "right_float16_p_one": int(np.sum(b.p_float16_diagnostic == 1))}


def main():
    rows = []
    for domain, role in (("BRSET", "SOURCE_TEST_LOCKED"),
                         ("mBRSET", "TARGET_TEST_LOCKED")):
        for contrast in (("lora", "head_only"), ("lora", "source"),
                         ("head_only", "source")):
            rows.append(pair(domain, role, contrast))
    output = BASE / "unified_experiment" / f"{TAG}_seed{SEED}_paired_bootstrap.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
