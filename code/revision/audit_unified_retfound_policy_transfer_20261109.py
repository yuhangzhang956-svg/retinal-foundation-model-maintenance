"""Apply already locked source and target development policies to both tests."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

BASE = Path(__file__).resolve().parent
TAG = "UNIFIED_R20261109_RTF"
SEED = 20260906


def main():
    maps_path = BASE / "unified_experiment" / f"{TAG}_seed{SEED}_locked_dev_maps_thresholds.json"
    maps = json.loads(maps_path.read_text(encoding="utf-8"))["maps_thresholds"]
    rows = []
    for method in ("source", "head_only", "lora"):
        for domain, role in (("BRSET", "SOURCE_TEST_LOCKED"),
                             ("mBRSET", "TARGET_TEST_LOCKED")):
            path = BASE / "private_predictions" / (
                f"{TAG}_seed{SEED}_{method}_{domain}_{role}_patients_PRIVATE.csv")
            frame = pd.read_csv(path, dtype={"patient_id": str})
            assert frame.patient_id.is_unique
            y = frame.y.to_numpy(dtype=int)
            z = frame.logit_margin_float32.to_numpy(dtype=float)
            assert len(frame) == (202 if domain == "BRSET" else 400)
            for policy, dev in (("SOURCE_POLICY", "BRSET"),
                                ("LOCAL_MAINTENANCE", "mBRSET")):
                fit = maps[f"{method}:{dev}"]
                p = expit(fit["intercept"] + fit["slope"] * z)
                pred = p >= fit["threshold"]
                tp = int(np.sum(pred & (y == 1)))
                fn = int(np.sum(~pred & (y == 1)))
                fp = int(np.sum(pred & (y == 0)))
                tn = int(np.sum(~pred & (y == 0)))
                assert tp + fn + fp + tn == len(frame)
                rows.append({"seed": SEED, "method": method, "test_domain": domain,
                             "policy": policy, "policy_fit_domain": dev,
                             "policy_dev_patients": fit["dev_patients"],
                             "patients": len(frame), "positive": int(y.sum()),
                             "threshold": fit["threshold"], "tp": tp, "fn": fn,
                             "fp": fp, "tn": tn,
                             "sensitivity": tp / (tp + fn),
                             "specificity": tn / (tn + fp),
                             "missed_rdr_per_1000": 1000 * fn / len(frame),
                             "false_referrals_per_1000": 1000 * fp / len(frame),
                             "auroc": roc_auc_score(y, p),
                             "brier": brier_score_loss(y, p),
                             "nll": log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))})
    output = BASE / "unified_experiment" / f"{TAG}_seed{SEED}_policy_transfer_summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
