"""Independent development-set threshold-only comparator for R2-3.

Uses recovered, executed Phase2A1 feature caches and its three saved source
decoder checkpoints. No training, no target-test threshold selection. Patient
predictions are private and are written only to private_predictions/.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
torch_vendor = Path(os.environ.get("R22_TORCH_VENDOR", str(BASE / "analyses" / "torch_vendor")))
sys.path.insert(0, str(torch_vendor))
sys.path.insert(0, str(BASE / "analyses" / "stats_vendor"))
torch_lib = torch_vendor / "torch" / "lib"
if os.name == "nt" and torch_lib.exists():
    os.add_dll_directory(str(torch_lib))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

torch.set_num_threads(4)
EPS = 1e-7
SEEDS = (20267001, 20267002, 20267003)


class VisionFMClsHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_bn = nn.BatchNorm1d(3072, eps=1e-6, momentum=0.99)
        self.classifier = nn.Sequential(
            nn.Linear(3072, 1536), nn.GELU(), nn.Dropout(p=0.1),
            nn.Linear(1536, 768), nn.GELU(), nn.Dropout(p=0.1),
            nn.Linear(768, 1),
        )

    def forward(self, x):
        return self.classifier(self.channel_bn(x)).squeeze(-1)


def load_cache(name):
    z = np.load(BASE / "private_feature_cache" / f"{name}_features.npz")
    x = z["X"].astype(np.float32)
    pid = z["patient_id"].astype(str)
    y = z["y_patient"].astype(int)
    assert x.shape[1] == 3072 and len(x) == len(pid) == len(y)
    return x, pid, y


@torch.inference_mode()
def predict_patient(model, name):
    x, pid, y = load_cache(name)
    model.eval()
    image_p = []
    for start in range(0, len(x), 512):
        image_p.append(torch.sigmoid(model(torch.from_numpy(x[start:start+512]))).numpy())
    image_p = np.concatenate(image_p)
    by_id = defaultdict(list)
    labels = {}
    for ident, label, score in zip(pid, y, image_p):
        ident = str(ident)
        if ident in labels:
            assert labels[ident] == int(label)
        labels[ident] = int(label)
        by_id[ident].append(score)
    ids = sorted(by_id)
    return ids, np.array([labels[i] for i in ids]), np.array([max(by_id[i]) for i in ids], dtype=np.float32)


def logit_np(p):
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y, p):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    m.fit(logit_np(p).reshape(-1, 1), np.asarray(y).astype(int))
    return m


def apply_platt(model, p):
    return model.predict_proba(logit_np(p).reshape(-1, 1))[:, 1]


def threshold_90(y, p):
    # Mirrors executed Phase2A1: among sensitivity >=0.90, choose greatest
    # specificity; strict > tie rule retains the first, lowest threshold.
    best = None
    y = np.asarray(y, dtype=int)
    for th in np.unique(p):
        pred = p >= th
        tp = int(np.sum(pred & (y == 1)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        fp = int(np.sum(pred & (y == 0)))
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        if sens >= 0.90 and (best is None or spec > best[0]):
            best = (spec, th)
    return float(best[1]) if best is not None else 0.0


def counts(y, pred):
    return {
        "tp": int(np.sum((y == 1) & pred)),
        "fp": int(np.sum((y == 0) & pred)),
        "tn": int(np.sum((y == 0) & ~pred)),
        "fn": int(np.sum((y == 1) & ~pred)),
    }


def frozen_predictions():
    p = BASE / "private_predictions" / "02_patient_predictions_VisionFM_source.csv"
    with p.open(newline="", encoding="utf-8-sig") as f:
        return {(int(r["seed"]), r["domain"], str(r["patient_id"])): r for r in csv.DictReader(f)}


def main():
    frozen = frozen_predictions()
    aggregate = []
    private = []
    domains = (
        ("BRSET", "BR_VAL", "BR_TEST", "sourcecal"),
        ("mBRSET", "MB_LOCAL100", "MB_TEST", "localcal"),
        ("DeepDRiD", "DD_VAL", "DD_TEST", "localcal"),
    )
    for seed in SEEDS:
        state = torch.load(
            BASE / "private_checkpoints" / f"VisionFM_BRSET_decoder_seed{seed}.pth",
            map_location="cpu", weights_only=False,
        )
        assert int(state["seed"]) == seed
        model = VisionFMClsHead()
        model.load_state_dict(state["state_dict"], strict=True)
        for domain, val_name, test_name, cal_field in domains:
            val_ids, val_y, val_p = predict_patient(model, val_name)
            test_ids, test_y, test_p = predict_patient(model, test_name)
            # Fit only on the appropriate development/calibration set.
            cal = fit_platt(val_y, val_p)
            val_p_cal = apply_platt(cal, val_p)
            test_p_cal = apply_platt(cal, test_p)
            th_raw = threshold_90(val_y, val_p)
            th_cal = threshold_90(val_y, val_p_cal)
            raw_decision = test_p >= th_raw
            cal_decision = test_p_cal >= th_cal
            raw_counts = counts(test_y, raw_decision)
            cal_counts = counts(test_y, cal_decision)
            frozen_raw_diff = []
            frozen_cal_diff = []
            for ident, label, raw, calibrated, raw_yes, cal_yes in zip(
                test_ids, test_y, test_p, test_p_cal, raw_decision, cal_decision
            ):
                ref = frozen[(seed, domain, ident)]
                assert int(ref["y"]) == int(label)
                frozen_raw_diff.append(abs(float(raw) - float(ref["p"])))
                frozen_cal_diff.append(abs(float(calibrated) - float(ref[f"p_{cal_field}"])))
                private.append({
                    "seed": seed, "domain": domain, "patient_id": ident, "y": int(label),
                    "p_raw_replay": float(raw), "p_cal_replay": float(calibrated),
                    "raw_threshold_decision": int(raw_yes),
                    "platt_threshold_decision": int(cal_yes),
                })
            row = {
                "seed": seed, "domain": domain, "validation_n": len(val_ids),
                "validation_positive": int(val_y.sum()), "test_n": len(test_ids),
                "test_positive": int(test_y.sum()), "threshold_only_raw": th_raw,
                "platt_then_threshold": th_cal,
                "platt_slope_replay": float(cal.coef_[0, 0]),
                "platt_intercept_replay": float(cal.intercept_[0]),
                "raw_vs_platt_decision_discordance": int(np.sum(raw_decision != cal_decision)),
                "raw_tp": raw_counts["tp"], "raw_fp": raw_counts["fp"],
                "raw_tn": raw_counts["tn"], "raw_fn": raw_counts["fn"],
                "platt_tp": cal_counts["tp"], "platt_fp": cal_counts["fp"],
                "platt_tn": cal_counts["tn"], "platt_fn": cal_counts["fn"],
                "max_raw_replay_vs_frozen_abs_diff": max(frozen_raw_diff),
                "max_cal_replay_vs_frozen_abs_diff": max(frozen_cal_diff),
                "numeric_runtime": f"torch_{torch.__version__}_cpu;sklearn_1.9.1",
            }
            aggregate.append(row)
            print(seed, domain, "discordance", row["raw_vs_platt_decision_discordance"],
                  "max_frozen_raw_diff", row["max_raw_replay_vs_frozen_abs_diff"])
    out = BASE / "analyses" / "r23_independent_validation_threshold_comparison.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        w.writeheader(); w.writerows(aggregate)
    priv = BASE / "private_predictions" / "r23_threshold_only_vs_platt_replay_PRIVATE.csv"
    with priv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(private[0]))
        w.writeheader(); w.writerows(private)
    print("aggregate", out, "private", priv)


if __name__ == "__main__":
    main()
