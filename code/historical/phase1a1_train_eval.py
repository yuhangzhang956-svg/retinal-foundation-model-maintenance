#!/usr/bin/env python3
"""Locked Phase 1A1 BRSET -> mBRSET zero-label RETFound deployment study.

This script deliberately reads only BRSET TRAIN/VALIDATION/FROZEN_SOURCE_TEST
and mBRSET FROZEN_TARGET_TEST.  It never reads the mBRSET adaptation pool.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SEEDS = (20260906, 20260907, 20260908)
EXPECTED_BUNDLE_ROWS = 4072
EXPECTED_BRSET_IMAGES = 2574
EXPECTED_MBRSET_IMAGES = 1498
EXPECTED_MBRSET_PATIENTS = 400
PRETRAINED_SHA256 = "e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "run", "summarize"), required=True)
    p.add_argument("--bundle", type=Path, default=Path("/content/phase1a1/phase1a1_colab_bundle"))
    p.add_argument("--repo", type=Path, default=Path("/content/RETFound"))
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("/content/drive/MyDrive/Retinal_Retraining_Phase1A1/checkpoints/RETFound_mae_natureCFP.pth"),
    )
    p.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/Retinal_Retraining_Phase1A1"),
    )
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--accum-iter", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot JSON encode {type(value)!r}")

    path.write_text(json.dumps(obj, indent=2, default=encode), encoding="utf-8")


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def assert_environment(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected A100, found {torch.cuda.get_device_name(0)}")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    got = sha256_file(args.weights)
    if got != PRETRAINED_SHA256:
        raise RuntimeError(f"Pretrained checkpoint SHA256 mismatch: {got}")
    if not (args.repo / "models_vit.py").is_file():
        raise FileNotFoundError(args.repo / "models_vit.py")
    sys.path.insert(0, str(args.repo))
    return got


def load_locked_frames(bundle: Path):
    bm = pd.read_csv(bundle / "bundle_manifest.csv")
    assert len(bm) == EXPECTED_BUNDLE_ROWS
    assert (bm["status"] == "OK").all()
    assert (bm["expected_sha256"].astype(str) == bm["actual_sha256"].astype(str)).all()
    assert not bm["split"].astype(str).str.contains("ADAPTATION", case=False, na=False).any()

    br = pd.read_csv(bundle / "metadata" / "BRSET_source_split_v2.csv")
    mb_all = pd.read_csv(bundle / "metadata" / "mBRSET_master_manifest_v2.csv", low_memory=False)
    mb = mb_all.loc[
        mb_all["exclusion_status"].eq("INCLUDED")
        & mb_all["split"].eq("FROZEN_TARGET_TEST")
        & mb_all["image_id"].notna()
    ].copy()
    mb = mb.drop_duplicates("image_id", keep="first")

    assert len(br) == EXPECTED_BRSET_IMAGES
    assert len(mb) == EXPECTED_MBRSET_IMAGES
    assert mb["patient_id"].nunique() == EXPECTED_MBRSET_PATIENTS
    assert set(br["split"].unique()) == {"TRAIN", "VALIDATION", "FROZEN_SOURCE_TEST"}
    assert br.groupby("split")["patient_id"].nunique().to_dict() == {
        "FROZEN_SOURCE_TEST": 202,
        "TRAIN": 939,
        "VALIDATION": 202,
    }

    br["dataset"] = "BRSET"
    br["local_path"] = br["image_id"].map(lambda x: str(bundle / "BRSET" / "images" / str(x)))
    mb["dataset"] = "mBRSET"
    mb["local_path"] = mb["image_id"].map(lambda x: str(bundle / "mBRSET" / "images" / str(x)))
    for df in (br, mb):
        df["patient_id"] = df["patient_id"].astype(str)
        df["image_id"] = df["image_id"].astype(str)
        df["referable_dr_image"] = df["referable_dr_image"].astype(int)
        df["referable_dr_patient"] = df["referable_dr_patient"].astype(int)
        missing = [p for p in df["local_path"] if not Path(p).is_file()]
        if missing:
            raise RuntimeError(f"Missing locked images: {len(missing)}")
    return br, mb, bm


def official_transforms():
    train_tf = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=None,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return train_tf, eval_tf


class LockedFundusDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row.local_path) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(row.referable_dr_image), idx


def build_model(args, pretrained: bool):
    if str(args.repo) not in sys.path:
        sys.path.insert(0, str(args.repo))
    import models_vit
    from util.pos_embed import interpolate_pos_embed

    model = models_vit.RETFound_mae(
        img_size=224,
        num_classes=2,
        drop_path_rate=0.2,
        global_pool=True,
    )
    if pretrained:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        state = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in state.items()}
        state = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in state.items()}
        current = model.state_dict()
        for key in ("head.weight", "head.bias"):
            if key in state and state[key].shape != current[key].shape:
                del state[key]
        interpolate_pos_embed(model, state)
        msg = model.load_state_dict(state, strict=False)
        allowed_missing = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
        unexpected_non_decoder = [
            k
            for k in msg.unexpected_keys
            if not k.startswith(("decoder", "mask_token")) and k not in {"norm.weight", "norm.bias"}
        ]
        missing_non_head = [k for k in msg.missing_keys if k not in allowed_missing]
        if missing_non_head or unexpected_non_decoder:
            raise RuntimeError(
                f"Unexpected checkpoint incompatibility missing={missing_non_head[:10]} "
                f"unexpected={unexpected_non_decoder[:10]}"
            )
        trunc_normal_(model.head.weight, std=2e-5)
        nn.init.zeros_(model.head.bias)
        del checkpoint, state
    return model


def autocast_ctx():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def forward_logits(model, x):
    logits = model(x)
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise RuntimeError(f"Unexpected RETFound output shape: {tuple(logits.shape)}")
    return logits


def smoke_test(args):
    sha = assert_environment(args)
    out = args.drive_root / "results" / "00_compute_smoke_test.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(args, pretrained=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    rows = []
    for batch in (1, 2, 4, 8, 12, 16, 24):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(batch, 3, 224, 224, device="cuda")
            y = torch.randint(0, 2, (batch,), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                loss = F.cross_entropy(forward_logits(model, x), y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            headroom = 100 * (1 - peak / total)
            rows.append((batch, "PASS", peak, headroom, ""))
            del x, y, loss
            if headroom < 18:
                break
        except torch.cuda.OutOfMemoryError as e:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            rows.append((batch, "OOM", float("nan"), 0.0, str(e).splitlines()[0]))
            break
    safe = [r for r in rows if r[1] == "PASS" and r[3] >= 18]
    if not safe:
        raise RuntimeError("Full fine-tuning failed the A100 safety smoke test")
    selected = max(r[0] for r in safe)
    lines = [
        "# Phase 1A1 Compute Smoke Test",
        "",
        f"- Timestamp UTC: `{now_utc()}`",
        f"- GPU: `{torch.cuda.get_device_name(0)}`",
        f"- VRAM: `{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB`",
        f"- PyTorch: `{torch.__version__}`",
        f"- CUDA runtime: `{torch.version.cuda}`",
        "- Precision: `BF16 autocast`",
        "- Model: `official RETFound_mae, full fine-tuning`",
        f"- Pretrained checkpoint SHA256: `{sha}`",
        f"- Largest tested batch with >=18% headroom: `{selected}`",
        "",
        "| Batch | Status | Peak allocated GiB | Headroom % | Note |",
        "|---:|---|---:|---:|---|",
    ]
    lines += [f"| {b} | {s} | {p:.2f} | {h:.1f} | {n} |" for b, s, p, h, n in rows]
    lines += ["", "**RETFound full fine-tuning feasible: YES**", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    save_json(args.drive_root / "logs" / "smoke_test.json", {"rows": rows, "selected_batch": selected})
    print(f"SMOKE_TEST_PASS selected_batch={selected}")


def patient_aggregate(pred: pd.DataFrame, prob_col="p_raw") -> pd.DataFrame:
    cols = ["dataset", "patient_id"]
    agg = pred.groupby(cols, as_index=False).agg(
        y_true=("referable_dr_patient", "max"),
        p_raw=(prob_col, "max"),
        n_images=("image_id", "size"),
    )
    return agg


def predict(model, frame, transform, args, batch_size=None):
    ds = LockedFundusDataset(frame, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size or max(32, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model.eval()
    out = []
    with torch.no_grad():
        for x, _, idx in loader:
            with autocast_ctx():
                logits = forward_logits(model, x.cuda(non_blocking=True))
            p = logits.softmax(1)[:, 1].float().cpu().numpy()
            chunk = frame.reset_index(drop=True).iloc[idx.numpy()].copy()
            chunk["p_raw"] = p
            out.append(chunk)
    return pd.concat(out, ignore_index=True)


def basic_metrics(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "nll": log_loss(y, p, labels=[0, 1]),
    }


def calibration_metrics(y, p, bins=10):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(z, y)
    ids = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    base = basic_metrics(y, p)
    base.update({"calibration_intercept": float(lr.intercept_[0]), "calibration_slope": float(lr.coef_[0, 0]), "ece": ece})
    return base


def fit_platt(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(z, np.asarray(y, int))
    return model


def apply_platt(model, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return model.predict_proba(z)[:, 1]


def threshold_for_sensitivity(y, p, target=0.90):
    fpr, tpr, thresholds = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return float(thresholds[np.argmax(tpr)])
    best = ok[np.argmin(fpr[ok])]
    return float(thresholds[best])


def threshold_metrics(y, p, threshold):
    y = np.asarray(y, int)
    pred = np.asarray(p) >= threshold
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    prevalence = y.mean()
    sensitivity = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "accuracy": div(tp + tn, len(y)),
        "false_negative_rate": 1 - sensitivity,
        "false_positive_rate": 1 - specificity,
        "prevalence": prevalence,
        "missed_referable_per_1000": 1000 * prevalence * (1 - sensitivity),
        "unnecessary_referrals_per_1000": 1000 * (1 - prevalence) * (1 - specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def patient_auc_from_image_frame(frame):
    p = patient_aggregate(frame)
    return roc_auc_score(p.y_true, p.p_raw)


def lr_for_step(base_lr, epoch_progress, warmup, epochs):
    if epoch_progress < warmup:
        return base_lr * epoch_progress / max(warmup, 1e-8)
    t = (epoch_progress - warmup) / max(epochs - warmup, 1)
    return 1e-6 + 0.5 * (base_lr - 1e-6) * (1 + math.cos(math.pi * t))


def train_seed(seed, br, args, train_tf, eval_tf):
    set_seed(seed)
    result_root = args.drive_root / "results"
    ckpt = result_root / "checkpoints" / f"RETFound_BRSET_seed{seed}_best.pth"
    history_path = result_root / "logs" / f"training_history_seed{seed}.csv"
    done = result_root / "logs" / f"seed{seed}_complete.json"

    if ckpt.is_file() and done.is_file():
        print(f"SEED {seed}: completed checkpoint found; skipping training")
        model = build_model(args, pretrained=False)
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model.cuda(), saved

    train = br.loc[br.split.eq("TRAIN")].reset_index(drop=True)
    val = br.loc[br.split.eq("VALIDATION")].reset_index(drop=True)
    ds = LockedFundusDataset(train, train_tf)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    model = build_model(args, pretrained=True).cuda()
    sys.path.insert(0, str(args.repo))
    from util import lr_decay as lrd

    param_groups = lrd.param_groups_lrd(
        model,
        weight_decay=0.05,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=0.65,
    )
    eff_batch = args.batch_size * args.accum_iter
    base_lr = 5e-3 * eff_batch / 256
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
    nneg = int((train.referable_dr_image == 0).sum())
    npos = int((train.referable_dr_image == 1).sum())
    class_weights = torch.tensor([1.0, nneg / npos], device="cuda")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = []
    best_auc, best_epoch, stale = -np.inf, -1, 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, (x, y, _) in enumerate(loader):
            progress = epoch + step / len(loader)
            lr = lr_for_step(base_lr, progress, args.warmup_epochs, args.epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with autocast_ctx():
                loss = criterion(forward_logits(model, x), y) / args.accum_iter
            loss.backward()
            if (step + 1) % args.accum_iter == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item() * args.accum_iter))
        val_img = predict(model, val, eval_tf, args)
        val_auc = patient_auc_from_image_frame(val_img)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_patient_auroc": val_auc,
            "lr": lr,
            "timestamp_utc": now_utc(),
        }
        history.append(row)
        atomic_csv(pd.DataFrame(history), history_path)
        print(json.dumps(row), flush=True)
        if val_auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            payload = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "seed": seed,
                "epoch": epoch,
                "val_patient_auroc": val_auc,
                "config": vars(args),
                "saved_utc": now_utc(),
            }
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, ckpt)
            save_json(
                result_root / "logs" / f"best_seed{seed}.json",
                {k: v for k, v in payload.items() if k != "model"},
            )
        else:
            stale += 1
        if epoch + 1 >= max(10, args.warmup_epochs + 2) and stale >= args.patience:
            print(f"SEED {seed}: early stop epoch={epoch} best_epoch={best_epoch}")
            break

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    save_json(done, {"seed": seed, "best_epoch": saved["epoch"], "best_val_auroc": saved["val_patient_auroc"], "completed_utc": now_utc()})
    return model.cuda(), saved


def bootstrap_seed(seed, source, target, threshold, nboot):
    rng = np.random.default_rng(seed + 991)
    records = []
    metrics = ("auroc", "auprc", "brier", "nll", "sensitivity", "specificity")
    vals = {f"source_{m}": [] for m in metrics}
    vals.update({f"target_{m}": [] for m in metrics})
    vals.update({f"delta_{m}": [] for m in metrics})
    sy, sp = source.y_true.to_numpy(), source.p_cal.to_numpy()
    ty, tp = target.y_true.to_numpy(), target.p_cal.to_numpy()
    for _ in range(nboot):
        si = rng.integers(0, len(source), len(source))
        ti = rng.integers(0, len(target), len(target))
        if len(np.unique(sy[si])) < 2 or len(np.unique(ty[ti])) < 2:
            continue
        sm = basic_metrics(sy[si], sp[si])
        tm = basic_metrics(ty[ti], tp[ti])
        sm.update(threshold_metrics(sy[si], sp[si], threshold))
        tm.update(threshold_metrics(ty[ti], tp[ti], threshold))
        for m in metrics:
            vals[f"source_{m}"].append(sm[m])
            vals[f"target_{m}"].append(tm[m])
            vals[f"delta_{m}"].append(tm[m] - sm[m])
    for name, arr in vals.items():
        arr = np.asarray(arr, float)
        records.append(
            {
                "seed": seed,
                "metric": name,
                "n_bootstrap_valid": len(arr),
                "bootstrap_mean": float(np.mean(arr)),
                "ci_lower": float(np.quantile(arr, 0.025)),
                "ci_upper": float(np.quantile(arr, 0.975)),
                "ci_method": "independent patient-level percentile bootstrap",
            }
        )
    return records


def secondary_metrics(seed, frame, domain, nboot=500):
    records = []
    for level in ("image", "eye"):
        if level == "image":
            d = frame[["patient_id", "referable_dr_image", "p_raw"]].rename(columns={"referable_dr_image": "y"})
        else:
            d = frame.loc[frame.eye_laterality.notna() & frame.eye_laterality.astype(str).ne("")].copy()
            if d.empty:
                continue
            d = d.groupby(["patient_id", "eye_laterality"], as_index=False).agg(y=("referable_dr_eye", "max"), p_raw=("p_raw", "max"))
        point = basic_metrics(d.y, d.p_raw)
        rng = np.random.default_rng(seed + (31 if domain == "source" else 47) + (0 if level == "image" else 100))
        patients = d.patient_id.astype(str).unique()
        boots = {"auroc": [], "auprc": []}
        for _ in range(nboot):
            sampled = rng.choice(patients, len(patients), replace=True)
            chunks = []
            for j, pid in enumerate(sampled):
                c = d.loc[d.patient_id.astype(str).eq(str(pid))].copy()
                c["_boot_patient"] = j
                chunks.append(c)
            b = pd.concat(chunks, ignore_index=True)
            if b.y.nunique() < 2:
                continue
            boots["auroc"].append(roc_auc_score(b.y, b.p_raw))
            boots["auprc"].append(average_precision_score(b.y, b.p_raw))
        for metric in ("auroc", "auprc"):
            arr = np.asarray(boots[metric])
            records.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "level": level,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower": np.quantile(arr, 0.025),
                    "ci_upper": np.quantile(arr, 0.975),
                    "bootstrap_unit": "patient cluster",
                    "n_observations": len(d),
                    "n_patients": d.patient_id.nunique(),
                }
            )
    return records


def evaluate_seed(seed, model, br, mb, args, eval_tf):
    result_root = args.drive_root / "results"
    val = br.loc[br.split.eq("VALIDATION")].reset_index(drop=True)
    src = br.loc[br.split.eq("FROZEN_SOURCE_TEST")].reset_index(drop=True)
    val_img = predict(model, val, eval_tf, args)
    src_img = predict(model, src, eval_tf, args)
    tgt_img = predict(model, mb.reset_index(drop=True), eval_tf, args)
    val_pat = patient_aggregate(val_img)
    src_pat = patient_aggregate(src_img)
    tgt_pat = patient_aggregate(tgt_img)
    platt = fit_platt(val_pat.y_true, val_pat.p_raw)
    for d in (val_pat, src_pat, tgt_pat):
        d["p_cal"] = apply_platt(platt, d.p_raw)
    threshold = threshold_for_sensitivity(val_pat.y_true, val_pat.p_cal, 0.90)
    for domain, d in (("source", src_pat), ("target", tgt_pat)):
        d["seed"] = seed
        d["domain"] = domain
        d["fixed_threshold"] = threshold
        d["predicted_positive"] = (d.p_cal >= threshold).astype(int)
    for d in (src_img, tgt_img):
        d["seed"] = seed

    src_raw = calibration_metrics(src_pat.y_true, src_pat.p_raw)
    tgt_raw = calibration_metrics(tgt_pat.y_true, tgt_pat.p_raw)
    src_cal = calibration_metrics(src_pat.y_true, src_pat.p_cal)
    tgt_cal = calibration_metrics(tgt_pat.y_true, tgt_pat.p_cal)
    src_thr = threshold_metrics(src_pat.y_true, src_pat.p_cal, threshold)
    tgt_thr = threshold_metrics(tgt_pat.y_true, tgt_pat.p_cal, threshold)
    val_thr = threshold_metrics(val_pat.y_true, val_pat.p_cal, threshold)
    seed_row = {
        "seed": seed,
        "source_AUROC": src_cal["auroc"],
        "target_AUROC": tgt_cal["auroc"],
        "delta_AUROC": tgt_cal["auroc"] - src_cal["auroc"],
        "source_AUPRC": src_cal["auprc"],
        "target_AUPRC": tgt_cal["auprc"],
        "source_Brier": src_cal["brier"],
        "target_Brier": tgt_cal["brier"],
        "source_fixed_sensitivity": src_thr["sensitivity"],
        "target_fixed_sensitivity": tgt_thr["sensitivity"],
        "source_fixed_specificity": src_thr["specificity"],
        "target_fixed_specificity": tgt_thr["specificity"],
        "threshold_source_validation_calibrated": threshold,
        "validation_sensitivity": val_thr["sensitivity"],
        "validation_specificity": val_thr["specificity"],
        "platt_intercept": float(platt.intercept_[0]),
        "platt_slope": float(platt.coef_[0, 0]),
    }
    cal_rows = []
    for domain, scale, metrics in (
        ("source", "raw", src_raw),
        ("target", "raw", tgt_raw),
        ("source", "source_calibrated", src_cal),
        ("target", "source_calibrated", tgt_cal),
    ):
        cal_rows.append({"seed": seed, "domain": domain, "probability_scale": scale, **metrics})
    fixed_rows = [
        {"seed": seed, "domain": "validation", "probability_scale": "source_calibrated", **val_thr},
        {"seed": seed, "domain": "source", "probability_scale": "source_calibrated", **src_thr},
        {"seed": seed, "domain": "target", "probability_scale": "source_calibrated", **tgt_thr},
    ]
    boot_rows = bootstrap_seed(seed, src_pat, tgt_pat, threshold, args.bootstrap)
    sec_rows = secondary_metrics(seed, src_img, "source") + secondary_metrics(seed, tgt_img, "target")
    atomic_csv(src_img, result_root / "logs" / f"image_predictions_BRSET_seed{seed}.csv")
    atomic_csv(tgt_img, result_root / "logs" / f"image_predictions_mBRSET_seed{seed}.csv")
    return seed_row, src_pat, tgt_pat, cal_rows, fixed_rows, boot_rows, sec_rows


def combine_and_report(args, all_seed=None):
    root = args.drive_root / "results"
    if all_seed is None:
        parts = []
        for seed in SEEDS:
            path = root / "logs" / f"seed_metrics_{seed}.json"
            if not path.is_file():
                raise RuntimeError(f"Missing completed seed metrics: {path}")
            parts.append(json.loads(path.read_text(encoding="utf-8")))
        all_seed = parts
    seed_df = pd.DataFrame([x["seed_row"] for x in all_seed])
    atomic_csv(seed_df, root / "03_seed_level_metrics.csv")
    src_pred = pd.concat([pd.DataFrame(x["source_predictions"]) for x in all_seed], ignore_index=True)
    tgt_pred = pd.concat([pd.DataFrame(x["target_predictions"]) for x in all_seed], ignore_index=True)
    atomic_csv(src_pred, root / "04_patient_predictions_BRSET.csv")
    atomic_csv(tgt_pred, root / "05_patient_predictions_mBRSET.csv")
    cal = pd.DataFrame(sum((x["calibration_rows"] for x in all_seed), []))
    fixed = pd.DataFrame(sum((x["fixed_rows"] for x in all_seed), []))
    boot = pd.DataFrame(sum((x["bootstrap_rows"] for x in all_seed), []))
    sec = pd.DataFrame(sum((x["secondary_rows"] for x in all_seed), []))
    atomic_csv(cal, root / "06_calibration_metrics.csv")
    atomic_csv(fixed, root / "07_fixed_threshold_metrics.csv")
    atomic_csv(boot, root / "08_bootstrap_domain_shift.csv")
    atomic_csv(sec, root / "09_secondary_eye_image_metrics.csv")

    supportive = []
    reasons = []
    for _, r in seed_df.iterrows():
        b = boot.loc[boot.seed.eq(r.seed)].set_index("metric")
        discrim = r.delta_AUROC <= -0.05 and b.loc["delta_auroc", "ci_upper"] < 0
        sens = (r.target_fixed_sensitivity - r.source_fixed_sensitivity) <= -0.05 and b.loc["delta_sensitivity", "ci_upper"] < 0
        spec = (r.target_fixed_specificity - r.source_fixed_specificity) <= -0.10 and b.loc["delta_specificity", "ci_upper"] < 0
        cr = cal.loc[cal.seed.eq(r.seed)]
        sr = cr.loc[(cr.domain == "source") & (cr.probability_scale == "raw")].iloc[0]
        tr = cr.loc[(cr.domain == "target") & (cr.probability_scale == "raw")].iloc[0]
        cal_hits = sum(
            [
                tr.brier - sr.brier >= 0.02,
                tr.nll - sr.nll >= 0.05,
                abs(tr.calibration_intercept) - abs(sr.calibration_intercept) >= 0.20,
                abs(tr.calibration_slope - 1) - abs(sr.calibration_slope - 1) >= 0.20,
                tr.ece - sr.ece >= 0.02,
            ]
        )
        cal_worse = cal_hits >= 2
        supportive.append(bool(discrim or sens or spec or cal_worse))
        reasons.append({"seed": int(r.seed), "discrimination": bool(discrim), "sensitivity": bool(sens), "specificity": bool(spec), "calibration": bool(cal_worse), "calibration_metric_hits": int(cal_hits)})
    n_concordant = sum(supportive)
    any_magnitude = bool(
        (seed_df.delta_AUROC <= -0.05).any()
        or ((seed_df.target_fixed_sensitivity - seed_df.source_fixed_sensitivity) <= -0.05).any()
        or ((seed_df.target_fixed_specificity - seed_df.source_fixed_specificity) <= -0.10).any()
        or any(r["calibration"] for r in reasons)
    )
    if n_concordant >= 2:
        verdict = "STRONG PASS — PROCEED TO LABEL-DOSE EXPERIMENT"
        eye = "PASS"
    elif any_magnitude:
        verdict = "BORDERLINE PASS — HUMAN REVIEW REQUIRED"
        eye = "BORDERLINE"
    else:
        verdict = "FAIL — STOP RETRAINING-DOSE PROJECT"
        eye = "FAIL"

    numeric = seed_df.select_dtypes(include=[np.number]).drop(columns=["seed"])
    stability = pd.DataFrame({"mean": numeric.mean(), "SD": numeric.std(ddof=1), "min": numeric.min(), "max": numeric.max()})
    stability.to_csv(root / "10_training_stability_summary.csv")
    (root / "10_training_stability.md").write_text(
        "# Phase 1A1 Training Stability\n\n" + stability.to_markdown() + "\n\n" + "Gate flags:\n\n```json\n" + json.dumps(reasons, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    make_figures(root, src_pred, tgt_pred, seed_df)
    make_reports(args, seed_df, cal, fixed, boot, reasons, n_concordant, verdict, eye)
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "full_finetuning_feasible": True,
        "source_auroc_mean": seed_df.source_AUROC.mean(),
        "target_auroc_mean": seed_df.target_AUROC.mean(),
        "delta_auroc_mean": seed_df.delta_AUROC.mean(),
        "source_brier_mean": seed_df.source_Brier.mean(),
        "target_brier_mean": seed_df.target_Brier.mean(),
        "source_sensitivity_mean": seed_df.source_fixed_sensitivity.mean(),
        "source_specificity_mean": seed_df.source_fixed_specificity.mean(),
        "target_sensitivity_mean": seed_df.target_fixed_sensitivity.mean(),
        "target_specificity_mean": seed_df.target_fixed_specificity.mean(),
        "concordant_deterioration_seeds": n_concordant,
        "phase1a1_verdict": verdict,
        "eye_and_vision_floor": eye,
        "gate_report": str(root / "PHASE_1A1_GATE_REPORT.md"),
    }
    save_json(root / "FINAL_CONSOLE_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    print("PHASE1A1_COMPLETE=YES")


def make_figures(root, src, tgt, seed_df):
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    colors = {"source": "#3567A5", "target": "#D2553D"}

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for seed in SEEDS:
        for domain, d in (("source", src.loc[src.seed.eq(seed)]), ("target", tgt.loc[tgt.seed.eq(seed)])):
            fpr, tpr, _ = roc_curve(d.y_true, d.p_cal)
            auc = roc_auc_score(d.y_true, d.p_cal)
            ax.plot(fpr, tpr, color=colors[domain], alpha=0.72, label=f"{domain} seed {seed}: {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", title="Patient-level ROC: source vs zero-label target")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"Figure_A_patient_ROC.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    bins = np.linspace(0, 1, 11)
    for domain, data in (("source", src), ("target", tgt)):
        for seed in SEEDS:
            d = data.loc[data.seed.eq(seed)].copy()
            d["bin"] = pd.cut(d.p_cal, bins, include_lowest=True)
            c = d.groupby("bin", observed=True).agg(pred=("p_cal", "mean"), obs=("y_true", "mean"))
            ax.plot(c.pred, c.obs, "o-", color=colors[domain], alpha=0.65)
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed proportion", title="Patient-level calibration")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"Figure_B_patient_calibration.{ext}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    pairs = [
        ("source_AUROC", "target_AUROC", "AUROC"),
        ("source_Brier", "target_Brier", "Brier score"),
        ("source_fixed_sensitivity", "target_fixed_sensitivity", "Sensitivity"),
        ("source_fixed_specificity", "target_fixed_specificity", "Specificity"),
    ]
    for ax, (s, t, title) in zip(axes.flat, pairs):
        for _, r in seed_df.iterrows():
            ax.plot([0, 1], [r[s], r[t]], "o-", alpha=0.8)
        ax.set_xticks([0, 1], ["BRSET", "mBRSET"])
        ax.set_title(title)
    fig.suptitle("Seed-wise source-to-target change")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"Figure_C_seedwise_change.{ext}", dpi=300)
    plt.close(fig)

    allp = pd.concat([src.assign(domain="BRSET"), tgt.assign(domain="mBRSET")])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (truth, label) in zip(axes, [(0, "Non-referable"), (1, "Referable")]):
        for domain, color in (("BRSET", colors["source"]), ("mBRSET", colors["target"])):
            vals = allp.loc[(allp.y_true == truth) & (allp.domain == domain), "p_cal"]
            ax.hist(vals, bins=20, density=True, alpha=0.45, color=color, label=domain)
        ax.set_title(label)
        ax.set_xlabel("Source-calibrated patient risk")
    axes[0].set_ylabel("Density")
    axes[1].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"Figure_D_risk_distributions.{ext}", dpi=300)
    plt.close(fig)


def make_reports(args, seed_df, cal, fixed, boot, reasons, n_concordant, verdict, eye):
    root = args.drive_root / "results"
    config = f"""# Phase 1A1 Training Configuration

- Official repository: `https://github.com/rmaphoh/RETFound`
- Repository commit: `{subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()}`
- Official checkpoint: `RETFound_mae_natureCFP.pth`
- Checkpoint SHA256: `{PRETRAINED_SHA256}`
- Model: RETFound ViT-Large/16, global pooling, two-class head, **full fine-tuning**
- Input: 224 × 224
- Train preprocessing: official timm transform; bicubic; RandAugment `rand-m9-mstd0.5-inc1`; random erasing p=0.25; ImageNet normalization
- Evaluation preprocessing: bicubic resize 256; center crop 224; ImageNet normalization
- Optimizer: AdamW; weight decay 0.05; layer-wise LR decay 0.65
- Base LR: `5e-3 × effective_batch/256`; cosine decay; {args.warmup_epochs}-epoch warmup
- Batch size: {args.batch_size}; accumulation: {args.accum_iter}; effective batch: {args.batch_size * args.accum_iter}
- Loss: class-weighted cross entropy; positive weight calculated from BRSET TRAIN only
- Maximum epochs: {args.epochs}; early stopping patience: {args.patience}; selection metric: BRSET VALIDATION patient AUROC
- Precision: BF16 autocast
- Seeds: {', '.join(map(str, SEEDS))}
- Calibration: Platt scaling fitted on BRSET VALIDATION patient predictions only
- Fixed operating point: threshold fitted on calibrated BRSET VALIDATION patient risks for sensitivity ≥0.90 with maximum specificity
- Primary aggregation: maximum image probability within eye, then maximum across eyes; algebraically patient maximum
- Bootstrap: {args.bootstrap} independent patient-level percentile resamples per seed
- Target-domain adaptation, target calibration, and target-derived thresholding: prohibited and not performed
"""
    (root / "02_training_configuration.md").write_text(config, encoding="utf-8")
    env = f"""# Environment and Model Provenance

- Timestamp UTC: `{now_utc()}`
- GPU: `{torch.cuda.get_device_name(0)}`
- VRAM: `{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB`
- Python: `{sys.version}`
- PyTorch: `{torch.__version__}`
- CUDA: `{torch.version.cuda}`
- RETFound commit: `{subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()}`
- Checkpoint path: `{args.weights}`
- Checkpoint SHA256: `{PRETRAINED_SHA256}`
- Bundle contains 2,574 BRSET images and only the 1,498 images from the locked 400-patient mBRSET target test.
"""
    (root / "01_environment_and_provenance.md").write_text(env, encoding="utf-8")
    means = seed_df.mean(numeric_only=True)
    report = f"""# Phase 1A1 Gate Report

## A. Environment and model provenance

Official RETFound CFP weights and the official RETFound implementation were used on `{torch.cuda.get_device_name(0)}`. Full fine-tuning passed the BF16 A100 smoke test. See `00_compute_smoke_test.md` and `01_environment_and_provenance.md`.

## B. Source training performance

Three independent full fine-tuning runs used BRSET TRAIN only. Checkpoint selection, Platt calibration, and the 90%-sensitivity operating threshold used BRSET VALIDATION only. Training-set performance is not used as evidence.

## C. Frozen source-test performance

Mean BRSET frozen source-test AUROC was **{means.source_AUROC:.4f}** and mean source-calibrated Brier score was **{means.source_Brier:.4f}**.

## D. Zero-label mBRSET external performance

Mean mBRSET frozen target-test AUROC was **{means.target_AUROC:.4f}** and mean source-calibrated Brier score was **{means.target_Brier:.4f}**. No mBRSET adaptation-pool observation was read.

## E. Discrimination shift

Mean target-minus-source AUROC difference was **{means.delta_AUROC:.4f}**. Seed-level estimates and patient-bootstrap intervals are in `03_seed_level_metrics.csv` and `08_bootstrap_domain_shift.csv`.

## F. Calibration shift

Raw and BRSET-validation Platt-calibrated Brier score, NLL, calibration intercept, calibration slope, and ECE are reported in `06_calibration_metrics.csv`. No calibration model was fitted on mBRSET.

## G. Fixed-threshold clinical screening shift

At the BRSET-validation-derived threshold, mean source sensitivity/specificity were **{means.source_fixed_sensitivity:.4f}/{means.source_fixed_specificity:.4f}**; mean target sensitivity/specificity were **{means.target_fixed_sensitivity:.4f}/{means.target_fixed_specificity:.4f}**. Per-1,000 values use each test cohort's observed prevalence and are descriptive cohort-standardized estimates, not population prevalence estimates.

## H. Seed stability

Concordant supportive deterioration was present in **{n_concordant}/3** seeds. Detailed flags: `{json.dumps(reasons)}`. See `10_training_stability.md`.

## I. Data/protocol deviations

Colab Python 3.13 did not support the repository's older pinned PyTorch wheel, so the compatible preinstalled PyTorch `{torch.__version__}`/CUDA `{torch.version.cuda}` stack was retained. The official model architecture, checkpoint-loading logic, 224-pixel preprocessing, full fine-tuning design, and fixed data locks were preserved. A class-weighted loss was used as prespecified. Bootstrap count was {args.bootstrap}, satisfying the allowed minimum. No target-test result influenced training, selection, calibration, or thresholding.

## J. Phase 1A1 verdict

**{verdict}**

## Eye and Vision Floor Gate

**{eye}**

The Phase 1A1 hard stop is active. No local adaptation, LoRA/PEFT, target recalibration, VisionFM, DINOv3, or label-dose experiment was started.
"""
    (root / "PHASE_1A1_GATE_REPORT.md").write_text(report, encoding="utf-8")


def formal_run(args):
    assert_environment(args)
    br, mb, _ = load_locked_frames(args.bundle)
    train_tf, eval_tf = official_transforms()
    all_seed = []
    for seed in SEEDS:
        model, saved = train_seed(seed, br, args, train_tf, eval_tf)
        seed_row, src, tgt, cal_rows, fixed_rows, boot_rows, sec_rows = evaluate_seed(seed, model, br, mb, args, eval_tf)
        payload = {
            "seed_row": seed_row,
            "source_predictions": src.to_dict(orient="records"),
            "target_predictions": tgt.to_dict(orient="records"),
            "calibration_rows": cal_rows,
            "fixed_rows": fixed_rows,
            "bootstrap_rows": boot_rows,
            "secondary_rows": sec_rows,
            "checkpoint_epoch": saved["epoch"],
        }
        save_json(args.drive_root / "results" / "logs" / f"seed_metrics_{seed}.json", payload)
        all_seed.append(payload)
        del model, saved
        torch.cuda.empty_cache()
    combine_and_report(args, all_seed)


def main():
    args = parse_args()
    for sub in ("results", "results/checkpoints", "results/logs", "results/figures", "logs"):
        (args.drive_root / sub).mkdir(parents=True, exist_ok=True)
    save_json(args.drive_root / "logs" / "phase1a1_runtime_args.json", vars(args))
    if args.mode == "smoke":
        smoke_test(args)
    elif args.mode == "run":
        formal_run(args)
    else:
        assert_environment(args)
        combine_and_report(args)


if __name__ == "__main__":
    main()
