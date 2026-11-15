
import os
import sys
import glob
import json
import time
import copy
import random
import zipfile
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# LOCKED SCIENTIFIC STATUS
# =========================================================

# Phase 1A1 seed 20260906:
BASE_SOURCE_AUC = 0.940230
BASE_TARGET_AUC = 0.850741
BASE_TARGET_BRIER = 0.118971
BASE_TARGET_SENS = 0.548387

# Naive full-FT N=100 historical comparator from Phase 1A2:
NAIVE_FULLFT_N100_AUC = 0.7720
NAIVE_FULLFT_N100_SOURCE_LOSS = -0.0750

ROOT = Path("/content/drive/MyDrive/Retinal_Retraining_Phase1A1")
OUT = ROOT / "results" / "phase1a2_rescue"
OUT.mkdir(parents=True, exist_ok=True)

REPO = "/content/RETFound"
sys.path.insert(0, REPO)

import timm
assert timm.__version__ == "0.9.2"

import models_vit
from util.datasets import build_transform

DEVICE = "cuda"
BATCH_SIZE = 24
NUM_WORKERS = 4

# First pre-specified trajectory from Phase 1A2:
DEV_TRAJECTORY = 20261001
DEV_BUDGET = 100

# Internal patient split, still within the 100-label budget
DEV_SPLIT_SEED = 20262001

# Three optimization seeds
TRAINING_SEEDS = [
    20263001,
    20263002,
    20263003,
]

MAX_EPOCHS = 20
PATIENCE = 4

IMG_EXTS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

# =========================================================
# PATHS
# =========================================================

BR_MAN_PATH = ROOT / "data/locked_metadata/BRSET_master_manifest_v2.csv"
MB_MAN_PATH = ROOT / "data/locked_metadata/mBRSET_master_manifest_v2.csv"
BR_SPLIT_PATH = ROOT / "data/locked_metadata/BRSET_source_split_v2.csv"

SAMPLING_PATH = ROOT / "results/phase1a2/01_sampling_trajectories.csv"

MB_ZIP = ROOT / "data/mbrset_1.0.0.zip"
P1_BUNDLE = ROOT / "data/phase1a1_colab_bundle.zip"

SOURCE_CKPT = (
    ROOT /
    "results/checkpoints/"
    "RETFound_BRSET_seed20260906_best.pth"
)

# =========================================================
# HELPERS
# =========================================================

def choose_col(df, names, required=True):
    lower = {str(c).lower(): c for c in df.columns}

    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]

    if required:
        raise KeyError(
            f"Cannot find columns {names}. "
            f"Available = {list(df.columns)}"
        )

    return None


def basename_any(x):
    return os.path.basename(
        str(x).replace("\\", "/")
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# LOAD LOCKED DATA
# =========================================================

br = pd.read_csv(BR_MAN_PATH)
mb = pd.read_csv(MB_MAN_PATH)
sampling = pd.read_csv(SAMPLING_PATH)

br_pid = choose_col(br, ["patient_id"])
mb_pid = choose_col(mb, ["patient_id"])

br_img = choose_col(
    br,
    ["image_path", "image_id", "filename", "file"]
)

mb_img = choose_col(
    mb,
    ["image_path", "image_id", "filename", "file"]
)

br_y_img = choose_col(br, ["referable_dr_image"])
mb_y_img = choose_col(mb, ["referable_dr_image"])

br_y_pt = choose_col(br, ["referable_dr_patient"])
mb_y_pt = choose_col(mb, ["referable_dr_patient"])

br_split_col = choose_col(br, ["split"])
mb_split_col = choose_col(mb, ["split"])

br["_split"] = br[br_split_col].astype(str).str.upper()
mb["_split"] = mb[mb_split_col].astype(str).str.upper()

# IMPORTANT: preserve Phase 1A0.5 analyzable lock
mb = mb[
    mb[mb_y_img].notna()
    & mb[mb_y_pt].notna()
].copy()

assert mb[mb_pid].nunique() == 1287
assert len(mb) == 4884

BR_TEST = br[
    br["_split"] == "FROZEN_SOURCE_TEST"
].copy()

MB_ADAPT = mb[
    mb["_split"] == "ADAPTATION_POOL"
].copy()

MB_PILOT_TEST = mb[
    mb["_split"] == "FROZEN_TARGET_TEST"
].copy()

assert BR_TEST[br_pid].nunique() == 202
assert MB_ADAPT[mb_pid].nunique() == 887
assert MB_PILOT_TEST[mb_pid].nunique() == 400

# =========================================================
# LOCK 100-PATIENT DEVELOPMENT COHORT
# =========================================================

s_pid = choose_col(sampling, ["patient_id"])
s_seed = choose_col(sampling, ["trajectory_seed"])
s_budget = choose_col(sampling, ["budget"])

dev_ids = (
    sampling[
        (sampling[s_seed].astype(int) == DEV_TRAJECTORY)
        &
        (sampling[s_budget].astype(int) == DEV_BUDGET)
    ][s_pid]
    .astype(str)
    .unique()
)

assert len(dev_ids) == 100

DEV100 = MB_ADAPT[
    MB_ADAPT[mb_pid].astype(str).isin(dev_ids)
].copy()

assert DEV100[mb_pid].nunique() == 100

pt = (
    DEV100[[mb_pid, mb_y_pt]]
    .drop_duplicates(mb_pid)
    .copy()
)

pt[mb_pid] = pt[mb_pid].astype(str)
pt[mb_y_pt] = pt[mb_y_pt].astype(int)

train_ids, val_ids = train_test_split(
    pt[mb_pid].values,
    test_size=0.30,
    random_state=DEV_SPLIT_SEED,
    stratify=pt[mb_y_pt].values,
)

train_ids = set(map(str, train_ids))
val_ids = set(map(str, val_ids))

DEV_TRAIN = DEV100[
    DEV100[mb_pid].astype(str).isin(train_ids)
].copy()

DEV_VAL = DEV100[
    DEV100[mb_pid].astype(str).isin(val_ids)
].copy()

assert DEV_TRAIN[mb_pid].nunique() == 70
assert DEV_VAL[mb_pid].nunique() == 30
assert train_ids.isdisjoint(val_ids)

split_out = pd.DataFrame({
    "patient_id": list(train_ids) + list(val_ids),
    "split": (
        ["LOCAL_TRAIN"] * len(train_ids)
        +
        ["LOCAL_VALIDATION"] * len(val_ids)
    )
})

split_out.to_csv(
    OUT / "01_locked_100_patient_development_split.csv",
    index=False
)

print("100-patient development cohort:")
print(" train =", DEV_TRAIN[mb_pid].nunique())
print(" val   =", DEV_VAL[mb_pid].nunique())

# =========================================================
# MATERIALIZE ONLY REQUIRED IMAGES
# =========================================================

DATA_ROOT = Path("/content/phase1a2_rescue_data")

if DATA_ROOT.exists():
    raise FileExistsError("Candidate safety guard: inspect or relocate existing extraction directory before rerun")

(DATA_ROOT / "mbrset").mkdir(parents=True)
(DATA_ROOT / "brset").mkdir(parents=True)

m_needed = set(
    basename_any(x)
    for x in pd.concat(
        [DEV100, MB_PILOT_TEST]
    )[mb_img]
)

with zipfile.ZipFile(MB_ZIP, "r") as z:

    member_map = {}

    for m in z.namelist():
        b = basename_any(m)

        if b in m_needed:
            member_map.setdefault(b, []).append(m)

    absent = m_needed - set(member_map)

    if absent:
        raise RuntimeError(
            f"Missing required mBRSET images: "
            f"{len(absent)}"
        )

    ambiguous = {
        k:v
        for k,v in member_map.items()
        if len(v) != 1
    }

    if ambiguous:
        raise RuntimeError(
            "Ambiguous mBRSET filenames."
        )

    for i, (b, members) in enumerate(
        member_map.items(), 1
    ):
        target = DATA_ROOT / "mbrset" / b

        with z.open(members[0]) as src:
            target.write_bytes(src.read())

        if i % 500 == 0:
            print("mBRSET extracted", i)


# BRSET frozen source test from Phase1A1 bundle
br_needed = set(
    basename_any(x)
    for x in BR_TEST[br_img]
)

with zipfile.ZipFile(P1_BUNDLE, "r") as z:

    member_map = {}

    for m in z.namelist():

        b = basename_any(m)

        if b in br_needed:
            member_map.setdefault(b, []).append(m)

    absent = br_needed - set(member_map)

    if absent:
        raise RuntimeError(
            f"Missing BRSET source-test images: "
            f"{len(absent)}"
        )

    for b, members in member_map.items():

        if len(members) != 1:
            raise RuntimeError(
                "Ambiguous BRSET filename."
            )

        target = DATA_ROOT / "brset" / b

        with z.open(members[0]) as src:
            target.write_bytes(src.read())


DEV100["_runtime_path"] = DEV100[mb_img].map(
    lambda x:
    str(DATA_ROOT / "mbrset" / basename_any(x))
)

DEV_TRAIN["_runtime_path"] = DEV_TRAIN[mb_img].map(
    lambda x:
    str(DATA_ROOT / "mbrset" / basename_any(x))
)

DEV_VAL["_runtime_path"] = DEV_VAL[mb_img].map(
    lambda x:
    str(DATA_ROOT / "mbrset" / basename_any(x))
)

MB_PILOT_TEST["_runtime_path"] = (
    MB_PILOT_TEST[mb_img].map(
        lambda x:
        str(DATA_ROOT / "mbrset" / basename_any(x))
    )
)

BR_TEST["_runtime_path"] = BR_TEST[br_img].map(
    lambda x:
    str(DATA_ROOT / "brset" / basename_any(x))
)

for df in [
    DEV100,
    DEV_TRAIN,
    DEV_VAL,
    MB_PILOT_TEST,
    BR_TEST,
]:
    assert df["_runtime_path"].map(
        os.path.exists
    ).all()

print("Image materialization: PASS")

# =========================================================
# SOURCE CHECKPOINT
# =========================================================

source_ckpt = torch.load(
    SOURCE_CKPT,
    map_location="cpu",
    weights_only=False
)

def extract_state(ckpt):

    if isinstance(ckpt, dict):

        for key in [
            "model",
            "state_dict",
            "model_state_dict",
        ]:
            if (
                key in ckpt
                and isinstance(ckpt[key], dict)
            ):
                state = ckpt[key]
                break
        else:
            state = ckpt

    else:
        state = ckpt

    out = {}

    for k,v in state.items():

        if not torch.is_tensor(v):
            continue

        if k.startswith("module."):
            k = k[7:]

        out[k] = v

    return out


SOURCE_STATE = extract_state(source_ckpt)

N_CLASSES = int(
    SOURCE_STATE["head.weight"].shape[0]
)

INPUT_SIZE = 224
DROP_PATH = 0.2
GLOBAL_POOL = True

# =========================================================
# TRANSFORMS
# =========================================================

tf_args = SimpleNamespace(
    input_size=224,
    color_jitter=None,
    aa="rand-m9-mstd0.5-inc1",
    reprob=0.25,
    remode="pixel",
    recount=1,
)

TRAIN_TF = build_transform(
    "train",
    tf_args
)

EVAL_TF = build_transform(
    "val",
    tf_args
)

# =========================================================
# DATASETS
# =========================================================

class FrameDataset(Dataset):

    def __init__(
        self,
        df,
        pid_col,
        y_img_col,
        y_pt_col,
        transform,
    ):

        self.df = (
            df.reset_index(drop=True).copy()
        )

        self.pid_col = pid_col
        self.y_img_col = y_img_col
        self.y_pt_col = y_pt_col
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, i):

        r = self.df.iloc[i]

        img = Image.open(
            r["_runtime_path"]
        ).convert("RGB")

        img = self.transform(img)

        return (
            img,
            int(r[self.y_img_col]),
            i,
        )


def make_loader(
    df,
    dataset_name,
    train=False,
    seed=1,
):

    if dataset_name == "mbrset":
        pid_col = mb_pid
        yi = mb_y_img
        yp = mb_y_pt
    else:
        pid_col = br_pid
        yi = br_y_img
        yp = br_y_pt

    ds = FrameDataset(
        df,
        pid_col,
        yi,
        yp,
        TRAIN_TF if train else EVAL_TF,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=(
            BATCH_SIZE if train else 64
        ),
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=g if train else None,
    ), ds


# =========================================================
# MODEL
# =========================================================

def fresh_source_model():

    model = models_vit.RETFound_mae(
        img_size=224,
        num_classes=N_CLASSES,
        drop_path_rate=DROP_PATH,
        global_pool=GLOBAL_POOL,
    )

    model.load_state_dict(
        SOURCE_STATE,
        strict=False
    )

    return model.cuda()


# =========================================================
# MINIMAL LoRA
# =========================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        base,
        r=8,
        alpha=16,
        dropout=0.05,
    ):

        super().__init__()

        self.base = base

        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout)

        # LoRA parameters must be created on the same device/dtype
        # as the wrapped pretrained Linear layer.
        device = base.weight.device
        dtype = base.weight.dtype

        self.A = nn.Parameter(
            torch.empty(
                r,
                base.in_features,
                device=device,
                dtype=dtype,
            )
        )

        self.B = nn.Parameter(
            torch.zeros(
                base.out_features,
                r,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.kaiming_uniform_(
            self.A,
            a=np.sqrt(5)
        )


    def forward(self, x):

        return (
            self.base(x)
            +
            F.linear(
                F.linear(
                    self.dropout(x),
                    self.A
                ),
                self.B
            )
            * self.scaling
        )


def configure_method(model, method):

    for p in model.parameters():
        p.requires_grad = False

    if method == "head_only":

        for p in model.head.parameters():
            p.requires_grad = True

        return 1e-3, 1e-4


    if method == "partial_last2":

        for block in model.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True

        for p in model.head.parameters():
            p.requires_grad = True

        if hasattr(model, "fc_norm"):
            for p in model.fc_norm.parameters():
                p.requires_grad = True

        return 1e-5, 1e-2


    if method == "lora":

        # Replace qkv + proj in all transformer blocks
        for block in model.blocks:

            block.attn.qkv = LoRALinear(
                block.attn.qkv,
                r=8,
                alpha=16
            )

            block.attn.proj = LoRALinear(
                block.attn.proj,
                r=8,
                alpha=16
            )

        for p in model.head.parameters():
            p.requires_grad = True

        return 1e-4, 1e-2


    if method == "full_lr1e6":

        for p in model.parameters():
            p.requires_grad = True

        return 1e-6, 5e-2


    if method == "full_lr3e6":

        for p in model.parameters():
            p.requires_grad = True

        return 3e-6, 5e-2


    raise ValueError(method)


# =========================================================
# PREDICTION
# =========================================================

def probs_from_logits(logits):

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.shape[1] == 1:
        return torch.sigmoid(
            logits[:,0]
        )

    return torch.softmax(
        logits,
        dim=1
    )[:,1]


@torch.inference_mode()
def predict_patient(
    model,
    df,
    dataset_name,
):

    loader, ds = make_loader(
        df,
        dataset_name,
        train=False
    )

    model.eval()

    rows = []

    for xb, _, ii in loader:

        xb = xb.cuda(
            non_blocking=True
        )

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16
        ):
            logits = model(xb)

        p = probs_from_logits(
            logits
        ).float().cpu().numpy()

        for idx, prob in zip(
            ii.numpy(),
            p
        ):

            r = ds.df.iloc[int(idx)]

            if dataset_name == "mbrset":
                pid = str(r[mb_pid])
                ypt = int(r[mb_y_pt])
            else:
                pid = str(r[br_pid])
                ypt = int(r[br_y_pt])

            rows.append({
                "patient_id": pid,
                "y": ypt,
                "p": float(prob)
            })

    im = pd.DataFrame(rows)

    pt = (
        im.groupby(
            "patient_id",
            as_index=False
        )
        .agg(
            y=("y", "first"),
            p=("p", "max")
        )
    )

    return pt


# =========================================================
# METRICS
# =========================================================

EPS = 1e-6

def logit_np(p):
    p = np.clip(
        np.asarray(p),
        EPS,
        1-EPS
    )
    return np.log(
        p / (1-p)
    )


def fit_platt(pt):

    X = logit_np(
        pt["p"].values
    ).reshape(-1,1)

    y = pt["y"].values

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000
    )

    model.fit(X, y)

    return model


def apply_platt(model, p):

    X = logit_np(p).reshape(-1,1)

    return model.predict_proba(X)[:,1]


def threshold_sens90(y, p):

    y = np.asarray(y).astype(int)
    p = np.asarray(p)

    best = None

    for th in np.unique(p):

        pred = (p >= th).astype(int)

        tn,fp,fn,tp = confusion_matrix(
            y,
            pred,
            labels=[0,1]
        ).ravel()

        sens = (
            tp / (tp+fn)
            if tp+fn else np.nan
        )

        spec = (
            tn / (tn+fp)
            if tn+fp else np.nan
        )

        if sens >= 0.90:

            if (
                best is None
                or spec > best[0]
            ):
                best = (
                    spec,
                    th
                )

    return (
        float(best[1])
        if best is not None
        else 0.0
    )


def calc_metrics(y, p, th=None):

    y = np.asarray(y).astype(int)
    p = np.clip(
        np.asarray(p),
        EPS,
        1-EPS
    )

    out = {
        "auroc": roc_auc_score(y,p),
        "auprc": average_precision_score(y,p),
        "brier": brier_score_loss(y,p),
        "nll": log_loss(
            y,p,labels=[0,1]
        ),
    }

    if th is not None:

        pred = (p >= th).astype(int)

        tn,fp,fn,tp = confusion_matrix(
            y,
            pred,
            labels=[0,1]
        ).ravel()

        out.update({
            "sensitivity":
                tp/(tp+fn)
                if tp+fn else np.nan,
            "specificity":
                tn/(tn+fp)
                if tn+fp else np.nan,
        })

    return out


# =========================================================
# TRAINING
# =========================================================

def criterion_from_train(df):

    y = df[mb_y_img].astype(int)

    n0 = int((y==0).sum())
    n1 = int((y==1).sum())

    weights = torch.tensor([
        len(y)/(2*n0),
        len(y)/(2*n1)
    ], device="cuda")

    return nn.CrossEntropyLoss(
        weight=weights
    )


def clone_state_cpu(model):

    return {
        k:v.detach().cpu().clone()
        for k,v in model.state_dict().items()
    }


def train_method(
    method,
    seed,
):

    set_seed(seed)

    model = fresh_source_model()

    lr, wd = configure_method(
        model,
        method
    )

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    n_trainable = sum(
        p.numel()
        for p in trainable
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=wd
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=lr*0.1
    )

    criterion = criterion_from_train(
        DEV_TRAIN
    )

    train_loader, _ = make_loader(
        DEV_TRAIN,
        "mbrset",
        train=True,
        seed=seed
    )

    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    no_improve = 0

    history = []

    for epoch in range(MAX_EPOCHS):

        model.train()

        losses = []

        for xb, yb, _ in train_loader:

            xb = xb.cuda(
                non_blocking=True
            )

            yb = yb.cuda(
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):

                logits = model(xb)

                if isinstance(
                    logits,
                    (tuple,list)
                ):
                    logits = logits[0]

                loss = criterion(
                    logits,
                    yb.long()
                )

            loss.backward()
            optimizer.step()

            losses.append(
                float(loss.detach().cpu())
            )

        scheduler.step()

        val_pt = predict_patient(
            model,
            DEV_VAL,
            "mbrset"
        )

        val_auc = roc_auc_score(
            val_pt["y"],
            val_pt["p"]
        )

        history.append({
            "epoch": epoch+1,
            "loss": np.mean(losses),
            "val_auc": val_auc,
        })

        print(
            method,
            seed,
            f"epoch={epoch+1}",
            f"loss={np.mean(losses):.4f}",
            f"val_auc={val_auc:.4f}",
            flush=True
        )

        if val_auc > best_auc + 1e-4:

            best_auc = val_auc
            best_epoch = epoch+1

            best_state = clone_state_cpu(
                model
            )

            no_improve = 0

        else:

            no_improve += 1

        if no_improve >= PATIENCE:
            break

    assert best_state is not None

    model.load_state_dict(
        best_state,
        strict=True
    )

    del best_state

    # Local validation calibration
    val_pt = predict_patient(
        model,
        DEV_VAL,
        "mbrset"
    )

    platt = fit_platt(val_pt)

    val_cal = apply_platt(
        platt,
        val_pt["p"].values
    )

    threshold = threshold_sens90(
        val_pt["y"],
        val_cal
    )

    # Development/pilot target test
    target_pt = predict_patient(
        model,
        MB_PILOT_TEST,
        "mbrset"
    )

    target_cal = apply_platt(
        platt,
        target_pt["p"].values
    )

    target_raw_metrics = calc_metrics(
        target_pt["y"],
        target_pt["p"]
    )

    target_cal_metrics = calc_metrics(
        target_pt["y"],
        target_cal,
        threshold
    )

    # Source retention:
    source_pt = predict_patient(
        model,
        BR_TEST,
        "brset"
    )

    source_auc = roc_auc_score(
        source_pt["y"],
        source_pt["p"]
    )

    row = {
        "method": method,
        "seed": seed,
        "trainable_parameters": n_trainable,
        "best_epoch": best_epoch,
        "local_val_auc": best_auc,
        "target_raw_auroc":
            target_raw_metrics["auroc"],
        "target_raw_auprc":
            target_raw_metrics["auprc"],
        "target_cal_brier":
            target_cal_metrics["brier"],
        "target_cal_nll":
            target_cal_metrics["nll"],
        "target_cal_sensitivity":
            target_cal_metrics["sensitivity"],
        "target_cal_specificity":
            target_cal_metrics["specificity"],
        "source_raw_auroc":
            source_auc,
        "source_auroc_change":
            source_auc - BASE_SOURCE_AUC,
        "threshold":
            threshold,
    }

    del model
    torch.cuda.empty_cache()

    return row, history


# =========================================================
# RECALIBRATION-ONLY STRATEGY
# =========================================================

print("\n=== RECALIBRATION ONLY ===")

source_model = fresh_source_model()

dev100_pt = predict_patient(
    source_model,
    DEV100,
    "mbrset"
)

recal = fit_platt(
    dev100_pt
)

dev100_cal = apply_platt(
    recal,
    dev100_pt["p"].values
)

recal_threshold = threshold_sens90(
    dev100_pt["y"],
    dev100_cal
)

pilot_pt = predict_patient(
    source_model,
    MB_PILOT_TEST,
    "mbrset"
)

pilot_cal = apply_platt(
    recal,
    pilot_pt["p"].values
)

recal_metrics = calc_metrics(
    pilot_pt["y"],
    pilot_cal,
    recal_threshold
)

recal_raw_auc = roc_auc_score(
    pilot_pt["y"],
    pilot_pt["p"]
)

recal_row = {
    "method":
        "recalibration_only_100",
    "seed": 0,
    "trainable_parameters": 0,
    "best_epoch": 0,
    "local_val_auc":
        roc_auc_score(
            dev100_pt["y"],
            dev100_pt["p"]
        ),
    "target_raw_auroc":
        recal_raw_auc,
    "target_raw_auprc":
        average_precision_score(
            pilot_pt["y"],
            pilot_pt["p"]
        ),
    "target_cal_brier":
        recal_metrics["brier"],
    "target_cal_nll":
        recal_metrics["nll"],
    "target_cal_sensitivity":
        recal_metrics["sensitivity"],
    "target_cal_specificity":
        recal_metrics["specificity"],
    "source_raw_auroc":
        BASE_SOURCE_AUC,
    "source_auroc_change": 0.0,
    "threshold":
        recal_threshold,
}

del source_model
torch.cuda.empty_cache()

# =========================================================
# TRAINABLE METHODS
# =========================================================

METHODS = [
    "head_only",
    "partial_last2",
    "lora",
    "full_lr1e6",
    "full_lr3e6",
]

RESULTS_PATH = OUT / "02_method_seed_metrics.csv"

existing = (
    pd.read_csv(RESULTS_PATH)
    if RESULTS_PATH.exists()
    else pd.DataFrame()
)

rows = []

if not (
    len(existing)
    and "method" in existing.columns
    and "recalibration_only_100"
        in set(existing["method"])
):
    rows.append(recal_row)

for method in METHODS:

    for seed in TRAINING_SEEDS:

        run_exists = False

        if len(existing):

            run_exists = (
                (
                    existing["method"] == method
                )
                &
                (
                    existing["seed"].astype(int)
                    == seed
                )
            ).any()

        if run_exists:
            print("SKIP", method, seed)
            continue

        print(
            "\n" + "="*80
        )

        print(
            "RUN",
            method,
            seed
        )

        print(
            "="*80
        )

        row, history = train_method(
            method,
            seed
        )

        rows.append(row)

        pd.DataFrame(history).to_csv(
            OUT /
            f"history_{method}_{seed}.csv",
            index=False
        )

        pd.DataFrame([row]).to_csv(
            RESULTS_PATH,
            mode="a",
            header=not RESULTS_PATH.exists(),
            index=False
        )

# ensure recalibration row persisted
if rows and not RESULTS_PATH.exists():
    pd.DataFrame(rows).to_csv(
        RESULTS_PATH,
        index=False
    )

elif (
    rows
    and recal_row in rows
):
    # if file exists due later runs
    current = pd.read_csv(RESULTS_PATH)

    if "recalibration_only_100" not in set(
        current["method"]
    ):
        pd.DataFrame([recal_row]).to_csv(
            RESULTS_PATH,
            mode="a",
            header=False,
            index=False
        )

results = pd.read_csv(
    RESULTS_PATH
)

# =========================================================
# METHOD SUMMARY
# =========================================================

summary_rows = []

for method, g in results.groupby("method"):

    summary_rows.append({
        "method": method,
        "n_runs": len(g),
        "median_target_auroc":
            g["target_raw_auroc"].median(),
        "min_target_auroc":
            g["target_raw_auroc"].min(),
        "max_target_auroc":
            g["target_raw_auroc"].max(),
        "median_target_brier":
            g["target_cal_brier"].median(),
        "median_target_sensitivity":
            g["target_cal_sensitivity"].median(),
        "median_target_specificity":
            g["target_cal_specificity"].median(),
        "median_source_auroc_change":
            g["source_auroc_change"].median(),
        "worst_source_auroc_change":
            g["source_auroc_change"].min(),
    })

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT / "03_method_summary.csv",
    index=False
)

# =========================================================
# RESCUE CRITERIA
# =========================================================

trainable_results = results[
    results["method"].isin(METHODS)
].copy()

trainable_results["safe"] = (
    (trainable_results["target_raw_auroc"]
        >= BASE_TARGET_AUC - 0.01)
    &
    (trainable_results["source_auroc_change"]
        >= -0.02)
)

trainable_results["benefit"] = (
    (
        trainable_results["target_raw_auroc"]
        >= BASE_TARGET_AUC + 0.015
    )
    |
    (
        trainable_results["target_cal_brier"]
        <= BASE_TARGET_BRIER - 0.015
    )
    |
    (
        trainable_results["target_cal_sensitivity"]
        >= BASE_TARGET_SENS + 0.10
    )
)

trainable_results["rescue"] = (
    trainable_results["safe"]
    &
    trainable_results["benefit"]
)

rescue_counts = (
    trainable_results
    .groupby("method")["rescue"]
    .sum()
)

strong_methods = [
    m
    for m,n in rescue_counts.items()
    if n >= 2
]

# recalibration-only signal
recal_brier = float(
    results[
        results["method"]
        == "recalibration_only_100"
    ]["target_cal_brier"].iloc[0]
)

recal_sens = float(
    results[
        results["method"]
        == "recalibration_only_100"
    ]["target_cal_sensitivity"].iloc[0]
)

recal_signal = (
    recal_brier
    <= BASE_TARGET_BRIER - 0.015
    or
    recal_sens
    >= BASE_TARGET_SENS + 0.10
)

if strong_methods:

    VERDICT = (
        "STRONG RESCUE SIGNAL — "
        "PROCEED TO STRATEGY-SPECIFIC LABEL-DOSE"
    )

elif recal_signal:

    VERDICT = (
        "CALIBRATION-ONLY RESCUE SIGNAL — "
        "CONDITIONAL GO"
    )

else:

    VERDICT = (
        "RESCUE FAILED — "
        "HUMAN REVIEW BEFORE CONTINUATION"
    )

# Eye and Vision minimum still requires
# external-domain replication
EYE_STATUS = (
    "CONDITIONAL — "
    "requires independent external-domain replication"
)

# =========================================================
# FIGURE
# =========================================================

plt.figure(figsize=(8,5))

plot_df = summary.sort_values(
    "median_target_auroc",
    ascending=False
)

plt.scatter(
    plot_df[
        "median_source_auroc_change"
    ],
    plot_df[
        "median_target_auroc"
    ],
)

for _,r in plot_df.iterrows():

    plt.annotate(
        r["method"],
        (
            r["median_source_auroc_change"],
            r["median_target_auroc"]
        )
    )

plt.axhline(
    BASE_TARGET_AUC,
    linestyle="--"
)

plt.axvline(
    -0.02,
    linestyle="--"
)

plt.xlabel(
    "Median source AUROC change"
)

plt.ylabel(
    "Median target AUROC"
)

plt.title(
    "Target recovery vs source retention"
)

plt.tight_layout()

plt.savefig(
    OUT /
    "figure_strategy_rescue.png",
    dpi=300
)

plt.savefig(
    OUT /
    "figure_strategy_rescue.pdf"
)

plt.close()

# =========================================================
# REPORT
# =========================================================

best = summary.sort_values(
    [
        "median_target_auroc",
        "median_source_auroc_change"
    ],
    ascending=[False, False]
).iloc[0]

report = f"""# PHASE 1A2-R1 ADAPTATION-STRATEGY RESCUE REPORT

## Design status

mBRSET is now treated as a METHOD-DEVELOPMENT DOMAIN.

The previously frozen 400-patient target set is a pilot/development
evaluation set and is no longer claimed as a final untouched confirmation set.

No new mBRSET confirmation cohort was created because the Phase 1A2
887-patient oracle previously used all 887 adaptation-pool patients.

Final independent confirmation must therefore come from an untouched
external dataset/domain.

## Locked development cohort

Trajectory seed: {DEV_TRAJECTORY}

Total locally labelled patients: 100

Local train patients: 70

Local validation patients: 30

## Historical baseline

Zero-label target AUROC: {BASE_TARGET_AUC:.4f}

Source AUROC: {BASE_SOURCE_AUC:.4f}

Naive full-fine-tuning N=100 target AUROC: {NAIVE_FULLFT_N100_AUC:.4f}

Naive full-fine-tuning N=100 mean source AUROC loss: {NAIVE_FULLFT_N100_SOURCE_LOSS:.4f}

## Strategies tested

- recalibration-only
- head-only tuning
- last-two-block partial fine-tuning
- LoRA
- full fine-tuning at LR 1e-6
- full fine-tuning at LR 3e-6

## Best method by development target AUROC

Method: {best['method']}

Median target AUROC: {best['median_target_auroc']:.4f}

Median target Brier: {best['median_target_brier']:.4f}

Median target sensitivity: {best['median_target_sensitivity']:.4f}

Median source AUROC change: {best['median_source_auroc_change']:+.4f}

## Reproducible rescue methods

{strong_methods}

## Phase 1A2-R1 verdict

**{VERDICT}**

## Eye and Vision floor

**{EYE_STATUS}**

The minimum publication floor is not considered secured by mBRSET
development results alone.

Before an Eye and Vision-level claim is made, the locked winning strategy
must be tested in at least one genuinely untouched external domain and
ultimately expanded to the planned multi-domain design.
"""

(OUT / "PHASE_1A2_R1_RESCUE_REPORT.md").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "="*80)
print("PHASE 1A2-R1 FINAL")
print("="*80)

print(summary.to_string(index=False))

print("\nStrong rescue methods:", strong_methods)

print("\nVERDICT:")
print(VERDICT)

print("\nEYE AND VISION FLOOR:")
print(EYE_STATUS)

print(
    "\nREPORT:",
    OUT / "PHASE_1A2_R1_RESCUE_REPORT.md"
)

print("\nSTOP. Do not start another domain automatically.")
