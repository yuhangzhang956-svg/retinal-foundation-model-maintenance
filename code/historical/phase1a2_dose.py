
import os
import sys
import glob
import json
import time
import math
import random
import zipfile
import hashlib
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

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
# LOCKED CONFIGURATION
# =========================================================

DRIVE_ROOT = "/content/drive/MyDrive/Retinal_Retraining_Phase1A1"
OUT = os.path.join(DRIVE_ROOT, "results", "phase1a2")
os.makedirs(OUT, exist_ok=True)

RETF_REPO = "/content/RETFound"
sys.path.insert(0, RETF_REPO)

import models_vit
import util.lr_decay as lrd
from util.datasets import build_transform

DEVICE = "cuda"
BATCH_SIZE = 24

# Pilot: same number of passes through each local labelled cohort.
# Fixed BEFORE frozen-target evaluation.
ADAPT_EPOCHS = 10

# Same full fine-tuning recipe across all budgets
FALLBACK_LR = 9.25e-5
FALLBACK_WEIGHT_DECAY = 0.05
FALLBACK_LAYER_DECAY = 0.65

TRAJECTORY_SEEDS = [
    20261001,
    20261002,
    20261003,
    20261004,
    20261005,
]

BUDGETS = [25, 100, 500]

SAVE_ADAPTED_CHECKPOINTS = False  # Pilot: avoid filling Google Drive
NUM_WORKERS = 4

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# =========================================================
# HELPERS
# =========================================================

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def find_files(filename):
    roots = [
        "/content/phase1a1",
        "/content",
        DRIVE_ROOT,
    ]
    out = []
    for root in roots:
        if not os.path.exists(root):
            continue
        out.extend(glob.glob(os.path.join(root, "**", filename), recursive=True))
    return sorted(set(out))

def require_one(filename):
    hits = find_files(filename)
    if not hits:
        raise FileNotFoundError(f"Cannot find {filename}")
    # prefer /content copy for speed
    hits.sort(key=lambda p: (0 if p.startswith("/content/phase1a1") else 1, len(p)))
    print(f"{filename}: {hits[0]}")
    return hits[0]

def choose_col(df, names, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    if required:
        raise KeyError(f"Cannot find any of columns {names}. Available:\n{list(df.columns)}")
    return None

def normalize_split(x):
    return str(x).strip().upper()

def basename_any(x):
    return os.path.basename(str(x).replace("\\", "/"))

# =========================================================
# LOCATE AUTHORITATIVE FILES
# =========================================================

BR_MAN_PATH = require_one("BRSET_master_manifest_v2.csv")
MB_MAN_PATH = require_one("mBRSET_master_manifest_v2.csv")
BR_SPLIT_PATH = require_one("BRSET_source_split_v2.csv")

br = pd.read_csv(BR_MAN_PATH)
mb = pd.read_csv(MB_MAN_PATH)
br_split = pd.read_csv(BR_SPLIT_PATH)

print("BR manifest:", br.shape)
print("mBRSET manifest:", mb.shape)

# Columns
br_pid = choose_col(br, ["patient_id"])
mb_pid = choose_col(mb, ["patient_id"])

br_img = choose_col(br, ["image_path", "image_id", "filename", "file"])
mb_img = choose_col(mb, ["image_path", "image_id", "filename", "file"])

br_y_img = choose_col(br, ["referable_dr_image"])
mb_y_img = choose_col(mb, ["referable_dr_image"])

br_y_pt = choose_col(br, ["referable_dr_patient"])
mb_y_pt = choose_col(mb, ["referable_dr_patient"])

br_split_col = choose_col(br, ["split"], required=False)
mb_split_col = choose_col(mb, ["split"])

# If BR manifest lacks split, merge locked source split
if br_split_col is None:
    sp_pid = choose_col(br_split, ["patient_id"])
    sp_img = choose_col(br_split, ["image_path", "image_id", "filename", "file"])
    sp_split = choose_col(br_split, ["split"])

    tmp = br_split[[sp_pid, sp_img, sp_split]].copy()
    tmp["_key"] = (
        tmp[sp_pid].astype(str) + "||" +
        tmp[sp_img].astype(str).map(basename_any)
    )

    br["_key"] = (
        br[br_pid].astype(str) + "||" +
        br[br_img].astype(str).map(basename_any)
    )

    mp = tmp.drop_duplicates("_key").set_index("_key")[sp_split]
    br["split"] = br["_key"].map(mp)
    br_split_col = "split"

br["_split"] = br[br_split_col].map(normalize_split)
mb["_split"] = mb[mb_split_col].map(normalize_split)

BR_TRAIN = br[br["_split"] == "TRAIN"].copy()
BR_VAL = br[br["_split"] == "VALIDATION"].copy()
BR_TEST = br[br["_split"] == "FROZEN_SOURCE_TEST"].copy()

# IMPORTANT:
# Phase 1A0.5 locked the analyzable mBRSET cohort at
# 1,287 patients / 4,884 images.
# Rows lacking valid image- or patient-level referable-DR labels
# are excluded BEFORE applying the frozen patient split.

MB_ANALYZABLE = mb[
    mb[mb_y_img].notna()
    & mb[mb_y_pt].notna()
].copy()

MB_ADAPT = MB_ANALYZABLE[
    MB_ANALYZABLE["_split"] == "ADAPTATION_POOL"
].copy()

MB_TEST = MB_ANALYZABLE[
    MB_ANALYZABLE["_split"] == "FROZEN_TARGET_TEST"
].copy()

# Hard locked-count assertions from Phase 1A0.5
assert MB_ANALYZABLE[mb_pid].nunique() == 1287
assert len(MB_ANALYZABLE) == 4884

assert MB_ADAPT[mb_pid].nunique() == 887
assert len(MB_ADAPT) == 3386

assert MB_TEST[mb_pid].nunique() == 400
assert len(MB_TEST) == 1498

print("\nLocked cohorts:")
print("BR train patients:", BR_TRAIN[br_pid].nunique(), "images:", len(BR_TRAIN))
print("BR val patients:", BR_VAL[br_pid].nunique(), "images:", len(BR_VAL))
print("BR test patients:", BR_TEST[br_pid].nunique(), "images:", len(BR_TEST))
print("MB adapt patients:", MB_ADAPT[mb_pid].nunique(), "images:", len(MB_ADAPT))
print("MB test patients:", MB_TEST[mb_pid].nunique(), "images:", len(MB_TEST))

assert BR_TRAIN[br_pid].nunique() == 939
assert BR_VAL[br_pid].nunique() == 202
assert BR_TEST[br_pid].nunique() == 202
assert MB_ADAPT[mb_pid].nunique() == 887
assert MB_TEST[mb_pid].nunique() == 400

# =========================================================
# IMAGE INDEX
# =========================================================

def image_index(roots):
    idx = {}
    for root in roots:
        if not os.path.exists(root):
            continue
        for dp, _, files in os.walk(root):
            for fn in files:
                if Path(fn).suffix.lower() in IMG_EXTS:
                    idx.setdefault(fn, []).append(os.path.join(dp, fn))
    return idx

SEARCH_ROOTS = [
    "/content/phase1a1",
    "/content/phase1a2_mbrset_extra",
]

idx = image_index(SEARCH_ROOTS)

def resolve_from_index(filename, dataset_hint=None):
    b = basename_any(filename)
    hits = idx.get(b, [])
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]

    if dataset_hint:
        hint = dataset_hint.lower()
        preferred = [
            p for p in hits
            if hint in p.lower()
            or ("brset" in hint and ("brazil" in p.lower() or "brset" in p.lower()))
            or ("mbrset" in hint and "mbrset" in p.lower())
        ]
        if len(preferred) == 1:
            return preferred[0]

    # exact duplicate filename ambiguity is not silently accepted
    return None

# =========================================================
# IF NEEDED, RESTORE PHASE1A1 BUNDLE
# =========================================================

def missing_count(df, img_col, hint):
    return sum(resolve_from_index(x, hint) is None for x in df[img_col])

need_existing = pd.concat([
    BR_TRAIN,
    BR_VAL,
    BR_TEST,
], ignore_index=True)

missing_br = missing_count(need_existing, br_img, "brset")
missing_mb_test = missing_count(MB_TEST, mb_img, "mbrset")

if missing_br or missing_mb_test:
    bundle_zips = glob.glob(
        os.path.join(DRIVE_ROOT, "**", "phase1a1_colab_bundle.zip"),
        recursive=True
    )
    if bundle_zips:
        print("Extracting existing Phase1A1 bundle...")
        dest = "/content/phase1a1_bundle_restore"
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(bundle_zips[0], "r") as z:
            z.extractall(dest)

        SEARCH_ROOTS.append(dest)
        idx = image_index(SEARCH_ROOTS)

# =========================================================
# EXTRACT MBRSET ADAPTATION IMAGES FROM FULL ARCHIVE IF NEEDED
# =========================================================

missing_adapt_names = [
    basename_any(x)
    for x in MB_ADAPT[mb_img]
    if resolve_from_index(x, "mbrset") is None
]

if missing_adapt_names:
    print(f"\nMissing mBRSET adaptation images: {len(missing_adapt_names)}")

    candidates = []
    for p in glob.glob(os.path.join(DRIVE_ROOT, "**", "*.zip"), recursive=True):
        name = os.path.basename(p).lower()
        if "mbrset" in name and "phase1a1_colab_bundle" not in name:
            candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "\nPHASE 1A2 DATA STOP:\n"
            "The locked 887-patient mBRSET ADAPTATION_POOL images are not present.\n"
            "Upload the full `mbrset_1.0.0.zip` (or an adaptation-only image ZIP) "
            f"somewhere under:\n{DRIVE_ROOT}\n"
            "Then rerun this script. No scientific analysis has started."
        )

    candidates.sort(key=os.path.getsize, reverse=True)
    zpath = candidates[0]

    print("Using mBRSET archive:", zpath)
    dest = "/content/phase1a2_mbrset_extra"
    os.makedirs(dest, exist_ok=True)

    needed = set(missing_adapt_names)

    with zipfile.ZipFile(zpath, "r") as z:
        member_map = {}
        for m in z.namelist():
            b = basename_any(m)
            if b in needed:
                member_map.setdefault(b, []).append(m)

        absent = sorted(needed - set(member_map))
        if absent:
            raise RuntimeError(
                f"{len(absent)} required adaptation images are absent from ZIP. "
                f"First examples: {absent[:10]}"
            )

        ambiguous = {k:v for k,v in member_map.items() if len(v) != 1}
        if ambiguous:
            raise RuntimeError(
                "Ambiguous duplicated filenames in mBRSET archive; manual review required."
            )

        print("Extracting only required adaptation images...")
        for i, (b, members) in enumerate(member_map.items(), 1):
            target = os.path.join(dest, b)
            if not os.path.exists(target):
                with z.open(members[0]) as src, open(target, "wb") as dst:
                    dst.write(src.read())

            if i % 500 == 0:
                print(f"  extracted {i}/{len(member_map)}")

    if dest not in SEARCH_ROOTS:
        SEARCH_ROOTS.append(dest)
    idx = image_index(SEARCH_ROOTS)

# Final image verification
for name, df, img_col, hint in [
    ("BR_TRAIN", BR_TRAIN, br_img, "brset"),
    ("BR_VAL", BR_VAL, br_img, "brset"),
    ("BR_TEST", BR_TEST, br_img, "brset"),
    ("MB_ADAPT", MB_ADAPT, mb_img, "mbrset"),
    ("MB_TEST", MB_TEST, mb_img, "mbrset"),
]:
    miss = [x for x in df[img_col] if resolve_from_index(x, hint) is None]
    print(name, "missing:", len(miss))
    if miss:
        raise RuntimeError(f"{name}: required images missing. STOP.")

# Add resolved runtime paths
BR_TRAIN["_runtime_path"] = BR_TRAIN[br_img].map(lambda x: resolve_from_index(x, "brset"))
BR_VAL["_runtime_path"] = BR_VAL[br_img].map(lambda x: resolve_from_index(x, "brset"))
BR_TEST["_runtime_path"] = BR_TEST[br_img].map(lambda x: resolve_from_index(x, "brset"))

MB_ADAPT["_runtime_path"] = MB_ADAPT[mb_img].map(lambda x: resolve_from_index(x, "mbrset"))
MB_TEST["_runtime_path"] = MB_TEST[mb_img].map(lambda x: resolve_from_index(x, "mbrset"))

# =========================================================
# SOURCE CHECKPOINT — seed 20260906 ONLY
# =========================================================

ckpts = []
for ext in ("*.pth", "*.pt"):
    ckpts.extend(
        glob.glob(
            os.path.join(DRIVE_ROOT, "results", "**", ext),
            recursive=True
        )
    )

seed_ckpts = [p for p in ckpts if "20260906" in p]

if not seed_ckpts:
    raise RuntimeError(
        "Cannot uniquely locate the Phase1A1 seed 20260906 source checkpoint."
    )

def ckpt_score(p):
    s = 0
    q = p.lower()
    if "best" in q:
        s += 10
    if "checkpoint" in q:
        s += 5
    if "phase1a1" in q:
        s += 3
    return s

seed_ckpts.sort(key=lambda p: (-ckpt_score(p), len(p)))

best_score = ckpt_score(seed_ckpts[0])
top = [p for p in seed_ckpts if ckpt_score(p) == best_score]

if len(top) > 1:
    print("\nCheckpoint candidates:")
    for p in top:
        print(" ", p)
    # Prefer most recently modified if multiple best-labelled candidates
    top.sort(key=os.path.getmtime, reverse=True)

SOURCE_CKPT = top[0]

print("\nSOURCE CHECKPOINT:", SOURCE_CKPT)
print("SHA256:", sha256_file(SOURCE_CKPT))

source_ckpt = torch.load(
    SOURCE_CKPT,
    map_location="cpu",
    weights_only=False
)

def extract_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            # maybe raw state dict
            state = ckpt
    else:
        state = ckpt

    state = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state.items()
        if torch.is_tensor(v)
    }

    if "head.weight" not in state and "model.head.weight" in state:
        state = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k,v in state.items()
        }

    return state

SOURCE_STATE = extract_state(source_ckpt)

if "head.weight" not in SOURCE_STATE:
    raise RuntimeError("Cannot identify classifier head in source checkpoint.")

N_CLASSES = int(SOURCE_STATE["head.weight"].shape[0])
print("Source classifier outputs:", N_CLASSES)

src_args = source_ckpt.get("args", None) if isinstance(source_ckpt, dict) else None

def getarg(name, default):
    if src_args is None:
        return default
    if isinstance(src_args, dict):
        return src_args.get(name, default)
    return getattr(src_args, name, default)

INPUT_SIZE = int(getarg("input_size", 224))
DROP_PATH = float(getarg("drop_path", 0.2))
GLOBAL_POOL = bool(getarg("global_pool", True))

LR = getarg("lr", None)
try:
    LR = float(LR) if LR is not None else FALLBACK_LR
except:
    LR = FALLBACK_LR

if LR <= 0 or LR > 0.01:
    LR = FALLBACK_LR

WEIGHT_DECAY = float(getarg("weight_decay", FALLBACK_WEIGHT_DECAY))
LAYER_DECAY = float(getarg("layer_decay", FALLBACK_LAYER_DECAY))

print("Adapt config:")
print(" input =", INPUT_SIZE)
print(" lr =", LR)
print(" wd =", WEIGHT_DECAY)
print(" layer_decay =", LAYER_DECAY)
print(" epochs =", ADAPT_EPOCHS)

# =========================================================
# TRANSFORMS — OFFICIAL RETFOUND PIPELINE
# =========================================================

tf_args = SimpleNamespace(
    input_size=INPUT_SIZE,
    color_jitter=getarg("color_jitter", None),
    aa=getarg("aa", "rand-m9-mstd0.5-inc1"),
    reprob=float(getarg("reprob", 0.25)),
    remode=getarg("remode", "pixel"),
    recount=int(getarg("recount", 1)),
)

TRAIN_TF = build_transform("train", tf_args)
EVAL_TF = build_transform("val", tf_args)

# =========================================================
# DATASETS
# =========================================================

class FrameDataset(Dataset):
    def __init__(self, df, image_col, y_col, transform):
        self.df = df.reset_index(drop=True).copy()
        self.image_col = image_col
        self.y_col = y_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["_runtime_path"]).convert("RGB")
        img = self.transform(img)
        y = int(row[self.y_col])
        return img, y, i

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fresh_model():
    model = models_vit.RETFound_mae(
        img_size=INPUT_SIZE,
        num_classes=N_CLASSES,
        drop_path_rate=DROP_PATH,
        global_pool=GLOBAL_POOL,
    )

    missing, unexpected = model.load_state_dict(SOURCE_STATE, strict=False)

    # Source checkpoint should already contain its classifier head
    bad_missing = [x for x in missing if "num_batches_tracked" not in x]
    if len(bad_missing) > 5 or len(unexpected) > 5:
        print("Load-state missing:", missing[:20])
        print("Load-state unexpected:", unexpected[:20])

    return model.to(DEVICE)

def probs_from_logits(logits):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.ndim == 1:
        logits = logits[:, None]

    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    elif logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    else:
        raise RuntimeError(f"Unexpected classifier output shape {logits.shape}")

def make_criterion(df, ycol):
    y = df[ycol].astype(int).values
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    if n0 == 0 or n1 == 0:
        raise RuntimeError("Adaptation image cohort has only one class.")

    if N_CLASSES == 1:
        pos_weight = torch.tensor([n0 / n1], dtype=torch.float32, device=DEVICE)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    weights = torch.tensor(
        [
            len(y) / (2 * n0),
            len(y) / (2 * n1),
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    return torch.nn.CrossEntropyLoss(weight=weights)

def train_fixed(model, df, ycol, train_seed):
    set_seed(train_seed)

    ds = FrameDataset(df, mb_img, ycol, TRAIN_TF)

    generator = torch.Generator()
    generator.manual_seed(train_seed)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )

    criterion = make_criterion(df, ycol)

    nwd = model.no_weight_decay() if hasattr(model, "no_weight_decay") else []
    param_groups = lrd.param_groups_lrd(
        model,
        weight_decay=WEIGHT_DECAY,
        no_weight_decay_list=nwd,
        layer_decay=LAYER_DECAY,
    )

    optimizer = torch.optim.AdamW(param_groups, lr=LR)

    history = []

    model.train()

    for epoch in range(ADAPT_EPOCHS):
        losses = []

        for xb, yb, _ in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                logits = model(xb)

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                if N_CLASSES == 1:
                    loss = criterion(
                        logits.view(-1),
                        yb.float()
                    )
                else:
                    loss = criterion(logits, yb.long())

            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        m = float(np.mean(losses))
        history.append(m)

        print(
            f"    epoch {epoch+1:02d}/{ADAPT_EPOCHS} "
            f"loss={m:.5f}",
            flush=True
        )

    return history

@torch.inference_mode()
def predict_df(model, df, imgcol, y_img_col, pid_col):
    ds = FrameDataset(df, imgcol, y_img_col, EVAL_TF)

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    out = []

    for xb, _, ii in loader:
        xb = xb.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            logits = model(xb)
            p = probs_from_logits(logits)

        p = p.float().cpu().numpy()

        for idx_local, prob in zip(ii.numpy(), p):
            r = ds.df.iloc[int(idx_local)]
            out.append({
                "patient_id": str(r[pid_col]),
                "y_patient": int(r[
                    br_y_pt if pid_col == br_pid else mb_y_pt
                ]),
                "y_image": int(r[y_img_col]),
                "p_image": float(prob),
            })

    image_pred = pd.DataFrame(out)

    # Locked clinical aggregation = maximum image/eye risk within patient.
    patient = (
        image_pred
        .groupby("patient_id", as_index=False)
        .agg(
            y_patient=("y_patient", "first"),
            p_raw=("p_image", "max"),
        )
    )

    return patient, image_pred

# =========================================================
# METRICS
# =========================================================

EPS = 1e-6

def logit_np(p):
    p = np.clip(np.asarray(p), EPS, 1-EPS)
    return np.log(p / (1-p))

def fit_platt(patient_df):
    X = logit_np(patient_df["p_raw"].values).reshape(-1, 1)
    y = patient_df["y_patient"].astype(int).values

    mod = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
    )
    mod.fit(X, y)
    return mod

def apply_platt(mod, p):
    X = logit_np(p).reshape(-1, 1)
    return mod.predict_proba(X)[:, 1]

def select_sens90_threshold(y, p, target=0.90):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)

    best = None

    for th in sorted(np.unique(p), reverse=True):
        pred = (p >= th).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0,1]
        ).ravel()

        sens = tp / (tp + fn) if (tp+fn) else np.nan
        spec = tn / (tn + fp) if (tn+fp) else np.nan

        if sens >= target:
            candidate = (spec, th, sens)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return 0.0

    return float(best[1])

def ece_score(y, p, n_bins=10):
    y = np.asarray(y)
    p = np.asarray(p)

    edges = np.linspace(0, 1, n_bins+1)
    ece = 0.0

    for i in range(n_bins):
        if i < n_bins-1:
            mask = (p >= edges[i]) & (p < edges[i+1])
        else:
            mask = (p >= edges[i]) & (p <= edges[i+1])

        if mask.sum() == 0:
            continue

        ece += (
            mask.mean()
            * abs(y[mask].mean() - p[mask].mean())
        )

    return float(ece)

def calibration_intercept_slope(y, p):
    X = logit_np(p).reshape(-1,1)

    try:
        m = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
        )
        m.fit(X, np.asarray(y).astype(int))
        return float(m.intercept_[0]), float(m.coef_[0][0])
    except Exception:
        return np.nan, np.nan

def metric_row(patient_df, p_col, threshold):
    y = patient_df["y_patient"].astype(int).values
    p = np.clip(patient_df[p_col].values, EPS, 1-EPS)

    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0,1]
    ).ravel()

    sens = tp / (tp + fn) if (tp+fn) else np.nan
    spec = tn / (tn + fp) if (tn+fp) else np.nan
    ppv  = tp / (tp + fp) if (tp+fp) else np.nan
    npv  = tn / (tn + fn) if (tn+fn) else np.nan

    ci, cs = calibration_intercept_slope(y, p)

    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "nll": log_loss(y, p, labels=[0,1]),
        "ece": ece_score(y, p),
        "calibration_intercept": ci,
        "calibration_slope": cs,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "fnr": 1-sens,
        "fpr": 1-spec,
    }

# =========================================================
# ZERO-LABEL SOURCE MODEL + SOURCE-ONLY CALIBRATION
# =========================================================

print("\n=== Reconstructing Phase1A1 seed-20260906 baseline ===")

source_model = fresh_model()

val_pt, _ = predict_df(
    source_model, BR_VAL, br_img, br_y_img, br_pid
)

source_test_pt, _ = predict_df(
    source_model, BR_TEST, br_img, br_y_img, br_pid
)

target0_pt, _ = predict_df(
    source_model, MB_TEST, mb_img, mb_y_img, mb_pid
)

PLATT = fit_platt(val_pt)

val_pt["p_cal"] = apply_platt(PLATT, val_pt["p_raw"].values)
source_test_pt["p_cal"] = apply_platt(
    PLATT, source_test_pt["p_raw"].values
)
target0_pt["p_cal"] = apply_platt(
    PLATT, target0_pt["p_raw"].values
)

FIXED_THRESHOLD = select_sens90_threshold(
    val_pt["y_patient"],
    val_pt["p_cal"],
    target=0.90,
)

print("Fixed source-validation threshold:", FIXED_THRESHOLD)

BASE_SOURCE = metric_row(
    source_test_pt,
    "p_cal",
    FIXED_THRESHOLD
)

BASE_TARGET = metric_row(
    target0_pt,
    "p_cal",
    FIXED_THRESHOLD
)

print("Seed 20260906 source AUROC:", BASE_SOURCE["auroc"])
print("Seed 20260906 target AUROC:", BASE_TARGET["auroc"])
print("Seed 20260906 target sensitivity:", BASE_TARGET["sensitivity"])

# Safety comparison to Phase1A1 recorded seed-level metrics if available
phase1_metrics = find_files("03_seed_level_metrics.csv")
if phase1_metrics:
    try:
        old = pd.read_csv(phase1_metrics[0])
        seed_col = choose_col(old, ["seed"])
        row = old[old[seed_col].astype(int) == 20260906]

        possible = [
            c for c in old.columns
            if "target" in c.lower()
            and "auroc" in c.lower()
        ]

        if len(row) and possible:
            old_auc = float(row.iloc[0][possible[0]])
            diff = abs(old_auc - BASE_TARGET["auroc"])
            print("Recorded Phase1A1 seed target AUROC:", old_auc)
            print("Re-inference difference:", diff)

            if diff > 0.015:
                raise RuntimeError(
                    "Baseline re-inference does not reproduce Phase1A1 closely enough. "
                    "STOP before target adaptation."
                )
    except RuntimeError:
        raise
    except Exception as e:
        print("Phase1A1 metric comparison skipped:", e)

# Persist baseline
pd.DataFrame([{
    "run_id": "N0_SOURCE_MODEL",
    "budget": 0,
    **{f"target_{k}":v for k,v in BASE_TARGET.items()},
    **{f"source_{k}":v for k,v in BASE_SOURCE.items()},
}]).to_csv(
    os.path.join(OUT, "00_zero_label_seed20260906.csv"),
    index=False
)

# =========================================================
# NESTED PATIENT SAMPLING
# =========================================================

pt_table = (
    MB_ADAPT[[mb_pid, mb_y_pt]]
    .drop_duplicates(subset=[mb_pid])
    .copy()
)

pt_table[mb_pid] = pt_table[mb_pid].astype(str)
pt_table[mb_y_pt] = pt_table[mb_y_pt].astype(int)

positive = pt_table.loc[
    pt_table[mb_y_pt] == 1, mb_pid
].tolist()

negative = pt_table.loc[
    pt_table[mb_y_pt] == 0, mb_pid
].tolist()

prevalence = len(positive) / len(pt_table)

sampling_rows = []

trajectory_sets = {}

for seed in TRAJECTORY_SEEDS:
    rng = np.random.default_rng(seed)

    pos = np.array(positive, dtype=object)
    neg = np.array(negative, dtype=object)

    rng.shuffle(pos)
    rng.shuffle(neg)

    trajectory_sets[seed] = {}

    for n in BUDGETS:
        n_pos = int(round(n * prevalence))
        n_pos = max(1, min(n_pos, len(pos)))
        n_neg = n - n_pos

        ids = list(pos[:n_pos]) + list(neg[:n_neg])
        ids = set(map(str, ids))

        trajectory_sets[seed][n] = ids

        for pid in sorted(ids):
            sampling_rows.append({
                "trajectory_seed": seed,
                "budget": n,
                "patient_id": pid,
                "referable_dr_patient":
                    int(pt_table.loc[
                        pt_table[mb_pid].astype(str) == pid,
                        mb_y_pt
                    ].iloc[0]),
            })

    assert trajectory_sets[seed][25].issubset(
        trajectory_sets[seed][100]
    )
    assert trajectory_sets[seed][100].issubset(
        trajectory_sets[seed][500]
    )

sampling_df = pd.DataFrame(sampling_rows)
sampling_df.to_csv(
    os.path.join(OUT, "01_sampling_trajectories.csv"),
    index=False
)

print("\nNested trajectories frozen.")

# =========================================================
# RUN/RESUME TABLE
# =========================================================

METRICS_PATH = os.path.join(
    OUT, "04_target_metrics_all_runs.csv"
)

SOURCE_PATH = os.path.join(
    OUT, "05_source_retention_all_runs.csv"
)

if os.path.exists(METRICS_PATH):
    completed_target = pd.read_csv(METRICS_PATH)
else:
    completed_target = pd.DataFrame()

if os.path.exists(SOURCE_PATH):
    completed_source = pd.read_csv(SOURCE_PATH)
else:
    completed_source = pd.DataFrame()

def is_completed(run_id):
    return (
        len(completed_target)
        and "run_id" in completed_target.columns
        and run_id in set(completed_target["run_id"].astype(str))
    )

def append_csv(path, row):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False
    )

# =========================================================
# ADAPTATION RUN FUNCTION
# =========================================================

run_records = []

def execute_adaptation(
    run_id,
    cohort_df,
    train_seed,
    budget,
    trajectory_seed,
    oracle=False,
):
    global completed_target, completed_source

    if is_completed(run_id):
        print("\nSKIP completed:", run_id)
        return

    print("\n" + "="*72)
    print("RUN:", run_id)
    print("patients:", cohort_df[mb_pid].nunique())
    print("images:", len(cohort_df))
    print("="*72)

    t0 = time.time()

    model = fresh_model()

    history = train_fixed(
        model,
        cohort_df,
        mb_y_img,
        train_seed=train_seed,
    )

    # Frozen target test
    target_pt, _ = predict_df(
        model,
        MB_TEST,
        mb_img,
        mb_y_img,
        mb_pid,
    )
    target_pt["p_cal"] = apply_platt(
        PLATT,
        target_pt["p_raw"].values
    )

    # Frozen source test: backward compatibility
    source_pt, _ = predict_df(
        model,
        BR_TEST,
        br_img,
        br_y_img,
        br_pid,
    )
    source_pt["p_cal"] = apply_platt(
        PLATT,
        source_pt["p_raw"].values
    )

    mt_raw = metric_row(
        target_pt,
        "p_raw",
        threshold=FIXED_THRESHOLD
    )
    mt_cal = metric_row(
        target_pt,
        "p_cal",
        threshold=FIXED_THRESHOLD
    )

    ms_raw = metric_row(
        source_pt,
        "p_raw",
        threshold=FIXED_THRESHOLD
    )
    ms_cal = metric_row(
        source_pt,
        "p_cal",
        threshold=FIXED_THRESHOLD
    )

    elapsed = time.time() - t0

    target_row = {
        "run_id": run_id,
        "trajectory_seed": trajectory_seed,
        "training_seed": train_seed,
        "budget": budget,
        "oracle": int(oracle),
        "patients": cohort_df[mb_pid].nunique(),
        "images": len(cohort_df),
        "positive_patients":
            cohort_df[[mb_pid,mb_y_pt]]
            .drop_duplicates(mb_pid)[mb_y_pt].sum(),
        "fixed_threshold": FIXED_THRESHOLD,
        "train_loss_final": history[-1],
        "runtime_min": elapsed/60,
        **{f"raw_{k}":v for k,v in mt_raw.items()},
        **{f"sourcecal_{k}":v for k,v in mt_cal.items()},
    }

    source_row = {
        "run_id": run_id,
        "trajectory_seed": trajectory_seed,
        "training_seed": train_seed,
        "budget": budget,
        "oracle": int(oracle),
        "patients": cohort_df[mb_pid].nunique(),
        "images": len(cohort_df),
        "source_AUROC_change":
            ms_cal["auroc"] - BASE_SOURCE["auroc"],
        **{f"raw_{k}":v for k,v in ms_raw.items()},
        **{f"sourcecal_{k}":v for k,v in ms_cal.items()},
    }

    append_csv(METRICS_PATH, target_row)
    append_csv(SOURCE_PATH, source_row)

    # Optional pilot checkpoint persistence
    if SAVE_ADAPTED_CHECKPOINTS:
        ckdir = os.path.join(OUT, "checkpoints")
        os.makedirs(ckdir, exist_ok=True)

        torch.save({
            "model": model.state_dict(),
            "run_id": run_id,
            "budget": budget,
            "trajectory_seed": trajectory_seed,
            "epochs": ADAPT_EPOCHS,
            "lr": LR,
        }, os.path.join(ckdir, run_id + ".pth"))

    del model
    torch.cuda.empty_cache()

    completed_target = pd.read_csv(METRICS_PATH)
    completed_source = pd.read_csv(SOURCE_PATH)

    print(
        f"  target AUROC={mt_cal['auroc']:.4f} "
        f"sens={mt_cal['sensitivity']:.4f} "
        f"Brier={mt_cal['brier']:.4f}"
    )
    print(
        f"  source AUROC={ms_cal['auroc']:.4f} "
        f"change={source_row['source_AUROC_change']:+.4f}"
    )
    print(f"  runtime={elapsed/60:.1f} min")

# =========================================================
# FIVE NESTED TRAJECTORIES
# =========================================================

for traj_seed in TRAJECTORY_SEEDS:

    for budget in BUDGETS:

        ids = trajectory_sets[traj_seed][budget]

        df_run = MB_ADAPT[
            MB_ADAPT[mb_pid].astype(str).isin(ids)
        ].copy()

        assert df_run[mb_pid].nunique() == budget

        run_id = f"traj{traj_seed}_N{budget}"

        execute_adaptation(
            run_id=run_id,
            cohort_df=df_run,
            train_seed=traj_seed,
            budget=budget,
            trajectory_seed=traj_seed,
            oracle=False,
        )

# =========================================================
# 887-PATIENT LOCAL ORACLE
# =========================================================

execute_adaptation(
    run_id="TARGET_LOCAL_ORACLE_N887",
    cohort_df=MB_ADAPT.copy(),
    train_seed=20261999,
    budget=887,
    trajectory_seed=20261999,
    oracle=True,
)

# =========================================================
# SUMMARIES
# =========================================================

target_all = pd.read_csv(METRICS_PATH)
source_all = pd.read_csv(SOURCE_PATH)

oracle = target_all[
    target_all["run_id"] == "TARGET_LOCAL_ORACLE_N887"
].iloc[0]

pilot = target_all[
    target_all["budget"].isin(BUDGETS)
].copy()

AUC0 = BASE_TARGET["auroc"]
SENS0 = BASE_TARGET["sensitivity"]
BRIER0 = BASE_TARGET["brier"]

AUC_ORACLE = float(oracle["sourcecal_auroc"])
SENS_ORACLE = float(oracle["sourcecal_sensitivity"])
BRIER_ORACLE = float(oracle["sourcecal_brier"])

def safe_frac(num, den):
    if abs(den) < 1e-8:
        return np.nan
    return num / den

pilot["auroc_recovery_fraction"] = pilot["sourcecal_auroc"].map(
    lambda x: safe_frac(x-AUC0, AUC_ORACLE-AUC0)
)

pilot["sensitivity_recovery_fraction"] = pilot["sourcecal_sensitivity"].map(
    lambda x: safe_frac(x-SENS0, SENS_ORACLE-SENS0)
)

pilot["brier_recovery_fraction"] = pilot["sourcecal_brier"].map(
    lambda x: safe_frac(BRIER0-x, BRIER0-BRIER_ORACLE)
)

pilot.to_csv(
    os.path.join(OUT, "07_recovery_fraction.csv"),
    index=False
)

summary_rows = []

for n, g in pilot.groupby("budget"):
    for metric in [
        "sourcecal_auroc",
        "sourcecal_auprc",
        "sourcecal_brier",
        "sourcecal_sensitivity",
        "sourcecal_specificity",
        "auroc_recovery_fraction",
    ]:
        x = g[metric].dropna().astype(float)
        summary_rows.append({
            "budget": n,
            "metric": metric,
            "mean": x.mean(),
            "median": x.median(),
            "sd": x.std(ddof=1),
            "min": x.min(),
            "max": x.max(),
            "q25": x.quantile(.25),
            "q75": x.quantile(.75),
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    os.path.join(OUT, "08_dose_response_summary.csv"),
    index=False
)

# =========================================================
# COHERENCE / GATE
# =========================================================

coherent = {}

for seed in TRAJECTORY_SEEDS:
    g = (
        pilot[pilot["trajectory_seed"] == seed]
        .set_index("budget")
    )

    a25 = float(g.loc[25, "sourcecal_auroc"])
    a100 = float(g.loc[100, "sourcecal_auroc"])
    a500 = float(g.loc[500, "sourcecal_auroc"])

    # monotonicity with small sampling/training tolerance
    coherent[seed] = bool(
        (a25 >= AUC0 - 0.01)
        and (a100 >= a25 - 0.01)
        and (a500 >= a100 - 0.01)
        and (a500 > AUC0)
    )

n_coherent = sum(coherent.values())

n500 = pilot[pilot["budget"] == 500]

med_auc500 = float(n500["sourcecal_auroc"].median())
med_sens500 = float(n500["sourcecal_sensitivity"].median())

auc_gain500 = med_auc500 - AUC0
sens_gain500 = med_sens500 - SENS0

source500 = source_all[source_all["budget"] == 500]
worst_source_loss = float(
    source500["source_AUROC_change"].min()
)

severe_forgetting_runs = int(
    (source500["source_AUROC_change"] < -0.05).sum()
)

if (
    n_coherent >= 3
    and (
        auc_gain500 >= 0.03
        or sens_gain500 >= 0.10
    )
    and severe_forgetting_runs <= 2
):
    VERDICT = "STRONG PASS — EXPAND TO FORMAL MULTI-METHOD DOSE STUDY"
    EYE_FLOOR = "PASS"

elif (
    auc_gain500 > 0
    or sens_gain500 > 0
):
    VERDICT = "BORDERLINE PASS — HUMAN REVIEW"
    EYE_FLOOR = "BORDERLINE"

else:
    VERDICT = "FAIL — LABEL-DOSE HYPOTHESIS NOT SUPPORTED"
    EYE_FLOOR = "FAIL"

# =========================================================
# FIGURES
# =========================================================

figdir = os.path.join(OUT, "figures")
os.makedirs(figdir, exist_ok=True)

def plot_trajectories(metric, ylabel, filename, baseline):
    plt.figure(figsize=(7,5))

    for seed in TRAJECTORY_SEEDS:
        g = (
            pilot[pilot["trajectory_seed"] == seed]
            .sort_values("budget")
        )

        x = [0] + g["budget"].tolist()
        y = [baseline] + g[metric].tolist()

        plt.plot(x, y, marker="o", alpha=.8)

    plt.xlabel("Locally labelled patients")
    plt.ylabel(ylabel)
    plt.title(ylabel + " dose-response")
    plt.grid(alpha=.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(figdir, filename + ".png"),
        dpi=300
    )
    plt.savefig(
        os.path.join(figdir, filename + ".pdf")
    )
    plt.close()

plot_trajectories(
    "sourcecal_auroc",
    "Patient-level AUROC",
    "figure_A_AUROC_dose",
    AUC0,
)

plot_trajectories(
    "sourcecal_sensitivity",
    "Fixed-threshold sensitivity",
    "figure_B_sensitivity_dose",
    SENS0,
)

plot_trajectories(
    "sourcecal_brier",
    "Brier score",
    "figure_C_brier_dose",
    BRIER0,
)

plt.figure(figsize=(7,5))

merge = pilot.merge(
    source_all[[
        "run_id",
        "source_AUROC_change"
    ]],
    on="run_id",
    how="left"
)

plt.scatter(
    merge["source_AUROC_change"],
    merge["sourcecal_auroc"] - AUC0,
)

plt.axvline(-0.02, linestyle="--")
plt.axvline(-0.05, linestyle="--")
plt.axhline(0, linestyle="--")

plt.xlabel("Source AUROC change after adaptation")
plt.ylabel("Target AUROC gain")
plt.title("Target recovery vs source retention")
plt.grid(alpha=.2)
plt.tight_layout()

plt.savefig(
    os.path.join(
        figdir,
        "figure_D_target_vs_source_tradeoff.png"
    ),
    dpi=300
)

plt.savefig(
    os.path.join(
        figdir,
        "figure_D_target_vs_source_tradeoff.pdf"
    )
)
plt.close()

plt.figure(figsize=(7,5))

for seed in TRAJECTORY_SEEDS:
    g = pilot[
        pilot["trajectory_seed"] == seed
    ].sort_values("budget")

    plt.plot(
        g["budget"],
        100*g["auroc_recovery_fraction"],
        marker="o"
    )

plt.axhline(100, linestyle="--")
plt.xlabel("Locally labelled patients")
plt.ylabel("AUROC recovery (%)")
plt.title("Recoverable AUROC restored")
plt.grid(alpha=.2)
plt.tight_layout()

plt.savefig(
    os.path.join(
        figdir,
        "figure_E_AUROC_recovery_fraction.png"
    ),
    dpi=300
)
plt.savefig(
    os.path.join(
        figdir,
        "figure_E_AUROC_recovery_fraction.pdf"
    )
)
plt.close()

# =========================================================
# PROTOCOL / RUN MANIFEST
# =========================================================

protocol = f"""# Phase 1A2 Locked Protocol

Source checkpoint: `{SOURCE_CKPT}`

Source checkpoint SHA256:
`{sha256_file(SOURCE_CKPT)}`

Training unit: IMAGE

Sampling / dose unit: PATIENT

Primary evaluation unit: PATIENT

Source checkpoint seed: 20260906

Trajectory seeds:
{TRAJECTORY_SEEDS}

Budgets:
{BUDGETS}

Local oracle:
887 patients

Nested rule:
25 subset 100 subset 500

Full fine-tuning epochs:
{ADAPT_EPOCHS}

Batch size:
{BATCH_SIZE}

Learning rate:
{LR}

Weight decay:
{WEIGHT_DECAY}

Layer decay:
{LAYER_DECAY}

Target validation:
NONE

Frozen target test used for training/model selection:
NO

Target-specific calibration:
NO

Primary clinical aggregation:
maximum predicted risk within patient.
"""

with open(
    os.path.join(OUT, "00_protocol.md"),
    "w",
    encoding="utf-8"
) as f:
    f.write(protocol)

run_manifest = target_all[
    [
        "run_id",
        "trajectory_seed",
        "training_seed",
        "budget",
        "oracle",
        "patients",
        "images",
        "runtime_min",
    ]
].copy()

run_manifest.to_csv(
    os.path.join(OUT, "03_run_manifest.csv"),
    index=False
)

# =========================================================
# REPORT
# =========================================================

def med(n, col):
    return float(
        pilot[pilot["budget"] == n][col].median()
    )

def recovery_pct(n):
    return 100 * med(n, "auroc_recovery_fraction")

report = f"""# PHASE 1A2 GATE REPORT

## Status

**{VERDICT}**

## A. Locked Phase 1A1 baseline

Source checkpoint seed: `20260906`

Zero-label target patient AUROC: **{AUC0:.4f}**

Zero-label fixed-threshold sensitivity: **{SENS0:.4f}**

Zero-label Brier score: **{BRIER0:.4f}**

## B. Label-dose design

Five independent nested patient-level trajectories were used.

Within every trajectory:

`N25 ⊂ N100 ⊂ N500`

The frozen 400-patient mBRSET target test was never used for training,
model selection, calibration fitting, or threshold selection.

## C. Target AUROC dose response

N=25 median AUROC: **{med(25, "sourcecal_auroc"):.4f}**

N=100 median AUROC: **{med(100, "sourcecal_auroc"):.4f}**

N=500 median AUROC: **{med(500, "sourcecal_auroc"):.4f}**

887-patient local-oracle AUROC: **{AUC_ORACLE:.4f}**

## D. AUROC recovery

N=25 median recovery: **{recovery_pct(25):.1f}%**

N=100 median recovery: **{recovery_pct(100):.1f}%**

N=500 median recovery: **{recovery_pct(500):.1f}%**

## E. Clinical screening sensitivity

Zero-label sensitivity: **{SENS0:.4f}**

N=25 median sensitivity: **{med(25, "sourcecal_sensitivity"):.4f}**

N=100 median sensitivity: **{med(100, "sourcecal_sensitivity"):.4f}**

N=500 median sensitivity: **{med(500, "sourcecal_sensitivity"):.4f}**

Oracle sensitivity: **{SENS_ORACLE:.4f}**

## F. Calibration

Zero-label Brier: **{BRIER0:.4f}**

N=25 median Brier: **{med(25, "sourcecal_brier"):.4f}**

N=100 median Brier: **{med(100, "sourcecal_brier"):.4f}**

N=500 median Brier: **{med(500, "sourcecal_brier"):.4f}**

Oracle Brier: **{BRIER_ORACLE:.4f}**

## G. Backward compatibility

Worst N=500 observed BRSET source AUROC change:

**{worst_source_loss:+.4f}**

N=500 severe-forgetting runs (>0.05 AUROC loss):

**{severe_forgetting_runs}/5**

## H. Sampling stability

Trajectories satisfying coherent positive dose pattern:

**{n_coherent}/5**

Details:

`{json.dumps(coherent)}`

## I. Phase 1A2 verdict

**{VERDICT}**

## Eye and Vision Floor Gate

**{EYE_FLOOR}**

This is a Phase 1A2 pilot gate. VisionFM, DINOv3, PEFT/LoRA,
MESSIDOR-2, and DeepDRiD have not been started.
"""

REPORT_PATH = os.path.join(
    OUT,
    "PHASE_1A2_GATE_REPORT.md"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# =========================================================
# FINAL CONSOLE SUMMARY
# =========================================================

print("\n\n" + "="*80)
print("PHASE 1A2 FINAL SUMMARY")
print("="*80)

print(f"1. N=0 target AUROC: {AUC0:.4f}")
print(f"2. N=25 median target AUROC: {med(25,'sourcecal_auroc'):.4f}")
print(f"3. N=100 median target AUROC: {med(100,'sourcecal_auroc'):.4f}")
print(f"4. N=500 median target AUROC: {med(500,'sourcecal_auroc'):.4f}")
print(f"5. Oracle target AUROC: {AUC_ORACLE:.4f}")
print(f"6. N=25 median AUROC recovery: {recovery_pct(25):.1f}%")
print(f"7. N=100 median AUROC recovery: {recovery_pct(100):.1f}%")
print(f"8. N=500 median AUROC recovery: {recovery_pct(500):.1f}%")
print(f"9. N=500 median fixed-threshold sensitivity: {med(500,'sourcecal_sensitivity'):.4f}")
print(f"10. Worst observed source AUROC loss: {worst_source_loss:+.4f}")
print(f"11. Coherent positive trajectories: {n_coherent}/5")
print(f"12. PHASE 1A2 VERDICT: {VERDICT}")
print(f"13. EYE AND VISION FLOOR: {EYE_FLOOR}")
print("REPORT:", REPORT_PATH)

print("\nSTOP. Do not start another model or domain.")
