
# ============================================================
# PHASE 2A2
# VisionFM × mBRSET 100-label representation-adaptation rescue
#
# Prespecified comparison:
#   - Existing no adaptation
#   - Existing recalibration-only
#   - LoRA qkv adaptation
#   - Aggressive full fine-tuning
#
# IMPORTANT:
#   * Same locked 100-patient mBRSET cohort
#   * No mBRSET test-based model selection
#   * Fixed 10 epochs
#   * mBRSET test evaluated once after training
#   * BRSET test simultaneously evaluates source forgetting
# ============================================================

import os
import sys
import gc
import re
import json
import copy
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ============================================================
# 0. CONFIG
# ============================================================

ROOT = Path(
    "/content/vfm_rescue_drive/MyDrive/"
    "Retinal_Retraining_Phase1A1"
)

assert ROOT.exists(), ROOT

VFM_REPO = Path(
    "/content/VisionFM"
)

VFM_WEIGHTS = Path(
    "/content/VFM_Fundus_weights.pth"
)

OUT = (
    ROOT
    / "results"
    / "phase2a2_visionfm_mbrset_rescue"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PRED_DIR = OUT / "predictions"
PRED_DIR.mkdir(
    exist_ok=True
)

SEEDS = [
    20267001,
    20267002,
    20267003,
]

METHODS = [
    "lora",
    "full_ft",
]

EPOCHS = 10

TRAIN_BATCH = 8
EVAL_BATCH = 32

FULL_LR = 1e-4
FULL_WD = 0.05

LORA_LR = 3e-4
LORA_WD = 0.01

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

N_BOOT = 2000

DEVICE = "cuda"

assert torch.cuda.is_available()

print("=" * 80)
print("PHASE 2A2")
print("VisionFM × mBRSET 100-label representation rescue")
print("=" * 80)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

# ============================================================
# 1. IMPORT VISIONFM
# ============================================================

sys.path.insert(
    0,
    str(VFM_REPO)
)

import models
import utils

# ============================================================
# 2. FILES
# ============================================================

BR_MAN = (
    ROOT
    / "data"
    / "locked_metadata"
    / "BRSET_master_manifest_v2.csv"
)

MB_MAN = (
    ROOT
    / "data"
    / "locked_metadata"
    / "mBRSET_master_manifest_v2.csv"
)

SAMPLING = (
    ROOT
    / "results"
    / "phase1a2"
    / "01_sampling_trajectories.csv"
)

MB_ZIP = (
    ROOT
    / "data"
    / "mbrset_1.0.0.zip"
)

bundle_hits = list(
    ROOT.rglob(
        "phase1a1_colab_bundle.zip"
    )
)

assert bundle_hits

BR_ZIP = bundle_hits[0]

PHASE2A1 = (
    ROOT
    / "results"
    / "phase2a1_visionfm"
    / "01_seed_level_visionfm_metrics.csv"
)

HEAD_DIR = (
    ROOT
    / "results"
    / "phase2a1_visionfm"
    / "decoder_checkpoints"
)

# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# 4. COLUMN HELPERS
# ============================================================

def choose_col(df, names):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for x in names:

        if x.lower() in lower:
            return lower[x.lower()]

    raise KeyError(
        f"Missing {names}; "
        f"available={list(df.columns)}"
    )


def basename_any(x):

    return os.path.basename(
        str(x).replace("\\", "/")
    )


# ============================================================
# 5. MANIFESTS
# ============================================================

br = pd.read_csv(
    BR_MAN,
    low_memory=False
)

mb = pd.read_csv(
    MB_MAN,
    low_memory=False
)

sampling = pd.read_csv(
    SAMPLING
)

br_pid = choose_col(
    br,
    ["patient_id"]
)

br_img = choose_col(
    br,
    [
        "image_path",
        "image_id",
        "filename",
        "file",
    ]
)

br_yi = choose_col(
    br,
    ["referable_dr_image"]
)

br_yp = choose_col(
    br,
    ["referable_dr_patient"]
)

br_split = choose_col(
    br,
    ["split"]
)

mb_pid = choose_col(
    mb,
    ["patient_id"]
)

mb_img = choose_col(
    mb,
    [
        "image_path",
        "image_id",
        "filename",
        "file",
    ]
)

mb_yi = choose_col(
    mb,
    ["referable_dr_image"]
)

mb_yp = choose_col(
    mb,
    ["referable_dr_patient"]
)

mb_split = choose_col(
    mb,
    ["split"]
)

br = br[
    br[br_yi].notna()
    &
    br[br_yp].notna()
].copy()

mb = mb[
    mb[mb_yi].notna()
    &
    mb[mb_yp].notna()
].copy()

br["_split"] = (
    br[br_split]
    .astype(str)
    .str.upper()
)

mb["_split"] = (
    mb[mb_split]
    .astype(str)
    .str.upper()
)

# ============================================================
# 6. LOCK BRSET TEST
# ============================================================

br_test = (
    br[
        br["_split"]
        == "FROZEN_SOURCE_TEST"
    ]
    .copy()
)

assert (
    br_test[br_pid]
    .nunique()
    == 202
)

# ============================================================
# 7. LOCK mBRSET TEST
# ============================================================

mb_test = (
    mb[
        mb["_split"]
        == "FROZEN_TARGET_TEST"
    ]
    .copy()
)

assert (
    mb_test[mb_pid]
    .nunique()
    == 400
)

# ============================================================
# 8. LOCK SAME PRESPECIFIED mBRSET N=100 COHORT
# ============================================================

s_pid = choose_col(
    sampling,
    ["patient_id"]
)

s_seed = choose_col(
    sampling,
    ["trajectory_seed"]
)

s_budget = choose_col(
    sampling,
    ["budget"]
)

local100_ids = set(
    sampling[
        (
            sampling[s_seed]
            .astype(int)
            == 20261001
        )
        &
        (
            sampling[s_budget]
            .astype(int)
            == 100
        )
    ][s_pid]
    .astype(str)
)

assert len(
    local100_ids
) == 100

mb_local100 = (
    mb[
        (
            mb["_split"]
            == "ADAPTATION_POOL"
        )
        &
        (
            mb[mb_pid]
            .astype(str)
            .isin(local100_ids)
        )
    ]
    .copy()
)

assert (
    mb_local100[mb_pid]
    .nunique()
    == 100
)

print("\nLOCKED COHORTS")

print(
    "BR_TEST:",
    br_test[br_pid].nunique(),
    "patients /",
    len(br_test),
    "images"
)

print(
    "MB_LOCAL100:",
    mb_local100[mb_pid].nunique(),
    "patients /",
    len(mb_local100),
    "images"
)

print(
    "MB_TEST:",
    mb_test[mb_pid].nunique(),
    "patients /",
    len(mb_test),
    "images"
)

# ============================================================
# 9. EXTRACT IMAGES TO LOCAL SSD
# ============================================================

LOCAL = Path(
    "/content/"
    "phase2a2_images"
)

if LOCAL.exists():
    raise FileExistsError("Candidate safety guard: inspect or relocate existing extraction directory before rerun")

LOCAL.mkdir(
    parents=True
)


def build_index(zpath):

    out = {}

    with zipfile.ZipFile(
        zpath,
        "r"
    ) as z:

        for x in z.namelist():

            if x.endswith("/"):
                continue

            b = Path(x).name

            out.setdefault(
                b,
                []
            ).append(x)

    return out


def extract_rows(
    zpath,
    df,
    image_col,
    dst,
    index,
):

    dst.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = []

    with zipfile.ZipFile(
        zpath,
        "r"
    ) as z:

        for value in df[
            image_col
        ]:

            b = basename_any(
                value
            )

            hits = index.get(
                b,
                []
            )

            if len(hits) != 1:

                raise RuntimeError(
                    f"{b}: "
                    f"{len(hits)} archive hits"
                )

            target = dst / b

            with z.open(
                hits[0]
            ) as src, \
                open(
                    target,
                    "wb"
                ) as out:

                shutil.copyfileobj(
                    src,
                    out
                )

            paths.append(
                str(target)
            )

    return paths


print("\nIndexing archives...")

br_index = build_index(
    BR_ZIP
)

mb_index = build_index(
    MB_ZIP
)

print("Extracting BR_TEST...")

br_test[
    "_runtime_path"
] = extract_rows(
    BR_ZIP,
    br_test,
    br_img,
    LOCAL / "BR_TEST",
    br_index,
)

print("Extracting MB_LOCAL100...")

mb_local100[
    "_runtime_path"
] = extract_rows(
    MB_ZIP,
    mb_local100,
    mb_img,
    LOCAL / "MB_LOCAL100",
    mb_index,
)

print("Extracting MB_TEST...")

mb_test[
    "_runtime_path"
] = extract_rows(
    MB_ZIP,
    mb_test,
    mb_img,
    LOCAL / "MB_TEST",
    mb_index,
)

print("✅ Image extraction complete")

# ============================================================
# 10. LOCKED VISIONFM FUNDUS PREPROCESSING
# ============================================================

mean, std = utils.get_stats(
    "Fundus"
)

transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean,
        std,
    ),
])

# ============================================================
# 11. DATASET
# ============================================================

class RetinalDataset(Dataset):

    def __init__(
        self,
        df,
        pid_col,
        image_col,
        yi_col,
        yp_col,
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
            .copy()
        )

        self.pid_col = pid_col
        self.image_col = image_col
        self.yi_col = yi_col
        self.yp_col = yp_col


    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        i,
    ):

        r = self.df.iloc[i]

        image = (
            Image.open(
                r["_runtime_path"]
            )
            .convert("RGB")
        )

        x = transform(
            image
        )

        return (
            x,
            torch.tensor(
                float(
                    r[self.yi_col]
                ),
                dtype=torch.float32,
            ),
            str(
                r[self.pid_col]
            ),
            int(
                r[self.yp_col]
            ),
            str(
                r[self.image_col]
            ),
        )


train_ds = RetinalDataset(
    mb_local100,
    mb_pid,
    mb_img,
    mb_yi,
    mb_yp,
)

br_test_ds = RetinalDataset(
    br_test,
    br_pid,
    br_img,
    br_yi,
    br_yp,
)

mb_test_ds = RetinalDataset(
    mb_test,
    mb_pid,
    mb_img,
    mb_yi,
    mb_yp,
)

# ============================================================
# 12. SOURCE DECODER — SAME AS PHASE 2A1
# ============================================================

class VisionFMClsHead(nn.Module):

    def __init__(
        self,
        embed_dim=3072,
    ):

        super().__init__()

        self.channel_bn = nn.BatchNorm1d(
            embed_dim,
            eps=1e-6,
            momentum=0.99,
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim // 2,
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                embed_dim // 2,
                embed_dim // 4,
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                embed_dim // 4,
                1,
            ),
        )


    def forward(self, x):

        x = self.channel_bn(x)

        return (
            self.classifier(x)
            .squeeze(-1)
        )


# ============================================================
# 13. FULL SOURCE MODEL
# ============================================================

class VisionFMClassifier(nn.Module):

    def __init__(
        self,
        encoder,
        head,
    ):

        super().__init__()

        self.encoder = encoder
        self.head = head


    def forward(self, x):

        blocks = (
            self.encoder
            .get_intermediate_layers(
                x,
                4
            )
        )

        feat = torch.cat(
            [
                b[:, 0]
                for b in blocks
            ],
            dim=-1,
        )

        return self.head(
            feat
        )


def build_source_model(seed):

    encoder = models.__dict__[
        "vit_base"
    ](
        img_size=[224],
        patch_size=16,
        num_classes=0,
        use_mean_pooling=False,
    )

    checkpoint = torch.load(
        VFM_WEIGHTS,
        map_location="cpu",
        weights_only=False,
    )

    assert "teacher" in checkpoint

    state = checkpoint[
        "teacher"
    ]

    state = {
        k
        .replace("module.", "")
        .replace("backbone.", ""):
        v

        for k,v in state.items()
    }

    missing, unexpected = (
        encoder.load_state_dict(
            state,
            strict=False,
        )
    )

    assert len(missing) == 0

    head = VisionFMClsHead(
        3072
    )

    head_file = (
        HEAD_DIR
        /
        f"VisionFM_BRSET_decoder_seed{seed}.pth"
    )

    h = torch.load(
        head_file,
        map_location="cpu",
        weights_only=False,
    )

    head.load_state_dict(
        h["state_dict"],
        strict=True,
    )

    return VisionFMClassifier(
        encoder,
        head,
    )


# ============================================================
# 14. LoRA
# ============================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        base,
        r=8,
        alpha=16,
        dropout=0.05,
    ):

        super().__init__()

        assert isinstance(
            base,
            nn.Linear
        )

        self.base = base

        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r

        self.scaling = (
            alpha / r
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.lora_A = nn.Parameter(
            torch.empty(
                r,
                base.in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                r,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=np.sqrt(5),
        )


    def forward(self, x):

        base_out = self.base(
            x
        )

        z = self.dropout(
            x
        )

        z = torch.nn.functional.linear(
            z,
            self.lora_A,
        )

        z = torch.nn.functional.linear(
            z,
            self.lora_B,
        )

        return (
            base_out
            +
            self.scaling * z
        )


def configure_lora(model):

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # LoRA over qkv in all transformer blocks
    count = 0

    assert hasattr(
        model.encoder,
        "blocks"
    )

    for block in model.encoder.blocks:

        old = block.attn.qkv

        assert isinstance(
            old,
            nn.Linear
        )

        block.attn.qkv = LoRALinear(
            old,
            r=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
        )

        count += 1

    # Keep downstream classification head trainable
    for p in model.head.parameters():
        p.requires_grad = True

    assert count == 12, (
        f"Expected 12 qkv LoRA modules, got {count}"
    )

    return count


def configure_full_ft(model):

    for p in model.parameters():
        p.requires_grad = True


# ============================================================
# 15. PARAMETER COUNTS
# ============================================================

def param_counts(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        total,
        trainable,
    )


# ============================================================
# 16. PATIENT-LEVEL AGGREGATION
# ============================================================

def patient_aggregate(
    pid,
    yp,
    p,
):

    df = pd.DataFrame({
        "patient_id":
            np.asarray(pid).astype(str),

        "y":
            np.asarray(yp).astype(int),

        "p":
            np.asarray(p),
    })

    out = (
        df
        .groupby(
            "patient_id",
            as_index=False,
        )
        .agg(
            y=("y", "first"),
            p=("p", "max"),
        )
    )

    return out


# ============================================================
# 17. PREDICTION
# ============================================================

@torch.inference_mode()
def predict_patient(
    model,
    dataset,
):

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model.eval()

    all_pid = []
    all_yp = []
    all_p = []

    for (
        x,
        _,
        pid,
        yp,
        _,
    ) in loader:

        x = x.cuda(
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            logits = model(
                x
            )

        p = (
            torch.sigmoid(
                logits
            )
            .float()
            .cpu()
            .numpy()
        )

        all_pid.extend(
            list(pid)
        )

        all_yp.extend(
            yp.numpy().tolist()
        )

        all_p.extend(
            p.tolist()
        )

    return patient_aggregate(
        all_pid,
        all_yp,
        all_p,
    )


# ============================================================
# 18. METRICS
# ============================================================

def calc_metrics(df):

    y = df["y"].to_numpy()
    p = df["p"].to_numpy()

    return {
        "auroc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "brier_raw":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),
    }


# ============================================================
# 19. PAIRED STRATIFIED BOOTSTRAP AUROC DELTA
# ============================================================

def paired_auc_delta_ci(
    base_df,
    adapted_df,
    seed,
    n_boot=N_BOOT,
):

    a = (
        base_df
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    b = (
        adapted_df
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    assert np.array_equal(
        a["patient_id"],
        b["patient_id"],
    )

    assert np.array_equal(
        a["y"],
        b["y"],
    )

    y = a["y"].to_numpy()

    p0 = a["p"].to_numpy()
    p1 = b["p"].to_numpy()

    pos = np.where(
        y == 1
    )[0]

    neg = np.where(
        y == 0
    )[0]

    rng = np.random.default_rng(
        seed
    )

    deltas = []

    for _ in range(
        n_boot
    ):

        ip = rng.choice(
            pos,
            size=len(pos),
            replace=True,
        )

        ineg = rng.choice(
            neg,
            size=len(neg),
            replace=True,
        )

        idx = np.concatenate(
            [ip, ineg]
        )

        yy = y[idx]

        d = (
            roc_auc_score(
                yy,
                p1[idx],
            )
            -
            roc_auc_score(
                yy,
                p0[idx],
            )
        )

        deltas.append(
            d
        )

    return (
        float(
            np.quantile(
                deltas,
                0.025,
            )
        ),

        float(
            np.quantile(
                deltas,
                0.975,
            )
        ),
    )


# ============================================================
# 20. TRAINING
# ============================================================

def train_model(
    model,
    method,
    seed,
):

    set_seed(seed)

    model = model.cuda()

    if method == "lora":

        n_lora = configure_lora(
            model
        )

        lr = LORA_LR
        wd = LORA_WD

        print(
            "LoRA qkv modules:",
            n_lora
        )

    elif method == "full_ft":

        configure_full_ft(
            model
        )

        lr = FULL_LR
        wd = FULL_WD

    else:
        raise ValueError(method)

    total, trainable = (
        param_counts(
            model
        )
    )

    print(
        f"Total params: "
        f"{total:,}"
    )

    print(
        f"Trainable params: "
        f"{trainable:,} "
        f"({100*trainable/total:.3f}%)"
    )

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=lr,
        weight_decay=wd,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=0,
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    history = []

    # --------------------------------------------------------
    # Fixed 10 epochs.
    # NO target test evaluation during training.
    # NO target-based early stopping.
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        losses = []

        for (
            x,
            yi,
            _,
            _,
            _,
        ) in loader:

            x = x.cuda(
                non_blocking=True
            )

            yi = yi.cuda(
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                logits = model(
                    x
                )

                loss = criterion(
                    logits,
                    yi,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        scheduler.step()

        row = {
            "seed": seed,
            "method": method,
            "epoch": epoch,
            "loss":
                float(
                    np.mean(
                        losses
                    )
                ),
            "lr":
                float(
                    optimizer
                    .param_groups[0]["lr"]
                ),
        }

        history.append(
            row
        )

        print(
            f"{method} "
            f"seed={seed} "
            f"epoch={epoch:02d}/{EPOCHS} "
            f"loss={row['loss']:.5f} "
            f"lr={row['lr']:.2e}"
        )

    return (
        model,
        pd.DataFrame(history),
        total,
        trainable,
    )


# ============================================================
# 21. LOAD EXISTING PHASE 2A1 REFERENCE
# ============================================================

reference = pd.read_csv(
    PHASE2A1
)

reference[
    "seed"
] = reference[
    "seed"
].astype(int)

# ============================================================
# 22. MAIN EXPERIMENT
# ============================================================

all_metrics = []
all_predictions = []
all_history = []

for seed in SEEDS:

    print(
        "\n" + "="*80
    )

    print(
        "SOURCE DECODER SEED",
        seed
    )

    print(
        "="*80
    )

    # --------------------------------------------------------
    # Build untouched source model
    # --------------------------------------------------------

    source = build_source_model(
        seed
    ).cuda()

    source.eval()

    # --------------------------------------------------------
    # Baseline image-level inference from original images
    # --------------------------------------------------------

    print(
        "\nBaseline inference..."
    )

    br_base = predict_patient(
        source,
        br_test_ds,
    )

    mb_base = predict_patient(
        source,
        mb_test_ds,
    )

    br_base_m = calc_metrics(
        br_base
    )

    mb_base_m = calc_metrics(
        mb_base
    )

    expected = (
        reference[
            reference["seed"]
            == seed
        ]
        .iloc[0]
    )

    print(
        "BR baseline AUROC:",
        round(
            br_base_m["auroc"],
            6
        ),
        "| Phase2A1:",
        round(
            expected[
                "BR_TEST_AUROC"
            ],
            6
        )
    )

    print(
        "MB baseline AUROC:",
        round(
            mb_base_m["auroc"],
            6
        ),
        "| Phase2A1:",
        round(
            expected[
                "MB_AUROC"
            ],
            6
        )
    )

    # float16 feature cache may induce tiny differences;
    # anything >0.01 needs investigation.
    assert (
        abs(
            br_base_m["auroc"]
            -
            expected[
                "BR_TEST_AUROC"
            ]
        )
        <= 0.01
    ), "BRSET source reconstruction mismatch"

    assert (
        abs(
            mb_base_m["auroc"]
            -
            expected[
                "MB_AUROC"
            ]
        )
        <= 0.01
    ), "mBRSET baseline reconstruction mismatch"

    # save baseline rows once per seed
    for domain,df in [
        ("BRSET", br_base),
        ("mBRSET", mb_base),
    ]:

        t = df.copy()

        t["seed"] = seed
        t["method"] = "no_adaptation"
        t["domain"] = domain

        all_predictions.append(
            t
        )

    # --------------------------------------------------------
    # Delete source model before adaptation builds
    # --------------------------------------------------------

    del source

    gc.collect()
    torch.cuda.empty_cache()

    # ========================================================
    # TWO PRESPECIFIED ADAPTATION METHODS
    # ========================================================

    for method in METHODS:

        print(
            "\n" + "-"*80
        )

        print(
            "METHOD:",
            method,
            "| SOURCE SEED:",
            seed
        )

        print(
            "-"*80
        )

        model = build_source_model(
            seed
        )

        (
            model,
            history,
            total_params,
            trainable_params,
        ) = train_model(
            model,
            method,
            seed,
        )

        all_history.append(
            history
        )

        # ----------------------------------------------------
        # First and only frozen-test evaluation after training
        # ----------------------------------------------------

        print(
            "\nFrozen BRSET test..."
        )

        br_adapt = predict_patient(
            model,
            br_test_ds,
        )

        print(
            "Frozen mBRSET test..."
        )

        mb_adapt = predict_patient(
            model,
            mb_test_ds,
        )

        br_m = calc_metrics(
            br_adapt
        )

        mb_m = calc_metrics(
            mb_adapt
        )

        source_delta = (
            br_m["auroc"]
            -
            br_base_m["auroc"]
        )

        target_delta = (
            mb_m["auroc"]
            -
            mb_base_m["auroc"]
        )

        br_ci = paired_auc_delta_ci(
            br_base,
            br_adapt,
            seed=
                seed
                +
                (
                    1000
                    if method == "lora"
                    else 2000
                ),
        )

        mb_ci = paired_auc_delta_ci(
            mb_base,
            mb_adapt,
            seed=
                seed
                +
                (
                    3000
                    if method == "lora"
                    else 4000
                ),
        )

        print(
            "\nRESULT"
        )

        print(
            "BR AUROC:",
            f"{br_base_m['auroc']:.4f}",
            "→",
            f"{br_m['auroc']:.4f}",
            "| Δ",
            f"{source_delta:+.4f}",
            "| 95% CI",
            f"[{br_ci[0]:+.4f}, {br_ci[1]:+.4f}]"
        )

        print(
            "MB AUROC:",
            f"{mb_base_m['auroc']:.4f}",
            "→",
            f"{mb_m['auroc']:.4f}",
            "| Δ",
            f"{target_delta:+.4f}",
            "| 95% CI",
            f"[{mb_ci[0]:+.4f}, {mb_ci[1]:+.4f}]"
        )

        all_metrics.append({

            "seed":
                seed,

            "method":
                method,

            "epochs":
                EPOCHS,

            "train_patients":
                100,

            "train_images":
                len(
                    mb_local100
                ),

            "total_params":
                total_params,

            "trainable_params":
                trainable_params,

            "BR_baseline_AUROC":
                br_base_m["auroc"],

            "BR_adapted_AUROC":
                br_m["auroc"],

            "BR_AUROC_change":
                source_delta,

            "BR_AUROC_change_CI_low":
                br_ci[0],

            "BR_AUROC_change_CI_high":
                br_ci[1],

            "BR_adapted_AUPRC":
                br_m["auprc"],

            "MB_baseline_AUROC":
                mb_base_m["auroc"],

            "MB_adapted_AUROC":
                mb_m["auroc"],

            "MB_AUROC_gain":
                target_delta,

            "MB_AUROC_gain_CI_low":
                mb_ci[0],

            "MB_AUROC_gain_CI_high":
                mb_ci[1],

            "MB_adapted_AUPRC":
                mb_m["auprc"],

            # Existing independent recalibration-only reference
            "MB_recalibration_success_reference":
                bool(
                    expected[
                        "MB_recalibration_success"
                    ]
                ),

            "MB_recalibration_Brier_before":
                float(
                    expected[
                        "MB_sourcecal_Brier"
                    ]
                ),

            "MB_recalibration_Brier_after":
                float(
                    expected[
                        "MB_localcal_Brier"
                    ]
                ),
        })

        for domain,df in [
            ("BRSET", br_adapt),
            ("mBRSET", mb_adapt),
        ]:

            t = df.copy()

            t["seed"] = seed
            t["method"] = method
            t["domain"] = domain

            all_predictions.append(
                t
            )

        del model

        gc.collect()
        torch.cuda.empty_cache()

# ============================================================
# 23. SAVE RAW RESULTS
# ============================================================

metrics = pd.DataFrame(
    all_metrics
)

predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

history = pd.concat(
    all_history,
    ignore_index=True,
)

metrics.to_csv(
    OUT
    /
    "01_seed_method_metrics.csv",
    index=False,
)

predictions.to_csv(
    OUT
    /
    "02_patient_predictions.csv",
    index=False,
)

history.to_csv(
    OUT
    /
    "03_training_history.csv",
    index=False,
)

# ============================================================
# 24. METHOD SUMMARY
# ============================================================

summary = (
    metrics
    .groupby(
        "method"
    )
    .agg(

        median_target_gain=
            (
                "MB_AUROC_gain",
                "median",
            ),

        mean_target_gain=
            (
                "MB_AUROC_gain",
                "mean",
            ),

        min_target_gain=
            (
                "MB_AUROC_gain",
                "min",
            ),

        max_target_gain=
            (
                "MB_AUROC_gain",
                "max",
            ),

        target_gain_ge_001_count=
            (
                "MB_AUROC_gain",
                lambda x:
                    int(
                        (
                            x >= 0.01
                        ).sum()
                    ),
            ),

        median_source_change=
            (
                "BR_AUROC_change",
                "median",
            ),

        worst_source_change=
            (
                "BR_AUROC_change",
                "min",
            ),
    )
    .reset_index()
)

# ============================================================
# 25. PRESPECIFIED STRATEGY GATES
# ============================================================

def strategy_gate(method):

    r = summary[
        summary["method"]
        == method
    ].iloc[0]

    target_effective = (
        r[
            "median_target_gain"
        ] >= 0.01
        and
        r[
            "target_gain_ge_001_count"
        ] >= 2
    )

    source_safe = (
        r[
            "median_source_change"
        ] >= -0.02
        and
        r[
            "worst_source_change"
        ] >= -0.03
    )

    return {
        "target_effective":
            bool(
                target_effective
            ),

        "source_safe":
            bool(
                source_safe
            ),

        "strategy_pass":
            bool(
                target_effective
                and
                source_safe
            ),
    }


lora_gate = strategy_gate(
    "lora"
)

full_gate = strategy_gate(
    "full_ft"
)

full_rows = metrics[
    metrics["method"]
    == "full_ft"
]

FULL_FT_HARM_SIGNAL = bool(
    (
        (
            full_rows[
                "MB_AUROC_gain"
            ] < 0
        ).sum()
        >= 2
    )
    or
    (
        (
            full_rows[
                "BR_AUROC_change"
            ] < -0.03
        ).sum()
        >= 2
    )
)

# ============================================================
# 26. FINAL INTERPRETATION STATE
# ============================================================

if (
    lora_gate[
        "strategy_pass"
    ]
    and
    not full_gate[
        "strategy_pass"
    ]
):

    VERDICT = (
        "CONSERVATIVE REPRESENTATION-ADAPTATION "
        "RESCUE REPLICATED — PASS"
    )

elif (
    lora_gate[
        "strategy_pass"
    ]
    and
    full_gate[
        "strategy_pass"
    ]
):

    VERDICT = (
        "REPRESENTATION-ADAPTATION RESCUE "
        "REPLICATED — BOTH METHODS EFFECTIVE"
    )

elif (
    not lora_gate[
        "strategy_pass"
    ]
    and
    full_gate[
        "strategy_pass"
    ]
):

    VERDICT = (
        "REPRESENTATION ADAPTATION BENEFIT PRESENT, "
        "BUT LoRA RESCUE NOT REPLICATED"
    )

else:

    VERDICT = (
        "100-LABEL REPRESENTATION RESCUE "
        "NOT REPLICATED IN VISIONFM"
    )

# ============================================================
# 27. REPORT
# ============================================================

report = f"""# PHASE 2A2 — VISIONFM × mBRSET REPRESENTATION RESCUE

## Prespecified purpose

This experiment tests whether the severe mBRSET discrimination
shift observed with VisionFM can be rescued by representation-level
adaptation using the same locked 100-patient local cohort.

The experiment was defined after completion of the frozen-backbone
external transport analysis but before observing any LoRA or
VisionFM full-fine-tuning result.

## Fixed design

Source:
BRSET-trained VisionFM decoder.

Target:
mBRSET.

Local adaptation cohort:
100 patients from the previously frozen trajectory
(seed 20261001).

Frozen target test:
400 mBRSET patients.

Source retention test:
202 BRSET patients.

Training unit:
image.

Primary evaluation unit:
patient.

Patient aggregation:
maximum image probability.

## Strategies

### LoRA
Attention qkv only.
Rank = {LORA_R}.
Alpha = {LORA_ALPHA}.
Dropout = {LORA_DROPOUT}.
Head also trainable.
Fixed 10 epochs.
LR = {LORA_LR}.

### Aggressive full fine-tuning
All VisionFM encoder and decoder parameters trainable.
Fixed 10 epochs.
LR = {FULL_LR}.

No target-test early stopping or target-test checkpoint selection
was permitted.

## Seed-level results

{metrics.to_string(index=False)}

## Strategy summary

{summary.to_string(index=False)}

## Prespecified gates

LoRA:
{json.dumps(lora_gate, indent=2)}

Full fine-tuning:
{json.dumps(full_gate, indent=2)}

Aggressive full-FT harm signal:
{FULL_FT_HARM_SIGNAL}

## Verdict

**{VERDICT}**

## Interpretation boundary

A positive LoRA result supports the hypothesis that a domain with
substantial discrimination loss may require representation-level
adaptation rather than probability recalibration alone.

A negative LoRA result means that 100 local patients were
insufficient to demonstrate a reproducible VisionFM
representation-level rescue under the prespecified configuration.

Aggressive full fine-tuning should only be described as harmful
if the observed target deterioration or source forgetting meets
the prespecified harm signal above.

No conclusion should be changed by choosing a target-test-favored
epoch, learning rate, or method after inspection of these frozen
test results.
"""

REPORT = (
    OUT
    /
    "PHASE_2A2_VISIONFM_MBRSET_RESCUE_REPORT.md"
)

REPORT.write_text(
    report,
    encoding="utf-8",
)

# ============================================================
# 28. FINAL OUTPUT
# ============================================================

print(
    "\n" + "="*80
)

print(
    "PHASE 2A2 FINAL"
)

print(
    "="*80
)

print(
    "\nSEED-LEVEL RESULTS:"
)

print(
    metrics[
        [
            "seed",
            "method",
            "BR_baseline_AUROC",
            "BR_adapted_AUROC",
            "BR_AUROC_change",
            "MB_baseline_AUROC",
            "MB_adapted_AUROC",
            "MB_AUROC_gain",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nMETHOD SUMMARY:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nLoRA GATE:",
    lora_gate
)

print(
    "FULL-FT GATE:",
    full_gate
)

print(
    "FULL-FT HARM SIGNAL:",
    FULL_FT_HARM_SIGNAL
)

print(
    "\nVERDICT:",
    VERDICT
)

print(
    "\nREPORT:",
    REPORT
)

print(
    "\n✅ PHASE 2A2 COMPLETE"
)