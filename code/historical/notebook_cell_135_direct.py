# ============================================================
# PHASE 2A0 — ONE-TIME VISIONFM FEATURE EXTRACTION
# ============================================================
#
# Purpose:
# Extract frozen VisionFM Fundus representations once for:
#
# 1. BRSET TRAIN
# 2. BRSET VALIDATION
# 3. BRSET FROZEN SOURCE TEST
# 4. mBRSET LOCAL-100 development cohort
# 5. mBRSET FROZEN TARGET TEST
# 6. DeepDRiD VALIDATION
# 7. DeepDRiD FROZEN CONFIRMATORY TEST
#
# After this completes, downstream decoder training,
# external validation, recalibration and statistics can
# be performed on CPU without repeatedly running VisionFM.
# ============================================================

from pathlib import Path
from io import BytesIO

import os
import sys
import re
import shutil
import zipfile
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ============================================================
# 0. LOCATE PROJECT ROOT
# ============================================================

PROJECT_NAME = "Retinal_Retraining_Phase1A1"

candidate_roots = [
    Path("/content/drive_c3/MyDrive") / PROJECT_NAME,
    Path("/content/drive_c2/MyDrive") / PROJECT_NAME,
    Path("/content/drive/MyDrive") / PROJECT_NAME,
    Path("/content/gdrive/MyDrive") / PROJECT_NAME,
]

ROOT = next(
    (p for p in candidate_roots if p.exists()),
    None
)

assert ROOT is not None, (
    "Cannot locate project Drive. "
    "Mount Google Drive first."
)

print("=" * 80)
print("PHASE 2A0 — VISIONFM FEATURE EXTRACTION")
print("=" * 80)
print("PROJECT ROOT:", ROOT)


# ============================================================
# 1. OUTPUT DIRECTORY
# ============================================================

OUT = (
    ROOT
    / "results"
    / "visionfm_feature_cache"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("OUTPUT:", OUT)


# ============================================================
# 2. AUTHORITATIVE PROJECT FILES
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

BR_SPLIT = (
    ROOT
    / "data"
    / "locked_metadata"
    / "BRSET_source_split_v2.csv"
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

DD_ZIP = (
    ROOT
    / "data"
    / "deepdrid"
    / "DeepDRiD-v1.1.zip"
)

DD_EVAL_MAN = (
    ROOT
    / "data"
    / "deepdrid"
    / "deepdrid_frozen_confirmatory_manifest.csv"
)

bundle_hits = list(
    ROOT.rglob(
        "phase1a1_colab_bundle.zip"
    )
)

assert bundle_hits, (
    "Cannot find phase1a1_colab_bundle.zip"
)

BR_BUNDLE = bundle_hits[0]

required_files = [
    BR_MAN,
    MB_MAN,
    BR_SPLIT,
    SAMPLING,
    MB_ZIP,
    DD_ZIP,
    DD_EVAL_MAN,
    BR_BUNDLE,
]

print("\n=== PROJECT FILE CHECK ===")

for p in required_files:
    print(
        "✅" if p.exists() else "❌",
        p
    )
    assert p.exists(), p

print("✅ All project files found")


# ============================================================
# 3. VISIONFM REPOSITORY + WEIGHTS
# ============================================================

VFM = Path("/content/VisionFM")

WEIGHTS = Path(
    "/content/VFM_Fundus_weights.pth"
)

assert VFM.exists(), (
    "VisionFM repo missing. "
    "Run the VisionFM setup/smoke-test cell first."
)

assert WEIGHTS.exists(), (
    "VisionFM Fundus weights missing. "
    "Run the VisionFM setup/smoke-test cell first."
)

sys.path.insert(
    0,
    str(VFM)
)

import models
import utils

print("\nVisionFM repo:", VFM)
print(
    "VisionFM weights MiB:",
    round(
        WEIGHTS.stat().st_size
        / 1024**2,
        2
    )
)


# ============================================================
# 4. BUILD FROZEN VISIONFM FUNDUS ENCODER
# ============================================================

model = models.__dict__["vit_base"](
    img_size=[224],
    patch_size=16,
    num_classes=0,
    use_mean_pooling=False,
)

checkpoint = torch.load(
    WEIGHTS,
    map_location="cpu",
    weights_only=False,
)

assert "teacher" in checkpoint

state = checkpoint["teacher"]

# Official pretrained checkpoint contains prefixes
state = {
    key
        .replace("module.", "")
        .replace("backbone.", ""):
    value

    for key, value
    in state.items()
}

missing, unexpected = (
    model.load_state_dict(
        state,
        strict=False
    )
)

print("\nVisionFM load result:")
print("Missing keys:", missing)
print(
    "Unexpected keys:",
    unexpected
)

# Backbone itself must be fully populated.
assert len(missing) == 0

model = model.cuda()
model.eval()

for p in model.parameters():
    p.requires_grad = False

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print("✅ VisionFM frozen encoder loaded")


# ============================================================
# 5. OFFICIAL FUNDS PREPROCESSING
# ============================================================

mean, std = utils.get_stats(
    "Fundus"
)

print("\nFundus mean:", mean)
print("Fundus std :", std)

transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=
            InterpolationMode.BICUBIC
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean,
        std
    ),
])

print("✅ VisionFM Fundus preprocessing locked")


# ============================================================
# 6. COLUMN HELPERS
# ============================================================

def choose_col(
    df,
    names,
    required=True
):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in names:

        if name.lower() in lower:
            return lower[
                name.lower()
            ]

    if required:

        raise KeyError(
            f"Cannot find any of {names}. "
            f"Available columns: "
            f"{list(df.columns)}"
        )

    return None


def basename_any(x):

    return os.path.basename(
        str(x).replace(
            "\\",
            "/"
        )
    )


# ============================================================
# 7. LOAD BRSET / mBRSET MANIFESTS
# ============================================================

br = pd.read_csv(
    BR_MAN
)

mb = pd.read_csv(
    MB_MAN
)

sampling = pd.read_csv(
    SAMPLING
)

print("\nBR manifest:", br.shape)
print("mBRSET manifest:", mb.shape)


# ------------------------------------------------------------
# BRSET columns
# ------------------------------------------------------------

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
        "file"
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

br_split_col = choose_col(
    br,
    ["split"]
)


# ------------------------------------------------------------
# mBRSET columns
# ------------------------------------------------------------

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
        "file"
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

mb_split_col = choose_col(
    mb,
    ["split"]
)


# ------------------------------------------------------------
# Keep only locked analyzable observations
# ------------------------------------------------------------

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
    br[br_split_col]
    .astype(str)
    .str.upper()
)

mb["_split"] = (
    mb[mb_split_col]
    .astype(str)
    .str.upper()
)


# ============================================================
# 8. LOCK BRSET COHORTS
# ============================================================

cohorts = {}

cohorts["BR_TRAIN"] = (
    br[
        br["_split"]
        == "TRAIN"
    ]
    .copy()
)

cohorts["BR_VAL"] = (
    br[
        br["_split"]
        == "VALIDATION"
    ]
    .copy()
)

cohorts["BR_TEST"] = (
    br[
        br["_split"]
        == "FROZEN_SOURCE_TEST"
    ]
    .copy()
)

assert (
    cohorts["BR_TRAIN"]
    [br_pid]
    .nunique()
    == 939
)

assert (
    cohorts["BR_VAL"]
    [br_pid]
    .nunique()
    == 202
)

assert (
    cohorts["BR_TEST"]
    [br_pid]
    .nunique()
    == 202
)

print("\nBRSET:")
print(
    "TRAIN:",
    cohorts["BR_TRAIN"][br_pid].nunique(),
    "patients /",
    len(cohorts["BR_TRAIN"]),
    "images"
)
print(
    "VAL:",
    cohorts["BR_VAL"][br_pid].nunique(),
    "patients /",
    len(cohorts["BR_VAL"]),
    "images"
)
print(
    "TEST:",
    cohorts["BR_TEST"][br_pid].nunique(),
    "patients /",
    len(cohorts["BR_TEST"]),
    "images"
)


# ============================================================
# 9. LOCK mBRSET FROZEN TEST
# ============================================================

cohorts["MB_TEST"] = (
    mb[
        mb["_split"]
        == "FROZEN_TARGET_TEST"
    ]
    .copy()
)

assert (
    cohorts["MB_TEST"]
    [mb_pid]
    .nunique()
    == 400
)

print(
    "\nmBRSET TEST:",
    cohorts["MB_TEST"][mb_pid].nunique(),
    "patients /",
    len(cohorts["MB_TEST"]),
    "images"
)


# ============================================================
# 10. LOCK SAME mBRSET N=100 DEVELOPMENT COHORT
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

mb100_ids = set(
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
    mb100_ids
) == 100

cohorts[
    "MB_LOCAL100"
] = (
    mb[
        (
            mb["_split"]
            == "ADAPTATION_POOL"
        )
        &
        (
            mb[mb_pid]
            .astype(str)
            .isin(
                mb100_ids
            )
        )
    ]
    .copy()
)

assert (
    cohorts[
        "MB_LOCAL100"
    ][mb_pid]
    .nunique()
    == 100
)

print(
    "mBRSET LOCAL100:",
    cohorts["MB_LOCAL100"][mb_pid].nunique(),
    "patients /",
    len(cohorts["MB_LOCAL100"]),
    "images"
)


# ============================================================
# 11. LOAD DeepDRiD VALIDATION
# ============================================================

with zipfile.ZipFile(
    DD_ZIP,
    "r"
) as z:

    names = z.namelist()

    val_csv_hits = [
        x for x in names
        if x.endswith(
            "regular_fundus_images/"
            "regular-fundus-validation/"
            "regular-fundus-validation.csv"
        )
    ]

    assert (
        len(val_csv_hits)
        == 1
    )

    val_csv = (
        val_csv_hits[0]
    )

    ddval = pd.read_csv(
        BytesIO(
            z.read(
                val_csv
            )
        )
    )

    ddval_members = [
        x for x in names
        if (
            "/regular-fundus-validation/"
            in x
            and
            Path(x).suffix.lower()
            == ".jpg"
        )
    ]

assert len(ddval) == 400

assert (
    ddval["patient_id"]
    .nunique()
    == 100
)

ddval[
    "referable_dr_patient"
] = (
    ddval[
        "patient_DR_Level"
    ]
    .astype(int)
    .ge(2)
    .astype(int)
)


# ------------------------------------------------------------
# Validation eye-level image label
# ------------------------------------------------------------

is_left = (
    ddval["image_id"]
    .astype(str)
    .str.contains(
        r"_l\d+$",
        regex=True
    )
)

ddval[
    "_eye_grade"
] = np.where(
    is_left,
    ddval[
        "left_eye_DR_Level"
    ],
    ddval[
        "right_eye_DR_Level"
    ]
)

# Earlier audit confirmed validation is fully labelled
assert (
    ddval[
        "_eye_grade"
    ]
    .notna()
    .all()
)

ddval[
    "referable_dr_image"
] = (
    ddval[
        "_eye_grade"
    ]
    .astype(int)
    .ge(2)
    .astype(int)
)


ddval[
    "filename"
] = (
    ddval[
        "image_id"
    ]
    .astype(str)
    + ".jpg"
)

dd_map = {
    Path(x).name: x
    for x in ddval_members
}

ddval[
    "zip_member"
] = (
    ddval[
        "filename"
    ]
    .map(
        dd_map
    )
)

assert (
    ddval[
        "zip_member"
    ]
    .notna()
    .all()
)

print(
    "\nDeepDRiD VAL:",
    ddval["patient_id"].nunique(),
    "patients /",
    len(ddval),
    "images"
)


# ============================================================
# 12. LOAD DeepDRiD FROZEN CONFIRMATORY TEST
# ============================================================

ddeval = pd.read_csv(
    DD_EVAL_MAN
)

assert len(
    ddeval
) == 400

assert (
    ddeval[
        "patient_id"
    ]
    .nunique()
    == 100
)

print(
    "DeepDRiD TEST:",
    ddeval["patient_id"].nunique(),
    "patients /",
    len(ddeval),
    "images"
)


# ============================================================
# 13. BUILD ZIP INDEXES
# ============================================================

def build_archive_index(
    zpath
):

    index = {}

    with zipfile.ZipFile(
        zpath,
        "r"
    ) as z:

        for member in z.namelist():

            if member.endswith("/"):
                continue

            b = Path(
                member
            ).name

            index.setdefault(
                b,
                []
            ).append(
                member
            )

    return index


print("\nIndexing BRSET bundle...")
br_archive_map = (
    build_archive_index(
        BR_BUNDLE
    )
)

print("Indexing mBRSET...")
mb_archive_map = (
    build_archive_index(
        MB_ZIP
    )
)


# ============================================================
# 14. LOCAL SSD IMAGE CACHE
# ============================================================

DATA = Path(
    "/content/"
    "visionfm_phase2a_images"
)

if DATA.exists():
    raise FileExistsError("Candidate safety guard: inspect or relocate existing extraction directory before rerun")

DATA.mkdir(
    parents=True
)


# ============================================================
# 15. EXTRACT BY BASENAME
# ============================================================

def extract_by_basename(
    zpath,
    df,
    image_col,
    outdir,
    archive_map
):

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    runtime_paths = []

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

            hits = archive_map.get(
                b,
                []
            )

            if len(hits) != 1:

                raise RuntimeError(
                    f"{b}: "
                    f"{len(hits)} "
                    f"archive matches"
                )

            target = (
                outdir / b
            )

            if not target.exists():

                with z.open(
                    hits[0]
                ) as src, \
                    open(
                        target,
                        "wb"
                    ) as dst:

                    shutil.copyfileobj(
                        src,
                        dst
                    )

            runtime_paths.append(
                str(target)
            )

    return runtime_paths


# ============================================================
# 16. EXTRACT BRSET
# ============================================================

for name in [
    "BR_TRAIN",
    "BR_VAL",
    "BR_TEST"
]:

    print(
        "\nExtracting",
        name
    )

    cohorts[name][
        "_runtime_path"
    ] = extract_by_basename(
        BR_BUNDLE,
        cohorts[name],
        br_img,
        DATA / name,
        br_archive_map
    )


# ============================================================
# 17. EXTRACT mBRSET
# ============================================================

for name in [
    "MB_LOCAL100",
    "MB_TEST"
]:

    print(
        "\nExtracting",
        name
    )

    cohorts[name][
        "_runtime_path"
    ] = extract_by_basename(
        MB_ZIP,
        cohorts[name],
        mb_img,
        DATA / name,
        mb_archive_map
    )


# ============================================================
# 18. EXTRACT DeepDRiD BY EXACT MEMBER
# ============================================================

def extract_explicit_members(
    zpath,
    df,
    member_col,
    outdir
):

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    runtime_paths = []

    with zipfile.ZipFile(
        zpath,
        "r"
    ) as z:

        names = set(
            z.namelist()
        )

        for member in df[
            member_col
        ]:

            assert (
                member
                in names
            ), member

            target = (
                outdir
                / Path(member).name
            )

            if not target.exists():

                with z.open(
                    member
                ) as src, \
                    open(
                        target,
                        "wb"
                    ) as dst:

                    shutil.copyfileobj(
                        src,
                        dst
                    )

            runtime_paths.append(
                str(target)
            )

    return runtime_paths


ddval[
    "_runtime_path"
] = extract_explicit_members(
    DD_ZIP,
    ddval,
    "zip_member",
    DATA / "DD_VAL"
)

ddeval[
    "_runtime_path"
] = extract_explicit_members(
    DD_ZIP,
    ddeval,
    "zip_member",
    DATA / "DD_TEST"
)


# ============================================================
# 19. IMAGE FILE CHECK
# ============================================================

for name, df in [
    ("BR_TRAIN", cohorts["BR_TRAIN"]),
    ("BR_VAL", cohorts["BR_VAL"]),
    ("BR_TEST", cohorts["BR_TEST"]),
    ("MB_LOCAL100", cohorts["MB_LOCAL100"]),
    ("MB_TEST", cohorts["MB_TEST"]),
    ("DD_VAL", ddval),
    ("DD_TEST", ddeval),
]:

    ok = (
        df[
            "_runtime_path"
        ]
        .map(
            os.path.exists
        )
        .all()
    )

    print(
        "✅" if ok else "❌",
        name,
        "images"
    )

    assert ok


# ============================================================
# 20. FEATURE DATASET
# ============================================================

class FeatureDataset(
    Dataset
):

    def __init__(
        self,
        df,
        patient_col,
        image_col,
        y_image_col,
        y_patient_col,
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
            .copy()
        )

        self.pid_col = (
            patient_col
        )

        self.image_col = (
            image_col
        )

        self.y_image_col = (
            y_image_col
        )

        self.y_patient_col = (
            y_patient_col
        )


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        i
    ):

        r = (
            self.df.iloc[i]
        )

        image = Image.open(
            r[
                "_runtime_path"
            ]
        ).convert(
            "RGB"
        )

        x = transform(
            image
        )

        patient_id = str(
            r[
                self.pid_col
            ]
        )

        image_id = str(
            r[
                self.image_col
            ]
        )

        yi = int(
            r[
                self.y_image_col
            ]
        )

        yp = int(
            r[
                self.y_patient_col
            ]
        )

        return (
            x,
            patient_id,
            image_id,
            yi,
            yp
        )


# ============================================================
# 21. FEATURE EXTRACTION FUNCTION
# ============================================================

@torch.inference_mode()
def extract_features(
    name,
    df,
    patient_col,
    image_col,
    y_image_col,
    y_patient_col,
):

    outfile = (
        OUT /
        f"{name}_features.npz"
    )

    # Resume / skip if complete
    if outfile.exists():

        try:

            old = np.load(
                outfile
            )

            if (
                len(
                    old[
                        "patient_id"
                    ]
                )
                == len(df)
                and
                old["X"].shape[1]
                == 3072
            ):

                print(
                    f"\n✅ SKIP {name}: "
                    f"feature cache already complete"
                )

                return

        except Exception:

            pass


    print(
        "\n" + "="*70
    )

    print(
        "EXTRACTING:",
        name
    )

    print(
        "images:",
        len(df),
        "| patients:",
        df[
            patient_col
        ].nunique()
    )

    ds = FeatureDataset(
        df,
        patient_col,
        image_col,
        y_image_col,
        y_patient_col,
    )

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    all_X = []
    all_pid = []
    all_image_id = []
    all_yi = []
    all_yp = []

    for batch_i, batch in enumerate(
        loader,
        1
    ):

        (
            xb,
            pids,
            image_ids,
            yi,
            yp
        ) = batch

        xb = xb.cuda(
            non_blocking=True
        )

        # VisionFM official-style representation:
        # CLS tokens from last four transformer blocks
        blocks = (
            model
            .get_intermediate_layers(
                xb,
                4
            )
        )

        features = (
            torch.cat(
                [
                    block[:, 0]
                    for block in blocks
                ],
                dim=-1
            )
        )

        assert (
            features.shape[1]
            == 3072
        )

        all_X.append(
            features
            .float()
            .cpu()
            .numpy()
        )

        all_pid.extend(
            list(
                pids
            )
        )

        all_image_id.extend(
            list(
                image_ids
            )
        )

        all_yi.extend(
            yi.numpy().tolist()
        )

        all_yp.extend(
            yp.numpy().tolist()
        )

        if batch_i % 20 == 0:

            print(
                f"  batches: "
                f"{batch_i}/"
                f"{len(loader)}"
            )

    X = np.concatenate(
        all_X,
        axis=0
    )

    assert (
        X.shape
        ==
        (
            len(df),
            3072
        )
    )

    np.savez_compressed(
        outfile,

        # float16 greatly reduces persistent file size
        X=X.astype(
            np.float16
        ),

        patient_id=np.asarray(
            all_pid,
            dtype=str
        ),

        image_id=np.asarray(
            all_image_id,
            dtype=str
        ),

        y_image=np.asarray(
            all_yi,
            dtype=np.int8
        ),

        y_patient=np.asarray(
            all_yp,
            dtype=np.int8
        ),
    )

    print(
        "✅ SAVED:",
        outfile.name
    )

    print(
        "shape:",
        X.shape
    )

    print(
        "size MiB:",
        round(
            outfile.stat().st_size
            /1024**2,
            2
        )
    )


# ============================================================
# 22. EXTRACT ALL 7 COHORTS
# ============================================================

extract_features(
    "BR_TRAIN",
    cohorts["BR_TRAIN"],
    br_pid,
    br_img,
    br_yi,
    br_yp,
)

extract_features(
    "BR_VAL",
    cohorts["BR_VAL"],
    br_pid,
    br_img,
    br_yi,
    br_yp,
)

extract_features(
    "BR_TEST",
    cohorts["BR_TEST"],
    br_pid,
    br_img,
    br_yi,
    br_yp,
)

extract_features(
    "MB_LOCAL100",
    cohorts["MB_LOCAL100"],
    mb_pid,
    mb_img,
    mb_yi,
    mb_yp,
)

extract_features(
    "MB_TEST",
    cohorts["MB_TEST"],
    mb_pid,
    mb_img,
    mb_yi,
    mb_yp,
)

extract_features(
    "DD_VAL",
    ddval,
    "patient_id",
    "image_id",
    "referable_dr_image",
    "referable_dr_patient",
)

extract_features(
    "DD_TEST",
    ddeval,
    "patient_id",
    "image_id",
    "referable_dr_image",
    "referable_dr_patient",
)


# ============================================================
# 23. FINAL CHECK
# ============================================================

expected = [
    "BR_TRAIN_features.npz",
    "BR_VAL_features.npz",
    "BR_TEST_features.npz",
    "MB_LOCAL100_features.npz",
    "MB_TEST_features.npz",
    "DD_VAL_features.npz",
    "DD_TEST_features.npz",
]

print(
    "\n" + "="*80
)

print(
    "PHASE 2A0 FINAL CHECK"
)

print(
    "="*80
)

n_ok = 0

for filename in expected:

    p = (
        OUT /
        filename
    )

    ok = p.exists()

    print(
        "✅" if ok else "❌",
        filename,
        (
            f"{p.stat().st_size/1024**2:.2f} MiB"
            if ok
            else ""
        )
    )

    if ok:
        n_ok += 1


print(
    "\nTOTAL:",
    n_ok,
    "/",
    len(expected)
)

assert (
    n_ok
    == len(expected)
)

print(
    "\n✅ VISIONFM PHASE 2A0 COMPLETE"
)

print(
    "✅ ALL 7 FEATURE CACHES SAVED TO GOOGLE DRIVE"
)

print(
    "✅ T4 GPU IS NO LONGER NEEDED FOR THIS PHASE"
)