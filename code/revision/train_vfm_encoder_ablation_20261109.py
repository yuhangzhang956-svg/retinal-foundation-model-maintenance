"""New VisionFM local-unfreeze ablations on the same locked roles.

This is not a replay of historical Phase2A2. Model selection uses only the
separate MODEL_VALIDATION patients. Target test is never opened in this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
import zipfile
from pathlib import Path

from gpu_single_batch_feasibility_20261109 import BASE, VisionFMClassifier, configure, torch, build_model
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT = BASE.parent
ZIP_PATH = Path(r"<LICENSED_DATA_ROOT>\mbrset_1.0.0.zip")
MEAN = (0.423737496137619, 0.2609460651874542, 0.128403902053833)
STD = (0.29482534527778625, 0.20167365670204163, 0.13668020069599152)
TRANSFORM = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ZipImages(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.archive = zipfile.ZipFile(ZIP_PATH)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        member = "mbrset_1.0.0/" + row.image_path.replace("\\", "/")
        with Image.open(io.BytesIO(self.archive.read(member))) as image:
            x = TRANSFORM(image.convert("RGB"))
        return x, torch.tensor(float(row.referable_dr_image), dtype=torch.float32), \
               str(row.patient_id), int(row.referable_dr_patient), str(row.image_id)


def load_roles():
    roles = pd.read_csv(BASE / "private_manifests" / "unified_experiment_patient_roles_PRIVATE.csv",
                        dtype={"patient_id": str})
    roles = roles[roles.dataset == "mBRSET"][["patient_id", "role"]]
    manifest = pd.read_csv(PROJECT / "outputs" / "audit" / "mBRSET_master_manifest_v2.csv",
                           dtype={"patient_id": str}, low_memory=False)
    manifest = manifest[manifest.exclusion_status == "INCLUDED"]
    merged = manifest.merge(roles, on="patient_id", validate="many_to_one")
    assert len(merged) == len(manifest)
    assert merged.groupby("patient_id").role.nunique().max() == 1
    train = merged[merged.role == "ADAPT_TRAIN"].copy()
    valid = merged[merged.role == "MODEL_VALIDATION"].copy()
    calibr = merged[merged.role == "CALIBRATION_THRESHOLD"].copy()
    assert (train.patient_id.nunique(), valid.patient_id.nunique(), calibr.patient_id.nunique()) == (100, 200, 200)
    assert set(train.patient_id).isdisjoint(set(valid.patient_id))
    assert set(train.patient_id).isdisjoint(set(calibr.patient_id))
    assert set(valid.patient_id).isdisjoint(set(calibr.patient_id))
    return train, valid, calibr


def numpy_legacy_scalar(*args):
    return np._core.multiarray.scalar(*args)


numpy_legacy_scalar.__module__ = "numpy.core.multiarray"
numpy_legacy_scalar.__name__ = "scalar"


def source_model(seed):
    from argparse import Namespace
    torch.serialization.add_safe_globals([
        numpy_legacy_scalar, np.dtype, type(np.dtype("float64")),
        type(np.dtype("float32")), type(np.dtype("int64")), Namespace,
    ])
    model: VisionFMClassifier = build_model("VisionFM")
    weight = BASE / "private_checkpoints" / "VFM_Fundus_weights.pth"
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True, mmap=True)
    state = {k.replace("module.", "").replace("backbone.", ""): v
             for k, v in checkpoint["teacher"].items()}
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    assert not missing, missing
    assert all(key.startswith("head.") for key in unexpected), unexpected
    del checkpoint, state
    head_path = BASE / "private_checkpoints" / f"VisionFM_BRSET_decoder_seed{seed}.pth"
    head_state = torch.load(head_path, map_location="cpu", weights_only=True)
    model.head.load_state_dict(head_state["state_dict"], strict=True)
    return model, weight, head_path


def predict(model, loader, seed, mode, role):
    model.eval()
    records = []
    with torch.inference_mode():
        for x, _yi, patient_id, yp, image_id in loader:
            logits = model(x.cuda(non_blocking=True)).float()
            probs = torch.sigmoid(logits)
            half = torch.sigmoid(logits.to(torch.float16)).to(torch.float32)
            for i in range(len(patient_id)):
                records.append({"seed": seed, "mode": mode, "role": role,
                                "patient_id": patient_id[i], "image_id": image_id[i],
                                "y": int(yp[i]), "logit_float32": float(logits[i].cpu()),
                                "p_float32": float(probs[i].cpu()),
                                "p_float16_diagnostic": float(half[i].cpu())})
    patients = {}
    for row in records:
        pid = row["patient_id"]
        if pid not in patients:
            patients[pid] = {"y": row["y"], "p": row["p_float32"]}
        else:
            assert patients[pid]["y"] == row["y"]
            patients[pid]["p"] = max(patients[pid]["p"], row["p_float32"])
    y = np.asarray([v["y"] for v in patients.values()], dtype=np.int64)
    p = np.asarray([v["p"] for v in patients.values()], dtype=np.float64)
    return records, {"patients": len(patients), "positive": int(y.sum()),
                     "auroc": float(roc_auc_score(y, p)),
                     "nll": float(log_loss(y, np.clip(p, 1e-7, 1-1e-7))),
                     "p_zero": int(np.count_nonzero(p == 0)), "p_one": int(np.count_nonzero(p == 1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["last1", "last3", "full_ft"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--tag", default="UNIFIED_R20261109_B3_ABL")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    args = p.parse_args()
    assert args.seed in (20267001, 20267002, 20267003)
    assert args.epochs == 10, "Main comparison requires 10 fixed candidate epochs"
    assert args.learning_rate in (1e-5, 3e-5)
    assert args.mode == "full_ft" or args.learning_rate == 1e-5
    assert torch.cuda.is_available()
    train, valid, calibr = load_roles()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, valid_ds, cal_ds = ZipImages(train), ZipImages(valid), ZipImages(calibr)
    train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    cal_loader = DataLoader(cal_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    model, pretrained, source_head = source_model(args.seed)
    total, trainable = configure(model, "VisionFM", args.mode)
    model = model.cuda()
    # Fixed for this exploratory ablation before this arm's own test evaluation.
    lr = args.learning_rate
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    run_id = f"{args.tag}_{args.mode}_seed{args.seed}"
    out_dir = BASE / "unified_experiment"
    out_dir.mkdir(exist_ok=True)
    config = {"run_id": run_id, "mode": args.mode, "seed": args.seed,
              "epochs": 10, "batch_size": 3, "learning_rate": lr,
              "weight_decay": 0.01, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR_Tmax10",
              "amp": "float16_GradScaler_initial128", "gradient_clipping_norm": 1.0,
              "protocol_timing": "revision-period exploratory; last1/last3 fixed LR and full-FT two-LR validation grid specified before any respective ablation test",
              "classification_head_trainable": True,
              "validation_rule": "highest patient AUROC; tie lowest NLL; then earliest epoch",
              "selection_data": "mBRSET MODEL_VALIDATION 200 patients",
              "calibration_data": "mBRSET CALIBRATION_THRESHOLD 200 patients, predictions saved but no fit here",
              "test_data_used": False, "target_test_opened": False,
              "patient_roles_sha256": sha(BASE / "private_manifests" / "unified_experiment_patient_roles_PRIVATE.csv"),
              "mbrset_archive_sha256": json.loads((BASE / "reproducibility" / "unified_role_image_integrity_20261109.json").read_text())["archive_sha256"],
              "official_fundus_weight_sha256": sha(pretrained), "source_head_sha256": sha(source_head),
              "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "train_images": len(train), "validation_images": len(valid), "calibration_images": len(calibr),
              "total_parameters": total, "trainable_parameters": trainable}
    config_path = out_dir / f"{run_id}_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    history = []
    best = None
    checkpoint_path = BASE / "private_checkpoints" / f"{run_id}_best.pth"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, 11):
        model.train()
        if args.mode == "head_only":
            model.encoder.eval()
        losses = []
        skipped = 0
        for x, yi, _pid, _yp, _image in train_loader:
            x, yi = x.cuda(non_blocking=True), yi.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            before = scaler.get_scale()
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x)
                loss = criterion(logit, yi)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < before:
                skipped += 1
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        _records, metrics = predict(model, valid_loader, args.seed, args.mode, "MODEL_VALIDATION")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "skipped_steps": skipped,
               "validation_patients": metrics["patients"], "validation_positive": metrics["positive"],
               "validation_auroc": metrics["auroc"], "validation_nll": metrics["nll"],
               "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20, 2),
               "elapsed_seconds": round(time.perf_counter()-started, 2)}
        history.append(row)
        score = (metrics["auroc"], -metrics["nll"], -epoch)
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "validation_metrics": metrics}
            state_cpu = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
            torch.save({"state_dict": state_cpu, "config": config,
                        "best_epoch": epoch, "best_validation_metrics": metrics}, checkpoint_path)
            del state_cpu
        print(json.dumps({"run_id": run_id, **row}, ensure_ascii=False), flush=True)
        history_path = out_dir / f"{run_id}_history.csv"
        with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerows(history)
    assert best is not None
    from torch.torch_version import TorchVersion
    torch.serialization.add_safe_globals([TorchVersion])
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.cuda().eval()
    val_records, val_metrics = predict(model, valid_loader, args.seed, args.mode, "MODEL_VALIDATION")
    cal_records, cal_metrics = predict(model, cal_loader, args.seed, args.mode, "CALIBRATION_THRESHOLD")
    for role, records in (("MODEL_VALIDATION", val_records), ("CALIBRATION_THRESHOLD", cal_records)):
        path = BASE / "private_predictions" / f"{run_id}_{role}_image_scores_PRIVATE.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    summary = {"run_id": run_id, "status": "TRAINING_AND_INDEPENDENT_DEV_INFERENCE_COMPLETE_TEST_NOT_OPENED",
               "best_epoch": best["epoch"], "best_validation_metrics": val_metrics,
               "calibration_role_counts_only_no_fit": {k: cal_metrics[k] for k in ("patients", "positive")},
               "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20, 2),
               "elapsed_seconds": round(time.perf_counter()-started, 2),
               "checkpoint_path": str(checkpoint_path.relative_to(BASE)), "checkpoint_sha256": sha(checkpoint_path),
               "config_path": str(config_path.relative_to(BASE)), "test_data_used": False}
    (out_dir / f"{run_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
