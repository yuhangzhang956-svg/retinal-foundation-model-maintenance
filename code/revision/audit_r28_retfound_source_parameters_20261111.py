"""Read saved checkpoint configs safely and verify RETFound source parameters."""
import csv
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE / "analyses" / "torch_211_vendor"), str(BASE / "analyses" / "torch_vendor")]
import torch  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


torch.serialization.add_safe_globals([(Path().resolve().__class__, "pathlib._local.PosixPath")])
checkpoint_paths = {
    "20260906": BASE / "private_checkpoints" / "RETFound_BRSET_seed20260906_best.pth",
    "20260908": Path.home() / "Downloads" / "RETFound_BRSET_seed20260908_best.pth",
}
remote = {r["filename"]: r for r in csv.DictReader(
    (BASE / "reproducibility" / "retfound_remote_model_hashes_20261109.csv").open(encoding="utf-8-sig"))}
records = []
for seed, path in checkpoint_paths.items():
    if not path.is_file():
        continue
    name = path.name
    actual_sha = sha(path)
    assert actual_sha.lower() == remote[name]["sha256"].lower()
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    c = obj["config"]
    numeric = {k: int(c[k]) for k in ("batch_size", "accum_iter", "epochs", "patience", "warmup_epochs", "num_workers", "bootstrap")}
    assert int(obj["seed"]) == int(seed)
    records.append({"filename": name, "seed": int(seed), "bytes": path.stat().st_size,
                    "sha256": actual_sha, "best_epoch_zero_based": int(obj["epoch"]),
                    "best_validation_patient_auroc": float(obj["val_patient_auroc"]),
                    "checkpoint_config": numeric,
                    "source_pretraining_checkpoint_name": Path(str(c["weights"])).name})

manifest = BASE.parent / "outputs" / "audit" / "BRSET_master_manifest_v2.csv"
with manifest.open(newline="", encoding="utf-8-sig") as f:
    train = [r for r in csv.DictReader(f) if r["split"] == "TRAIN" and r["exclusion_status"] == "INCLUDED"]
pos = sum(int(float(r["referable_dr_image"])) for r in train)
neg = len(train) - pos
assert len(train) == 1802 and pos > 0 and records
code = BASE.parent / "phase1a1_train_eval.py"
source = code.read_text(encoding="utf-8")
for snippet in ("base_lr = 5e-3 * eff_batch / 256", "class_weights = torch.tensor([1.0, nneg / npos]", "nn.CrossEntropyLoss(weight=class_weights)"):
    assert snippet in source
out = {"status":"VERIFIED_EXECUTED_CHECKPOINT_CONFIG_AND_CODE", "checkpoint_configs":records,
       "source_script_sha256":sha(code), "train_image_counts":{"negative":neg,"positive":pos},
       "class_weight_by_executed_code":[1.0,neg/pos], "class_weight_unit":"BRSET source-training image rows, not patients",
       "base_learning_rate_formula":"5e-3 * (batch_size * accum_iter) / 256",
       "base_learning_rate_at_saved_config":5e-3*records[0]["checkpoint_config"]["batch_size"]*records[0]["checkpoint_config"]["accum_iter"]/256,
       "checkpoint_seed07":"REMOTE_HASH_ONLY_NO_LOCAL_CONFIG_READ",
       "historical_repo_commit":"NOT_RECOVERED"}
dest=BASE/"reproducibility"/"r28_retfound_source_parameters_20261111.json"
dest.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"status":out["status"],"local_checkpoints":len(records),"base_lr":out["base_learning_rate_at_saved_config"],"class_weight_positive":neg/pos}))
