# Local module and safe-copy review

The two ablation evaluators named below were present in an earlier internal package but are excluded from this repository. Their initial missing imports exist in the internal revision project, but those modules have additional local and third-party dependencies. Adding only the direct imports would give a misleading executable claim. See manifests/exclusion_log.csv and the internal dependency audit. Their import chains were rechecked for this closeout; no substitute evaluator was written or represented as historical code.

- evaluate_retfound_ablation_20261110.py: evaluate_retfound_unified_pair_20261109 -> train_retfound_unified_20261109 / evaluate_vfm_unified_pair_20261109 -> further project and RETFound vendor modules
- evaluate_vfm_ablation_20261109.py: evaluate_vfm_unified_pair_20261109 -> train_visionfm_unified_20261109 -> VisionFM vendor modules

The four historical scripts with fixed-directory shutil.rmtree calls have candidate-only fail-closed guards. A pre-existing directory raises FileExistsError and is never deleted by these copies. The historical execution sources and earlier candidate copies remain unchanged. These changes alter executable behaviour and their hashes; they are not claimed to be byte-identical to executed notebook cells. Exact original-to-safe-copy SHA256 pairs are in `manifests/historical_safe_copy_map.csv`.

Other included scripts may still require external RETFound util or VisionFM utils/model code from the rights holder and the authorised local project layout. This repository is not a self-contained reproduction.

The v2 R1-1 script has a separate smoke-only environment switch, RETINAL_USE_ENV_TORCH=1, and a separate RETINAL_SMOKE_OUTPUT path. This is a later portability change; it does not describe the original three-seed execution.
