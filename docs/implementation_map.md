# Implementation map and provenance boundary

| Research component | Historical code | Revision code | Actual execution evidence |
|---|---|---|---|
| RETFound BRSET source | phase1a1_train_eval.py | run_r11_matched_source_20261112.py is a separate frozen-encoder arm | Historical source checkpoint configs; R1-1 real-run report |
| VisionFM frozen features and MLP | notebook_cell_135_direct.py; phase2a1_visionfm_cpu.py | train_visionfm_unified_20261109.py | Executed notebook extraction index; revision run ledger |
| Historical head-only and dose | phase2a3_visionfm_head_only.py; phase2a2_visionfm_mbrset_rescue.py; phase2a4_visionfm_deepdrid_dose.py | train_visionfm_unified_20261109.py; train_retfound_unified_20261109.py | Executed notebook cell index; controlled revision outputs |
| LoRA and partial/full unfreezing | Historical adaptation in recovered scripts and source results | train_visionfm_unified_20261109.py; train_retfound_unified_20261109.py; train_vfm_encoder_ablation_20261109.py; train_retfound_encoder_ablation_20261110.py | Internal commands and checkpoint histories; not every copied script has direct per-file log proof |
| Cohort and role control | Historical source bundle and notebook code | lock_unified_experiment_roles_20261109.py; r21_cohort_audit.py; analyze_recovered_100_subset.py | Internal split audit and historical local100 record |
| Patient aggregation and inference | Historical scripts contain per-patient aggregation | train/evaluate scripts; run_r11_matched_source_20261112.py | R1-1 true local run; existing unified inference audit |
| Platt maps and thresholds | phase1c2_deepdrid_recalibration.py and archived notebook extracts | recover_calibration_maps.py; replay_visionfm_validation_thresholds.py; replay_retfound_deepdrid_validation_thresholds.py | Mapping/threshold audit; original fit objects not recovered |
| Discrimination, bootstrap, clinical metrics and McNemar | Historical outputs and clinical-consequences extract | analyze_unified_retfound_pair_20261109.py; analyze_unified_vfm_pair_20261109.py; analyze_r24_two_stage_bootstrap.py; analyze_r26_final_clinical_20261111.py | Internal result audits; not all original statistical objects available |

The copied historical code is not silently replaced by revision code. Script provenance and safe-copy hashes are recorded in manifests/unlinked_28_script_provenance.csv and manifests/historical_safe_copy_map.csv. Generic historical Colab mount paths are retained as forensic code context; no local Windows user paths, cloud-file identifiers or credentials are included.
