# Reproduction workflow

1. Obtain each dataset under its source terms. Obtain RETFound and VisionFM code and weights separately; verify the official RETFound byte hash. Prepare licensed local image paths and the original patient-level split manifest.
2. For historic RETFound source development, use code/historical/phase1a1_train_eval.py and the values in configs/retfound_source_config.yaml. Its Colab-oriented paths remain part of historical provenance; copy the code into an authorised environment and configure paths before running.
3. For historic VisionFM source and later historic head-only, calibration and dose stages, inspect the executed-cell extracts in code/historical/ and manifests/historical_notebook_extraction_index.csv. These are forensic code extracts, not a turnkey restored notebook.
4. For revision controlled comparisons, use code/revision/ with the exact per-run state and role manifests. The internal command ledger and per-run outputs remain authoritative.
5. For the bounded R1-1 revision arm, set local environment variables described in docs/model_weights.md, place the R1-1 source script in the revision project layout, then run it with modes smoke, extract, train, test and external in that order. Source/target tests use validation-selected heads; patient prediction outputs stay local.
6. For the no-data interface check, run python -I examples/smoke_stdlib_contract.py. This does not load an ophthalmic model or reproduce clinical performance.

For the previously tested clean-environment weight check, set RETINAL_USE_ENV_TORCH=1 and RETINAL_SMOKE_OUTPUT to a new local JSON path, then run the R1-1 script with --mode smoke. This bypasses the original local torch vendor injection and permits a CPU fallback. It is an interface/single-batch test only. Do not treat its output as the historical executed version or as a new scientific result.

The repository lacks restricted data, weight files, historical patient membership, calibration fit objects and third-party model repositories. It cannot regenerate all paper results on its own. A public clone and synthetic entrypoint must be checked separately; real-patient use requires independently authorised inputs.
