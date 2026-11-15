# Verification and release limits

Six historical code files were extracted from executed notebook cells. The historical Colab repository commit and some environment details remain unrecovered. All 28 scripts in the unlinked provenance register lack proven execution-time source-byte SHA256; present-day hashes do not close that gap. Historical code is separated from revision-period code. Four historical scripts were guarded against deleting an existing fixed directory; their safe copies are not byte-identical to the historical originals.

A separate clean Python 3.12.14 CPU environment installed the scientific requirements. Pip check, core imports and the synthetic contract passed. The official RETFound weight and two authorised BRSET images passed a CPU encoder forward, producing a [2,1024] feature tensor. See manifests/clean_environment_test_summary.json. GPU installation, candidate-based training, full patient prediction export, calibration, statistics and full historical pipeline reproduction were not tested in that environment.

Some historical scripts load checkpoints through torch.load without a restrictive weights_only setting. Run them only with trusted, authorised checkpoints and check known hashes. No repository code makes untrusted pickle objects safe.

The author confirmed rights for public display of study code and nonidentifying parameter summaries. No reuse license or final citation metadata was selected. This repository excludes raw images, identity mappings, individual predictions, model weights and upstream vendor modules. A public clone and synthetic smoke test verify access and a minimal interface, not full real-patient reproduction.
