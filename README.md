# Retinal foundation-model maintenance: research code

This repository contains study code for source development, external evaluation, recalibration, head-only adaptation, LoRA, last-block unfreezing, conservative full fine-tuning, referral-threshold selection, and clinical operating-point analysis for diabetic retinopathy screening. Historical code and revision-period additions are separated in code/historical and code/revision. The provenance registers distinguish saved source bytes from proven execution-time bytes.

## Minimal check

Run: python -I examples/smoke_stdlib_contract.py

This checks a synthetic patient-role and aggregation contract. It does not reproduce a real-patient result. The scientific dependencies are listed in requirements.txt; environment.yml is an untested Conda template. See docs/reproduction_workflow.md for the real-data sequence and docs/limitations.md for the verified testing scope.

## Data and weights

Researchers must obtain each resource from its owner and satisfy its access terms:

- BRSET v1.0.2: https://physionet.org/content/brazilian-ophthalmological/1.0.2/
- mBRSET v1.0: https://physionet.org/content/mbrset/1.0/
- MESSIDOR-2: https://www.adcis.net/en/third-party/messidor2/
- DeepDRiD: https://github.com/deepdrdoc/DeepDRiD
- RETFound implementation: https://github.com/rmaphoh/RETFound
- RETFound official natureCFP weight: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
- VisionFM implementation and Fundus weight instructions: https://github.com/ABILab-CUHK/VisionFM

This repository has no fundus images, patient identifiers, patient-level predictions, model weights, or upstream vendor modules. BRSET and mBRSET are credentialed-access resources. See docs/data_access.md and docs/model_weights.md.

## Finding the implementation

- docs/implementation_map.md links study tasks to historical and revision code.
- docs/training_parameters.md and manifests/parameter_provenance.csv give evidence-linked training values.
- docs/calibration_and_thresholds.md explains patient aggregation and the nonidentifying parameter tables in parameters/.
- docs/provenance_levels.md and manifests/unlinked_28_script_provenance.csv document where execution-time source hashes cannot be recovered.
- docs/dependency_and_exclusion_review.md names the two omitted ablation evaluators and the four guarded historical copies.

The code requires separately licensed data, exact private patient-role manifests, upstream implementations, and model weights for real-patient runs. It is not a self-contained reproduction of every manuscript result.

## Rights

The author team approved public display of its study code. No open-source reuse license or final CITATION.cff has been approved. Public readability does not by itself grant permission to copy, modify, or redistribute beyond applicable law and platform terms. See docs/rights_and_attribution.md.
