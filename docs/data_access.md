# Data access and local input contract

Real datasets must be obtained by each researcher under the original terms. This repository contains no fundus images, patient mapping, individual labels or predictions.

| Dataset | Official source | Access and redistribution boundary |
|---|---|---|
| BRSET v1.0.2 | https://physionet.org/content/brazilian-ophthalmological/1.0.2/ | Credentialing, CITI training and a signed DUA. Do not redistribute the archive or identity mappings. |
| mBRSET v1.0 | https://physionet.org/content/mbrset/1.0/ | Credentialing, CITI training and a signed DUA. Do not redistribute the archive or identity mappings. |
| MESSIDOR-2 | https://www.adcis.net/en/third-party/messidor2/ | Official download requires a personal-information form. Exact redistribution terms require individual review; no images bundled. |
| DeepDRiD | https://github.com/deepdrdoc/DeepDRiD | Repository displays CC BY-SA 4.0. No images or labels copied; users must follow the source terms. |

Real execution expects local image archives or directories, patient/image identifiers, the source dataset split and rDR labels. BRSET source roles have 939/202/202 patients. The revision mBRSET design is 100 adaptation, 200 validation, 200 separate calibration/threshold development and 400 locked test patients. Historic local100 and revision adaptation100 are the same 100 patients (100/100 verified overlap), but belong to different analysis protocols. Exact membership and patient IDs remain in the authorised internal project and are absent here. The split verification code can validate a locally supplied manifest but cannot reconstruct private membership from aggregate counts alone.

The official dataset pages are the authority for access conditions. Derived patient-level data are excluded pending licence and privacy review.
