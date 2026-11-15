# Calibration, thresholds and patient aggregation

The original Platt maps were algebraically reconstructed from archived outputs; original fitted objects were not recovered. The revision design fits calibration and selects referral thresholds on independent development patients, never on locked target-test patients. The historic 100-patient dose and the later 100/200/200/400 design must not be merged.

For R2-5, the shared complete composite rule P is an additional exploratory analysis: positive Brier gain with bootstrap CI lower bound above zero, NLL improvement, slope closer to one and balanced-accuracy loss no greater than 0.05. Rule A uses Brier gain at least 0.01 with the other conditions unchanged; it is a sensitivity analysis. Neither is an original prespecified clinical-safety rule.

For screening evaluation, image probabilities are aggregated to a patient by the maximum eligible image probability. A monotonic Platt map with a correspondingly mapped threshold leaves binary decisions mathematically equivalent; numerical output differences are separately audited. Patient predictions remain controlled internal artifacts. The original fitted objects were not recovered; the published historical coefficients are reconstructions from frozen probability pairs, not original fit objects.

The parameters directory contains historical_recovered_calibration_maps.csv (21 reconstructed maps) and revision_locked_development_maps_thresholds.csv (42 records read from 13 saved development-only map/threshold files). The latter records source file names and SHA256, seeds, model arms, development roles, Platt intercepts and slopes, regularization parameters, selected thresholds and nonidentifying development-set summaries. No target-test labels were used to fit these maps.
