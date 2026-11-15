# Script execution provenance

`manifests/unlinked_28_script_provenance.csv` assigns a separate evidence tier to each of the 28 scripts whose candidate bytes cannot be linked to an execution-time SHA256. The `conclusion` field now describes the same tier for every row. `runtime_sha256=NOT_RECOVERED` and `public_copy_execution_identity=UNVERIFIED` apply to **all 28**. A present-day source hash proves only the present-day bytes. For the 14 named execution/output records, `manifests/execution_record_14_reconciliation.csv` gives the exact log line and currently present result artifact.

| Tier | Meaning | Count |
| --- | --- | ---: |
| EXECUTION_RECORD_WITH_OUTPUT | A contemporaneous run log records the named command and result | 14 |
| EXECUTION_LOG_NO_RUNTIME_BYTE_HASH | A historical output log exists, but runtime source bytes were not hashed | 2 |
| COMMAND_AND_RESULT_RECORD | Command and resulting role-lock report are recorded, without runtime byte hash | 1 |
| RESULT_REPORT_NAMES_SCRIPT | An analysis report identifies the script and results, but does not freeze runtime bytes | 5 |
| COMMAND_RECORD_NO_RUNTIME_BYTE_HASH | A shell command names the script without a linked result in the selected record; linkage to the candidate bytes is unproved | 4 |
| SOURCE_FILE_ONLY | No named execution record was located in the selected contemporaneous ledgers | 2 |

The **six different historical VisionFM notebook extracts** are indexed separately in `manifests/historical_notebook_extraction_index.csv` with executed cell IDs and extracted-source hashes. Those notebook cell records should not be merged into the 28-script ledger or mistaken for execution-time hashes of later revised copies. Four historical safe copies differ in bytes and behaviour from their recovered originals; see `manifests/historical_safe_copy_map.csv`.

These tiers do not prove full reproduction, nor do they imply that every listed source file was executed. For executable use, obtain the separate licensed vendor modules, weights and data as described in `docs/model_weights.md` and `docs/data_access.md`.
