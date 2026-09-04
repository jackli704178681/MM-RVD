# MM-RVD Final Supplementary Parameter Closure

Created: 2026-08-17T15:40:57

## Scope

This closure did not train models, rerun inference, modify the manuscript, modify result files, or modify model code. It only generated a new supplementary parameter-closure directory.

## Internal Provenance Used

- Baseline parameters: `unified_17_session_authentic_rerun_a6_20260808_185006`, `src/mm_rvd/baselines.py`, and `scripts/run_unified_17_session_authentic_rerun_a6.py`.
- Current MM-RVD implementation facts: `mmrvd_secondgen_development_a9_20260814_171047` and `mmrvd_r6_authentic_ablation_a11_20260816_065146`.
- Missingness protocol: `src/mm_rvd/phase2b_mask_bank.py`, `scripts/run_global_authentic_baseline_smoke_a5.py`, and the locked ablation evaluator.
- Environment: run logs from the above directories plus a read-only GPFA virtual-environment probe.

## Public Terminology Sanitization

The public CSV tables avoid internal phase labels, internal split labels, historical model names, and internal candidate IDs. Public split terminology is restricted to training set, internal validation set, and evaluation set.

## Completion Status

S2_COMPLETE = PARTIAL  
S3_COMPLETE = PASS  
S4_COMPLETE = PASS  
S5_EXPERIMENT_ENVIRONMENT = PARTIAL  
UNRESOLVED_PARAMETERS = 9  
REAL_BLOCKERS = 0  
FINAL_SUPPLEMENTARY_PARAMETER_STATUS = READY_WITH_UNRESOLVED

## Notes

S2 remains PARTIAL because several package-level or environment-specific parameters were not explicitly recorded in the formal artifacts. S5 remains PARTIAL because CPU, RAM, GPU model, GPU memory, and cuDNN were not found in actual experiment logs and were not backfilled from the current audit runtime.

## Output Hashes

- `01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv`: `532bda68af0cd6581cc810e305799c721d723f533f79c40d6fdaf39f7b5762a7`
- `02_FINAL_SUPP_TABLE_S3_MISSINGNESS_EVALUATION_PROTOCOL.csv`: `7e5d1598e724c70b40b6d251caf56a22e84e50a89004570ccd68b102fe443cbb`
- `03_FINAL_SUPP_TABLE_S4_ABLATION_IMPLEMENTATION.csv`: `a304a586fd24be7226e51ead686907fcc51deb925ed14bc436e17690a82550ab`
- `04_FINAL_SUPP_TABLE_S5_SOFTWARE_HARDWARE.csv`: `b3daf6a4c94dd974841c80da3babd7e13b0d9aabb9195a58307f2506ab2bfffb`
- `06_FINAL_UNRESOLVED_PARAMETERS.csv`: `ab0ec3fc9a445ea9659179bc802849e69311c35145e8806a96a4982b4589f3d6`
