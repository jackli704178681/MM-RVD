# Reproducibility Workflow

This repository preserves the frozen code, protocol records, manifests and source data used for the MM-RVD submission. It does not redistribute raw Allen VBN or CRCNS pvc-11 data.

1. Download the original datasets from the official providers listed in `docs/DATA_SOURCES.md`.
2. Prepare the software environment described in `environment/versions.txt`.
3. Construct 25-bin trial neural responses using the preprocessing/data-construction scripts referenced in `docs/FILE_MANIFEST.md`.
4. Load the frozen train/validation/held-out split manifests from `manifests/splits/`.
5. Load the frozen observation-mask manifests from `manifests/masks/`.
6. Run or recalculate MM-RVD using the frozen R6 configuration in `configs/main/`.
7. Load/evaluate frozen baseline states or reproduce the baseline training/evaluation paths using the scripts under `scripts/main_experiment/`.
8. Apply paired structured-observation masks.
9. Calculate balanced accuracy and chance-normalized balanced accuracy (CN-BalAcc).
10. Aggregate in the order: mask replicate -> seed -> session -> animal -> dataset.
11. Calculate Five-Missing Mean (FMM) and Five-Missing Worst (FMW).
12. Reproduce Figure 2 and Table 2 from `source_data/figure_2/` and `source_data/tables/TABLE2_FINAL_SOURCE.csv`.
13. Reproduce Figure 3 and Table 3 from `source_data/figure_3/` and A11 ablation source tables.
14. Redraw Supplementary Figure S1 from `source_data/supplementary_figure_s1/SUPP_FIG_S1_FIGURE_SOURCE.csv`.

Supplementary Figure S1 should normally be redrawn from frozen source CSVs. Full severity inference is documented in `scripts/supplementary_s1/run_supplementary_figure_s1_severity_sweep.py` and `audit/supplementary_s1/`, but rerunning it is not required to regenerate the figure.

## Baseline implementation mapping

|Baseline|Source implementation|Training entrypoint|Evaluation entrypoint|Configuration|Frozen-state manifest|Result source|
|---|---|---|---|---|---|---|
|Mean-rate linear SVM|src/mm_rvd/baselines.py (MeanRateLinearSVM)|scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py|scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py; scripts/supplementary_s1/run_supplementary_figure_s1_severity_sweep.py for S1 frozen SVM evaluation|source_data/tables/01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv|manifests/model_states/FINAL_R4_SEVEN_MODEL_RESULTS_FREEZE_MANIFEST.json|results/main_summary/*; source_data/tables/TABLE2_FINAL_SOURCE.csv|
|SVD64-logistic|src/mm_rvd/baselines.py (SVD64LogReg)|scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py|scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py|source_data/tables/01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv|audit/model_state/R09_SOLVER_AUDIT_REPORT.md; manifests/model_states/FINAL_R4_SEVEN_MODEL_RESULTS_FREEZE_MANIFEST.json|results/main_summary/*; source_data/tables/TABLE2_FINAL_SOURCE.csv|
|supervised CEBRA + linear classifier|src/mm_rvd/baselines.py (CEBRAFlatLogReg); src/mm_rvd/cebra_api_adapter.py; src/mm_rvd/r1_formal_executor.py (cebra_features)|scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py|scripts/main_experiment/run_R3_one_shot_heldout_evaluation.py; scripts/supplementary_s1/run_supplementary_figure_s1_severity_sweep.py|source_data/tables/01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv|manifests/model_states/FINAL_R2_FOUR_BASELINE_MODEL_FREEZE_MANIFEST.json|results/main_summary/*; results/supplementary_s1/*|
|GRU-D-inspired recurrent decoder|src/mm_rvd/fair_baseline_r1.py (build_neural_model/FormalGRUD); src/mm_rvd/baselines.py (GRULightweight for A5/A6 compatibility)|src/mm_rvd/r1_formal_executor.py; scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py|scripts/main_experiment/run_R3_one_shot_heldout_evaluation.py|source_data/tables/01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv|manifests/model_states/FINAL_R2_FOUR_BASELINE_MODEL_FREEZE_MANIFEST.json|results/main_summary/*; source_data/tables/TABLE2_FINAL_SOURCE.csv|
|Lightweight TCN|src/mm_rvd/fair_baseline_r1.py (build_neural_model/FormalTCN); src/mm_rvd/baselines.py (TCNLightweight for A5/A6 compatibility)|src/mm_rvd/r1_formal_executor.py; scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py|scripts/main_experiment/run_R3_one_shot_heldout_evaluation.py|source_data/tables/01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv|manifests/model_states/FINAL_R2_FOUR_BASELINE_MODEL_FREEZE_MANIFEST.json|results/main_summary/*; source_data/tables/TABLE2_FINAL_SOURCE.csv|
|Position-aware Lightweight Transformer|src/mm_rvd/fair_baseline_r1.py (build_neural_model); src/authentic_baselines/lightweight_transformer.py (LightweightTransformerDecoder)|src/mm_rvd/r1_formal_executor.py; scripts/main_experiment/run_unified_17_session_authentic_rerun_a6.py|scripts/main_experiment/run_R3_one_shot_heldout_evaluation.py|source_data/tables/01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv|manifests/model_states/FINAL_R2_FOUR_BASELINE_MODEL_FREEZE_MANIFEST.json|results/main_summary/*; source_data/tables/TABLE2_FINAL_SOURCE.csv|
