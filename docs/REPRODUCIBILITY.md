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
