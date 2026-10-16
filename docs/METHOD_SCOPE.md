# Method and result scope

## Current 2026-10-16 study

The current model is the A9-selected MM-RVD R6 path: FIT-only unit standardization, five 100 ms observed-only temporal mean/standard-deviation windows, FIT-derived fixed PCA, query-mask-matched FIT training reference, class prototypes, pooled shrinkage covariance and covariance-aware discrimination. The current missingness task is **sorted recording-unit unavailability for the complete 0-0.5 s window**. It is not random time-bin deletion, raw electrode-voltage failure, or missing-signal reconstruction.

Current datasets: Allen VBN (14 sessions/12 mice), CRCNS pvc-11 (3 sessions/3 macaques), and RS Center-Out (one macaque, five split repeats). AC requires verified physical channel geometry and was not evaluated for RS. RC/AC and their RM controls match the *number of removed units*, not the fraction of selected channels. FIT, INNER and SCREENING have distinct roles. AB4 and AB5 are INNER mechanism ablations and are not external baselines.

Current-study result tables are not part of this code-and-documentation update. The old `source_data/figure_2/`, `source_data/figure_3/`, `results/main_summary/`, `results/ablation_summary/` and Supplementary S1 files represent an earlier 17-session, partly different missingness protocol. They must not be silently relabeled as current Figure 2-5 evidence.

## Reproducibility boundary

The public repository contains method source and earlier frozen records, but not all current frozen model states, masks, trial splits, RS execution adapters, current result tables or raw provider datasets. It does not by itself reproduce all formal predictions from raw data. See `docs/REPRODUCIBILITY.md`.
