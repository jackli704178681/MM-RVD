# Reproducibility and provenance

## Current aggregate results

`source_data/current_20261016/` contains four selected aggregate CSVs and a source manifest with SHA-256 bindings to the frozen input aggregates. These files contain no neural response matrix, trial-level labels or predictions, split keys, or fitted state. The historical tables remain available for provenance but do not represent the three-dataset current study.

The Allen/CRCNS eight-method SCREENING comparison excludes the internal `M1` -Mask ablation. The RS frozen source uses `M1` for the main MM-RVD method; model IDs must not be joined across these output families without the mapping in the current-data README. Mechanism-ablation values come from INNER, not SCREENING.

To regenerate these summaries from locally held *frozen aggregate CSVs only*, run:

```text
python scripts/aggregation/export_current_public_summaries.py \
  --formal-dataset <DATASET_LEVEL_RESULTS.csv> \
  --rs-unit <RS_P3_ALL_MODELS_SEVERITY_SUMMARY.csv> \
  --rs-channel <RS_P4_ALL_MODELS_CHANNEL_LOSS_SUMMARY.csv> \
  --ablation <formal_results_all_models.csv> \
  --output source_data/current_20261016
```

The exporter checks the expected scope and writes source SHA-256 values without absolute local paths. It does not fit, infer, generate masks, inspect SCREENING responses, or recalculate metrics from per-trial labels. This aggregate-data export is not a complete end-to-end independent replication. The frozen input files and raw provider data are not distributed in this repository. Fitted-state binaries are also not distributed; see `docs/MODEL_STATE_SCOPE.md`.

## Historical material

The earlier R4/A11 and Supplementary S1 procedures remain in `scripts/`, `configs/`, `manifests/`, `results/` and `source_data/`. They were built for a 17-session visual-data study, including older temporal/joint missingness and Five-Missing Mean/Worst endpoints. They are retained for provenance and should not be used as the current three-dataset Figure 2-5 protocol. Historical source SHA-256 listings do not cover the new current-data export.

Model checkpoints and raw neural datasets are not redistributed. Original datasets should be obtained from the providers listed in `docs/DATA_SOURCES.md`; provider access terms apply. No claim of online latency, prospective cross-day stability, or matched-hardware baseline speedup follows from these files.
