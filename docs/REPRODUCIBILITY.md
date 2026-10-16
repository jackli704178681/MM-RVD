# Reproducibility and provenance

## Current-result boundary

Current-study aggregate results are not distributed in this repository update. The historical tables remain available for provenance but do not represent the three-dataset current study.

The Allen/CRCNS eight-method SCREENING comparison excludes the internal `M1` -Mask ablation. The RS frozen source uses `M1` for the main MM-RVD method; model IDs must not be joined across these output families without an explicit mapping. Mechanism-ablation values come from INNER, not SCREENING.

Holders of the *frozen aggregate CSVs* may use the optional local exporter:

```text
python scripts/aggregation/export_current_public_summaries.py \
  --formal-dataset <DATASET_LEVEL_RESULTS.csv> \
  --rs-unit <RS_P3_ALL_MODELS_SEVERITY_SUMMARY.csv> \
  --rs-channel <RS_P4_ALL_MODELS_CHANNEL_LOSS_SUMMARY.csv> \
  --ablation <formal_results_all_models.csv> \
  --output <LOCAL_OUTPUT_DIRECTORY>
```

The exporter checks the expected scope and writes source SHA-256 values without absolute local paths. It does not fit, infer, generate masks, inspect SCREENING responses, or recalculate metrics from per-trial labels. Its output remains local and is not part of this release. This is not a complete end-to-end independent replication. The frozen inputs and raw provider data are not distributed in this repository.

## Historical material

The earlier R4/A11 and Supplementary S1 procedures remain in `scripts/`, `configs/`, `manifests/`, `results/` and `source_data/`. They were built for a 17-session visual-data study, including older temporal/joint missingness and Five-Missing Mean/Worst endpoints. They are retained for provenance and should not be used as the current three-dataset Figure 2-5 protocol. Historical source SHA-256 listings do not cover the current three-dataset results.

Model checkpoints and raw neural datasets are not redistributed. Original datasets should be obtained from the providers listed in `docs/DATA_SOURCES.md`; provider access terms apply. No claim of online latency, prospective cross-day stability, or matched-hardware baseline speedup follows from these files.
