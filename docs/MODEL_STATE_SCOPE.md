# Frozen model-state scope

The final Allen VBN and CRCNS pvc-11 SCREENING execution uses a state-level registry, not one checkpoint per method. Its effective execution registry contains 493 state rows across M0-M8: 34 each for M0-M3, 85 each for M4-M7, and 17 for M8. M0 denotes MM-RVD R6. The RS Center-Out extension has a separate, dataset-specific state and seed protocol; its state identifiers must not be inferred from the Allen/CRCNS M0-M8 numbering.

Historical freeze manifests under `manifests/model_states/` document earlier selected checkpoints and hashes, but those manifests alone do not identify every state used by the final three-dataset study. In particular, a smoke-test seed is not a substitute for the frozen state policy.

Checkpoint binaries are **not** in this repository. The locally verified Allen/CRCNS execution registry references hundreds of fitted states; the directly located M0, M1 and M4-M7 files alone total approximately 4.6 GiB, and an M0 file can exceed 100 MiB. Some other registry entries denote reconstructed or composite states rather than a single portable binary. Copying a partial set of binaries would misleadingly appear to provide complete inference reproducibility. The public aggregate tables therefore provide result inspection, not raw-data-to-prediction reproduction.

The formal state registry and its hashes should be consulted alongside the split, mask, metric and aggregation freezes before any independent replay. Do not select states using SCREENING performance.
