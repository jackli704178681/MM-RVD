# MM-RVD

**Mask-Matched Neural Population Decoding under Recording-Unit Missingness**

MM-RVD decodes eight task classes when some *sorted recording units* are unavailable for an entire 0-0.5 s trial window. The method uses observed-unit temporal statistics, a fixed FIT-derived PCA coordinate system, and the same observation mask for the query and its FIT training reference. Class prototypes and a shared covariance/precision estimate are constructed in that matched space. It does not reconstruct missing spikes or refit PCA for each mask.

This repository holds code, protocol records, and historical result summaries. **The manuscript and current-study result tables are not included in this update.** Earlier R4/A11 and temporal-missingness materials remain for provenance; they are not the current study's main result protocol. See [method scope](docs/METHOD_SCOPE.md) before using a historical script or table.

## Current study

| Dataset | Task | Sessions | Animals | Classification trials | Channel conditions |
|---|---|---:|---:|---:|---|
| Allen Visual Behavior Neuropixels | Natural-image identity, 8 classes | 14 | 12 mice | 129,694 | RC, AC and exact-count RM |
| CRCNS pvc-11 | Grating direction, 8 classes | 3 | 3 macaques | 4,800 | RC, AC and exact-count RM |
| RS Center-Out | Reach direction, 8 classes | 1 | 1 macaque | 391 | RC and exact-count RM; no AC |

Total: **18 recording datasets/sessions, 16 animals, 134,885 classification trials**. RS has five fixed *trial-partition repeats*, not five animals. Each session is fitted and evaluated separately. Source links are in [data sources](docs/DATA_SOURCES.md).

Each response covers 25 non-overlapping 20 ms bins after the task event. The 25 bins form five 100 ms windows; each observable unit contributes a mean and population standard deviation per window, producing a fixed `10N` feature layout. An unavailable unit is unavailable in all 25 bins, not at scattered time points. Missing-unit positions use placeholders only to preserve the layout.

FIT is used for fitting and data-dependent statistics, INNER for model-state selection and mechanism ablation, and independent SCREENING for the frozen external-method comparison. The main metric is eight-class chance-normalized balanced accuracy, `(BA - 1/8) / (1 - 1/8)`. Aggregation is mask replicate, model state/seed, session, animal, then dataset; no cross-species pooled score is reported.

## Evaluation scope

- **Unit loss:** CLEAN; random units (RU); high- and low-discriminability units (HD/LD), ranked using FIT-only evidence.
- **Channel-related unit loss:** random occupied channels (RC); adjacent occupied channels (AC) only where physical geometry is verified; exact removed-unit-count random controls (RM). RS has RC/RM but no AC.
- **External comparators:** mean-rate Linear SVM, SVD64 + Logistic Regression, CEBRA, GRU-D, TCN, position-aware Transformer, and MNP-E. MM-RVD plus these seven methods form the eight-method comparison. The historical `-Mask` method is an ablation, not an eighth external comparator.
- **Mechanism ablation (INNER):** AB0 full MM-RVD; AB4 removes mask matching from the training reference; AB5 replaces covariance-aware distance with dimension-normalized squared Euclidean distance. These are not SCREENING baseline results.

## Repository map

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): scope and interpretation of the available code and historical summaries.
- [`scripts/aggregation/export_current_public_summaries.py`](scripts/aggregation/export_current_public_summaries.py): optional local exporter for holders of frozen aggregate inputs; no current result tables are published here.
- `src/`, `scripts/`, `configs/`, `manifests/`: retained implementation and historical frozen material. Their older R4/A11/S1 results do **not** represent current results.
- `source_data/figure_2/`, `source_data/figure_3/`, `results/main_summary/`, `results/ablation_summary/`: September 2026 legacy output retained for traceability, not current Figure 2-5 data.

The historical code in this repository is **not yet a self-contained end-to-end runner for all three current datasets**: some frozen state, split, mask and RS execution artifacts remain outside this public update. The documentation should not be taken as full raw-data reproducibility or evidence of prospective cross-day unit loss.

## Authors and citation

Bo Li, Qing-Zhi He, Jun-Cai Zhu, Xiao-Ke Niu, Peng Wu, Ru-Peng Zhang, Zheng Xu, Jiang-Tao Wang, and Zhi-Zhong Wang. School of Electrical and Information Engineering, Zhengzhou University, Zhengzhou, China.

The manuscript is in preparation; see `CITATION.cff` for software citation metadata. The software license is pending author confirmation (`LICENSE_PENDING.txt`).
