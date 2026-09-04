# MM-RVD: Mask-matched prototype decoding under structured observation missingness in visual neural populations

## Authors
Bo Li, Qing-Zhi He, Jun-Cai Zhu, Xiao-Ke Niu, Peng Wu, Ru-Peng Zhang, Zheng Xu, Jiang-Tao Wang, and Zhi-Zhong Wang.

School of Electrical and Information Engineering, Zhengzhou University, Zhengzhou 450001, Henan, China.

Correspondence: Jiang-Tao Wang (jiangtaowang@zzu.edu.cn) and Zhi-Zhong Wang (wzz1982@zzu.edu.cn).

## Overview
MM-RVD handles structured neural observation missingness by constructing observed-only temporal statistics and a training reference space matched to the currently observable unit-time evidence. The method performs low-dimensional projection and covariance-aware prototype decoding without explicitly imputing missing neural responses or retraining a classifier for each missingness pattern.

This repository contains the code, frozen protocol records, source data, manifests and audit materials needed to reproduce the reported MM-RVD analyses. Raw neural datasets and manuscript files are not redistributed here.

## Datasets
- Allen Visual Behavior Neuropixels: 14 sessions, 12 mice, 129,694 trials.
- CRCNS pvc-11: 3 sessions, 3 macaques, 4,800 trials.
- Total: 17 sessions, 15 animals, 134,494 classification trials.

Session lists are in `manifests/sessions/`.

## Main missingness conditions
`CLEAN`, `U30`, `SW-U30`, `T5`, `B5`, and `J30-5`.

Primary endpoints include CN-BalAcc, Five-Missing Mean (FMM), and Five-Missing Worst (FMW).

## Supplementary severity analysis
Supplementary Figure S1 evaluates post-freeze severity sensitivity:

- Unit missing: 10%, 20%, 30%, 40%, and 50%.
- Contiguous temporal missing: 40, 60, 100, 140, and 200 ms.

The S1 audit status is preserved as: preflight PASS, anchor replay PASS, full sweep PASS, 3927/3927 tasks DONE, and 3927/3927 raw result rows OK.

## Repository structure
- `src/` and `scripts/`: implementation and reproduction entry points.
- `configs/`: frozen protocol and model configuration files.
- `manifests/`: session, split, mask and selected-state manifests.
- `source_data/`: source CSVs for tables, Figure 2, Figure 3 and Supplementary Figure S1.
- `results/`: main, ablation and supplementary summary outputs.
- `audit/`: convergence, model-state, aggregation and release audits.
- `docs/`: data source, reproducibility, file-manifest and method-scope notes.
- `checksums/`: SHA256 manifest for released files.

## Environment
The submitted analyses were run with Python 3.10.20, PyTorch 2.5.1, CUDA 12.4, scikit-learn 1.7.2 and CEBRA 0.6.1 on Windows 11 Professional 64-bit with an Intel Core i7-14700KF CPU, 128 GB RAM and an NVIDIA GeForce RTX 4070 Ti SUPER GPU with 16 GB GPU memory.

## Data access
The raw Allen VBN and CRCNS pvc-11 datasets must be downloaded from their official providers. See `docs/DATA_SOURCES.md`.

## Reproducibility
Start with `docs/REPRODUCIBILITY.md`. Supplementary Figure S1 can be redrawn from frozen CSV source data without rerunning the 3927 severity inference jobs.

## Citation
See `CITATION.cff`.

## License
The software license is pending author confirmation. See `LICENSE_PENDING.txt`.
