# MM-RVD: Mask-matched prototype decoding under structured observation missingness in visual neural populations

## Authors
Bo Li, Qing-Zhi He, Jun-Cai Zhu, Xiao-Ke Niu, Peng Wu, Ru-Peng Zhang, Zheng Xu, Jiang-Tao Wang, and Zhi-Zhong Wang.

School of Electrical and Information Engineering, Zhengzhou University, Zhengzhou 450001, Henan, China.

Correspondence: Jiang-Tao Wang (jiangtaowang@zzu.edu.cn) and Zhi-Zhong Wang (wzz1982@zzu.edu.cn).

## Overview

Neural population decoders rely on a consistent set of recorded units and time-resolved neural responses. In practical electrophysiological recordings, however, some neural units may become unavailable and parts of the response time course may be missing because of recording instability, unit loss, channel failure, or transient data-quality problems. These structured observation losses change the neural evidence available to a decoder and can cause a mismatch between the representation learned during training and the signals available at test time.

MM-RVD is a neural population decoding framework designed for this setting. Instead of reconstructing or imputing missing neural responses, it builds neural features only from the observations that remain available and matches the training reference representation to the same observable unit-time structure. Decoding is then performed in a low-dimensional population representation using class prototypes and covariance-aware distances.

The method is evaluated on two visual neural population datasets from mice and macaques under controlled unit-wise, temporal, and joint observation-missing conditions. This repository contains the implementation, frozen experimental protocols, baseline models, manifests, source data, evaluation outputs, and reproducibility audits corresponding to the reported analyses. Raw neural datasets and manuscript files are not redistributed.

## Datasets

### Allen Visual Behavior Neuropixels

The Allen Visual Behavior Neuropixels dataset used in this study comprises 14 recording sessions from 12 mice, with 129,694 classification trials.

Official data access:

- [Allen Visual Behavior Neuropixels dataset](https://allenswdb.github.io/physiology/ephys/visual-behavior/VBN-Dataset.html)
- [Allen Visual Behavior Neuropixels session data](https://allenswdb.github.io/physiology/ephys/visual-behavior/VBN-SessionData.html)
- [AllenSDK documentation](https://alleninstitute.github.io/AllenSDK/)
- [Allen Brain Observatory Open Data registry](https://registry.opendata.aws/allen-brain-observatory/)

The data can be accessed programmatically through the AllenSDK `VisualBehaviorNeuropixelsProjectCache`. The exact 14 session IDs used in this study are provided in `manifests/sessions/` and `docs/DATA_SOURCES.md`.

### CRCNS pvc-11

The CRCNS pvc-11 dataset used in this study comprises 3 recording sessions from 3 macaques, with 4,800 classification trials.

Official data access:

- [CRCNS pvc-11 dataset page](https://crcns.org/data-sets/vc/pvc-11/about/)
- [CRCNS pvc-11 NERSC download](https://portal.nersc.gov/project/crcns/download/pvc-11)

The three sessions used in this study are `data_monkey1_gratings`, `data_monkey2_gratings`, and `data_monkey3_gratings`. A CRCNS account may be required by the data provider. Exact dataset and session information is also provided in `manifests/sessions/` and `docs/DATA_SOURCES.md`.

### Analysis scope

Across the two datasets, the reported analyses include 17 recording sessions from 15 animals and 134,494 classification trials.

Raw neural datasets are not redistributed in this repository. Users should obtain the original data directly from the official providers using the links above.

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
