# Current three-dataset aggregate results

These tables are derived from existing, frozen **aggregate** files. They contain no neural response matrix, trial-level label/prediction, split key, or fitted model. `source_manifest.json` records the SHA-256 of each input aggregate, not a link to a raw participant record.

| File | Scope | Rows |
|---|---|---:|
| `screening_allen_crcns_dataset.csv` | Independent SCREENING, Allen VBN and CRCNS pvc-11, MM-RVD plus seven external comparators | 592 |
| `screening_rs_unit_loss.csv` | Independent SCREENING, RS CLEAN/RU/HD/LD, eight methods, five fixed partition repeats summarized | 152 |
| `screening_rs_rc_rm.csv` | Independent SCREENING, RS RC and exact-count RM, eight methods; **no AC** | 32 |
| `inner_ablation_dataset.csv` | INNER AB0/AB4/AB5, three datasets; conditions differ by dataset | 78 |

**ID warning:** Allen/CRCNS formal `M0` is MM-RVD and `M1` is the internal -Mask ablation. This export excludes `M1` from the eight-method comparison. RS P3/P4 sources use `M1` for MM-RVD and `M2`-`M8` for the seven external comparators. Thus model IDs across these two families are **not directly joinable**; use names or the scope-specific mapping. AB0/AB4/AB5 are separate INNER identifiers.

RS's `n_splits=5` measures partition sensitivity for **one macaque**, not five animal replicates. Allen and CRCNS dataset-level rows were aggregated at the animal level in the frozen formal protocol; the exporter reads those final rows and does not re-aggregate trials. CN-BalAcc is the eight-class chance-normalized balanced accuracy, `(BA - 0.125) / 0.875`. Negative values, if present, are retained.

For the study's figure numbering, Figure 2/Table 2 concern the eight-method comparison; Figure 3/Table 3 concern RU/HD/LD; Figure 4/Table 4 concern RC/AC versus exact-count RM; and Figure 5/Table 5 concern AB4/AB5 mechanism ablations. The files above provide aggregate inputs for those analyses, not finished figures.

The September R4/A11 and S1 results elsewhere in this repository are historical and address a different set of missingness conditions. Do not combine their Figure/Table numbers with the current study's Figure 2-5 numbering.
