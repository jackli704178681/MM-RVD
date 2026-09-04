# Baseline implementation navigation

Actual baseline implementations are maintained under `src/mm_rvd/`.
This directory provides navigation only; it does not duplicate implementation code.

| Baseline | Implementation file |
|---|---|
| Mean-rate linear SVM | src/mm_rvd/baselines.py (MeanRateLinearSVM) |
| SVD64-logistic | src/mm_rvd/baselines.py (SVD64LogReg) |
| supervised CEBRA + linear classifier | src/mm_rvd/baselines.py (CEBRAFlatLogReg); src/mm_rvd/cebra_api_adapter.py; src/mm_rvd/r1_formal_executor.py (cebra_features) |
| GRU-D-inspired recurrent decoder | src/mm_rvd/fair_baseline_r1.py (build_neural_model/FormalGRUD); src/mm_rvd/baselines.py (GRULightweight for A5/A6 compatibility) |
| Lightweight TCN | src/mm_rvd/fair_baseline_r1.py (build_neural_model/FormalTCN); src/mm_rvd/baselines.py (TCNLightweight for A5/A6 compatibility) |
| Position-aware Lightweight Transformer | src/mm_rvd/fair_baseline_r1.py (build_neural_model); src/authentic_baselines/lightweight_transformer.py (LightweightTransformerDecoder) |
