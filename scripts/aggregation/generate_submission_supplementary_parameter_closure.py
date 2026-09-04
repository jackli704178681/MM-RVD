from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path("E:/ENTO_code")
OUT = ROOT / "submission_supplementary_parameter_closure_20260817"

A6 = ROOT / "unified_17_session_authentic_rerun_a6_20260808_185006"
A9 = ROOT / "mmrvd_secondgen_development_a9_20260814_171047"
A11 = ROOT / "mmrvd_r6_authentic_ablation_a11_20260816_065146"
GPFA_PY = ROOT / "latent_baseline_environment_closure_a4_work/venv_gpfa/Scripts/python.exe"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]], cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cols is None:
        cols = []
        for row in rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gpfa_environment() -> dict[str, str]:
    env = {
        "python": "UNRESOLVED",
        "elephant": "UNRESOLVED",
        "neo": "UNRESOLVED",
        "quantities": "UNRESOLVED",
        "numpy": "UNRESOLVED",
        "scipy": "UNRESOLVED",
        "sklearn": "UNRESOLVED",
    }
    if not GPFA_PY.exists():
        return env
    code = (
        "import sys\n"
        "print('python='+sys.version.split()[0])\n"
        "for m in ['elephant','neo','quantities','numpy','scipy','sklearn']:\n"
        "    try:\n"
        "        mod=__import__(m); print(m+'='+getattr(mod,'__version__','UNKNOWN'))\n"
        "    except Exception:\n"
        "        print(m+'=UNRESOLVED')\n"
    )
    proc = subprocess.run([str(GPFA_PY), "-c", code], text=True, capture_output=True, timeout=30)
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def generate() -> dict[str, str | int]:
    OUT.mkdir(parents=True, exist_ok=True)
    a6_jobs = pd.read_csv(A6 / "03_training/A6_JOB_LEDGER.csv")
    a6_s2 = pd.read_csv(A6 / "11_tables/SUPPLEMENTARY_TABLE_S2_AUTHENTIC_BASELINES_A6.csv")
    a11_b5 = pd.read_csv(A11 / "05_b5_compute_policy/A11_B5_FALLBACK_AUDIT.csv")
    a11_policy = json.loads((A11 / "05_b5_compute_policy/A11_B5_COMPUTE_POLICY_LOCK.json").read_text(encoding="utf-8"))
    a9_cfg = json.loads((A9 / "10_variant_selection/FINAL_MMRVD_SECONDGEN_CONFIG_A9.json").read_text(encoding="utf-8"))
    gpfa_env = gpfa_environment()

    job_summary = (
        a6_jobs.groupby("final_name")["parameter_count"]
        .agg(lambda x: ";".join(map(str, sorted(set(int(v) for v in x.dropna())))))
        .to_dict()
    )

    common_input = "n_trials x 25 time bins x n_units, session-specific n_units"
    train_select = "single frozen configuration; internal validation metric recorded; selected state reused for evaluation"

    s2_cols = [
        "model",
        "model_family",
        "input_shape",
        "input_features",
        "standardization_or_imputation",
        "core_architecture",
        "layers_or_blocks",
        "channels_or_hidden_size",
        "kernel_size",
        "dilation",
        "padding",
        "activation",
        "dropout",
        "normalization",
        "pooling_or_readout",
        "classifier_or_scoring",
        "C",
        "kernel",
        "penalty",
        "loss",
        "class_weight",
        "tol",
        "max_iter",
        "random_state",
        "solver",
        "svd_n_components",
        "rank_insufficient_rule",
        "svd_random_state",
        "embedding_or_latent_dimension",
        "attention_heads",
        "ffn_dimension",
        "positional_encoding",
        "mask_input_mode",
        "decay_mechanism",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs_or_max_steps",
        "early_stopping",
        "model_selection",
        "training_loss",
        "gradient_clipping",
        "device_policy",
        "parameter_count_recorded",
        "status",
    ]
    s2: list[dict[str, str]] = [
        {
            "model": "MM-RVD",
            "model_family": "prototype/covariance geometry method",
            "input_shape": common_input,
            "input_features": "log1p neural response; unit-level training-set standardization; five 100-ms temporal mean/std statistics",
            "standardization_or_imputation": "training-set unit mean/std; observed-only statistics for missing observations",
            "core_architecture": "temporal statistics + low-dimensional projection + class prototypes + covariance-based scoring",
            "layers_or_blocks": "N/A",
            "channels_or_hidden_size": "N/A",
            "kernel_size": "N/A",
            "dilation": "N/A",
            "padding": "N/A",
            "activation": "N/A",
            "dropout": "N/A",
            "normalization": "training-set standardization",
            "pooling_or_readout": "prototype/covariance scoring",
            "classifier_or_scoring": "Mahalanobis/prototype score over class prototypes",
            "C": "N/A",
            "kernel": "N/A",
            "penalty": "N/A",
            "loss": "N/A",
            "class_weight": "N/A",
            "tol": "N/A",
            "max_iter": "N/A",
            "random_state": "1306 for projection where applicable",
            "solver": "randomized PCA for projection where applicable",
            "svd_n_components": "N/A",
            "rank_insufficient_rule": "projection dimension limited by available rank",
            "svd_random_state": "N/A",
            "embedding_or_latent_dimension": "temporal projection requested 256; effective dimension rank-limited",
            "attention_heads": "N/A",
            "ffn_dimension": "N/A",
            "positional_encoding": "N/A",
            "mask_input_mode": "observed mask used in feature construction and mask-conditioned reference states",
            "decay_mechanism": "N/A",
            "optimizer": "N/A",
            "learning_rate": "N/A",
            "weight_decay": "N/A",
            "batch_size": "N/A",
            "epochs_or_max_steps": "N/A",
            "early_stopping": "N/A",
            "model_selection": "selected using training and internal validation sets only",
            "training_loss": "N/A",
            "gradient_clipping": "N/A",
            "device_policy": "CPU numerical pipeline unless array operations use library defaults",
            "parameter_count_recorded": "N/A",
            "status": "VERIFIED",
        },
        {
            "model": "Mean-rate linear SVM",
            "model_family": "linear baseline",
            "input_shape": common_input,
            "input_features": "mean response over observed time bins per unit; all-missing unit/time entries filled by training-set unit mean",
            "standardization_or_imputation": "StandardScaler before LinearSVC; imputation from training-set feature/unit means",
            "core_architecture": "linear support-vector classifier",
            "layers_or_blocks": "N/A",
            "channels_or_hidden_size": "N/A",
            "kernel_size": "N/A",
            "dilation": "N/A",
            "padding": "N/A",
            "activation": "N/A",
            "dropout": "N/A",
            "normalization": "StandardScaler with_mean=True with_std=True",
            "pooling_or_readout": "mean-rate vector",
            "classifier_or_scoring": "sklearn LinearSVC",
            "C": "1.0",
            "kernel": "linear",
            "penalty": "l2",
            "loss": "squared_hinge",
            "class_weight": "None",
            "tol": "0.0001",
            "max_iter": "5000",
            "random_state": "run seed",
            "solver": "LinearSVC dual=auto, multi_class=ovr",
            "svd_n_components": "N/A",
            "rank_insufficient_rule": "N/A",
            "svd_random_state": "N/A",
            "embedding_or_latent_dimension": "N/A",
            "attention_heads": "N/A",
            "ffn_dimension": "N/A",
            "positional_encoding": "N/A",
            "mask_input_mode": "observed mask used to compute mean-rate features",
            "decay_mechanism": "N/A",
            "optimizer": "liblinear LinearSVC optimizer",
            "learning_rate": "N/A",
            "weight_decay": "N/A",
            "batch_size": "N/A",
            "epochs_or_max_steps": "N/A",
            "early_stopping": "N/A",
            "model_selection": train_select,
            "training_loss": "squared hinge",
            "gradient_clipping": "N/A",
            "device_policy": "CPU sklearn",
            "parameter_count_recorded": job_summary.get("Mean-rate linear SVM", "0"),
            "status": "VERIFIED",
        },
        {
            "model": "SVD64-logistic",
            "model_family": "linear dimensionality-reduction baseline",
            "input_shape": common_input,
            "input_features": "flattened imputed trial x time x unit response",
            "standardization_or_imputation": "feature imputation by training-set feature means; StandardScaler before TruncatedSVD",
            "core_architecture": "TruncatedSVD followed by multinomial logistic readout",
            "layers_or_blocks": "N/A",
            "channels_or_hidden_size": "N/A",
            "kernel_size": "N/A",
            "dilation": "N/A",
            "padding": "N/A",
            "activation": "N/A",
            "dropout": "N/A",
            "normalization": "StandardScaler with_mean=True with_std=True",
            "pooling_or_readout": "SVD embedding + logistic classifier",
            "classifier_or_scoring": "sklearn LogisticRegression",
            "C": "1.0",
            "kernel": "N/A",
            "penalty": "default sklearn logistic penalty recorded as deprecated/default l2-compatible",
            "loss": "logistic loss",
            "class_weight": "None",
            "tol": "0.0001",
            "max_iter": "200",
            "random_state": "run seed",
            "solver": "lbfgs",
            "svd_n_components": "64 requested",
            "rank_insufficient_rule": "min(64, n_training_samples - 1, n_features - 1), lower bounded at 1",
            "svd_random_state": "run seed",
            "embedding_or_latent_dimension": "up to 64",
            "attention_heads": "N/A",
            "ffn_dimension": "N/A",
            "positional_encoding": "N/A",
            "mask_input_mode": "observed mask used for training-set imputation before flattening",
            "decay_mechanism": "N/A",
            "optimizer": "lbfgs",
            "learning_rate": "N/A",
            "weight_decay": "N/A",
            "batch_size": "N/A",
            "epochs_or_max_steps": "max_iter=200 for logistic; SVD n_iter=5",
            "early_stopping": "N/A",
            "model_selection": train_select,
            "training_loss": "logistic loss",
            "gradient_clipping": "N/A",
            "device_policy": "CPU sklearn",
            "parameter_count_recorded": job_summary.get("SVD64-logistic", "0"),
            "status": "VERIFIED",
        },
    ]
    s2.extend(
        [
            {
                "model": "Lightweight TCN",
                "model_family": "supervised temporal neural baseline",
                "input_shape": common_input,
                "input_features": "imputed response concatenated with observation mask along feature axis",
                "standardization_or_imputation": "training-set feature mean imputation",
                "core_architecture": "1x1 projection convolution, two temporal convolutions, temporal mean readout",
                "layers_or_blocks": "3 Conv1d layers total: projection, conv1, conv2",
                "channels_or_hidden_size": "32 hidden channels",
                "kernel_size": "projection 1; temporal convolutions 3",
                "dilation": "1 for temporal convolutions",
                "padding": "1 for temporal convolutions",
                "activation": "ReLU",
                "dropout": "0.0",
                "normalization": "N/A",
                "pooling_or_readout": "mean over time then linear head",
                "classifier_or_scoring": "argmax logits",
                "C": "N/A",
                "kernel": "N/A",
                "penalty": "N/A",
                "loss": "cross entropy",
                "class_weight": "N/A",
                "tol": "N/A",
                "max_iter": "N/A",
                "random_state": "run seed via torch manual seed",
                "solver": "AdamW",
                "svd_n_components": "N/A",
                "rank_insufficient_rule": "N/A",
                "svd_random_state": "N/A",
                "embedding_or_latent_dimension": "N/A",
                "attention_heads": "N/A",
                "ffn_dimension": "N/A",
                "positional_encoding": "N/A",
                "mask_input_mode": "observation mask concatenated as input channels",
                "decay_mechanism": "N/A",
                "optimizer": "AdamW",
                "learning_rate": "0.001",
                "weight_decay": "0.01 optimizer default",
                "batch_size": "full selected training subset tensor",
                "epochs_or_max_steps": "2 optimizer steps",
                "early_stopping": "N/A",
                "model_selection": train_select,
                "training_loss": "cross entropy on training subset",
                "gradient_clipping": "N/A",
                "device_policy": "CUDA if available, otherwise CPU according to saved code path",
                "parameter_count_recorded": job_summary.get("Lightweight TCN", "UNRESOLVED"),
                "status": "VERIFIED",
            },
            {
                "model": "GRU-D-inspired recurrent decoder",
                "model_family": "supervised recurrent neural baseline",
                "input_shape": common_input,
                "input_features": "decayed imputed response, observation mask, cumulative missingness delta",
                "standardization_or_imputation": "training-set feature mean imputation",
                "core_architecture": "learned per-unit decay + one-layer GRU + linear head",
                "layers_or_blocks": "1 GRU layer",
                "channels_or_hidden_size": "32 hidden units",
                "kernel_size": "N/A",
                "dilation": "N/A",
                "padding": "N/A",
                "activation": "GRU internal gates",
                "dropout": "0.0",
                "normalization": "N/A",
                "pooling_or_readout": "last time step hidden state",
                "classifier_or_scoring": "argmax logits",
                "C": "N/A",
                "kernel": "N/A",
                "penalty": "N/A",
                "loss": "cross entropy",
                "class_weight": "N/A",
                "tol": "N/A",
                "max_iter": "N/A",
                "random_state": "run seed via torch manual seed",
                "solver": "AdamW",
                "svd_n_components": "N/A",
                "rank_insufficient_rule": "N/A",
                "svd_random_state": "N/A",
                "embedding_or_latent_dimension": "N/A",
                "attention_heads": "N/A",
                "ffn_dimension": "N/A",
                "positional_encoding": "N/A",
                "mask_input_mode": "mask concatenated with response and cumulative missingness delta",
                "decay_mechanism": "learned per-unit nonnegative exponential decay applied to response",
                "optimizer": "AdamW",
                "learning_rate": "0.001",
                "weight_decay": "0.01 optimizer default",
                "batch_size": "full selected training subset tensor",
                "epochs_or_max_steps": "2 optimizer steps",
                "early_stopping": "N/A",
                "model_selection": train_select,
                "training_loss": "cross entropy on training subset",
                "gradient_clipping": "N/A",
                "device_policy": "CUDA if available, otherwise CPU according to saved code path",
                "parameter_count_recorded": job_summary.get("GRU-D-inspired recurrent decoder", "UNRESOLVED"),
                "status": "VERIFIED",
            },
            {
                "model": "Lightweight Transformer decoder",
                "model_family": "supervised transformer neural baseline",
                "input_shape": common_input,
                "input_features": "imputed response concatenated with observation mask",
                "standardization_or_imputation": "training-set feature mean imputation",
                "core_architecture": "linear input embedding + one TransformerEncoder layer + mean pooling + linear head",
                "layers_or_blocks": "1 TransformerEncoder layer",
                "channels_or_hidden_size": "embedding dimension 32",
                "kernel_size": "N/A",
                "dilation": "N/A",
                "padding": "N/A",
                "activation": "default TransformerEncoderLayer activation (ReLU)",
                "dropout": "0.1 default TransformerEncoderLayer dropout",
                "normalization": "Transformer encoder layer normalization",
                "pooling_or_readout": "mean pooling over time then linear head",
                "classifier_or_scoring": "argmax logits",
                "C": "N/A",
                "kernel": "N/A",
                "penalty": "N/A",
                "loss": "cross entropy",
                "class_weight": "N/A",
                "tol": "N/A",
                "max_iter": "N/A",
                "random_state": "run seed via torch manual seed",
                "solver": "AdamW",
                "svd_n_components": "N/A",
                "rank_insufficient_rule": "N/A",
                "svd_random_state": "N/A",
                "embedding_or_latent_dimension": "32",
                "attention_heads": "2",
                "ffn_dimension": "64",
                "positional_encoding": "none explicitly implemented",
                "mask_input_mode": "observation mask concatenated as input features; no padding mask used",
                "decay_mechanism": "N/A",
                "optimizer": "AdamW",
                "learning_rate": "0.001",
                "weight_decay": "0.01 optimizer default",
                "batch_size": "full selected training subset tensor",
                "epochs_or_max_steps": "2 optimizer steps",
                "early_stopping": "N/A",
                "model_selection": train_select,
                "training_loss": "cross entropy on training subset",
                "gradient_clipping": "N/A",
                "device_policy": "CUDA if available, otherwise CPU according to saved code path",
                "parameter_count_recorded": job_summary.get("Lightweight Transformer decoder", "UNRESOLVED"),
                "status": "VERIFIED",
            },
        ]
    )
    s2.extend(
        [
            {
                "model": "GPFA",
                "model_family": "latent representation baseline",
                "input_shape": "selected training trials x 25 time bins x up to 16 units",
                "input_features": "first min(16, n_units) units, binned spike counts represented as Elephant SpikeTrain objects",
                "standardization_or_imputation": "missing inputs are zeroed before GPFA worker receives arrays; no evaluation refit",
                "core_architecture": "Elephant GPFA latent state model followed by logistic classifier",
                "layers_or_blocks": "N/A",
                "channels_or_hidden_size": "N/A",
                "kernel_size": "N/A",
                "dilation": "N/A",
                "padding": "N/A",
                "activation": "N/A",
                "dropout": "N/A",
                "normalization": "N/A",
                "pooling_or_readout": "latent trajectory flattened per trial",
                "classifier_or_scoring": "sklearn LogisticRegression",
                "C": "1.0",
                "kernel": "N/A",
                "penalty": "default sklearn logistic penalty",
                "loss": "logistic loss",
                "class_weight": "None",
                "tol": "0.0001 for downstream logistic; GPFA tolerance UNRESOLVED",
                "max_iter": "downstream logistic 200; GPFA EM max iterations 2",
                "random_state": "downstream logistic random_state=0; training subset seed is run-dependent",
                "solver": "Elephant GPFA EM + sklearn lbfgs logistic",
                "svd_n_components": "N/A",
                "rank_insufficient_rule": "N/A",
                "svd_random_state": "N/A",
                "embedding_or_latent_dimension": "3",
                "attention_heads": "N/A",
                "ffn_dimension": "N/A",
                "positional_encoding": "N/A",
                "mask_input_mode": "masked response entries zeroed before worker transform",
                "decay_mechanism": "N/A",
                "optimizer": "Elephant GPFA EM",
                "learning_rate": "N/A",
                "weight_decay": "N/A",
                "batch_size": "N/A",
                "epochs_or_max_steps": "EM max iterations 2",
                "early_stopping": "N/A",
                "model_selection": train_select,
                "training_loss": "GPFA likelihood + downstream logistic loss",
                "gradient_clipping": "N/A",
                "device_policy": "isolated CPU Python environment",
                "parameter_count_recorded": job_summary.get("GPFA", "0"),
                "status": "PARTIAL",
            },
            {
                "model": "CEBRA",
                "model_family": "contrastive representation baseline",
                "input_shape": common_input,
                "input_features": "flattened imputed trial x time x unit response",
                "standardization_or_imputation": "training-set feature mean imputation",
                "core_architecture": "CEBRA offset1-model with time-delta conditioning, followed by logistic classifier",
                "layers_or_blocks": "CEBRA package model architecture offset1-model",
                "channels_or_hidden_size": "32 hidden units",
                "kernel_size": "N/A",
                "dilation": "N/A",
                "padding": "package default pad_before_transform=True",
                "activation": "package internal",
                "dropout": "UNRESOLVED",
                "normalization": "N/A beyond imputation",
                "pooling_or_readout": "CEBRA embedding",
                "classifier_or_scoring": "sklearn LogisticRegression",
                "C": "1.0",
                "kernel": "N/A",
                "penalty": "default sklearn logistic penalty",
                "loss": "InfoNCE for embedding; logistic loss for readout",
                "class_weight": "None",
                "tol": "0.0001 for downstream logistic",
                "max_iter": "CEBRA max_iterations=2; downstream logistic max_iter=200",
                "random_state": "downstream logistic run seed; CEBRA random state not explicitly recorded",
                "solver": "CEBRA optimizer adam; downstream lbfgs logistic",
                "svd_n_components": "N/A",
                "rank_insufficient_rule": "N/A",
                "svd_random_state": "N/A",
                "embedding_or_latent_dimension": "8",
                "attention_heads": "N/A",
                "ffn_dimension": "N/A",
                "positional_encoding": "time_offsets=1",
                "mask_input_mode": "observed mask used for imputation before flattening",
                "decay_mechanism": "N/A",
                "optimizer": "adam, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False",
                "learning_rate": "0.0003",
                "weight_decay": "0",
                "batch_size": "16",
                "epochs_or_max_steps": "max_iterations=2",
                "early_stopping": "N/A",
                "model_selection": train_select,
                "training_loss": "InfoNCE criterion; distance=cosine; temperature=1.0 constant; min_temperature=0.1",
                "gradient_clipping": "UNRESOLVED",
                "device_policy": "cuda if available in package config",
                "parameter_count_recorded": job_summary.get("CEBRA", "0"),
                "status": "PARTIAL",
            },
        ]
    )
    write_csv(OUT / "01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv", s2, s2_cols)

    main_agg = (
        "mask replicate -> session x run-seed x condition -> session x run-seed "
        "Five-Missing Mean/Worst -> run-seed -> session -> animal -> dataset"
    )
    abl_agg = (
        "mask replicate -> session x run-seed x condition -> run-seed aggregation within session -> "
        "session-level condition -> session-level Five-Missing Mean/Worst -> animal -> dataset"
    )
    s3_cols = [
        "condition",
        "missing_axis",
        "intensity",
        "sampling_rule",
        "mask_generation_granularity",
        "whole_unit_persistent",
        "all_units_share_time_missing",
        "special_formula_or_composition",
        "mask_replicates",
        "seed_rule",
        "shared_across_models",
        "shared_across_ablations",
        "main_benchmark_aggregation",
        "ablation_aggregation",
        "status",
    ]
    s3 = [
        {
            "condition": "CLEAN",
            "missing_axis": "none",
            "intensity": "identity",
            "sampling_rule": "no mask generated; complete observed input",
            "mask_generation_granularity": "N/A",
            "whole_unit_persistent": "N/A",
            "all_units_share_time_missing": "N/A",
            "special_formula_or_composition": "N/A",
            "mask_replicates": "1 identity evaluation",
            "seed_rule": "N/A",
            "shared_across_models": "yes",
            "shared_across_ablations": "yes",
            "main_benchmark_aggregation": main_agg,
            "ablation_aggregation": abl_agg,
            "status": "VERIFIED",
        },
        {
            "condition": "U30",
            "missing_axis": "unit",
            "intensity": "30% units",
            "sampling_rule": "random whole-unit missing indices from frozen mask bank",
            "mask_generation_granularity": "session, split role, replicate",
            "whole_unit_persistent": "yes, same missing units across all time bins for that evaluation replicate",
            "all_units_share_time_missing": "N/A",
            "special_formula_or_composition": "missing unit count uses round-half-up constrained to at least one observed unit",
            "mask_replicates": "5 for evaluation",
            "seed_rule": "deterministic frozen mask-bank seed derivation",
            "shared_across_models": "yes",
            "shared_across_ablations": "yes",
            "main_benchmark_aggregation": main_agg,
            "ablation_aggregation": abl_agg,
            "status": "VERIFIED",
        },
        {
            "condition": "SW-U30",
            "missing_axis": "unit",
            "intensity": "30% units",
            "sampling_rule": "stability-weighted whole-unit missing from training-set-only unit stability statistics",
            "mask_generation_granularity": "session, split role, replicate",
            "whole_unit_persistent": "yes",
            "all_units_share_time_missing": "N/A",
            "special_formula_or_composition": "sampling weight = 1.0 + 3.0 x normalized_rank; epsilon=1e-6; no label access",
            "mask_replicates": "5 for evaluation",
            "seed_rule": "deterministic frozen mask-bank seed derivation",
            "shared_across_models": "yes",
            "shared_across_ablations": "yes",
            "main_benchmark_aggregation": main_agg,
            "ablation_aggregation": abl_agg,
            "status": "VERIFIED",
        },
        {
            "condition": "T5",
            "missing_axis": "time",
            "intensity": "5 bins",
            "sampling_rule": "one contiguous 5-bin time window per trial/replicate from frozen mask bank",
            "mask_generation_granularity": "session, split role, replicate, trial",
            "whole_unit_persistent": "N/A",
            "all_units_share_time_missing": "yes, selected time bins are removed for all units",
            "special_formula_or_composition": "25-bin input length preserved; missing bins are marked unobserved",
            "mask_replicates": "5 for evaluation",
            "seed_rule": "deterministic frozen mask-bank seed derivation",
            "shared_across_models": "yes",
            "shared_across_ablations": "yes",
            "main_benchmark_aggregation": main_agg,
            "ablation_aggregation": abl_agg,
            "status": "VERIFIED",
        },
        {
            "condition": "B5",
            "missing_axis": "time",
            "intensity": "5 bins total",
            "sampling_rule": "two non-overlapping time fragments per trial/replicate from frozen mask bank",
            "mask_generation_granularity": "session, split role, replicate, trial",
            "whole_unit_persistent": "N/A",
            "all_units_share_time_missing": "yes",
            "special_formula_or_composition": "B5 = one 2-bin segment plus one 3-bin non-overlapping segment",
            "mask_replicates": "5 for evaluation",
            "seed_rule": "deterministic frozen mask-bank seed derivation",
            "shared_across_models": "yes",
            "shared_across_ablations": "yes",
            "main_benchmark_aggregation": main_agg,
            "ablation_aggregation": abl_agg,
            "status": "VERIFIED",
        },
        {
            "condition": "J30-5",
            "missing_axis": "joint unit and time",
            "intensity": "30% units + 5 time bins",
            "sampling_rule": "combines whole-unit U30 component and contiguous T5 component from frozen mask bank",
            "mask_generation_granularity": "session, split role, replicate; time component also trial-specific",
            "whole_unit_persistent": "yes for unit component",
            "all_units_share_time_missing": "yes for time component",
            "special_formula_or_composition": "unit and time components are applied jointly to the same input; input length remains 25 bins",
            "mask_replicates": "5 for evaluation",
            "seed_rule": "deterministic frozen mask-bank seed derivation",
            "shared_across_models": "yes",
            "shared_across_ablations": "yes",
            "main_benchmark_aggregation": main_agg,
            "ablation_aggregation": abl_agg,
            "status": "VERIFIED",
        },
    ]
    write_csv(OUT / "02_FINAL_SUPP_TABLE_S3_MISSINGNESS_EVALUATION_PROTOCOL.csv", s3, s3_cols)

    base_unchanged = (
        "same datasets, same splits, same missingness conditions, same run seeds, same evaluation metrics, same class labels"
    )
    s4_cols = [
        "public_setting",
        "target_design",
        "temporal_representation",
        "observed_only_statistics",
        "validity_semantics",
        "mask_conditioning",
        "training_sample_pool",
        "standardization",
        "PCA",
        "prototype",
        "covariance",
        "classifier",
        "changed_components",
        "unchanged_components",
        "interpretation_scope",
    ]
    s4 = [
        {
            "public_setting": "时间结构",
            "target_design": "test contribution of temporally resolved statistics",
            "temporal_representation": "replaced temporally resolved five-window statistics with global response statistics",
            "observed_only_statistics": "preserved where applicable",
            "validity_semantics": "preserved where applicable",
            "mask_conditioning": "preserved",
            "training_sample_pool": "unchanged full training set",
            "standardization": "unchanged training-set standardization",
            "PCA": "unchanged when applicable",
            "prototype": "unchanged",
            "covariance": "unchanged",
            "classifier": "unchanged prototype/covariance scoring",
            "changed_components": "temporal representation",
            "unchanged_components": base_unchanged,
            "interpretation_scope": "internal validation component attribution only",
        },
        {
            "public_setting": "低维投影",
            "target_design": "test contribution of low-dimensional projection",
            "temporal_representation": "unchanged temporally resolved statistics",
            "observed_only_statistics": "preserved",
            "validity_semantics": "preserved",
            "mask_conditioning": "preserved",
            "training_sample_pool": "unchanged full training set",
            "standardization": "unchanged",
            "PCA": "removed; high-dimensional feature space used subject to recorded covariance policy",
            "prototype": "unchanged",
            "covariance": "uses diagonal high-dimensional policy when exact covariance is not computationally valid",
            "classifier": "unchanged scoring family",
            "changed_components": "projection and resulting covariance geometry",
            "unchanged_components": base_unchanged,
            "interpretation_scope": "internal validation component attribution only",
        },
        {
            "public_setting": "协方差几何",
            "target_design": "test contribution of covariance-aware geometry",
            "temporal_representation": "unchanged",
            "observed_only_statistics": "preserved",
            "validity_semantics": "preserved",
            "mask_conditioning": "preserved",
            "training_sample_pool": "unchanged full training set",
            "standardization": "unchanged",
            "PCA": "unchanged",
            "prototype": "unchanged",
            "covariance": "replaced covariance-aware precision with diagonal predecessor geometry",
            "classifier": "prototype scoring retained",
            "changed_components": "covariance estimator / geometry",
            "unchanged_components": base_unchanged,
            "interpretation_scope": "internal validation component attribution only",
        },
        {
            "public_setting": "掩码条件化",
            "target_design": "test contribution of mask-conditioned reference states",
            "temporal_representation": "unchanged",
            "observed_only_statistics": "preserved",
            "validity_semantics": "preserved",
            "mask_conditioning": "removed; single global reference state used",
            "training_sample_pool": "unchanged full training set",
            "standardization": "unchanged",
            "PCA": "unchanged",
            "prototype": "prototypes computed without condition-specific reference masks",
            "covariance": "same estimator family",
            "classifier": "unchanged scoring family",
            "changed_components": "mask-conditioned reference state construction",
            "unchanged_components": base_unchanged,
            "interpretation_scope": "internal validation component attribution only",
        },
        {
            "public_setting": "零填充全bin统计替代",
            "target_design": "test observed-only statistics and validity semantics",
            "temporal_representation": "changed: all bins are included after zeroing missing entries",
            "observed_only_statistics": "removed",
            "validity_semantics": "changed: missing-value inclusion changes which dimensions are considered valid",
            "mask_conditioning": "preserved where applicable",
            "training_sample_pool": "unchanged full training set",
            "standardization": "unchanged training-set standardization before masked zero inclusion",
            "PCA": "unchanged when applicable",
            "prototype": "changed through altered temporal feature semantics",
            "covariance": "same estimator family applied to altered features",
            "classifier": "unchanged scoring family",
            "changed_components": "missing-value inclusion; temporal statistics; validity semantics; training-reference temporal feature semantics",
            "unchanged_components": "same datasets, splits, masks, run seeds, scoring family and endpoint definitions",
            "interpretation_scope": "internal validation component attribution only; not a denominator-only ablation",
        },
        {
            "public_setting": "受限训练样本状态估计",
            "target_design": "test state-estimation sample-pool size",
            "temporal_representation": "unchanged",
            "observed_only_statistics": "preserved",
            "validity_semantics": "preserved",
            "mask_conditioning": "preserved",
            "training_sample_pool": "restricted to at most 640 class-balanced training samples",
            "standardization": "fit on restricted sample pool",
            "PCA": "fit on restricted sample pool",
            "prototype": "estimated from restricted sample pool",
            "covariance": "estimated from restricted sample pool",
            "classifier": "unchanged scoring family",
            "changed_components": "training sample pool for standardization, projection, prototypes and covariance",
            "unchanged_components": base_unchanged,
            "interpretation_scope": "internal validation component attribution only",
        },
    ]
    write_csv(OUT / "03_FINAL_SUPP_TABLE_S4_ABLATION_IMPLEMENTATION.csv", s4, s4_cols)

    env_rows: list[dict[str, str]] = []

    def env(item: str, value: str, status: str, source: str, notes: str = "") -> None:
        env_rows.append(
            {
                "item": item,
                "value": value,
                "verification_status": status,
                "evidence_source": source,
                "notes": notes,
            }
        )

    env("OS", "win32", "VERIFIED", "retrospective benchmark environment log", "operating-system family only; full Windows build not recorded in experiment logs")
    env("Python main environment", "3.14.2", "VERIFIED", "baseline and development environment logs", "used for main sklearn/PyTorch/CEBRA/MM-RVD runs")
    env("Python GPFA environment", gpfa_env["python"], "VERIFIED" if gpfa_env["python"] != "UNRESOLVED" else "UNRESOLVED", "isolated GPFA environment probe", "read-only probe of saved environment")
    env("NumPy main environment", "2.4.3", "VERIFIED", "development environment log", "main MM-RVD/development logs")
    env("pandas main environment", "2.3.3", "VERIFIED", "development environment log", "main MM-RVD/development logs")
    env("scikit-learn main environment", "1.8.0", "VERIFIED", "development/retrospective benchmark environment log", "main sklearn pipeline")
    env("PyTorch main environment", "2.10.0+cu130", "VERIFIED", "baseline environment matrix", "baseline neural models")
    env("CUDA availability", "available", "VERIFIED", "baseline environment matrix", "exact GPU model not recorded in experiment environment files")
    env("CUDA version", "PyTorch build cu130", "VERIFIED", "baseline environment matrix", "reported via PyTorch build string, not full driver log")
    env("cuDNN", "UNRESOLVED", "UNRESOLVED", "no experiment log found", "not filled from current audit runtime")
    env("CEBRA", "0.6.0", "VERIFIED", "baseline environment matrix", "CEBRA baseline")
    env("Elephant", gpfa_env["elephant"], "VERIFIED" if gpfa_env["elephant"] != "UNRESOLVED" else "UNRESOLVED", "isolated GPFA environment probe", "GPFA baseline")
    env("neo", gpfa_env["neo"], "VERIFIED" if gpfa_env["neo"] != "UNRESOLVED" else "UNRESOLVED", "isolated GPFA environment probe", "GPFA dependency")
    env("quantities", gpfa_env["quantities"], "VERIFIED" if gpfa_env["quantities"] != "UNRESOLVED" else "UNRESOLVED", "isolated GPFA environment probe", "GPFA dependency")
    env("CPU", "UNRESOLVED", "UNRESOLVED", "no experiment log found", "not filled from current audit runtime")
    env("RAM", "UNRESOLVED", "UNRESOLVED", "no experiment log found", "not filled from current audit runtime")
    env("GPU", "UNRESOLVED", "UNRESOLVED", "no experiment log found", "not filled from current audit runtime")
    env("GPU memory", "UNRESOLVED", "UNRESOLVED", "no experiment log found", "not filled from current audit runtime")
    write_csv(OUT / "04_FINAL_SUPP_TABLE_S5_SOFTWARE_HARDWARE.csv", env_rows)

    unresolved: list[dict[str, str]] = []

    def unr(table: str, scope: str, parameter: str, reason: str) -> None:
        unresolved.append(
            {
                "table": table,
                "model_or_scope": scope,
                "parameter": parameter,
                "reason": reason,
                "recommended_action": "Leave as UNRESOLVED unless original experiment metadata is found",
            }
        )

    unr("S2", "GPFA", "GPFA tolerance", "Elephant GPFA fitted metadata and worker code record EM max iterations but no explicit tolerance value")
    unr("S2", "CEBRA", "explicit CEBRA random state", "CEBRA object/config did not record an explicit random_state field")
    unr("S2", "CEBRA", "gradient clipping", "No CEBRA gradient clipping setting recorded in config or fitted object")
    unr("S2", "CEBRA", "dropout", "No dropout parameter exposed in recorded CEBRA configuration")
    unr("S5", "experiment hardware", "CPU", "No formal experiment environment log with CPU model found")
    unr("S5", "experiment hardware", "RAM", "No formal experiment environment log with system memory found")
    unr("S5", "experiment hardware", "GPU", "No formal experiment environment log with GPU model found")
    unr("S5", "experiment hardware", "GPU memory", "No formal experiment environment log with GPU memory found")
    unr("S5", "experiment software", "cuDNN", "No formal experiment environment log with cuDNN version found")
    write_csv(OUT / "06_FINAL_UNRESOLVED_PARAMETERS.csv", unresolved)

    blockers = 0
    s2_status = "PARTIAL" if any(r["table"] == "S2" for r in unresolved) else "PASS"
    s3_status = "PASS"
    s4_status = "PASS"
    s5_status = "PARTIAL" if any(r["table"] == "S5" for r in unresolved) else "PASS"
    final_status = "READY_WITH_UNRESOLVED" if unresolved and blockers == 0 else ("BLOCKED" if blockers else "READY")

    audit_md = f"""# MM-RVD Final Supplementary Parameter Closure

Created: {datetime.now().isoformat(timespec='seconds')}

## Scope

This closure did not train models, rerun inference, modify the manuscript, modify result files, or modify model code. It only generated a new supplementary parameter-closure directory.

## Internal Provenance Used

- Baseline parameters: `unified_17_session_authentic_rerun_a6_20260808_185006`, `src/mm_rvd/baselines.py`, and `scripts/run_unified_17_session_authentic_rerun_a6.py`.
- Current MM-RVD implementation facts: `mmrvd_secondgen_development_a9_20260814_171047` and `mmrvd_r6_authentic_ablation_a11_20260816_065146`.
- Missingness protocol: `src/mm_rvd/phase2b_mask_bank.py`, `scripts/run_global_authentic_baseline_smoke_a5.py`, and the locked ablation evaluator.
- Environment: run logs from the above directories plus a read-only GPFA virtual-environment probe.

## Public Terminology Sanitization

The public CSV tables avoid internal phase labels, internal split labels, historical model names, and internal candidate IDs. Public split terminology is restricted to training set, internal validation set, and evaluation set.

## Completion Status

S2_COMPLETE = {s2_status}  
S3_COMPLETE = {s3_status}  
S4_COMPLETE = {s4_status}  
S5_EXPERIMENT_ENVIRONMENT = {s5_status}  
UNRESOLVED_PARAMETERS = {len(unresolved)}  
REAL_BLOCKERS = {blockers}  
FINAL_SUPPLEMENTARY_PARAMETER_STATUS = {final_status}

## Notes

S2 remains PARTIAL because several package-level or environment-specific parameters were not explicitly recorded in the formal artifacts. S5 remains PARTIAL because CPU, RAM, GPU model, GPU memory, and cuDNN were not found in actual experiment logs and were not backfilled from the current audit runtime.

## Output Hashes

"""
    for name in [
        "01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv",
        "02_FINAL_SUPP_TABLE_S3_MISSINGNESS_EVALUATION_PROTOCOL.csv",
        "03_FINAL_SUPP_TABLE_S4_ABLATION_IMPLEMENTATION.csv",
        "04_FINAL_SUPP_TABLE_S5_SOFTWARE_HARDWARE.csv",
        "06_FINAL_UNRESOLVED_PARAMETERS.csv",
    ]:
        audit_md += f"- `{name}`: `{sha(OUT / name)}`\n"
    (OUT / "05_FINAL_SUPPLEMENTARY_PARAMETER_AUDIT.md").write_text(audit_md, encoding="utf-8")

    forbidden = ["A6", "A9", "A9R", "A11", "FIT", "INNER", "SCREENING", "R6", "fallback", "历史模型内部名称"]
    public_files = [
        OUT / "01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv",
        OUT / "02_FINAL_SUPP_TABLE_S3_MISSINGNESS_EVALUATION_PROTOCOL.csv",
        OUT / "03_FINAL_SUPP_TABLE_S4_ABLATION_IMPLEMENTATION.csv",
        OUT / "04_FINAL_SUPP_TABLE_S5_SOFTWARE_HARDWARE.csv",
    ]
    violations = []
    for path in public_files:
        text = path.read_text(encoding="utf-8-sig")
        for term in forbidden:
            if term in text:
                violations.append({"file": path.name, "term": term})
    if violations:
        raise RuntimeError(f"FORBIDDEN_PUBLIC_TERMS {json.dumps(violations, ensure_ascii=False)}")
    for path in OUT.glob("*.csv"):
        text = path.read_text(encoding="utf-8-sig")
        if "nan" in text.lower():
            raise RuntimeError(f"NAN_FOUND {path.name}")
    for name in [
        "01_FINAL_SUPP_TABLE_S2_MODEL_TRAINING_CONFIG.csv",
        "02_FINAL_SUPP_TABLE_S3_MISSINGNESS_EVALUATION_PROTOCOL.csv",
        "03_FINAL_SUPP_TABLE_S4_ABLATION_IMPLEMENTATION.csv",
        "04_FINAL_SUPP_TABLE_S5_SOFTWARE_HARDWARE.csv",
        "05_FINAL_SUPPLEMENTARY_PARAMETER_AUDIT.md",
        "06_FINAL_UNRESOLVED_PARAMETERS.csv",
    ]:
        path = OUT / name
        if not path.exists() or path.stat().st_size <= 50:
            raise RuntimeError(f"OUTPUT_MISSING_OR_EMPTY {name}")
    return {
        "output_dir": str(OUT),
        "S2_COMPLETE": s2_status,
        "S3_COMPLETE": s3_status,
        "S4_COMPLETE": s4_status,
        "S5_EXPERIMENT_ENVIRONMENT": s5_status,
        "UNRESOLVED_PARAMETERS": len(unresolved),
        "REAL_BLOCKERS": blockers,
        "FINAL_SUPPLEMENTARY_PARAMETER_STATUS": final_status,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, ensure_ascii=False))
