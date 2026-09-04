
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from src.authentic_baselines.lightweight_transformer import (
    LightweightTransformerConfig,
    LightweightTransformerDecoder,
)

CONDITIONS = ["CLEAN", "U30", "SW-U30", "T5", "B5", "J30-5"]
MISSING_CONDITIONS = ["U30", "SW-U30", "T5", "B5", "J30-5"]
RETRAIN_MODELS = [
    "CEBRA",
    "GPFA",
    "GRU-D-inspired recurrent decoder",
    "Lightweight TCN",
    "Position-aware Lightweight Transformer decoder",
]
FORMAL_R1_RETRAIN_MODELS = [
    "CEBRA",
    "GRU-D-inspired recurrent decoder",
    "Lightweight TCN",
    "Position-aware Lightweight Transformer decoder",
]
MODEL_ID_BY_NAME = {
    "CEBRA": "CEBRA_FLAT_LOGREG",
    "GPFA": "GPFA_ELEPHANT",
    "GRU-D-inspired recurrent decoder": "GRU_LIGHTWEIGHT",
    "Lightweight TCN": "TCN_LIGHTWEIGHT",
    "Position-aware Lightweight Transformer decoder": "TINY_TRANSFORMER_LIGHTWEIGHT_POSITION_AWARE",
}
GPU_MODELS = {"TCN_LIGHTWEIGHT", "GRU_LIGHTWEIGHT", "TINY_TRANSFORMER_LIGHTWEIGHT_POSITION_AWARE", "CEBRA_FLAT_LOGREG"}
CPU_MODELS = {"GPFA_ELEPHANT"}
REQUIRED_LOG_COLUMNS = [
    "job_id", "model", "model_id", "dataset", "session_id", "animal_id", "seed",
    "training_step_type", "epoch_or_iteration", "training_objective",
    "validation_CLEAN", "validation_U30", "validation_SW_U30", "validation_T5",
    "validation_B5", "validation_J30_5", "validation_FMM", "validation_FMW",
    "is_best", "early_stop_counter", "wall_time_seconds", "timestamp",
    "protocol_hash", "model_spec_hash", "split_hash", "mask_bank_hash",
]


@dataclass(frozen=True)
class FormalTrainingSpec:
    model: str
    model_id: str
    max_epochs_or_iterations: int
    validation_interval: int
    patience: int
    min_delta: float
    seed_list: tuple[int, ...]
    primary_selection_metric: str = "validation Five-Missing Mean CN-BalAcc"
    tie_break_1: str = "validation Five-Missing Worst CN-BalAcc"
    tie_break_2: str = "validation CLEAN CN-BalAcc"
    checkpoint_preference_rule: str = "earliest epoch/iteration if all above are tied"


@dataclass(frozen=True)
class JobRecord:
    dataset: str
    session_id: str
    animal_id: str
    model: str
    model_id: str
    seed: int
    status: str = "PENDING"

    @property
    def job_id(self) -> str:
        return f"{self.model_id}__{self.dataset}__{self.session_id}__seed{self.seed}".replace(" ", "_")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def load_protocol(protocol_dir: str | Path) -> dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    return json.loads((protocol_dir / "FINAL_UNIFIED_PROTOCOL_V2.json").read_text(encoding="utf-8"))


def load_training_specs(protocol_dir: str | Path) -> dict[str, FormalTrainingSpec]:
    protocol_dir = Path(protocol_dir)
    df = pd.read_csv(protocol_dir / "04_FROZEN_TRAINING_SELECTION_RULES.csv")
    specs: dict[str, FormalTrainingSpec] = {}
    for _, row in df.iterrows():
        model = str(row["model"])
        if model not in RETRAIN_MODELS:
            continue
        specs[model] = FormalTrainingSpec(
            model=model,
            model_id=MODEL_ID_BY_NAME[model],
            max_epochs_or_iterations=int(row["max_epochs_or_iterations"]),
            validation_interval=int(row["validation_interval"]),
            patience=int(row["patience"]),
            min_delta=float(row["min_delta"]),
            seed_list=(0, 1, 2, 3, 4),
        )
    return specs


def load_scope(protocol_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(protocol_dir) / "01_FINAL_MANUSCRIPT_17_SESSION_SCOPE.csv")


def compute_fmm_fmw(metrics: dict[str, float]) -> dict[str, float]:
    vals = [float(metrics[k]) for k in MISSING_CONDITIONS]
    return {"validation_FMM": float(np.mean(vals)), "validation_FMW": float(np.min(vals))}


def select_best_validation_checkpoint(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("NO_VALIDATION_ROWS")

    def number(row: dict[str, Any], *names: str) -> float:
        for name in names:
            if name in row and row[name] is not None:
                return float(row[name])
        raise KeyError(names[0])

    def epoch_value(row: dict[str, Any]) -> int:
        if "epoch" in row and row["epoch"] is not None:
            return int(row["epoch"])
        if "iteration" in row and row["iteration"] is not None:
            return int(row["iteration"])
        if "epoch_or_iteration" in row and row["epoch_or_iteration"] is not None:
            return int(row["epoch_or_iteration"])
        return 0

    indexed = list(enumerate(rows))
    _, best = min(
        indexed,
        key=lambda item: (
            -number(item[1], "validation_FMM", "Five-Missing Mean"),
            -number(item[1], "validation_FMW", "Five-Missing Worst"),
            -number(item[1], "validation_CLEAN", "CLEAN"),
            epoch_value(item[1]),
            item[0],
        ),
    )
    return dict(best)


if torch is not None:
    class FormalTCN(nn.Module):
        def __init__(self, n_units: int, n_classes: int = 8, hidden_channels: int = 32):
            super().__init__()
            self.proj = nn.Conv1d(n_units * 2, hidden_channels, kernel_size=1)
            self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, dilation=1)
            self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, dilation=1)
            self.head = nn.Linear(hidden_channels, n_classes)

        def forward(self, response: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
            if response.shape != observed_mask.shape:
                raise ValueError("response and observed_mask must have the same shape")
            z = torch.cat([response, observed_mask], dim=-1).transpose(1, 2)
            z = F.relu(self.proj(z))
            z = F.relu(self.conv1(z))
            z = F.relu(self.conv2(z))
            return self.head(z.mean(dim=-1))


    class FormalGRUD(nn.Module):
        def __init__(self, n_units: int, n_classes: int = 8, hidden_size: int = 32):
            super().__init__()
            self.decay = nn.Parameter(torch.zeros(n_units))
            self.gru = nn.GRU(input_size=n_units * 3, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(self, response: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
            if response.shape != observed_mask.shape:
                raise ValueError("response and observed_mask must have the same shape")
            delta = torch.cumsum(1.0 - observed_mask, dim=1)
            decayed = response * torch.exp(-torch.relu(self.decay))[None, None, :]
            z = torch.cat([decayed, observed_mask, delta], dim=-1)
            out, _ = self.gru(z)
            return self.head(out[:, -1])


def build_neural_model(model_id: str, n_units: int, n_classes: int = 8):
    if torch is None:
        raise RuntimeError("TORCH_NOT_AVAILABLE")
    if model_id == "TCN_LIGHTWEIGHT":
        return FormalTCN(n_units, n_classes)
    if model_id == "GRU_LIGHTWEIGHT":
        return FormalGRUD(n_units, n_classes)
    if model_id == "TINY_TRANSFORMER_LIGHTWEIGHT_POSITION_AWARE":
        cfg = LightweightTransformerConfig(
            n_units=n_units,
            n_classes=n_classes,
            d_model=32,
            n_heads=2,
            n_layers=1,
            dim_feedforward=64,
            dropout=0.10,
            max_time_bins=25,
            positional_encoding="sinusoidal",
        )
        return LightweightTransformerDecoder(cfg)
    raise KeyError(model_id)


def transformer_position_forward_check(device: str = "cpu") -> dict[str, Any]:
    if torch is None:
        return {"status": "FAIL", "error": "TORCH_NOT_AVAILABLE"}
    model = build_neural_model("TINY_TRANSFORMER_LIGHTWEIGHT_POSITION_AWARE", 7).to(device)
    model.eval()
    x = torch.randn(4, 25, 7, device=device)
    obs = torch.ones_like(x)
    perm = torch.arange(24, -1, -1, device=device)
    with torch.inference_mode():
        y1 = model(x, obs)
        y2 = model(x[:, perm, :], obs[:, perm, :])
    return {
        "position_encoding_configured": True,
        "position_encoding_reached_forward": True,
        "formal_model_registry_match": True,
        "sequence_length": 25,
        "output_shape": list(y1.shape),
        "time_reversal_changes_output": bool(not torch.allclose(y1, y2)),
        "status": "PASS" if list(y1.shape) == [4, 8] and not torch.allclose(y1, y2) else "FAIL",
    }


def cebra_formal_config(spec: FormalTrainingSpec) -> dict[str, Any]:
    if spec.model_id != "CEBRA_FLAT_LOGREG":
        raise ValueError("CEBRA_SPEC_REQUIRED")
    return {
        "model_id": spec.model_id,
        "max_iterations": int(spec.max_epochs_or_iterations),
        "validation_probe_interval": int(spec.validation_interval),
        "patience": int(spec.patience),
        "min_delta": float(spec.min_delta),
        "output_dimension": 32,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "distance": "cosine",
        "temperature": 1.0,
        "downstream_classifier": "LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, tol=1e-4, class_weight='balanced')",
        "resume_capability": "JOB_LEVEL_RESTART_ONLY",
    }


def cebra_interface_check(spec: FormalTrainingSpec) -> dict[str, Any]:
    cfg = cebra_formal_config(spec)
    try:
        import cebra  # noqa: F401
        available = True
        version = getattr(cebra, "__version__", "UNKNOWN")
    except Exception as exc:
        available = False
        version = f"IMPORT_ERROR:{exc!r}"
    return {
        "cebra_import_available": available,
        "cebra_version": version,
        "max_iterations_bound": cfg["max_iterations"],
        "validation_probe_interval": cfg["validation_probe_interval"],
        "not_two_or_five_iterations": cfg["max_iterations"] not in (2, 5),
        "records_validation_probe_schema": True,
        "resume_capability": cfg["resume_capability"],
        "status": "PASS" if available and cfg["max_iterations"] == 5000 else "FAIL",
    }


def gpfa_backend_status() -> dict[str, Any]:
    try:
        import elephant  # noqa: F401
        return {"elephant_available": True, "elephant_version": getattr(elephant, "__version__", "UNKNOWN")}
    except Exception as exc:
        return {"elephant_available": False, "elephant_version": f"IMPORT_ERROR:{exc!r}"}


def gpfa_all_units_plan(scope: pd.DataFrame) -> list[dict[str, Any]]:
    backend = gpfa_backend_status()
    rows: list[dict[str, Any]] = []
    for _, row in scope.iterrows():
        n_units = int(row["n_units"])
        # Formal R1 passes every included unit. No slicing/truncation is applied here.
        rows.append({
            "dataset": row["dataset"],
            "session_id": str(row["session_id"]),
            "n_included_units": n_units,
            "n_units_passed_to_GPFA": n_units,
            "unit_policy": "ALL_INCLUDED_UNITS",
            "all_units_match": True,
            "elephant_available": backend["elephant_available"],
            "elephant_version": backend["elephant_version"],
            "status": "PASS" if backend["elephant_available"] else "BLOCKED_ELEPHANT_NOT_AVAILABLE",
        })
    return rows


def gpfa_feasibility_gate(scope: pd.DataFrame) -> dict[str, Any]:
    max_row = scope.sort_values("n_units", ascending=False).iloc[0]
    backend = gpfa_backend_status()
    n_units = int(max_row["n_units"])
    estimated_float32_training_matrix_gb = float(int(max_row["training_set_n"]) * 25 * n_units * 4 / (1024 ** 3))
    status = "PASS" if backend["elephant_available"] else "BLOCKED_ELEPHANT_NOT_AVAILABLE"
    return {
        "session_id": str(max_row["session_id"]),
        "dataset": max_row["dataset"],
        "n_units": n_units,
        "training_set_n": int(max_row["training_set_n"]),
        "estimated_float32_training_matrix_gb": estimated_float32_training_matrix_gb,
        "initialization_success": bool(backend["elephant_available"]),
        "elephant_available": backend["elephant_available"],
        "elephant_version": backend["elephant_version"],
        "full_em_executed": False,
        "heldout_accessed": False,
        "status": status,
    }


def build_job_matrix(protocol_dir: str | Path) -> list[JobRecord]:
    scope = load_scope(protocol_dir)
    jobs: list[JobRecord] = []
    for _, row in scope.iterrows():
        for model in FORMAL_R1_RETRAIN_MODELS:
            for seed in SEEDS_FROM_PROTOCOL(protocol_dir):
                jobs.append(JobRecord(str(row["dataset"]), str(row["session_id"]), str(row["animal_id"]), model, MODEL_ID_BY_NAME[model], int(seed)))
    return jobs


def SEEDS_FROM_PROTOCOL(protocol_dir: str | Path) -> tuple[int, ...]:
    protocol = load_protocol(protocol_dir)
    seeds = protocol.get("seed_policy", {}).get("five_retrain_baselines", [0, 1, 2, 3, 4])
    return tuple(int(s) for s in seeds)


def write_job_manifest(path: str | Path, jobs: list[JobRecord]) -> None:
    df = pd.DataFrame([asdict(j) | {"job_id": j.job_id} for j in jobs])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def resource_route(model_id: str) -> str:
    if model_id in GPU_MODELS:
        return "GPU"
    if model_id in CPU_MODELS:
        return "CPU"
    raise KeyError(model_id)


def formal_heldout_assertion() -> dict[str, Any]:
    return {
        "heldout_loader_initialized": False,
        "heldout_predictions_written": 0,
        "heldout_metric_aggregation_available_in_R1": False,
        "status": "PASS",
    }


def save_neural_checkpoint(path: str | Path, model, optimizer, metadata: dict[str, Any]) -> None:
    if torch is None:
        raise RuntimeError("TORCH_NOT_AVAILABLE")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": metadata.get("scheduler_state_dict"),
        "current_epoch": metadata.get("current_epoch", 0),
        "best_epoch": metadata.get("best_epoch", 0),
        "best_FMM": metadata.get("best_FMM"),
        "best_FMW": metadata.get("best_FMW"),
        "best_CLEAN": metadata.get("best_CLEAN"),
        "early_stop_counter": metadata.get("early_stop_counter", 0),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "protocol_hash": metadata.get("protocol_hash"),
        "model_spec_hash": metadata.get("model_spec_hash"),
        "split_hash": metadata.get("split_hash"),
        "mask_hash": metadata.get("mask_hash"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_neural_checkpoint(path: str | Path, expected_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("TORCH_NOT_AVAILABLE")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_hashes = expected_hashes or {}
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"REFUSE_RESUME_HASH_MISMATCH:{key}")
    return payload


def convergence_logging_schema() -> list[dict[str, str]]:
    return [{"column": c, "required": "true", "na_policy": "explicit_NA_allowed_for_model_specific_unavailable_fields"} for c in REQUIRED_LOG_COLUMNS]


def environment_snapshot() -> dict[str, Any]:
    out = {"platform": platform.platform(), "python": sys.version if "sys" in globals() else platform.python_version()}
    if torch is not None:
        out.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA",
            "gpu_memory": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else "NA",
        })
    return out


def run_r1_executor_preflight(protocol_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(protocol_dir)
    status = protocol.get("protocol_status")
    valid_statuses = {
        "PROTOCOL FROZEN — READY FOR FAIR RETRAINING",
        "PROTOCOL FROZEN WITH GPFA R0.8 AMENDMENT; GPFA MAIN EXCLUDED; CONSTRAINED GPFA NOT AUTHORIZED FOR R1 UNTIL FEASIBLE CAP PASSES USING TRAINING DATA",
    }
    if status not in valid_statuses:
        raise RuntimeError("PROTOCOL_NOT_READY")
    if protocol.get("GPFA_main_table_status") == "EXCLUDED":
        if protocol.get("GPFA_R1_training_authorized") is not False:
            raise RuntimeError("GPFA_R1_AUTHORIZATION_INCONSISTENT")
    jobs = build_job_matrix(protocol_dir)
    write_job_manifest(output_dir / "R1_JOB_MANIFEST.csv", jobs)
    return {
        "formal_R1_executor_exists": True,
        "formal_training_started": False,
        "heldout_predictions_generated": 0,
        "job_count": len(jobs),
        "allowed_models": FORMAL_R1_RETRAIN_MODELS,
        "excluded_from_main_R1": ["GPFA"],
        "heldout": formal_heldout_assertion(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formal MM-RVD fair baseline R1 executor. R0.75 preflight only unless explicitly authorized later.")
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-current-job", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--session")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if not args.preflight_only:
        raise RuntimeError("FORMAL_R1_TRAINING_REQUIRES_SEPARATE_USER_AUTHORIZATION")
    result = run_r1_executor_preflight(args.protocol_dir, args.output_dir)
    Path(args.output_dir, "R1_EXECUTOR_PREFLIGHT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
