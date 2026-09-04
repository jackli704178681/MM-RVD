from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import scripts.run_global_authentic_baseline_smoke_a5 as a5
import scripts.run_unified_17_session_authentic_rerun_a6 as a6
from src.mm_rvd.evaluator import balanced_accuracy, chance_normalized_balanced_accuracy
from src.mm_rvd.fair_baseline_r1 import (
    CONDITIONS,
    FormalTrainingSpec,
    FORMAL_R1_RETRAIN_MODELS,
    MISSING_CONDITIONS,
    REQUIRED_LOG_COLUMNS,
    build_neural_model,
    cebra_formal_config,
    environment_snapshot,
    select_best_validation_checkpoint,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


CLASS_COUNT = 8
REPLICATES = [0, 1, 2, 3, 4]
OUTPUT_SUBDIRS = [
    "checkpoints",
    "convergence_logs",
    "validation_predictions",
    "validation_metrics",
    "job_metadata",
    "failure_logs",
    "environment",
    "hashes",
]
FINAL_STATUS_VALUES = {"COMPLETE", "FAILED", "FAILED_RESOURCE", "BLOCKED"}
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def hash_path(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).replace("\\", "/").encode("utf-8"))
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        if not columns:
            columns = ["status"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_job(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["job_id"] = str(row.get("job_id") or row.get("job_key"))
    out["job_key"] = out["job_id"]
    out["seed"] = int(row["seed"])
    out["session_id"] = str(row["session_id"])
    out["status"] = str(row.get("status") or "PENDING")
    return out


def ensure_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in OUTPUT_SUBDIRS:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def validate_preflight(preflight_dir: Path) -> dict[str, Any]:
    decision = json.loads((preflight_dir / "FINAL_PREFLIGHT_V4_1_DECISION.json").read_text(encoding="utf-8"))
    expected = "PRETRAINING_CHECK_PASSED_READY_TO_START_PHASE_R1_FAIR_RETRAINING"
    if decision.get("status") != expected or decision.get("final_preflight_status") != expected:
        raise RuntimeError("R1_PREFLIGHT_STATUS_NOT_PASS")
    if decision.get("blockers") != [] or decision.get("pretraining_blockers") != []:
        raise RuntimeError("R1_PREFLIGHT_BLOCKERS_PRESENT")
    if decision.get("formal_training_started") is not False:
        raise RuntimeError("R1_PREFLIGHT_ALREADY_STARTED")
    if int(decision.get("heldout_predictions_generated", -1)) != 0:
        raise RuntimeError("R1_PREFLIGHT_HELDOUT_NOT_ZERO")
    return decision


def validate_job_matrix(job_matrix: Path) -> list[dict[str, Any]]:
    rows = [normalize_job(r) for r in read_csv_rows(job_matrix)]
    keys = [r["job_id"] for r in rows]
    if len(rows) != 340:
        raise RuntimeError(f"R1_JOB_COUNT_MISMATCH:{len(rows)}")
    if len(set(keys)) != len(keys):
        raise RuntimeError("R1_DUPLICATE_JOB_KEYS")
    models = {r["model"] for r in rows}
    if models != set(FORMAL_R1_RETRAIN_MODELS):
        raise RuntimeError(f"R1_MODEL_SET_MISMATCH:{sorted(models)}")
    if any("GPFA" in r["model"] or "SVM" in r["model"] or "SVD" in r["model"] or "MM-RVD" in r["model"] for r in rows):
        raise RuntimeError("R1_FORBIDDEN_MODEL_IN_JOB_MATRIX")
    return rows


def load_r1_specs(rebind_dir: Path) -> dict[str, FormalTrainingSpec]:
    registry = rebind_dir / "03_FINAL_RETRAIN_MODEL_REGISTRY.csv"
    if not registry.exists():
        raise RuntimeError(f"R1_RETRAIN_REGISTRY_MISSING:{registry}")
    specs: dict[str, FormalTrainingSpec] = {}
    for row in read_csv_rows(registry):
        if str(row.get("formal_R1_included")) != "True":
            continue
        model = str(row["model"])
        specs[model] = FormalTrainingSpec(
            model=model,
            model_id=str(row["model_id"]),
            max_epochs_or_iterations=int(row["max_epochs_or_iterations"]),
            validation_interval=int(row["validation_interval"]),
            patience=int(row["patience"]),
            min_delta=float(row["min_delta"]),
            seed_list=tuple(int(x.strip()) for x in str(row["seed_list"]).split(",") if x.strip()),
        )
    if set(specs) != set(FORMAL_R1_RETRAIN_MODELS):
        raise RuntimeError(f"R1_SPEC_MODEL_SET_MISMATCH:{sorted(specs)}")
    return specs


def require_cuda() -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("TORCH_NOT_AVAILABLE")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_REQUIRED_BUT_CUDA_UNAVAILABLE")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "vram_total_bytes": int(props.total_memory),
    }


def session_spec_map() -> dict[tuple[str, str], a5.SessionSpec]:
    specs: dict[tuple[str, str], a5.SessionSpec] = {}
    for spec in a6.session_specs():
        display = a6.display_dataset(spec.dataset)
        specs[(display, spec.session_id)] = spec
    return specs


def load_role(spec: a5.SessionSpec, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, trial_ids, _ = a5.load_arrays(spec)
    idx = a5.split_indices(spec, role, trial_ids)
    return x[idx].astype(np.float32), y[idx].astype(np.int64), trial_ids[idx].astype(np.int64)


def condition_arrays(
    spec: a5.SessionSpec,
    role: str,
    x: np.ndarray,
    trial_ids: np.ndarray,
    condition: str,
    replicate: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    return a5.make_observation_state(spec, x, trial_ids, role, condition, replicate)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    ba = balanced_accuracy(y_true, y_pred, CLASS_COUNT)
    return {
        "balanced_accuracy": float(ba),
        "cn_balacc": float(chance_normalized_balanced_accuracy(ba, CLASS_COUNT)),
    }


def validation_metrics_for_predictor(
    spec: a5.SessionSpec,
    job: dict[str, Any],
    x_inner: np.ndarray,
    y_inner: np.ndarray,
    trial_inner: np.ndarray,
    predict_fn,
    output_dir: Path,
    checkpoint_ref: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    by_condition: dict[str, list[float]] = {}
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep in reps:
            cx, cobs, mask_hash = condition_arrays(spec, "inner", x_inner, trial_inner, condition, rep)
            y_pred = np.asarray(predict_fn(cx, cobs), dtype=np.int64)
            met = metrics_from_predictions(y_inner, y_pred)
            by_condition.setdefault(condition, []).append(met["cn_balacc"])
            pred_path = (
                output_dir
                / "validation_predictions"
                / str(job["dataset"]).replace(" ", "_")
                / str(job["session_id"])
                / str(job["model_id"])
                / f"seed{job['seed']}"
                / Path(checkpoint_ref).stem
                / f"{condition}__rep{rep}.npz"
            )
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            if not pred_path.exists():
                np.savez_compressed(
                    pred_path,
                    data_role="validation",
                    trial_ids=trial_inner,
                    y_true=y_inner,
                    y_pred=y_pred,
                    condition=condition,
                    replicate=rep,
                    checkpoint_ref=checkpoint_ref,
                )
            rows.append({
                "job_id": job["job_id"],
                "model": job["model"],
                "model_id": job["model_id"],
                "dataset": job["dataset"],
                "session_id": job["session_id"],
                "animal_id": job["animal_id"],
                "seed": job["seed"],
                "condition": condition,
                "replicate": rep,
                "balanced_accuracy": met["balanced_accuracy"],
                "cn_balacc": met["cn_balacc"],
                "mask_hash": mask_hash,
                "checkpoint_ref": checkpoint_ref,
                "prediction_path": str(pred_path),
                "data_role": "validation",
                "status": "OK",
            })
    summary = {f"validation_{c.replace('-', '_')}": float(np.mean(vals)) for c, vals in by_condition.items()}
    missing_vals = [summary[f"validation_{c.replace('-', '_')}"] for c in MISSING_CONDITIONS]
    summary["validation_FMM"] = float(np.mean(missing_vals))
    summary["validation_FMW"] = float(np.min(missing_vals))
    return rows, summary


def append_convergence(output_dir: Path, row: dict[str, Any]) -> None:
    path = output_dir / "03_R1_TRAINING_CONVERGENCE_LONG.csv"
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_LOG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "NA") for k in REQUIRED_LOG_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp)
    torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.replace(path)
    return sha256_file(path)


def build_model_and_optimizer(model_id: str, n_units: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_neural_model(model_id, n_units, CLASS_COUNT).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer


def torch_predict(model, x: np.ndarray, obs: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device="cuda")
            ob = torch.as_tensor(obs[start:start + batch_size], dtype=torch.float32, device="cuda")
            logits = model(xb, ob)
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def train_neural_job(
    output_dir: Path,
    protocol_hashes: dict[str, str],
    spec_obj,
    spec: a5.SessionSpec,
    job: dict[str, Any],
) -> dict[str, Any]:
    x_fit, y_fit, trial_fit = load_role(spec, "fit")
    x_inner, y_inner, trial_inner = load_role(spec, "inner")
    fit_obs = np.ones_like(x_fit, dtype=np.float32)
    model, optimizer = build_model_and_optimizer(str(job["model_id"]), int(x_fit.shape[2]), int(job["seed"]))
    batch_size = 64
    best: dict[str, Any] | None = None
    early = 0
    rows_all: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(job["seed"]))
    started = time.perf_counter()
    max_epochs = int(spec_obj.max_epochs_or_iterations)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(x_fit))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(x_fit[idx], dtype=torch.float32, device="cuda")
            ob = torch.as_tensor(fit_obs[idx], dtype=torch.float32, device="cuda")
            yb = torch.as_tensor(y_fit[idx], dtype=torch.long, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb, ob), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        ckpt_path = output_dir / "checkpoints" / str(job["dataset"]).replace(" ", "_") / str(job["session_id"]) / str(job["model_id"]) / f"seed{job['seed']}" / f"checkpoint_epoch_{epoch:03d}.pt"
        ck_hash = atomic_torch_save(ckpt_path, {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_epoch": epoch,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
            "job": job,
            **protocol_hashes,
        })
        val_rows, val = validation_metrics_for_predictor(
            spec,
            job,
            x_inner,
            y_inner,
            trial_inner,
            lambda cx, co: torch_predict(model, cx, co),
            output_dir,
            str(ckpt_path),
        )
        rows_all.extend(val_rows)
        candidate = {"epoch": epoch, "checkpoint_path": str(ckpt_path), "checkpoint_sha256": ck_hash, **val}
        current_best = select_best_validation_checkpoint([best, candidate] if best else [candidate])
        is_best = current_best.get("checkpoint_path") == str(ckpt_path)
        if is_best:
            best = candidate
            early = 0
            best_path = ckpt_path.parent / "best_checkpoint.pt"
            atomic_torch_save(best_path, torch.load(ckpt_path, map_location="cpu", weights_only=False) | {"best_epoch": epoch})
        else:
            early += 1
        conv = {
            "job_id": job["job_id"],
            "model": job["model"],
            "model_id": job["model_id"],
            "dataset": job["dataset"],
            "session_id": job["session_id"],
            "animal_id": job["animal_id"],
            "seed": job["seed"],
            "training_step_type": "epoch",
            "epoch_or_iteration": epoch,
            "training_objective": float(np.mean(losses)),
            "validation_CLEAN": val["validation_CLEAN"],
            "validation_U30": val["validation_U30"],
            "validation_SW_U30": val["validation_SW_U30"],
            "validation_T5": val["validation_T5"],
            "validation_B5": val["validation_B5"],
            "validation_J30_5": val["validation_J30_5"],
            "validation_FMM": val["validation_FMM"],
            "validation_FMW": val["validation_FMW"],
            "is_best": bool(is_best),
            "early_stop_counter": early,
            "wall_time_seconds": round(time.perf_counter() - started, 4),
            "timestamp": utc_now(),
            **protocol_hashes,
        }
        append_convergence(output_dir, conv)
        write_progress(output_dir, job, epoch, max_epochs, conv, str(ckpt_path))
        if early >= int(spec_obj.patience) or should_stop(output_dir):
            break
    atomic_csv(output_dir / "validation_metrics" / f"{job['job_id']}.csv", rows_all)
    return {
        **job,
        "status": "COMPLETE",
        "selected_epoch_or_iteration": int(best["epoch"]),
        "validation_FMM": best["validation_FMM"],
        "validation_FMW": best["validation_FMW"],
        "validation_CLEAN": best["validation_CLEAN"],
        "checkpoint_path": best["checkpoint_path"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "wall_time_seconds": round(time.perf_counter() - started, 4),
    }


def cebra_features(x: np.ndarray, obs: np.ndarray, fit_mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if fit_mean is None:
        denom = np.maximum(obs.sum(axis=0), 1.0)
        fit_mean = (x * obs).sum(axis=0) / denom
    z = np.array(x, dtype=np.float32, copy=True)
    missing = obs <= 0
    z[missing] = np.broadcast_to(fit_mean, z.shape)[missing]
    return z.reshape(len(z), -1).astype(np.float32), fit_mean.astype(np.float32)


def train_cebra_job(
    output_dir: Path,
    protocol_hashes: dict[str, str],
    spec_obj,
    spec: a5.SessionSpec,
    job: dict[str, Any],
) -> dict[str, Any]:
    with warnings_as_errors_disabled():
        import cebra
    if not torch.cuda.is_available():
        raise RuntimeError("CEBRA_GPU_REQUIRED_BUT_CUDA_UNAVAILABLE")
    x_fit, y_fit, trial_fit = load_role(spec, "fit")
    x_inner, y_inner, trial_inner = load_role(spec, "inner")
    fit_obs = np.ones_like(x_fit, dtype=np.float32)
    fit_x, fit_mean = cebra_features(x_fit, fit_obs)
    cfg = cebra_formal_config(spec_obj)
    started = time.perf_counter()
    model = cebra.CEBRA(
        model_architecture="offset1-model",
        device="cuda",
        conditional="time_delta",
        max_iterations=int(cfg["max_iterations"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        output_dimension=int(cfg["output_dimension"]),
        distance=str(cfg["distance"]),
        temperature=float(cfg["temperature"]),
        verbose=False,
    )
    model.fit(fit_x, y_fit)
    fit_z = np.asarray(model.transform(fit_x), dtype=np.float32)
    scaler = StandardScaler()
    fit_zs = scaler.fit_transform(fit_z)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4, class_weight="balanced", random_state=int(job["seed"]))
    clf.fit(fit_zs, y_fit)
    ckpt_dir = output_dir / "checkpoints" / str(job["dataset"]).replace(" ", "_") / str(job["session_id"]) / str(job["model_id"]) / f"seed{job['seed']}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint_iteration_5000.pkl"
    with (ckpt_path.with_suffix(".pkl.tmp")).open("wb") as handle:
        pickle.dump({"cebra": model, "scaler": scaler, "classifier": clf, "fit_feature_mean": fit_mean, "job": job, **protocol_hashes}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    ckpt_path.with_suffix(".pkl.tmp").replace(ckpt_path)
    ck_hash = sha256_file(ckpt_path)

    def predict(cx: np.ndarray, co: np.ndarray) -> np.ndarray:
        cxf, _ = cebra_features(cx, co, fit_mean)
        z = np.asarray(model.transform(cxf), dtype=np.float32)
        return clf.predict(scaler.transform(z))

    val_rows, val = validation_metrics_for_predictor(spec, job, x_inner, y_inner, trial_inner, predict, output_dir, str(ckpt_path))
    atomic_csv(output_dir / "validation_metrics" / f"{job['job_id']}.csv", val_rows)
    conv = {
        "job_id": job["job_id"],
        "model": job["model"],
        "model_id": job["model_id"],
        "dataset": job["dataset"],
        "session_id": job["session_id"],
        "animal_id": job["animal_id"],
        "seed": job["seed"],
        "training_step_type": "iteration_block",
        "epoch_or_iteration": int(cfg["max_iterations"]),
        "training_objective": "NA",
        "validation_CLEAN": val["validation_CLEAN"],
        "validation_U30": val["validation_U30"],
        "validation_SW_U30": val["validation_SW_U30"],
        "validation_T5": val["validation_T5"],
        "validation_B5": val["validation_B5"],
        "validation_J30_5": val["validation_J30_5"],
        "validation_FMM": val["validation_FMM"],
        "validation_FMW": val["validation_FMW"],
        "is_best": True,
        "early_stop_counter": 0,
        "wall_time_seconds": round(time.perf_counter() - started, 4),
        "timestamp": utc_now(),
        **protocol_hashes,
    }
    append_convergence(output_dir, conv)
    write_progress(output_dir, job, int(cfg["max_iterations"]), int(cfg["max_iterations"]), conv, str(ckpt_path))
    return {
        **job,
        "status": "COMPLETE",
        "selected_epoch_or_iteration": int(cfg["max_iterations"]),
        "validation_FMM": val["validation_FMM"],
        "validation_FMW": val["validation_FMW"],
        "validation_CLEAN": val["validation_CLEAN"],
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ck_hash,
        "wall_time_seconds": round(time.perf_counter() - started, 4),
    }


class warnings_as_errors_disabled:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore")
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def should_stop(output_dir: Path) -> bool:
    return STOP_REQUESTED or (output_dir / "STOP_AFTER_CURRENT_JOB.flag").exists()


def write_progress(output_dir: Path, job: dict[str, Any], epoch: int, max_epoch: int, row: dict[str, Any], checkpoint_path: str) -> None:
    event = {
        "timestamp": utc_now(),
        "dataset": job["dataset"],
        "session_id": job["session_id"],
        "model_id": job["model_id"],
        "model": job["model"],
        "training_seed": job["seed"],
        "epoch": epoch,
        "max_epoch": max_epoch,
        "loss": row.get("training_objective", "NA"),
        "inner_metric": row.get("validation_FMM", "NA"),
        "best_epoch": "SEE_SELECTED_CHECKPOINTS",
        "elapsed_time": row.get("wall_time_seconds", "NA"),
        "ETA": "NA",
        "GPU_memory": torch.cuda.max_memory_allocated(0) if torch is not None and torch.cuda.is_available() else "NA",
        "checkpoint_path": checkpoint_path,
        "status": "RUNNING",
    }
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "training_progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_json(log_dir / "training_progress_latest.json", event)
    atomic_text(
        ROOT / "reports" / "mm_rvd_v1" / "R1_LIVE_TRAINING_STATUS.md",
        "# R1 live training status\n\n"
        f"- status: `{event['status']}`\n"
        f"- job: `{job['job_id']}`\n"
        f"- epoch/iteration: `{epoch}/{max_epoch}`\n"
        f"- validation FMM: `{event['inner_metric']}`\n"
        f"- checkpoint: `{checkpoint_path}`\n",
    )


def update_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_csv(output_dir / "01_R1_JOB_MANIFEST.csv", rows, ["job_key", "job_id", "model", "model_id", "dataset", "session_id", "animal_id", "seed", "status", "checkpoint_path", "checkpoint_sha256", "selected_epoch_or_iteration", "validation_FMM", "validation_FMW", "validation_CLEAN", "wall_time_seconds", "failure"])


def summarize(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    atomic_csv(output_dir / "02_R1_JOB_COMPLETENESS.csv", [{"status": k, "count": v} for k, v in sorted(counts.items())])
    selected = [r for r in rows if r.get("status") == "COMPLETE"]
    atomic_csv(output_dir / "05_R1_SELECTED_CHECKPOINTS.csv", selected)
    atomic_json(output_dir / "14_R1_HELDOUT_FIREWALL.json", {
        "heldout_loader_initialized": False,
        "heldout_predictions_generated": 0,
        "heldout_metrics_computed": False,
        "heldout_used_for_selection": False,
        "status": "PASS",
    })
    failures = [r for r in rows if r.get("status") in {"FAILED", "FAILED_RESOURCE", "BLOCKED"}]
    atomic_csv(output_dir / "11_R1_FAILURE_LOG.csv", failures)
    atomic_json(output_dir / "R1_TRAINING_DECISION.json", {
        "complete_jobs": counts.get("COMPLETE", 0),
        "failed_jobs": len(failures),
        "pending_jobs": counts.get("PENDING", 0),
        "running_jobs": counts.get("RUNNING", 0),
        "heldout_predictions_generated": 0,
        "formal_training_started": True,
        "status": "RUNNING_OR_COMPLETE" if not failures else "PHASE_R1_BLOCKED",
    })


def handle_signal(signum, frame):  # noqa: ANN001
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGINT, handle_signal)
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except Exception:
        pass
    output_dir = Path(args.output_dir)
    ensure_output(output_dir)
    preflight = validate_preflight(Path(args.preflight_dir))
    gpu = require_cuda()
    specs_by_name = load_r1_specs(Path(args.job_matrix).parent)
    jobs = validate_job_matrix(Path(args.job_matrix))
    protocol_hashes = {
        "protocol_hash": sha256_file(Path(args.protocol_dir) / "FINAL_UNIFIED_PROTOCOL_V2_1_GPFA_AMENDMENT.json") if (Path(args.protocol_dir) / "FINAL_UNIFIED_PROTOCOL_V2_1_GPFA_AMENDMENT.json").exists() else sha256_file(Path(args.protocol_dir) / "FINAL_UNIFIED_PROTOCOL_V2.json"),
        "model_spec_hash": stable_hash({k: asdict(v) for k, v in specs_by_name.items()}),
        "split_hash": preflight.get("training_split_hash", "NA"),
        "mask_bank_hash": preflight.get("mask_bank_hash", "NA"),
    }
    if args.model:
        jobs = [j for j in jobs if j["model"] == args.model or j["model_id"] == args.model]
    if args.session:
        jobs = [j for j in jobs if j["session_id"] == str(args.session)]
    if args.seed is not None:
        jobs = [j for j in jobs if int(j["seed"]) == int(args.seed)]
    existing = {r["job_id"]: r for r in read_csv_rows(output_dir / "01_R1_JOB_MANIFEST.csv")}
    merged = []
    for job in jobs:
        prior = existing.get(job["job_id"])
        merged.append({**job, **prior} if prior else job)
    update_manifest(output_dir, merged)
    atomic_json(output_dir / "environment" / "R1_ENVIRONMENT_START.json", environment_snapshot() | gpu | {"network_required_for_training": False, "offline_training_expected": True})
    atomic_json(output_dir / "R1_PROCESS_INFO.json", {
        "pid": os.getpid(),
        "start_time": utc_now(),
        "command": " ".join(sys.argv),
        "working_directory": str(ROOT),
        "stdout_log": str(output_dir / "R1_MASTER_LOG.txt"),
        "stderr_log": str(output_dir / "R1_MASTER_ERR.txt"),
        "formal_training_started": True,
        "heldout_predictions_generated": 0,
    })
    spec_map = session_spec_map()
    for idx, job in enumerate(merged):
        if job.get("status") == "COMPLETE":
            continue
        if job.get("status") in {"FAILED", "FAILED_RESOURCE", "BLOCKED"} and not args.retry_failed:
            continue
        if should_stop(output_dir):
            break
        job["status"] = "RUNNING"
        update_manifest(output_dir, merged)
        try:
            spec = spec_map[(job["dataset"], job["session_id"])]
            spec_obj = specs_by_name[job["model"]]
            if job["model_id"] == "CEBRA_FLAT_LOGREG":
                result = train_cebra_job(output_dir, protocol_hashes, spec_obj, spec, job)
            else:
                result = train_neural_job(output_dir, protocol_hashes, spec_obj, spec, job)
            job.update(result)
        except RuntimeError as exc:
            text = repr(exc)
            job["status"] = "FAILED_RESOURCE" if "out of memory" in text.lower() or "cuda" in text.lower() and "memory" in text.lower() else "FAILED"
            job["failure"] = text
            atomic_text(output_dir / "failure_logs" / f"{job['job_id']}.txt", text)
        except Exception as exc:  # pragma: no cover - long-run safety path
            job["status"] = "FAILED"
            job["failure"] = repr(exc)
            atomic_text(output_dir / "failure_logs" / f"{job['job_id']}.txt", repr(exc))
        finally:
            torch.cuda.empty_cache()
            update_manifest(output_dir, merged)
            summarize(output_dir, merged)
        if args.stop_after_current_job or should_stop(output_dir):
            break
    summarize(output_dir, merged)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-RVD R1 formal fair baseline retraining executor.")
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--job-matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stop-after-current-job", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--session")
    parser.add_argument("--seed", type=int)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
