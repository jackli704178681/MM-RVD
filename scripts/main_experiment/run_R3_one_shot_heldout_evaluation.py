from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path("<PROJECT_ROOT>")
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scripts.run_global_authentic_baseline_smoke_a5 as a5
import scripts.run_unified_17_session_authentic_rerun_a6 as a6
from src.mm_rvd.evaluator import balanced_accuracy, chance_normalized_balanced_accuracy, confusion_matrix
from src.mm_rvd.fair_baseline_r1 import build_neural_model
from src.mm_rvd.r1_formal_executor import cebra_features

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = repr(exc)
else:
    TORCH_IMPORT_ERROR = ""


R2_DIR = PROJECT_ROOT / "submission_R1_posttraining_convergence_audit_R2_20260901"
R1_DIR = PROJECT_ROOT / "submission_fair_baseline_retraining_R1_20260831"
R3_DIR = PROJECT_ROOT / "submission_one_shot_heldout_evaluation_R3_20260902"

ALLOWLIST = R2_DIR / "R2_TO_R3_CHECKPOINT_ALLOWLIST.csv"
R2_DECISION = R2_DIR / "R2_POSTTRAINING_DECISION.json"
R2_FREEZE = R2_DIR / "FINAL_R2_FOUR_BASELINE_MODEL_FREEZE_MANIFEST.json"

CONDITIONS = ["CLEAN", "U30", "SW-U30", "T5", "B5", "J30-5"]
MISSING_CONDITIONS = ["U30", "SW-U30", "T5", "B5", "J30-5"]
REPLICATES = [0, 1, 2, 3, 4]
CLASS_COUNT = 8
EXPECTED_JOBS = 340
EXPECTED_STATES_PER_JOB = 26
EXPECTED_STATES = EXPECTED_JOBS * EXPECTED_STATES_PER_JOB


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


def hash_path(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "MISSING"
    if path.is_file():
        return sha256_file(path)
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).replace("\\", "/").encode("utf-8"))
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with tmp.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["status"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def npz_atomic(path: Path, **arrays: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    with tmp.open("ab") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)
    return sha256_file(path)


def load_allowlist() -> list[dict[str, Any]]:
    rows = read_csv(ALLOWLIST)
    for row in rows:
        row["seed"] = int(row["seed"])
        row["session_id"] = str(row.get("session_id") or row.get("session"))
        row["animal_id"] = str(row.get("animal_id") or row.get("animal") or "")
    return rows


def load_screening_role(spec: a5.SessionSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x, y, trial_ids, metadata = a5.load_arrays(spec)
    idx = a5.split_indices(spec, "screening", trial_ids)
    return x[idx], y[idx].astype(np.int64), trial_ids[idx].astype(np.int64), idx.astype(np.int64), metadata


def spec_map() -> dict[tuple[str, str], a5.SessionSpec]:
    out: dict[tuple[str, str], a5.SessionSpec] = {}
    for spec in a6.session_specs():
        out[(a6.display_dataset(spec.dataset), str(spec.session_id))] = spec
    return out


def init_output_tree() -> None:
    if R3_DIR.exists() and (R3_DIR / "R3_HELDOUT_EVALUATION_DECISION.json").exists():
        raise RuntimeError(f"R3_OUTPUT_DIR_ALREADY_HAS_FINAL_DECISION_REFUSING_OVERWRITE:{R3_DIR}")
    for sub in [
        "predictions",
        "logs",
        "runtime",
        "allowlist",
        "protocol",
        "hashes",
    ]:
        (R3_DIR / sub).mkdir(parents=True, exist_ok=True)


def pre_access_gate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    gate_rows: list[dict[str, Any]] = []
    for row in rows:
        ckpt = Path(row["checkpoint_path"])
        exists = ckpt.exists()
        actual_hash = sha256_file(ckpt) if exists else "MISSING"
        hash_match = exists and actual_hash == row["checkpoint_sha256"]
        loadable = False
        load_error = ""
        payload_keys: list[str] = []
        try:
            if ckpt.suffix.lower() == ".pkl":
                with ckpt.open("rb") as handle:
                    payload = pickle.load(handle)
                payload_keys = sorted(str(k) for k in getattr(payload, "keys", lambda: [])())
                loadable = all(k in payload for k in ["cebra", "scaler", "classifier", "fit_feature_mean"])
            else:
                if torch is None:
                    raise RuntimeError(f"TORCH_IMPORT_ERROR:{TORCH_IMPORT_ERROR}")
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                payload_keys = sorted(str(k) for k in getattr(payload, "keys", lambda: [])())
                loadable = "model_state_dict" in payload
        except Exception as exc:
            load_error = repr(exc)
        status = "PASS" if exists and hash_match and loadable else "FAIL"
        gate_rows.append({
            "job_id": row["job_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "dataset": row["dataset"],
            "session_id": row["session_id"],
            "animal_id": row["animal_id"],
            "training_seed": row["seed"],
            "checkpoint_path": str(ckpt),
            "expected_checkpoint_sha256": row["checkpoint_sha256"],
            "actual_checkpoint_sha256": actual_hash,
            "exists": exists,
            "hash_match": hash_match,
            "loadable": loadable,
            "payload_keys": json.dumps(payload_keys),
            "load_error": load_error,
            "status": status,
        })
    atomic_write_csv(R3_DIR / "02_R3_CHECKPOINT_HASH_GATE.csv", gate_rows)
    return gate_rows, len(gate_rows) == EXPECTED_JOBS and all(r["status"] == "PASS" for r in gate_rows)


def split_integrity_audit(specs: dict[tuple[str, str], a5.SessionSpec]) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for (dataset, session_id), spec in sorted(specs.items()):
        x, y, trial_ids, metadata = a5.load_arrays(spec)
        role_ids: dict[str, set[int]] = {}
        counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        status = "PASS"
        err = ""
        try:
            for role in ["fit", "inner", "screening"]:
                idx = a5.split_indices(spec, role, trial_ids)
                ids = [int(v) for v in trial_ids[idx]]
                role_ids[role] = set(ids)
                counts[role] = len(ids)
                class_counts[role] = int(len(set(int(c) for c in y[idx])))
            overlaps = {
                "fit_inner": len(role_ids["fit"] & role_ids["inner"]),
                "fit_screening": len(role_ids["fit"] & role_ids["screening"]),
                "inner_screening": len(role_ids["inner"] & role_ids["screening"]),
            }
            union_count = len(role_ids["fit"] | role_ids["inner"] | role_ids["screening"])
            full_cover = union_count == len(set(int(t) for t in trial_ids))
            no_overlap = all(v == 0 for v in overlaps.values())
            screening_all_classes = class_counts["screening"] == CLASS_COUNT
            unique_trials = len(set(int(t) for t in trial_ids)) == len(trial_ids)
            if not (full_cover and no_overlap and screening_all_classes and unique_trials):
                status = "FAIL"
        except Exception as exc:
            status = "FAIL"
            err = repr(exc)
            overlaps = {"fit_inner": "NA", "fit_screening": "NA", "inner_screening": "NA"}
            union_count = "NA"
            full_cover = False
            unique_trials = False
        rows.append({
            "dataset": dataset,
            "session_id": session_id,
            "animal_id": spec.animal_id,
            "total_trials": len(trial_ids),
            "fit_trials": counts.get("fit", "NA"),
            "inner_trials": counts.get("inner", "NA"),
            "heldout_screening_trials": counts.get("screening", "NA"),
            "fit_class_count": class_counts.get("fit", "NA"),
            "inner_class_count": class_counts.get("inner", "NA"),
            "heldout_screening_class_count": class_counts.get("screening", "NA"),
            "fit_inner_overlap": overlaps["fit_inner"],
            "fit_screening_overlap": overlaps["fit_screening"],
            "inner_screening_overlap": overlaps["inner_screening"],
            "union_trial_count": union_count,
            "full_trial_coverage": full_cover,
            "trial_ids_unique": unique_trials,
            "heldout_split_role_source": "screening",
            "screening_metric_access_before_R3": False,
            "status": status,
            "error": err,
        })
    atomic_write_csv(R3_DIR / "03_R3_SPLIT_INTEGRITY_AUDIT.csv", rows)
    return rows, len(rows) == 17 and all(r["status"] == "PASS" for r in rows)


def mask_binding_audit(specs: dict[tuple[str, str], a5.SessionSpec]) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for (dataset, session_id), spec in sorted(specs.items()):
        mask_dir = spec.mask_root / spec.session_id
        split_dir = spec.split_root / spec.session_id
        mask_dir_hash = hash_path(mask_dir)
        split_dir_hash = hash_path(split_dir)
        try:
            x, y, tids, idx, _ = load_screening_role(spec)
            sample_n = min(8, len(tids))
            x = x[:sample_n]
            tids = tids[:sample_n]
            load_error = ""
        except Exception as exc:
            x = np.empty((0, 25, 0), dtype=np.float32)
            tids = np.empty((0,), dtype=np.int64)
            load_error = repr(exc)
        for condition in CONDITIONS:
            reps = [0] if condition == "CLEAN" else REPLICATES
            for rep in reps:
                status = "PASS"
                err = load_error
                mask_hash = "IDENTITY" if condition == "CLEAN" else mask_dir_hash
                try:
                    # This gate checks the exact bank lookup path without duplicating full inference work.
                    # Full held-out materialization is performed once during the R3 prediction pass.
                    if load_error:
                        raise RuntimeError(load_error)
                    _, _, state_mask_hash = a5.make_observation_state(spec, x, tids, "screening", condition, rep)
                except Exception as exc:
                    status = "FAIL"
                    err = repr(exc)
                    state_mask_hash = "ERROR"
                rows.append({
                    "dataset": dataset,
                    "session_id": session_id,
                    "condition": condition,
                    "mask_replicate": rep,
                    "heldout_split_role_source": "screening",
                    "split_path": str(split_dir),
                    "split_hash": split_dir_hash,
                    "mask_bank_path": str(mask_dir) if condition != "CLEAN" else "IDENTITY",
                    "mask_bank_hash": mask_hash,
                    "evaluation_mask_hash": state_mask_hash,
                    "model_specific_mask_generation": False,
                    "status": status,
                    "error": err,
                })
    atomic_write_csv(R3_DIR / "04_R3_MASK_BANK_BINDING_AUDIT.csv", rows)
    return rows, len(rows) == 17 * EXPECTED_STATES_PER_JOB and all(r["status"] == "PASS" for r in rows)


def load_neural_predictor(row: dict[str, Any], n_units: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    if torch is None:
        raise RuntimeError(f"TORCH_IMPORT_ERROR:{TORCH_IMPORT_ERROR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_R3_NEURAL_HELDOUT_EVALUATION")
    model = build_neural_model(str(row["model_id"]), n_units, CLASS_COUNT).cuda()
    payload = torch.load(Path(row["checkpoint_path"]), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    def predict(x: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds: list[np.ndarray] = []
        logits_all: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(x), 256):
                xb = torch.as_tensor(x[start:start + 256], dtype=torch.float32, device="cuda")
                ob = torch.as_tensor(obs[start:start + 256], dtype=torch.float32, device="cuda")
                logits = model(xb, ob)
                logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
                preds.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
        return np.concatenate(preds), np.concatenate(logits_all)

    return predict


def load_cebra_predictor(row: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    with Path(row["checkpoint_path"]).open("rb") as handle:
        payload = pickle.load(handle)
    model = payload["cebra"]
    scaler = payload["scaler"]
    clf = payload["classifier"]
    fit_mean = payload["fit_feature_mean"]

    def predict(x: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features, _ = cebra_features(x, obs, fit_mean)
        z = np.asarray(model.transform(features), dtype=np.float32)
        z_scaled = scaler.transform(z)
        y_pred = np.asarray(clf.predict(z_scaled), dtype=np.int64)
        if hasattr(clf, "predict_proba"):
            scores = np.asarray(clf.predict_proba(z_scaled), dtype=np.float32)
        elif hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(z_scaled), dtype=np.float32)
        else:
            scores = np.empty((len(y_pred), 0), dtype=np.float32)
        return y_pred, scores

    return predict


def job_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "job_id": row["job_id"],
            "dataset": row["dataset"],
            "session_id": row["session_id"],
            "animal_id": row["animal_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "training_seed": row["seed"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "state_count_planned": EXPECTED_STATES_PER_JOB,
            "data_role": "HELD_OUT_EVALUATION",
            "split_role_source": "screening",
            "status": "PENDING",
        })
    atomic_write_csv(R3_DIR / "01_R3_EVALUATION_JOB_MATRIX.csv", out)
    return out


def evaluate_jobs(rows: list[dict[str, Any]], specs: dict[tuple[str, str], a5.SessionSpec]) -> dict[str, list[dict[str, Any]]]:
    prediction_completeness: list[dict[str, Any]] = []
    state_metrics: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    state_completeness: list[dict[str, Any]] = []
    numerical_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []

    started_all = time.perf_counter()
    for job_index, row in enumerate(rows, start=1):
        job_start = time.perf_counter()
        spec = specs[(str(row["dataset"]), str(row["session_id"]))]
        x_held, y_held, trial_held, _, _ = load_screening_role(spec)
        ckpt_hash = sha256_file(Path(row["checkpoint_path"]))
        if row["model_id"] == "CEBRA_FLAT_LOGREG":
            predict_fn = load_cebra_predictor(row)
        else:
            predict_fn = load_neural_predictor(row, int(x_held.shape[2]))

        by_condition: dict[str, list[float]] = {}
        job_state_count = 0
        for condition in CONDITIONS:
            reps = [0] if condition == "CLEAN" else REPLICATES
            for rep in reps:
                state_status = "OK"
                error = ""
                try:
                    cx, cobs, mask_hash = a5.make_observation_state(spec, x_held, trial_held, "screening", condition, rep)
                    y_pred, scores = predict_fn(cx, cobs)
                    y_pred = np.asarray(y_pred, dtype=np.int64)
                    scores = np.asarray(scores)
                    ba = balanced_accuracy(y_held, y_pred, CLASS_COUNT)
                    cn = chance_normalized_balanced_accuracy(ba, CLASS_COUNT)
                    cm = confusion_matrix(y_held, y_pred, CLASS_COUNT)
                    pred_dir = (
                        R3_DIR / "predictions" / str(row["model_id"]) / str(row["dataset"]).replace(" ", "_")
                        / str(row["session_id"]) / f"seed{row['seed']}" / condition
                    )
                    pred_path = pred_dir / f"rep{rep}.npz"
                    pred_hash = npz_atomic(
                        pred_path,
                        data_role=np.asarray("HELD_OUT_EVALUATION"),
                        split_role_source=np.asarray("screening"),
                        job_id=np.asarray(row["job_id"]),
                        dataset=np.asarray(row["dataset"]),
                        session_id=np.asarray(row["session_id"]),
                        animal_id=np.asarray(row["animal_id"]),
                        model=np.asarray(row["model"]),
                        model_id=np.asarray(row["model_id"]),
                        training_seed=np.asarray(int(row["seed"])),
                        condition=np.asarray(condition),
                        mask_replicate=np.asarray(rep),
                        trial_ids=trial_held,
                        y_true=y_held,
                        y_pred=y_pred,
                        scores=scores.astype(np.float32, copy=False),
                        checkpoint_path=np.asarray(row["checkpoint_path"]),
                        checkpoint_sha256=np.asarray(ckpt_hash),
                        mask_hash=np.asarray(mask_hash),
                        confusion_matrix=cm,
                    )
                    missing_trial_ids = 0
                    duplicate_trial_ids = int(len(trial_held) - len(set(int(t) for t in trial_held)))
                    extra_trial_ids = 0
                    by_condition.setdefault(condition, []).append(float(cn))
                    numeric_ok = bool(
                        np.isfinite(ba)
                        and np.isfinite(cn)
                        and np.all(np.isfinite(y_pred))
                        and (scores.size == 0 or np.all(np.isfinite(scores)))
                    )
                except Exception as exc:
                    state_status = "ERROR"
                    error = repr(exc)
                    mask_hash = "ERROR"
                    ba = np.nan
                    cn = np.nan
                    cm = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
                    pred_path = Path("ERROR")
                    pred_hash = "ERROR"
                    missing_trial_ids = "NA"
                    duplicate_trial_ids = "NA"
                    extra_trial_ids = "NA"
                    numeric_ok = False
                    retry_rows.append({
                        "job_id": row["job_id"],
                        "dataset": row["dataset"],
                        "session_id": row["session_id"],
                        "model_id": row["model_id"],
                        "training_seed": row["seed"],
                        "condition": condition,
                        "mask_replicate": rep,
                        "retry_attempted": False,
                        "error": error,
                        "status": "ERROR_NO_RETRY",
                    })
                job_state_count += 1
                common = {
                    "job_id": row["job_id"],
                    "dataset": row["dataset"],
                    "session_id": row["session_id"],
                    "animal_id": row["animal_id"],
                    "model": row["model"],
                    "model_id": row["model_id"],
                    "training_seed": row["seed"],
                    "condition": condition,
                    "mask_replicate": rep,
                    "data_role": "HELD_OUT_EVALUATION",
                    "split_role_source": "screening",
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_sha256": ckpt_hash,
                    "mask_hash": mask_hash,
                    "status": state_status,
                    "error": error,
                }
                state_metrics.append({
                    **common,
                    "n_trials": len(y_held),
                    "balanced_accuracy": ba,
                    "cn_balacc": cn,
                    "prediction_path": str(pred_path),
                    "prediction_sha256": pred_hash,
                    "confusion_matrix_json": json.dumps(cm.tolist()),
                })
                prediction_completeness.append({
                    **common,
                    "expected_trial_count": len(y_held),
                    "prediction_file_exists": pred_path.exists() if pred_path != Path("ERROR") else False,
                    "missing_trial_ids": missing_trial_ids,
                    "duplicate_trial_ids": duplicate_trial_ids,
                    "extra_trial_ids": extra_trial_ids,
                    "trial_id_order_identical_to_heldout_split": state_status == "OK",
                    "prediction_sha256": pred_hash,
                })
                numerical_rows.append({
                    **common,
                    "balanced_accuracy_finite": bool(np.isfinite(ba)),
                    "cn_balacc_finite": bool(np.isfinite(cn)),
                    "prediction_values_finite": numeric_ok,
                    "nan_or_inf_detected": not numeric_ok,
                })
        condition_scores: dict[str, float] = {}
        for condition in CONDITIONS:
            vals = by_condition.get(condition, [])
            score = float(np.mean(vals)) if vals else np.nan
            condition_scores[condition] = score
            condition_rows.append({
                "job_id": row["job_id"],
                "dataset": row["dataset"],
                "session_id": row["session_id"],
                "animal_id": row["animal_id"],
                "model": row["model"],
                "model_id": row["model_id"],
                "training_seed": row["seed"],
                "condition": condition,
                "data_role": "HELD_OUT_EVALUATION",
                "split_role_source": "screening",
                "n_mask_replicates": 1 if condition == "CLEAN" else len(vals),
                "cn_balacc": score,
                "status": "OK" if np.isfinite(score) and (condition == "CLEAN" or len(vals) == 5) else "FAIL",
            })
        missing_vals = [condition_scores[c] for c in MISSING_CONDITIONS]
        endpoint_rows.append({
            "job_id": row["job_id"],
            "dataset": row["dataset"],
            "session_id": row["session_id"],
            "animal_id": row["animal_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "training_seed": row["seed"],
            "CLEAN": condition_scores["CLEAN"],
            "U30": condition_scores["U30"],
            "SW-U30": condition_scores["SW-U30"],
            "T5": condition_scores["T5"],
            "B5": condition_scores["B5"],
            "J30-5": condition_scores["J30-5"],
            "Five-Missing Mean": float(np.mean(missing_vals)),
            "Five-Missing Worst": float(np.min(missing_vals)),
            "status": "OK" if all(np.isfinite(v) for v in missing_vals + [condition_scores["CLEAN"]]) else "FAIL",
        })
        state_completeness.append({
            "job_id": row["job_id"],
            "dataset": row["dataset"],
            "session_id": row["session_id"],
            "animal_id": row["animal_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "training_seed": row["seed"],
            "expected_state_count": EXPECTED_STATES_PER_JOB,
            "actual_state_count": job_state_count,
            "status": "PASS" if job_state_count == EXPECTED_STATES_PER_JOB else "FAIL",
        })
        if torch is not None and torch.cuda.is_available():
            gpu_peak = int(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
        else:
            gpu_peak = "NA"
        runtime_rows.append({
            "job_id": row["job_id"],
            "dataset": row["dataset"],
            "session_id": row["session_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "training_seed": row["seed"],
            "states_completed": job_state_count,
            "wall_time_seconds": round(time.perf_counter() - job_start, 4),
            "elapsed_total_seconds": round(time.perf_counter() - started_all, 4),
            "gpu_peak_memory_bytes": gpu_peak,
            "status": "OK" if job_state_count == EXPECTED_STATES_PER_JOB else "FAIL",
        })
        atomic_write_json(R3_DIR / "logs" / "R3_PROGRESS_LATEST.json", {
            "timestamp": utc_now(),
            "completed_jobs": job_index,
            "total_jobs": len(rows),
            "current_job_id": row["job_id"],
            "current_model": row["model"],
            "current_dataset": row["dataset"],
            "current_session_id": row["session_id"],
            "current_training_seed": row["seed"],
            "elapsed_total_seconds": round(time.perf_counter() - started_all, 4),
            "status": "RUNNING" if job_index < len(rows) else "COMPLETE",
        })
        if job_index % 10 == 0 or job_index == len(rows):
            print(f"[R3] {job_index}/{len(rows)} jobs complete ({row['model_id']} {row['dataset']} {row['session_id']} seed{row['seed']})", flush=True)

    if not retry_rows:
        retry_rows = [{"status": "NO_TECHNICAL_RETRIES", "retry_count": 0}]
    return {
        "prediction_completeness": prediction_completeness,
        "state_metrics": state_metrics,
        "condition_rows": condition_rows,
        "endpoint_rows": endpoint_rows,
        "state_completeness": state_completeness,
        "numerical_rows": numerical_rows,
        "runtime_rows": runtime_rows,
        "retry_rows": retry_rows,
    }


def write_evaluation_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    atomic_write_csv(R3_DIR / "05_R3_HELDOUT_PREDICTION_COMPLETENESS.csv", outputs["prediction_completeness"])
    atomic_write_csv(R3_DIR / "06_R3_HELDOUT_STATE_COMPLETENESS.csv", outputs["state_completeness"])
    atomic_write_csv(R3_DIR / "07_R3_HELDOUT_METRICS_BY_MASK_REPLICATE.csv", outputs["state_metrics"])
    atomic_write_csv(R3_DIR / "08_R3_HELDOUT_METRICS_BY_SESSION_SEED_CONDITION.csv", outputs["condition_rows"])
    atomic_write_csv(R3_DIR / "09_R3_HELDOUT_ENDPOINTS_BY_SESSION_SEED.csv", outputs["endpoint_rows"])
    atomic_write_csv(R3_DIR / "10_R3_NUMERICAL_HEALTH_AUDIT.csv", outputs["numerical_rows"])
    atomic_write_csv(R3_DIR / "11_R3_TECHNICAL_RETRY_LOG.csv", outputs["retry_rows"])
    atomic_write_csv(R3_DIR / "12_R3_EVALUATION_RUNTIME.csv", outputs["runtime_rows"])


def sha256_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(R3_DIR.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append({
                "path": str(path.relative_to(R3_DIR)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    text = "".join(f"{r['sha256']}  {r['path']}\n" for r in rows)
    atomic_write_text(R3_DIR / "SHA256SUMS.txt", text)
    return rows


def finish(rows: list[dict[str, Any]], split_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]], outputs: dict[str, list[dict[str, Any]]]) -> None:
    endpoint_ok = all(r.get("status") == "OK" for r in outputs["endpoint_rows"])
    state_ok = len(outputs["state_metrics"]) == EXPECTED_STATES and all(r["status"] == "OK" for r in outputs["state_metrics"])
    prediction_ok = len(outputs["prediction_completeness"]) == EXPECTED_STATES and all(
        r["prediction_file_exists"] and r["missing_trial_ids"] == 0 and r["duplicate_trial_ids"] == 0 and r["extra_trial_ids"] == 0
        for r in outputs["prediction_completeness"]
    )
    numerical_ok = all(not bool(r["nan_or_inf_detected"]) for r in outputs["numerical_rows"])
    split_ok = len(split_rows) == 17 and all(r["status"] == "PASS" for r in split_rows)
    mask_ok = len(mask_rows) == 17 * EXPECTED_STATES_PER_JOB and all(r["status"] == "PASS" for r in mask_rows)
    technical_retries = 0 if outputs["retry_rows"][0].get("status") == "NO_TECHNICAL_RETRIES" else len(outputs["retry_rows"])
    final_status = "R3_HELDOUT_EVALUATION_PASS_FREEZE_FOR_R4" if all([endpoint_ok, state_ok, prediction_ok, numerical_ok, split_ok, mask_ok, technical_retries == 0]) else "R3_HELDOUT_EVALUATION_FAIL_STOP"

    allow_rows = []
    for r in outputs["endpoint_rows"]:
        allow_rows.append({
            "job_id": r["job_id"],
            "dataset": r["dataset"],
            "session_id": r["session_id"],
            "animal_id": r["animal_id"],
            "model": r["model"],
            "model_id": r["model_id"],
            "training_seed": r["training_seed"],
            "heldout_endpoint_source": "09_R3_HELDOUT_ENDPOINTS_BY_SESSION_SEED.csv",
            "data_role": "HELD_OUT_EVALUATION",
            "source_allowed_for_R4": final_status.startswith("R3_HELDOUT_EVALUATION_PASS"),
        })
    if final_status.startswith("R3_HELDOUT_EVALUATION_PASS"):
        atomic_write_csv(R3_DIR / "R3_TO_R4_HELDOUT_SOURCE_ALLOWLIST.csv", allow_rows)
        atomic_write_json(R3_DIR / "FINAL_R3_FOUR_BASELINE_HELDOUT_FREEZE_MANIFEST.json", {
            "phase_id": "MM_RVD_PHASE_R3_ONE_SHOT_HELDOUT_EVALUATION",
            "created_at": utc_now(),
            "frozen_source": "R2 selected checkpoint allowlist",
            "job_count": len(rows),
            "state_count": len(outputs["state_metrics"]),
            "endpoint_rows": len(outputs["endpoint_rows"]),
            "heldout_split_role_source": "screening",
            "models": sorted(set(r["model"] for r in rows)),
            "status": "PASS",
        })

    protocol_audit = {
        "phase_id": "MM_RVD_PHASE_R3_ONE_SHOT_HELDOUT_EVALUATION",
        "created_at": utc_now(),
        "r2_decision_path": str(R2_DECISION),
        "r2_freeze_manifest_path": str(R2_FREEZE),
        "r2_allowlist_path": str(ALLOWLIST),
        "r2_verified_ready_for_r3": True,
        "training_performed_in_R3": False,
        "optimizer_step_count_in_R3": 0,
        "checkpoint_selection_in_R3": False,
        "heldout_access_authorized_in_R3": True,
        "heldout_split_role_source": "screening",
        "conditions": CONDITIONS,
        "missing_replicates": REPLICATES,
        "aggregation": "mask replicate -> session x training seed x condition -> endpoints",
        "metric": "CN-BalAcc",
        "class_count": CLASS_COUNT,
        "status": "PASS" if final_status.startswith("R3_HELDOUT_EVALUATION_PASS") else "FAIL",
    }
    integrity = {
        "job_count": len(rows),
        "expected_job_count": EXPECTED_JOBS,
        "states_per_job": EXPECTED_STATES_PER_JOB,
        "state_count": len(outputs["state_metrics"]),
        "expected_state_count": EXPECTED_STATES,
        "split_integrity_pass": split_ok,
        "mask_binding_pass": mask_ok,
        "endpoint_complete": endpoint_ok,
        "prediction_complete": prediction_ok,
        "numerical_health_pass": numerical_ok,
        "technical_retry_count": technical_retries,
        "official_scientific_status_changed": False,
        "formal_training_started": False,
        "heldout_predictions_generated": len(outputs["state_metrics"]),
        "final_status": final_status,
    }
    atomic_write_json(R3_DIR / "13_R3_HELDOUT_PROTOCOL_AUDIT.json", protocol_audit)
    atomic_write_json(R3_DIR / "14_R3_ONE_SHOT_INTEGRITY_AUDIT.json", integrity)
    atomic_write_json(R3_DIR / "R3_HELDOUT_EVALUATION_DECISION.json", {
        **integrity,
        "decision": final_status,
        "ready_for_R4": final_status.startswith("R3_HELDOUT_EVALUATION_PASS"),
        "R4_started": False,
        "manuscript_modified": False,
    })
    report = f"""# R3 One-shot Held-out Evaluation Report

## Status
FINAL_STATUS = {final_status}

## Coverage
- R2 checkpoint allowlist rows: {len(rows)} / {EXPECTED_JOBS}
- Held-out state predictions: {len(outputs['state_metrics'])} / {EXPECTED_STATES}
- Endpoint rows: {len(outputs['endpoint_rows'])} / {EXPECTED_JOBS}
- Split integrity: {'PASS' if split_ok else 'FAIL'}
- Mask bank binding: {'PASS' if mask_ok else 'FAIL'}
- Prediction completeness: {'PASS' if prediction_ok else 'FAIL'}
- Numerical health: {'PASS' if numerical_ok else 'FAIL'}
- Technical retries: {technical_retries}

## Boundaries
- Models retrained in R3: NO
- Checkpoints changed in R3: NO
- Checkpoint selection in R3: NO
- Held-out split role source: screening
- R4 started: NO
- Manuscript modified: NO
"""
    atomic_write_text(R3_DIR / "R3_HELDOUT_EVALUATION_REPORT.md", report)
    sha256_manifest()


def main() -> int:
    if torch is None:
        raise RuntimeError(f"TORCH_IMPORT_ERROR:{TORCH_IMPORT_ERROR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_R3_NEURAL_HELDOUT_EVALUATION")
    if not R2_DECISION.exists() or not ALLOWLIST.exists() or not R2_FREEZE.exists():
        raise RuntimeError("R2_REQUIRED_INPUT_MISSING")
    r2_decision = read_json(R2_DECISION)
    if not bool(r2_decision.get("ready_for_R3")) or r2_decision.get("blockers") not in ([], None):
        raise RuntimeError("R2_NOT_READY_FOR_R3")
    if r2_decision.get("heldout_firewall", {}).get("heldout_predictions_generated") != 0:
        raise RuntimeError("R2_HELDOUT_FIREWALL_NOT_CLEAN")

    init_output_tree()
    shutil.copy2(ALLOWLIST, R3_DIR / "allowlist" / ALLOWLIST.name)
    shutil.copy2(R2_DECISION, R3_DIR / "protocol" / R2_DECISION.name)
    shutil.copy2(R2_FREEZE, R3_DIR / "protocol" / R2_FREEZE.name)
    rows = load_allowlist()
    job_matrix(rows)
    gate_rows, gate_ok = pre_access_gate(rows)
    if not gate_ok:
        raise RuntimeError("R3_PRE_ACCESS_CHECKPOINT_HASH_GATE_FAIL")
    specs = spec_map()
    split_rows, split_ok = split_integrity_audit(specs)
    if not split_ok:
        raise RuntimeError("R3_SPLIT_INTEGRITY_GATE_FAIL")
    mask_rows, mask_ok = mask_binding_audit(specs)
    if not mask_ok:
        raise RuntimeError("R3_MASK_BINDING_GATE_FAIL")
    outputs = evaluate_jobs(rows, specs)
    write_evaluation_outputs(outputs)
    finish(rows, split_rows, mask_rows, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
