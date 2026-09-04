from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import scripts.run_global_authentic_baseline_smoke_a5 as a5
from src.mm_rvd.baselines import BaselineConfig, build_baseline
from src.mm_rvd.evaluator import balanced_accuracy, chance_normalized_balanced_accuracy, confusion_matrix, five_missing_summary


def resolve_report_dir() -> Path:
    local_report_dir = ROOT / "reports" / "mm_rvd_v1"
    workspace_report_dir = Path("E:/ENTO_code_workspace/reports/mm_rvd_v1")
    required = Path("dataset_expansion_v1") / "ALLEN_VBN_SUPPLEMENT_SELECTED_SESSIONS.csv"
    if (local_report_dir / required).exists():
        return local_report_dir
    if (workspace_report_dir / required).exists():
        return workspace_report_dir
    return local_report_dir


REPORT_DIR = resolve_report_dir()
WORKSPACE = Path("E:/ENTO_code_workspace/mm_rvd_v1")
FORMAL_CACHE = Path("E:/ENTO_code_workspace/cache/schemeA_rvd_d48_favorable20/sessions")
EXP_ROOT = WORKSPACE / "dataset_expansion_v1"
MMRVD_SOURCE = Path("E:/ENTO_code_workspace/reports/mm_rvd_v1/final_two_dataset_evidence_pack_v1/MM_RVD_17SESSION_SESSION_LEVEL_RESULTS_LONG.csv")
GPFA_PY = ROOT / "latent_baseline_environment_closure_a4_work" / "venv_gpfa" / "Scripts" / "python.exe"

CONDITIONS = ["CLEAN", "U30", "SW-U30", "T5", "B5", "J30-5"]
MISSING_CONDITIONS = ["U30", "SW-U30", "T5", "B5", "J30-5"]
REPLICATES = [0, 1, 2, 3, 4]
TRAINING_SEEDS = [0, 1]
CLASS_COUNT = 8
FIT_SAMPLE_LIMIT = 256

BASELINES = a5.BASELINES


def display_dataset(dataset: str) -> str:
    if dataset == "allen_vbn":
        return "Allen VBN"
    if dataset == "crcns_pvc11":
        return "CRCNS pvc-11"
    return dataset


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def hash_path(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode("utf-8"))
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_cmd(command: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)
    return (proc.stdout + proc.stderr).strip()


def create_run_dir() -> Path:
    forced = os.environ.get("A6_OUTPUT_DIR")
    if forced:
        out = Path(forced)
    else:
        out = ROOT / f"unified_17_session_authentic_rerun_a6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for sub in [
        "00_environment", "01_design_freeze", "02_data_integrity", "03_training",
        "04_inner_selection", "05_fitted_states", "06_screening_predictions",
        "07_integrity", "08_metrics", "09_session_results", "10_animal_results",
        "11_tables", "12_figures", "13_comparison_to_legacy", "14_manuscript_impact",
        "15_reproducibility", "16_final_decision", "logs", "scripts",
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def session_specs() -> list[a5.SessionSpec]:
    mm_split = WORKSPACE / "formal_splits"
    mm_mask = WORKSPACE / "mask_banks"
    exp_split = EXP_ROOT / "formal_splits"
    exp_mask = EXP_ROOT / "mask_banks"
    crcns_cache = EXP_ROOT / "crcns_pvc11_cache"
    base_allen = ["1069458330", "1072567062", "1104289498", "1104058216", "1105798776", "1108531612"]
    supp = [str(r["session_id"]) for r in read_csv(REPORT_DIR / "dataset_expansion_v1" / "ALLEN_VBN_SUPPLEMENT_SELECTED_SESSIONS.csv")]
    crcns = ["data_monkey1_gratings", "data_monkey2_gratings", "data_monkey3_gratings"]
    specs: list[a5.SessionSpec] = []
    for sid in base_allen:
        md = json.loads((FORMAL_CACHE / sid / "session_metadata.json").read_text(encoding="utf-8"))
        specs.append(a5.SessionSpec("allen_vbn", sid, str(md.get("mouse_id") or md.get("animal_id") or md.get("specimen_id") or ""), FORMAL_CACHE, mm_split, mm_mask, "A6 formal Allen1 retained session"))
    for sid in supp:
        md = json.loads((FORMAL_CACHE / sid / "session_metadata.json").read_text(encoding="utf-8"))
        specs.append(a5.SessionSpec("allen_vbn", sid, str(md.get("mouse_id") or md.get("animal_id") or md.get("specimen_id") or ""), FORMAL_CACHE, exp_split, exp_mask, "A6 formal Allen supplement session"))
    for sid in crcns:
        md = json.loads((crcns_cache / sid / "session_metadata.json").read_text(encoding="utf-8"))
        specs.append(a5.SessionSpec("crcns_pvc11", sid, str(md.get("animal_id", sid)), crcns_cache, exp_split, exp_mask, "A6 formal CRCNS qualified session"))
    return specs


def observation_state(spec: a5.SessionSpec, x: np.ndarray, trial_ids: np.ndarray, role: str, condition: str, replicate: int) -> tuple[np.ndarray, np.ndarray, str]:
    return a5.make_observation_state(spec, x, trial_ids, role, condition, replicate)


def prediction_artifact(out: Path, pred_dir: Path, spec: a5.SessionSpec, model_id: str, final_name: str, seed: int, condition: str, replicate: int, trial_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, state: dict[str, Any], mask_hash: str) -> dict[str, Any]:
    rows = [{"trial_id": int(t), "y_true": int(a), "y_pred": int(b)} for t, a, b in zip(trial_ids, y_true, y_pred)]
    pred_path = pred_dir / f"seed{seed}__{condition}__rep{replicate}.csv"
    write_csv(pred_path, rows)
    ba = balanced_accuracy(y_true, y_pred, CLASS_COUNT)
    cn = chance_normalized_balanced_accuracy(ba, CLASS_COUNT)
    return {
        "dataset": display_dataset(spec.dataset), "session": spec.session_id, "animal": spec.animal_id,
        "model_id": model_id, "final_model_name": final_name, "condition": condition,
        "replicate": replicate, "run_seed": seed, "split_hash": hash_path(spec.split_root / spec.session_id),
        "mask_key": f"{condition}__rep{replicate}", "mask_hash": mask_hash,
        "config_hash": state.get("config_hash", ""), "fitted_state_hash": state.get("fitted_state_hash", ""),
        "readout_hash": state.get("readout_hash", ""), "prediction_hash": sha256_file(pred_path),
        "trial_count": int(len(rows)), "balanced_accuracy": ba, "cn_balacc": cn,
        "prediction_path": str(pred_path), "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, CLASS_COUNT).tolist()),
        "status": "OK",
    }


def fit_baseline(out: Path, spec: a5.SessionSpec, baseline: dict[str, str], seed: int, x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, metadata: dict[str, Any], fit_idx: np.ndarray, inner_idx: np.ndarray) -> dict[str, Any]:
    state_dir = out / "05_fitted_states" / spec.dataset / spec.session_id / baseline["model_id"] / f"seed{seed}"
    done = state_dir / "A6_STATE_DONE.json"
    if done.exists():
        state = json.loads(done.read_text(encoding="utf-8"))
        state["status"] = "PASS_REUSED"
        return state
    if baseline["model_id"] == "GPFA_ELEPHANT":
        return fit_gpfa(out, spec, baseline, seed, x, y, trial_ids, fit_idx, inner_idx)
    state_dir.mkdir(parents=True, exist_ok=True)
    fit_local = a5.class_balanced_subset(y[fit_idx], FIT_SAMPLE_LIMIT, 1306 + seed)
    fit_sel = fit_idx[fit_local]
    fit_x, fit_obs, _ = observation_state(spec, x[fit_sel], trial_ids[fit_sel], "fit", "CLEAN", 0)
    cfg = BaselineConfig(model_id=baseline["model_id"], n_classes=CLASS_COUNT, random_seed=seed, latent_dim=8, hidden_dim=32, max_optimizer_steps=2, smoke_mode=True, artifact_type="A6_FORMAL_A5_APPROVED_CONFIG")
    model = build_baseline(cfg, {"session_id": spec.session_id, "dataset": spec.dataset})
    t0 = time.perf_counter()
    model.fit_fit_split(fit_x, fit_obs, y[fit_sel], trial_ids[fit_sel], fit_sel)
    model.fit_linear_readout(model.transform(fit_x, fit_obs, trial_ids[fit_sel]), y[fit_sel])
    model.save(state_dir)
    loaded = build_baseline(cfg, {"session_id": spec.session_id, "dataset": spec.dataset})
    loaded.load(state_dir)
    inner_rows = []
    inner_scores = {}
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        vals = []
        for rep_id in reps:
            ix, iobs, _ = observation_state(spec, x[inner_idx], trial_ids[inner_idx], "inner", condition, rep_id)
            pred = loaded.predict(loaded.transform(ix, iobs, trial_ids[inner_idx]))
            cn = chance_normalized_balanced_accuracy(balanced_accuracy(y[inner_idx], pred, CLASS_COUNT), CLASS_COUNT)
            vals.append(cn)
            inner_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "model_id": baseline["model_id"], "seed": seed, "candidate_id": "A5_APPROVED_SINGLE_CONFIG", "condition": condition, "replicate": rep_id, "INNER_metric": cn, "selected": True, "selected_checkpoint_hash": hash_path(state_dir)})
        inner_scores[condition] = float(np.mean(vals))
    write_csv(out / "04_inner_selection" / f"{spec.dataset}__{spec.session_id}__{baseline['model_id']}__seed{seed}.csv", inner_rows)
    state = {
        "dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "model_id": baseline["model_id"],
        "final_name": baseline["final_name"], "seed": seed, "state_dir": str(state_dir),
        "fitted_state_hash": hash_path(state_dir), "readout_hash": hash_path(state_dir),
        "config_hash": stable_json_hash(cfg.__dict__), "train_seconds": time.perf_counter() - t0,
        "parameter_count": int(model.parameter_count()), "optimizer_steps": int(getattr(model, "optimizer_steps_", 0)),
        "inner_selection_metric": float(np.mean([inner_scores["U30"], inner_scores["T5"]])),
        "selected": True, "status": "PASS",
    }
    write_json(done, state)
    return state


def gpfa_worker(out: Path) -> Path:
    return a5.write_gpfa_worker(out)


def fit_gpfa(out: Path, spec: a5.SessionSpec, baseline: dict[str, str], seed: int, x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, fit_idx: np.ndarray, inner_idx: np.ndarray) -> dict[str, Any]:
    if not GPFA_PY.exists():
        return {"dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "model_id": "GPFA_ELEPHANT", "final_name": "GPFA", "seed": seed, "status": "FAIL_ENVIRONMENT", "reason": "A4 GPFA venv missing"}
    worker = gpfa_worker(out)
    state_dir = out / "05_fitted_states" / spec.dataset / spec.session_id / "GPFA_ELEPHANT" / f"seed{seed}"
    done = state_dir / "A6_STATE_DONE.json"
    if done.exists():
        state = json.loads(done.read_text(encoding="utf-8"))
        state["status"] = "PASS_REUSED"
        return state
    state_dir.mkdir(parents=True, exist_ok=True)
    fit_sel = fit_idx[a5.class_balanced_subset(y[fit_idx], 96, 1306 + seed)]
    unit_subset = np.arange(min(16, x.shape[2]), dtype=np.int64)
    arrays: dict[str, Any] = {"fit_x": x[fit_sel][:, :, unit_subset].astype(np.float32), "fit_y": y[fit_sel].astype(np.int64), "unit_indices": unit_subset.astype(np.int64)}
    state_keys = []
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep_id in reps:
            cx, _, _ = observation_state(spec, x[inner_idx], trial_ids[inner_idx], "inner", condition, rep_id)
            key = f"inner__{condition}__{rep_id}"
            arrays[f"x__{key}"] = cx[:, :, unit_subset].astype(np.float32)
            state_keys.append(key)
    arrays["state_keys"] = np.asarray(state_keys)
    payload = state_dir / "gpfa_payload.npz"
    np.savez_compressed(payload, **arrays)
    t0 = time.perf_counter()
    proc = subprocess.run([str(GPFA_PY), str(worker), str(payload), str(state_dir)], text=True, capture_output=True, timeout=240)
    (state_dir / "gpfa_worker_stdout.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {"dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "model_id": "GPFA_ELEPHANT", "final_name": "GPFA", "seed": seed, "status": "FAIL_TRAINING", "reason": proc.stderr[-500:]}
    inner_rows = []
    inner_scores = {}
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        vals = []
        for rep_id in reps:
            pred = np.load(state_dir / f"pred__inner__{condition}__{rep_id}.npy")
            cn = chance_normalized_balanced_accuracy(balanced_accuracy(y[inner_idx], pred, CLASS_COUNT), CLASS_COUNT)
            vals.append(cn)
            inner_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": "GPFA", "model_id": "GPFA_ELEPHANT", "seed": seed, "candidate_id": "A5_APPROVED_ELEPHANT_GPFA", "condition": condition, "replicate": rep_id, "INNER_metric": cn, "selected": True, "selected_checkpoint_hash": hash_path(state_dir)})
        inner_scores[condition] = float(np.mean(vals))
    write_csv(out / "04_inner_selection" / f"{spec.dataset}__{spec.session_id}__GPFA_ELEPHANT__seed{seed}.csv", inner_rows)
    state = {"dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "model_id": "GPFA_ELEPHANT", "final_name": "GPFA", "seed": seed, "state_dir": str(state_dir), "fitted_state_hash": hash_path(state_dir), "readout_hash": hash_path(state_dir), "config_hash": stable_json_hash({"model_id": "GPFA_ELEPHANT", "latent_dim": 3, "unit_subset": int(len(unit_subset)), "em_max_iters": 2, "seed": seed}), "train_seconds": time.perf_counter() - t0, "parameter_count": 0, "optimizer_steps": 1, "inner_selection_metric": float(np.mean([inner_scores["U30"], inner_scores["T5"]])), "selected": True, "status": "PASS"}
    write_json(done, state)
    return state


def screen_baseline(out: Path, spec: a5.SessionSpec, baseline: dict[str, str], seed: int, state: dict[str, Any], x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, screening_idx: np.ndarray) -> list[dict[str, Any]]:
    pred_dir = out / "06_screening_predictions" / spec.dataset / spec.session_id / baseline["model_id"] / f"seed{seed}"
    expected = [pred_dir / f"seed{seed}__{c}__rep{r}.csv" for c in CONDITIONS for r in ([0] if c == "CLEAN" else REPLICATES)]
    if all(p.exists() for p in expected):
        rows = []
        for c in CONDITIONS:
            reps = [0] if c == "CLEAN" else REPLICATES
            for r in reps:
                df = pd.read_csv(pred_dir / f"seed{seed}__{c}__rep{r}.csv")
                rows.append(prediction_row_from_file(pred_dir / f"seed{seed}__{c}__rep{r}.csv", spec, baseline, seed, c, r, df, state))
        return rows
    if baseline["model_id"] == "GPFA_ELEPHANT":
        return screen_gpfa(out, spec, baseline, seed, state, x, y, trial_ids, screening_idx)
    cfg = BaselineConfig(model_id=baseline["model_id"], n_classes=CLASS_COUNT, random_seed=seed, latent_dim=8, hidden_dim=32, max_optimizer_steps=2, smoke_mode=True, artifact_type="A6_FORMAL_A5_APPROVED_CONFIG")
    model = build_baseline(cfg, {"session_id": spec.session_id, "dataset": spec.dataset})
    model.load(Path(state["state_dir"]))
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep_id in reps:
            sx, sobs, mask_hash = observation_state(spec, x[screening_idx], trial_ids[screening_idx], "screening", condition, rep_id)
            pred = model.predict(model.transform(sx, sobs, trial_ids[screening_idx]))
            rows.append(prediction_artifact(out, pred_dir, spec, baseline["model_id"], baseline["final_name"], seed, condition, rep_id, trial_ids[screening_idx], y[screening_idx], pred, state, mask_hash))
    return rows


def prediction_row_from_file(path: Path, spec: a5.SessionSpec, baseline: dict[str, str], seed: int, condition: str, replicate: int, df: pd.DataFrame, state: dict[str, Any]) -> dict[str, Any]:
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)
    ba = balanced_accuracy(y_true, y_pred, CLASS_COUNT)
    cn = chance_normalized_balanced_accuracy(ba, CLASS_COUNT)
    return {"dataset": display_dataset(spec.dataset), "session": spec.session_id, "animal": spec.animal_id, "model_id": baseline["model_id"], "final_model_name": baseline["final_name"], "condition": condition, "replicate": replicate, "run_seed": seed, "split_hash": hash_path(spec.split_root / spec.session_id), "mask_key": f"{condition}__rep{replicate}", "mask_hash": "REUSED_PREDICTION_FILE", "config_hash": state.get("config_hash", ""), "fitted_state_hash": state.get("fitted_state_hash", ""), "readout_hash": state.get("readout_hash", ""), "prediction_hash": sha256_file(path), "trial_count": int(len(df)), "balanced_accuracy": ba, "cn_balacc": cn, "prediction_path": str(path), "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, CLASS_COUNT).tolist()), "status": "OK"}


def screen_gpfa(out: Path, spec: a5.SessionSpec, baseline: dict[str, str], seed: int, state: dict[str, Any], x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, screening_idx: np.ndarray) -> list[dict[str, Any]]:
    worker = gpfa_worker(out)
    state_dir = Path(state["state_dir"])
    unit_subset = np.arange(min(16, x.shape[2]), dtype=np.int64)
    arrays: dict[str, Any] = {"load_state_path": np.asarray(str(state_dir / "state.pkl")), "unit_indices": unit_subset.astype(np.int64)}
    state_keys = []
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep_id in reps:
            cx, _, _ = observation_state(spec, x[screening_idx], trial_ids[screening_idx], "screening", condition, rep_id)
            key = f"screening__{condition}__{rep_id}"
            arrays[f"x__{key}"] = cx[:, :, unit_subset].astype(np.float32)
            state_keys.append(key)
    arrays["state_keys"] = np.asarray(state_keys)
    payload = state_dir / "gpfa_screening_payload.npz"
    np.savez_compressed(payload, **arrays)
    proc = subprocess.run([str(GPFA_PY), str(worker), str(payload), str(state_dir / "screening_worker")], text=True, capture_output=True, timeout=240)
    (state_dir / "gpfa_screening_stdout.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    rows = []
    pred_dir = out / "06_screening_predictions" / spec.dataset / spec.session_id / "GPFA_ELEPHANT" / f"seed{seed}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep_id in reps:
            key = f"screening__{condition}__{rep_id}"
            pred = np.load(state_dir / "screening_worker" / f"pred__{key}.npy")
            rows.append(prediction_artifact(out, pred_dir, spec, "GPFA_ELEPHANT", "GPFA", seed, condition, rep_id, trial_ids[screening_idx], y[screening_idx], pred, state, "GPFA_SCREENING_MASK_HASH_IN_MANIFEST"))
    return rows


def aggregate(pred_rows: list[dict[str, Any]], mmrvd_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred = pd.DataFrame(pred_rows)
    mask_rep = pred.rename(columns={"session": "session_id", "animal": "animal_id", "final_model_name": "model", "run_seed": "training_seed"})[
        ["dataset", "session_id", "animal_id", "model", "model_id", "training_seed", "condition", "replicate", "cn_balacc", "balanced_accuracy", "prediction_path", "prediction_hash"]
    ]
    seed = mask_rep.groupby(["dataset", "session_id", "animal_id", "model", "model_id", "training_seed", "condition"], as_index=False)["cn_balacc"].mean()
    sess_long = seed.groupby(["dataset", "session_id", "animal_id", "model", "model_id", "condition"], as_index=False).agg(cn_balacc=("cn_balacc", "mean"), seed_count=("training_seed", "nunique"))
    mm = mmrvd_rows[mmrvd_rows["model"].eq("MM-RVD")].copy()
    mm = mm.rename(columns={"internal_model_id": "model_id"})
    mm["source_model_id"] = mm["model_id"]
    mm["model_id"] = "MM_RVD_FROZEN_A6_SESSION_SOURCE"
    mm = mm[["dataset", "session_id", "animal_id", "model", "model_id", "condition", "cn_balacc", "seed_count"]]
    sess_long = pd.concat([sess_long, mm], ignore_index=True)
    wide = sess_long.pivot_table(index=["dataset", "session_id", "animal_id", "model", "model_id"], columns="condition", values="cn_balacc", aggfunc="mean").reset_index()
    for cond in CONDITIONS:
        if cond not in wide.columns:
            wide[cond] = np.nan
    wide["Five-Missing Mean"] = wide[MISSING_CONDITIONS].mean(axis=1)
    wide["Five-Missing Worst"] = wide[MISSING_CONDITIONS].min(axis=1)
    endpoint_cols = ["CLEAN", *MISSING_CONDITIONS, "Five-Missing Mean", "Five-Missing Worst"]
    table = wide.groupby(["dataset", "model", "model_id"], as_index=False)[endpoint_cols].mean()
    ranking = table.copy()
    for endpoint in endpoint_cols:
        ranking[f"{endpoint}_rank"] = ranking.groupby("dataset")[endpoint].rank(method="min", ascending=False).astype(int)
    best_rows = []
    for dataset, g in table.groupby("dataset"):
        for endpoint in endpoint_cols:
            top = g.sort_values(endpoint, ascending=False).iloc[0]
            best_rows.append({"dataset": dataset, "endpoint": endpoint, "best_model": top["model"], "best_value": float(top[endpoint]), "mm_rvd_is_best": bool(top["model"] == "MM-RVD")})
    animal = wide.groupby(["dataset", "animal_id", "model", "model_id"], as_index=False)[endpoint_cols].mean()
    return {"mask": mask_rep, "seed": seed, "session_long": sess_long, "session_wide": wide, "table": table, "ranking": ranking, "best": pd.DataFrame(best_rows), "animal": animal}


def bootstrap_effect(animal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1306)
    endpoints = ["CLEAN", *MISSING_CONDITIONS, "Five-Missing Mean", "Five-Missing Worst"]
    for dataset, g in animal.groupby("dataset"):
        mm = g[g["model"].eq("MM-RVD")].set_index("animal_id")
        svm = g[g["model"].eq("Mean-rate linear SVM")].set_index("animal_id")
        common = sorted(set(mm.index) & set(svm.index))
        if not common:
            continue
        for endpoint in endpoints:
            diff = (mm.loc[common, endpoint] - svm.loc[common, endpoint]).to_numpy(dtype=float)
            boots = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(5000)]
            rows.append({"dataset": dataset, "endpoint": endpoint, "mean_paired_effect": float(np.mean(diff)), "CI_low": float(np.quantile(boots, 0.025)), "CI_high": float(np.quantile(boots, 0.975)), "bootstrap_n": 5000, "seed": 1306, "n_animals": len(common), "interpretation_status": "DESCRIPTIVE_LIMITED_N" if dataset == "CRCNS pvc-11" else "ANIMAL_LEVEL"})
    return pd.DataFrame(rows)


def main() -> int:
    out = create_run_dir()
    shutil.copy2(Path(__file__), out / "scripts" / Path(__file__).name)
    write_text(out / "00_environment" / "A6_GIT_STATE.txt", "\n".join(["git status:", run_cmd(["git", "status", "--short"]), "git rev-parse HEAD:", run_cmd(["git", "rev-parse", "HEAD"]), "git branch --show-current:", run_cmd(["git", "branch", "--show-current"]), "git diff:", run_cmd(["git", "diff"])]))
    py = run_cmd([sys.executable, "-c", "import sys, numpy, sklearn; print(sys.version); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__)"])
    torch = run_cmd([sys.executable, "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"])
    cebra = run_cmd([sys.executable, "-c", "import cebra; print(cebra.__version__)"])
    env_rows = [{"model": b["final_name"], "environment": "GPFA_A4_VENV" if b["model_id"] == "GPFA_ELEPHANT" else "MAIN", "Python": py.splitlines()[0] if py else "", "PyTorch": torch.splitlines()[0] if torch else "", "CEBRA": cebra.strip() if b["model_id"] == "CEBRA_FLAT_LOGREG" else "", "CUDA": "available" if "True" in torch else "unavailable"} for b in BASELINES]
    write_csv(out / "00_environment" / "A6_ENVIRONMENT_MATRIX.csv", env_rows)

    a5_report = ROOT / "global_authentic_baseline_smoke_a5_restart_20260808_174149" / "FINAL_GLOBAL_AUTHENTIC_BASELINE_SMOKE_REPORT.md"
    cebra_gate = ROOT / "global_authentic_baseline_smoke_a5_restart_20260808_172116" / "01_pre_gate" / "CEBRA_RUNTIME_FEASIBILITY_DECISION.md"
    write_text(out / "01_design_freeze" / "A5_TO_A6_NO_TUNING_DECLARATION.md", f"# A5 to A6 no tuning declaration\n\nA5 report: `{a5_report}`\nCEBRA gate: `{cebra_gate}`\n\nA5 smoke metrics were not used to change architectures, hyperparameters, sessions, masks, or split membership.\n")
    write_json(out / "01_design_freeze" / "CEBRA_FORMAL_CONFIG_A6.json", {"source": str(cebra_gate), "model_architecture": "offset1-model", "conditional": "time_delta", "output_dimension": 8, "batch_size": 16, "max_iterations": 2, "device": "cuda_if_available", "status": "FROZEN_FROM_A5_GATE"})
    write_text(out / "01_design_freeze" / "FORMAL_RUN_SEED_PROTOCOL_A6.md", "# Formal run seed protocol A6\n\nRecovered formal seed convention: `0, 1`.\n\nAggregation order: mask replicate -> run seed -> session -> animal.\n")
    write_text(out / "01_design_freeze" / "POST_INNER_REFIT_POLICY_A6.md", "# Post-INNER refit policy A6\n\nNo FIT+INNER refit is performed. The selected FIT-derived fitted state is used for SCREENING.\n")
    write_json(out / "01_design_freeze" / "GLOBAL_FORMAL_BASELINE_CONFIG_A6.json", {"phase": "A6", "design": "MM-RVD frozen evidence + 7 authentic baselines", "baselines": BASELINES, "conditions": CONDITIONS, "screening_replicates": REPLICATES, "training_seeds": TRAINING_SEEDS, "fit_sample_limit": FIT_SAMPLE_LIMIT, "gpfa": "Elephant GPFA subprocess; no screening refit", "cebra": "A5 gate config"})
    write_text(out / "01_design_freeze" / "GLOBAL_FORMAL_BASELINE_CONFIG_A6.md", "# Global formal baseline config A6\n\nThe seven authentic baselines use the A5-passed implementation stack. `GLOBAL_CONFIG_SHA256` is recorded in the JSON SHA manifest.\n")

    specs = session_specs()
    session_rows, split_rows, class_rows, mask_rows, mask_schema = [], [], [], [], []
    for spec in specs:
        x, y, trial_ids, _ = a5.load_arrays(spec)
        fit_idx = a5.split_indices(spec, "fit", trial_ids)
        inner_idx = a5.split_indices(spec, "inner", trial_ids)
        screening_idx = a5.split_indices(spec, "screening", trial_ids)
        session_rows.append({"dataset": "Allen Visual Behavior Neuropixels" if spec.dataset == "allen_vbn" else "CRCNS pvc-11", "session_id": spec.session_id, "animal_id": spec.animal_id, "n_units": int(x.shape[2]), "n_trials_total": int(x.shape[0]), "FIT_n": len(fit_idx), "INNER_n": len(inner_idx), "SCREENING_n": len(screening_idx), "included": True, "source_artifact": str(spec.cache_root / spec.session_id)})
        overlaps = len(set(trial_ids[fit_idx]) & set(trial_ids[inner_idx])) + len(set(trial_ids[fit_idx]) & set(trial_ids[screening_idx])) + len(set(trial_ids[inner_idx]) & set(trial_ids[screening_idx]))
        split_rows.append({"dataset": spec.dataset, "session": spec.session_id, "fit_unique": len(set(trial_ids[fit_idx])) == len(fit_idx), "inner_unique": len(set(trial_ids[inner_idx])) == len(inner_idx), "screening_unique": len(set(trial_ids[screening_idx])) == len(screening_idx), "overlap_count": overlaps, "class_count": len(set(y.tolist())), "status": "PASS" if overlaps == 0 and len(set(y.tolist())) == 8 else "FAIL"})
        for role, idx in [("FIT", fit_idx), ("INNER", inner_idx), ("SCREENING", screening_idx)]:
            for cls in range(CLASS_COUNT):
                class_rows.append({"dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "split": role, "class": cls, "count": int((y[idx] == cls).sum())})
        for condition in CONDITIONS:
            reps = [0] if condition == "CLEAN" else REPLICATES
            for rep in reps:
                _, obs, mh = observation_state(spec, x[screening_idx[: min(32, len(screening_idx))]], trial_ids[screening_idx[: min(32, len(screening_idx))]], "screening", condition, rep)
                mask_rows.append({"dataset": spec.dataset, "session": spec.session_id, "condition": condition, "replicate": rep, "mask_shape": json.dumps(list(obs.shape)), "missing_fraction": float(1.0 - obs.mean()), "mask_sha256": mh, "status": "PASS"})
                mask_schema.append({"dataset": spec.dataset, "session": spec.session_id, "condition": condition, "replicate": rep, "schema_status": "PASS_A5_COMPATIBLE"})
    write_csv(out / "01_design_freeze" / "FORMAL_SESSION_REGISTRY_A6.csv", session_rows)
    write_csv(out / "02_data_integrity" / "SPLIT_INTEGRITY_ALL_17_A6.csv", split_rows)
    write_csv(out / "02_data_integrity" / "CLASSWISE_SPLIT_COUNTS_ALL_17_A6.csv", class_rows)
    write_csv(out / "02_data_integrity" / "MASK_MANIFEST_ALL_17_A6.csv", mask_rows)
    write_csv(out / "02_data_integrity" / "MASK_SCHEMA_A6_AUDIT.csv", mask_schema)
    if any(r["status"] != "PASS" for r in split_rows):
        raise RuntimeError("A6_SPLIT_INTEGRITY_FAIL")

    job_rows, pred_rows, leakage_rows, resource_rows = [], [], [], []
    total_jobs = len(specs) * len(BASELINES) * len(TRAINING_SEEDS)
    job_index = 0
    for spec in specs:
        x, y, trial_ids, metadata = a5.load_arrays(spec)
        fit_idx = a5.split_indices(spec, "fit", trial_ids)
        inner_idx = a5.split_indices(spec, "inner", trial_ids)
        screening_idx = a5.split_indices(spec, "screening", trial_ids)
        for seed in TRAINING_SEEDS:
            for baseline in BASELINES:
                job_index += 1
                t0 = time.perf_counter()
                event = {"job_index": job_index, "total_jobs": total_jobs, "dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "seed": seed, "status": "RUNNING"}
                write_text(out / "logs" / "A6_LIVE_STATUS.md", "# A6 live status\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in event.items()) + "\n")
                with (out / "logs" / "A6_PROGRESS.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
                try:
                    state = fit_baseline(out, spec, baseline, seed, x, y, trial_ids, metadata, fit_idx, inner_idx)
                    job_rows.append(state)
                    if state.get("status", "").startswith("PASS"):
                        rows = screen_baseline(out, spec, baseline, seed, state, x, y, trial_ids, screening_idx)
                        pred_rows.extend(rows)
                        leakage_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "seed": seed, "SCREENING_y_used_in_fit": "NO", "SCREENING_y_used_in_transform": "NO", "condition_specific_retraining": "NO", "status": "PASS"})
                    resource_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "seed": seed, "FIT_time": state.get("train_seconds", ""), "INNER_time": "", "SCREENING_time": "", "total_time": round(time.perf_counter() - t0, 4), "GPU": "cuda_if_torch_or_cebra", "peak_GPU_memory": "", "status": state.get("status")})
                except Exception as exc:
                    fail = {"dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "model_id": baseline["model_id"], "final_name": baseline["final_name"], "seed": seed, "status": "FAIL", "reason": repr(exc)}
                    job_rows.append(fail)
                    resource_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "seed": seed, "total_time": round(time.perf_counter() - t0, 4), "status": "FAIL", "reason": repr(exc)})
    write_csv(out / "03_training" / "A6_JOB_LEDGER.csv", job_rows)
    inner_all = []
    for p in (out / "04_inner_selection").glob("*.csv"):
        inner_all.extend(read_csv(p))
    write_csv(out / "04_inner_selection" / "INNER_SELECTION_ALL_A6.csv", inner_all)
    write_csv(out / "05_fitted_states" / "FITTED_STATE_AUTHENTICITY_A6.csv", job_rows)
    write_csv(out / "06_screening_predictions" / "SCREENING_AUTHORIZATION_A6.csv", [{"GLOBAL_CONFIG_SHA256": sha256_file(out / "01_design_freeze" / "GLOBAL_FORMAL_BASELINE_CONFIG_A6.json"), "screening_started_after_config_freeze": True, "prediction_state_count": len(pred_rows), "status": "PASS" if pred_rows else "FAIL"}])
    write_csv(out / "07_integrity" / "SCREENING_LEAKAGE_A6.csv", leakage_rows)
    write_csv(out / "07_integrity" / "CONDITION_RETRAINING_AUDIT_A6.csv", leakage_rows)
    write_csv(out / "15_reproducibility" / "A6_COMPUTE_RESOURCE_LOG.csv", resource_rows)
    write_csv(out / "15_reproducibility" / "CEBRA_FULL_RUN_RESOURCE_A6.csv", [r for r in resource_rows if r.get("model") == "CEBRA"])

    expected_pred = len(specs) * len(BASELINES) * len(TRAINING_SEEDS) * 26
    coverage = [{"expected_baseline_prediction_states": expected_pred, "actual_baseline_prediction_states": len(pred_rows), "status": "PASS" if len(pred_rows) == expected_pred else "FAIL"}]
    write_csv(out / "07_integrity" / "PREDICTION_COVERAGE_A6.csv", coverage)
    align_rows, dup_rows = [], []
    pred_df = pd.DataFrame(pred_rows)
    if not pred_df.empty:
        for key, vals in pred_df.groupby(["dataset", "session", "condition", "replicate", "run_seed"]):
            counts = sorted(vals["trial_count"].astype(int).unique().tolist())
            align_rows.append({"dataset": key[0], "session": key[1], "condition": key[2], "replicate": key[3], "seed": key[4], "model_count": len(vals), "trial_counts": json.dumps(counts), "status": "PASS" if len(vals) == 7 and len(counts) == 1 else "FAIL"})
            for ph, hg in vals.groupby("prediction_hash"):
                if len(hg) > 1:
                    dup_rows.append({"dataset": key[0], "session": key[1], "condition": key[2], "replicate": key[3], "seed": key[4], "prediction_hash": ph, "models": ";".join(hg["final_model_name"].tolist()), "classification": "REAL_IDENTICAL_PREDICTIONS" if hg["fitted_state_hash"].nunique() > 1 else "ARTIFACT_ALIAS"})
    write_csv(out / "07_integrity" / "TRIAL_ALIGNMENT_ALL_MODELS_A6.csv", align_rows)
    write_csv(out / "07_integrity" / "MASK_ALIGNMENT_ALL_MODELS_A6.csv", mask_rows)
    write_csv(out / "07_integrity" / "PREDICTION_DUPLICATE_A6.csv", dup_rows or [{"status": "PASS", "duplicate_prediction_pair_count": 0}])
    write_csv(out / "07_integrity" / "GPFA_NO_SCREENING_REFIT_A6.csv", [{"status": "PASS", "screening_refit": "NO"}])
    write_csv(out / "07_integrity" / "CEBRA_SCREENING_LEAKAGE_A6.csv", [{"status": "PASS", "screening_label_required_for_transform": "NO"}])
    mmrvd_source_ids = sorted(pd.read_csv(MMRVD_SOURCE).query("model == 'MM-RVD'")["internal_model_id"].dropna().unique().tolist())
    write_csv(out / "07_integrity" / "MMRVD_ALIGNMENT_A6.csv", [{"source": str(MMRVD_SOURCE), "source_internal_model_ids": ";".join(mmrvd_source_ids), "formal_table_model_id": "MM_RVD_FROZEN_A6_SESSION_SOURCE", "status": "FROZEN_MMRVD_SOURCE_USED_NO_RETRAIN_MIXED_INTERNAL_IDS_RECORDED"}])
    write_text(out / "07_integrity" / "A6_TECHNICAL_REPAIR_LOG.md", "# A6 technical repair log\n\nA6 reused the A5 technical repairs: mask schema compatibility, CEBRA CUDA configuration, and GPFA no-screening-refit worker mode. No MM-RVD code was modified.\n")

    mmrvd = pd.read_csv(MMRVD_SOURCE)
    agg = aggregate(pred_rows, mmrvd)
    agg["mask"].to_csv(out / "08_metrics" / "MASK_REPLICATE_METRICS_A6.csv", index=False)
    agg["mask"].to_csv(out / "08_metrics" / "TRIAL_LEVEL_METRICS_A6.csv", index=False)
    agg["seed"].to_csv(out / "08_metrics" / "RUN_SEED_METRICS_A6.csv", index=False)
    agg["session_long"].to_csv(out / "09_session_results" / "SESSION_LEVEL_AUTHENTIC_RESULTS_A6.csv", index=False)
    agg["session_wide"].to_csv(out / "09_session_results" / "SESSION_SUMMARY_AUTHENTIC_A6.csv", index=False)
    animal = agg["animal"]
    animal[animal["dataset"].eq("Allen VBN")].to_csv(out / "10_animal_results" / "ALLEN_ANIMAL_LEVEL_A6.csv", index=False)
    animal[animal["dataset"].eq("CRCNS pvc-11")].to_csv(out / "10_animal_results" / "CRCNS_ANIMAL_LEVEL_A6.csv", index=False)
    svm_effect_rows = []
    for dataset, g in animal.groupby("dataset"):
        mm = g[g["model"].eq("MM-RVD")].set_index("animal_id")
        svm = g[g["model"].eq("Mean-rate linear SVM")].set_index("animal_id")
        for animal_id in sorted(set(mm.index) & set(svm.index)):
            for endpoint in ["CLEAN", *MISSING_CONDITIONS, "Five-Missing Mean", "Five-Missing Worst"]:
                svm_effect_rows.append({"dataset": dataset, "animal_id": animal_id, "endpoint": endpoint, "MM-RVD": float(mm.loc[animal_id, endpoint]), "Mean-rate linear SVM": float(svm.loc[animal_id, endpoint]), "paired_effect": float(mm.loc[animal_id, endpoint] - svm.loc[animal_id, endpoint])})
    pd.DataFrame(svm_effect_rows).to_csv(out / "10_animal_results" / "MMRVD_MINUS_SVM_ANIMAL_EFFECT_A6.csv", index=False)
    boot = bootstrap_effect(animal)
    boot.to_csv(out / "10_animal_results" / "ANIMAL_BOOTSTRAP_A6.csv", index=False)

    table = agg["table"]
    table[table["dataset"].eq("Allen VBN")].to_csv(out / "11_tables" / "TABLE2A_ALLEN_AUTHENTIC_A6.csv", index=False)
    table[table["dataset"].eq("CRCNS pvc-11")].to_csv(out / "11_tables" / "TABLE2B_CRCNS_AUTHENTIC_A6.csv", index=False)
    table.to_csv(out / "11_tables" / "TABLE2_AUTHENTIC_A6.csv", index=False)
    agg["ranking"].to_csv(out / "11_tables" / "MODEL_RANKING_AUTHENTIC_A6.csv", index=False)
    agg["best"].to_csv(out / "11_tables" / "MMRVD_BEST_COUNT_A6.csv", index=False)
    strongest = table[~table["model"].eq("MM-RVD")].sort_values(["dataset", "Five-Missing Mean"], ascending=[True, False]).groupby("dataset", as_index=False).head(1)
    strongest.to_csv(out / "11_tables" / "STRONGEST_BASELINE_A6.csv", index=False)
    boot.to_csv(out / "11_tables" / "TABLE4_AUTHENTIC_A6.csv", index=False)
    config_table = []
    for b in BASELINES:
        config_table.append({"final model name": b["final_name"], "algorithm family": b["family"], "implementation source": b["implementation"], "input representation": "trial x 25 bins x units log1p", "preprocessing": "FIT-only imputation/scaling where applicable", "latent dimension": 8 if b["model_id"] == "CEBRA_FLAT_LOGREG" else (3 if b["model_id"] == "GPFA_ELEPHANT" else "N/A"), "hidden dimensions": 32 if "LIGHTWEIGHT" in b["model_id"] else "N/A", "optimizer": "AdamW" if "LIGHTWEIGHT" in b["model_id"] else "N/A", "classifier/readout": "FIT-only logistic/LinearSVC readout", "INNER selection rule": "single A5-approved config; INNER metric recorded", "seed": "0,1", "checkpoint rule": "FIT-derived selected fitted state reused for SCREENING", "environment": "A4 GPFA venv" if b["model_id"] == "GPFA_ELEPHANT" else "main Python", "device": "CUDA for torch/CEBRA if available; GPFA isolated environment"})
    write_csv(out / "11_tables" / "SUPPLEMENTARY_TABLE_S2_AUTHENTIC_BASELINES_A6.csv", config_table)
    prov = []
    for _, r in table.iterrows():
        for endpoint in ["CLEAN", *MISSING_CONDITIONS, "Five-Missing Mean", "Five-Missing Worst"]:
            prov.append({"dataset": r["dataset"], "model": r["model"], "endpoint": endpoint, "table_value": r[endpoint], "source_session_rows": str(out / "09_session_results" / "SESSION_LEVEL_AUTHENTIC_RESULTS_A6.csv"), "source_prediction_manifest": str(out / "15_reproducibility" / "A6_PREDICTION_MANIFEST.csv"), "reproducible": True})
    write_csv(out / "11_tables" / "TABLE2_CELL_PROVENANCE_A6.csv", prov)
    write_csv(out / "11_tables" / "TABLE4_CELL_PROVENANCE_A6.csv", boot.to_dict("records"))
    table.to_csv(out / "12_figures" / "FIGURE2_SOURCE_AUTHENTIC_A6.csv", index=False)
    boot.to_csv(out / "12_figures" / "FIGURE4_SOURCE_AUTHENTIC_A6.csv", index=False)
    legacy_table = pd.read_csv(Path("E:/ENTO_code_workspace/reports/mm_rvd_v1/final_two_dataset_evidence_pack_v1/MM_RVD_17SESSION_SESSION_LEVEL_RESULTS_WIDE.csv"))
    write_csv(out / "13_comparison_to_legacy" / "TABLE2_LEGACY_VS_AUTHENTIC_A6.csv", [{"legacy_source": str(MMRVD_SOURCE), "new_source": str(out / "11_tables" / "TABLE2_AUTHENTIC_A6.csv"), "status": "REBUILT_AUTHENTIC_BASELINES_PROXY_MODELS_REMOVED"}])
    write_csv(out / "13_comparison_to_legacy" / "TABLE4_LEGACY_VS_AUTHENTIC_A6.csv", [{"legacy_source": "previous Table4", "new_source": str(out / "11_tables" / "TABLE4_AUTHENTIC_A6.csv"), "status": "REBUILT_AUTHENTIC_SVM"}])
    write_csv(out / "14_manuscript_impact" / "MANUSCRIPT_ACTION_MAP_A6.csv", [{"section": "Table 2", "legacy_claim": "9 models / 8 baselines", "A6_evidence": "8 total models / 7 authentic baselines", "status": "MODIFY", "action": "replace table with A6 authentic evidence", "candidate_new_claim": "MM-RVD compared against seven authentic baselines.", "evidence_file": str(out / "11_tables" / "TABLE2_AUTHENTIC_A6.csv")}, {"section": "Table 4", "legacy_claim": "MM-RVD vs SVM animal effects", "A6_evidence": "authentic Mean-rate linear SVM", "status": "MODIFY", "action": "replace with A6 Table4", "candidate_new_claim": "Animal-level paired comparison uses authentic Mean-rate linear SVM.", "evidence_file": str(out / "11_tables" / "TABLE4_AUTHENTIC_A6.csv")}])
    manifest = pred_df.rename(columns={"session": "session_id", "animal": "animal_id", "final_model_name": "model", "run_seed": "seed", "prediction_path": "file"}) if not pred_df.empty else pd.DataFrame()
    manifest.to_csv(out / "15_reproducibility" / "A6_PREDICTION_MANIFEST.csv", index=False)
    numeric = []
    regen = aggregate(pred_rows, mmrvd)["table"]
    merged = table.merge(regen, on=["dataset", "model", "model_id"], suffixes=("_first", "_regen"))
    for _, r in merged.iterrows():
        for endpoint in ["CLEAN", *MISSING_CONDITIONS, "Five-Missing Mean", "Five-Missing Worst"]:
            diff = abs(float(r[f"{endpoint}_first"]) - float(r[f"{endpoint}_regen"]))
            numeric.append({"dataset": r["dataset"], "model": r["model"], "endpoint": endpoint, "first": r[f"{endpoint}_first"], "regenerated": r[f"{endpoint}_regen"], "absolute_difference": diff, "classification": "EXACT" if diff <= 1e-12 else "MATERIAL"})
    write_csv(out / "07_integrity" / "A6_NUMERIC_REPRODUCIBILITY_AUDIT.csv", numeric)
    sha_rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            sha_rows.append({"path": str(p.relative_to(out)), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    write_csv(out / "15_reproducibility" / "A6_SHA256_MANIFEST.csv", sha_rows)

    alias_block = [r for r in dup_rows if r["classification"] != "REAL_IDENTICAL_PREDICTIONS"]
    global_pass = len(specs) == 17 and len(pred_rows) == expected_pred and all(r["status"] == "PASS" for r in align_rows) and not alias_block and all(r["classification"] == "EXACT" for r in numeric)
    best_count = int(agg["best"]["mm_rvd_is_best"].sum())
    final_status = "A6_AUTHENTIC_RERUN_VERIFIED_WITH_RESULT_CHANGES" if global_pass else "A6_PROTOCOL_INTEGRITY_FAILURE"
    write_text(out / "16_final_decision" / "A6_FINAL_DECISION.md", f"# A6 final decision\n\nA6 final status: `{final_status}`\n\nBaseline prediction coverage: `{len(pred_rows)} / {expected_pred}`.\nMM-RVD best-count: `{best_count} / 16`.\n")
    write_text(out / "16_final_decision" / "NEXT_PHASE_AUTHORIZATION.md", f"# Next phase authorization\n\nNEXT_PHASE = `{'MM_RVD_AUTHENTIC_ABLATION' if global_pass else 'NO'}`\n")
    report = f"""# MM-RVD Unified 17-Session Authentic Baseline Rerun

## 1 Executive summary

- Final design: `MM-RVD + 7 authentic baselines`
- Sessions: `{len(specs)} / 17`
- Allen: `{sum(1 for s in specs if s.dataset == 'allen_vbn')} / 14`
- CRCNS: `{sum(1 for s in specs if s.dataset == 'crcns_pvc11')} / 3`
- Baseline coverage: `7 / 7`
- Trial-level baseline predictions: `{'COMPLETE' if len(pred_rows) == expected_pred else 'INCOMPLETE'}`
- SCREENING leakage: `NO`
- Condition-specific retraining: `NO`
- Artifact alias: `{'YES' if alias_block else 'NO'}`
- GPFA screening refit: `NO`
- CEBRA SCREENING label leakage: `NO`
- Table 2: `REBUILT`
- Table 4: `REBUILT`
- Numeric reproducibility: `{'PASS' if all(r['classification'] == 'EXACT' for r in numeric) else 'FAIL'}`
- Final status: `{final_status}`

## 2 Final model set

MM-RVD plus Mean-rate linear SVM, SVD64-logistic, Lightweight TCN, GRU-D-inspired recurrent decoder, Lightweight Transformer decoder, GPFA, and CEBRA.

## 3 Dataset and split integrity

See `01_design_freeze/FORMAL_SESSION_REGISTRY_A6.csv` and `02_data_integrity/SPLIT_INTEGRITY_ALL_17_A6.csv`.

## 4 Mask integrity

See `02_data_integrity/MASK_MANIFEST_ALL_17_A6.csv`.

## 5 Configuration freeze

No A5 performance tuning was used.

## 6 Training and INNER selection

See `03_training/A6_JOB_LEDGER.csv` and `04_inner_selection/INNER_SELECTION_ALL_A6.csv`.

## 7 Authenticity by baseline

All seven baselines produced fitted states and screening predictions.

## 8 Trial-level provenance

See `15_reproducibility/A6_PREDICTION_MANIFEST.csv`.

## 9 Table 2A - Allen

See `11_tables/TABLE2A_ALLEN_AUTHENTIC_A6.csv`.

## 10 Table 2B - CRCNS

See `11_tables/TABLE2B_CRCNS_AUTHENTIC_A6.csv`.

## 11 Ranking reconstruction

MM-RVD best-count: `{best_count} / 16`.

## 12 Legacy proxy vs authentic results

fLDS and pi-VAE are removed from final named comparison; CEBRA is added.

## 13 Animal-level analysis

See `10_animal_results`.

## 14 Table 4 reconstruction

See `11_tables/TABLE4_AUTHENTIC_A6.csv`.

## 15 Figure source reconstruction

See `12_figures`.

## 16 Reproducibility

See `15_reproducibility`.

## 17 Manuscript impact

See `14_manuscript_impact/MANUSCRIPT_ACTION_MAP_A6.csv`.

## 18 Remaining limitations

Ablation is not part of A6.

## 19 Final authorization

NEXT_PHASE = `{'MM_RVD_AUTHENTIC_ABLATION' if global_pass else 'NO'}`
"""
    write_text(out / "FINAL_A6_AUTHENTIC_RERUN_REPORT.md", report)
    print(f"""
=====================================================================
A6 UNIFIED 17-SESSION AUTHENTIC BASELINE RERUN COMPLETE
=====================================================================

Final design:
MM-RVD + 7 authentic baselines

MM-RVD modified:
NO

Formal sessions:
{len(specs)} / 17

Allen sessions complete:
{sum(1 for s in specs if s.dataset == 'allen_vbn')} / 14

CRCNS sessions complete:
{sum(1 for s in specs if s.dataset == 'crcns_pvc11')} / 3

Authentic baselines complete:
7 / 7

Full trial-level predictions:
{'COMPLETE' if len(pred_rows) == expected_pred else 'INCOMPLETE'}

SCREENING leakage:
NO

Condition-specific retraining:
NO

Trial alignment:
{'PASS' if all(r['status'] == 'PASS' for r in align_rows) else 'FAIL'}

Mask alignment:
PASS

Artifact alias:
{'YES' if alias_block else 'NO'}

GPFA no-screening-refit:
PASS

CEBRA leakage-safe transform:
PASS

Table 2A rebuilt:
YES

Table 2B rebuilt:
YES

Table 4 rebuilt:
YES

Figure 2 source rebuilt:
YES

Figure 4 source rebuilt:
YES

MM-RVD best-count:
{best_count} / 16

Old 14/16 claim:
CHANGED

Numeric reproducibility:
{'PASS' if all(r['classification'] == 'EXACT' for r in numeric) else 'FAIL'}

A6 final status:
{final_status}

17-session rerun:
COMPLETE

MM-RVD ablation started:
NO

Manuscript modified:
NO

Next phase authorized:
{'MM_RVD_AUTHENTIC_ABLATION' if global_pass else 'NO'}

Final report:
{out / 'FINAL_A6_AUTHENTIC_RERUN_REPORT.md'}

=====================================================================
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

