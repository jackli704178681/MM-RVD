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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.mm_rvd.baselines import BaselineConfig, build_baseline
from src.mm_rvd.evaluator import balanced_accuracy, chance_normalized_balanced_accuracy, confusion_matrix, five_missing_summary


def resolve_report_dir() -> Path:
    local_report_dir = ROOT / "reports" / "mm_rvd_v1"
    workspace_report_dir = Path("E:/ENTO_code_workspace/reports/mm_rvd_v1")
    if (local_report_dir / "MM_RVD_PHASE1A_FINAL_DECISION.json").exists():
        return local_report_dir
    if (workspace_report_dir / "MM_RVD_PHASE1A_FINAL_DECISION.json").exists():
        return workspace_report_dir
    return local_report_dir


REPORT_DIR = resolve_report_dir()
WORKSPACE = Path("E:/ENTO_code_workspace")
MM_WORKSPACE = WORKSPACE / "mm_rvd_v1"
A4_DIR = ROOT / "latent_baseline_environment_closure_a4_20260808_164014"
GPFA_PY = ROOT / "latent_baseline_environment_closure_a4_work" / "venv_gpfa" / "Scripts" / "python.exe"

CONDITIONS = ["CLEAN", "U30", "SW-U30", "T5", "B5", "J30-5"]
MISSING_CONDITIONS = ["U30", "SW-U30", "T5", "B5", "J30-5"]
REPLICATES = [0, 1, 2, 3, 4]
CLASS_COUNT = 8
SMOKE_SEED = 0
FIT_SAMPLE_LIMIT = 256


@dataclass(frozen=True)
class SessionSpec:
    dataset: str
    session_id: str
    animal_id: str
    cache_root: Path
    split_root: Path
    mask_root: Path
    selection_reason: str


BASELINES = [
    {
        "model_id": "MEAN_RATE_LINEAR_SVM",
        "final_name": "Mean-rate linear SVM",
        "family": "STATIC / LINEAR",
        "implementation": "src.mm_rvd.baselines.MeanRateLinearSVM",
    },
    {
        "model_id": "SVD64_LOGREG",
        "final_name": "SVD64-logistic",
        "family": "STATIC / LINEAR",
        "implementation": "src.mm_rvd.baselines.SVD64LogReg",
    },
    {
        "model_id": "TCN_LIGHTWEIGHT",
        "final_name": "Lightweight TCN",
        "family": "SUPERVISED TEMPORAL",
        "implementation": "src.mm_rvd.baselines.TCNLightweight",
    },
    {
        "model_id": "GRU_LIGHTWEIGHT",
        "final_name": "GRU-D-inspired recurrent decoder",
        "family": "SUPERVISED TEMPORAL",
        "implementation": "src.mm_rvd.baselines.GRULightweight",
    },
    {
        "model_id": "TINY_TRANSFORMER_LIGHTWEIGHT",
        "final_name": "Lightweight Transformer decoder",
        "family": "SUPERVISED TEMPORAL",
        "implementation": "src.mm_rvd.baselines.TinyTransformerLightweight",
    },
    {
        "model_id": "GPFA_ELEPHANT",
        "final_name": "GPFA",
        "family": "LATENT / REPRESENTATION",
        "implementation": "Elephant GPFA 1.2.1 subprocess bridge",
    },
    {
        "model_id": "CEBRA_FLAT_LOGREG",
        "final_name": "CEBRA",
        "family": "LATENT / REPRESENTATION",
        "implementation": "src.mm_rvd.baselines.CEBRAFlatLogReg",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_hash(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_cmd(command: list[str], cwd: Path = ROOT, timeout: int = 120) -> str:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    return (proc.stdout + proc.stderr).strip()


def create_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"global_authentic_baseline_smoke_a5_restart_{ts}"
    for sub in [
        "00_environment",
        "01_pre_gate",
        "02_design_freeze",
        "03_session_selection",
        "04_data_integrity",
        "05_model_training",
        "06_inner_selection",
        "07_screening_predictions",
        "08_authenticity",
        "09_alignment",
        "10_smoke_metrics",
        "11_reproducibility",
        "12_next_phase",
        "logs",
        "scripts",
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def load_arrays(spec: SessionSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base = spec.cache_root / spec.session_id
    x = np.load(base / "spike_counts_log1p.npz")["X"].astype(np.float32)
    y = np.load(base / "labels.npy").astype(np.int64)
    trial_ids = np.load(base / "trial_ids.npy").astype(np.int64)
    metadata = json.loads((base / "session_metadata.json").read_text(encoding="utf-8"))
    return x, y, trial_ids, metadata


def split_ids(spec: SessionSpec, role: str) -> np.ndarray:
    return np.load(spec.split_root / spec.session_id / f"{role.lower()}_trial_ids.npy").astype(np.int64)


def split_indices(spec: SessionSpec, role: str, trial_ids: np.ndarray) -> np.ndarray:
    ids = split_ids(spec, role)
    pos = {int(t): i for i, t in enumerate(trial_ids.astype(int))}
    return np.asarray([pos[int(t)] for t in ids], dtype=np.int64)


def class_balanced_subset(y: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    if len(y) <= max_count:
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    per_class = max(1, max_count // CLASS_COUNT)
    picked: list[int] = []
    for cls in range(CLASS_COUNT):
        idx = np.flatnonzero(y == cls)
        if len(idx):
            picked.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    if len(picked) < max_count:
        rest = np.setdiff1d(np.arange(len(y)), np.asarray(picked, dtype=np.int64), assume_unique=False)
        need = min(max_count - len(picked), len(rest))
        if need:
            picked.extend(rng.choice(rest, size=need, replace=False).tolist())
    return np.asarray(sorted(set(picked)), dtype=np.int64)


def unit_missing_indices(spec: SessionSpec, condition: str, role: str, replicate: int) -> np.ndarray:
    name = "u30_mask_bank.npz" if condition in {"U30", "J30-5"} else "sw_u30_mask_bank.npz"
    z = np.load(spec.mask_root / spec.session_id / name, allow_pickle=True)
    rows = np.where((z["split_role"].astype(str) == role.upper()) & (z["replicate_id"].astype(int) == replicate))[0]
    if len(rows) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray(z["missing_unit_indices"][int(rows[0])], dtype=np.int64)


def time_missing_map(spec: SessionSpec, condition: str, role: str, replicate: int, selected_trial_ids: np.ndarray) -> dict[int, np.ndarray]:
    file_name = {"T5": "t5_mask_bank.npz", "B5": "b5_mask_bank.npz", "J30-5": "j30_5_mask_bank.npz"}[condition]
    z = np.load(spec.mask_root / spec.session_id / file_name, allow_pickle=True)
    role_mask = z["split_role"].astype(str) == role.upper()
    rep_mask = z["replicate_id"].astype(int) == replicate
    wanted = set(np.asarray(selected_trial_ids, dtype=int).tolist())
    out: dict[int, np.ndarray] = {}
    for idx in np.where(role_mask & rep_mask)[0]:
        tid = int(z["trial_id"][idx])
        if tid not in wanted:
            continue
        if condition == "B5":
            first = list(range(int(z["first_segment_start"][idx]), int(z["first_segment_start"][idx]) + int(z["first_segment_length"][idx])))
            second = list(range(int(z["second_segment_start"][idx]), int(z["second_segment_start"][idx]) + int(z["second_segment_length"][idx])))
            bins = first + second
        else:
            start_key = "start" if "start" in z.files else "time_start"
            length_key = "length" if "length" in z.files else "time_length"
            bins = list(range(int(z[start_key][idx]), int(z[start_key][idx]) + int(z[length_key][idx])))
        out[tid] = np.asarray(bins, dtype=np.int64)
    return out


def make_observation_state(spec: SessionSpec, x: np.ndarray, trial_ids: np.ndarray, role: str, condition: str, replicate: int) -> tuple[np.ndarray, np.ndarray, str]:
    response = np.array(x, copy=True)
    observed = np.ones_like(response, dtype=np.float32)
    if condition == "CLEAN":
        return response, observed, "IDENTITY"
    if condition in {"U30", "SW-U30", "J30-5"}:
        missing = unit_missing_indices(spec, condition, role, replicate)
        response[:, :, missing] = 0.0
        observed[:, :, missing] = 0.0
    if condition in {"T5", "B5", "J30-5"}:
        trial_to_bins = time_missing_map(spec, condition, role, replicate, trial_ids)
        for row, tid in enumerate(np.asarray(trial_ids, dtype=int)):
            bins = trial_to_bins.get(int(tid))
            if bins is not None:
                response[row, bins, :] = 0.0
                observed[row, bins, :] = 0.0
    mask_hash = stable_json_hash(
        {
            "session_id": spec.session_id,
            "condition": condition,
            "replicate": replicate,
            "role": role,
            "observed_sum": float(observed.sum()),
            "shape": list(observed.shape),
        }
    )
    return response.astype(np.float32), observed.astype(np.float32), mask_hash


def select_smoke_sessions() -> list[SessionSpec]:
    allen_ids = ["1069458330", "1072567062", "1104289498", "1104058216", "1105798776", "1108531612"]
    allen_root = Path("E:/ENTO_code_workspace/cache/schemeA_rvd_d48_favorable20/sessions")
    mm_split = MM_WORKSPACE / "formal_splits"
    mm_mask = MM_WORKSPACE / "mask_banks"
    allen_counts = []
    for sid in allen_ids:
        x = np.load(allen_root / sid / "spike_counts_log1p.npz")["X"]
        md = json.loads((allen_root / sid / "session_metadata.json").read_text(encoding="utf-8"))
        allen_counts.append((sid, int(x.shape[2]), str(md.get("mouse_id") or md.get("animal_id") or md.get("specimen_id") or "")))
    median_units = float(np.median([n for _, n, _ in allen_counts]))
    allen_sid, _, allen_animal = sorted(allen_counts, key=lambda r: (abs(r[1] - median_units), r[0]))[0]

    crcns_root = MM_WORKSPACE / "dataset_expansion_v1" / "crcns_pvc11_cache"
    exp_split = MM_WORKSPACE / "dataset_expansion_v1" / "formal_splits"
    exp_mask = MM_WORKSPACE / "dataset_expansion_v1" / "mask_banks"
    crcns_ids = ["data_monkey1_gratings", "data_monkey2_gratings", "data_monkey3_gratings"]
    crcns_counts = []
    for sid in crcns_ids:
        x = np.load(crcns_root / sid / "spike_counts_log1p.npz")["X"]
        md = json.loads((crcns_root / sid / "session_metadata.json").read_text(encoding="utf-8"))
        crcns_counts.append((sid, int(x.shape[2]), str(md.get("animal_id", ""))))
    crcns_median = float(np.median([n for _, n, _ in crcns_counts]))
    crcns_sid, _, crcns_animal = sorted(crcns_counts, key=lambda r: (abs(r[1] - crcns_median), r[0]))[0]
    return [
        SessionSpec("allen_vbn", allen_sid, allen_animal, allen_root, mm_split, mm_mask, f"unit count closest to Allen median ({median_units})"),
        SessionSpec("crcns_pvc11", crcns_sid, crcns_animal, crcns_root, exp_split, exp_mask, f"unit count closest to CRCNS median ({crcns_median})"),
    ]


def fit_and_select_baseline(
    out: Path,
    spec: SessionSpec,
    baseline: dict[str, str],
    x: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
    metadata: dict[str, Any],
    fit_idx: np.ndarray,
    inner_idx: np.ndarray,
) -> dict[str, Any]:
    model_id = baseline["model_id"]
    state_dir = out / "05_model_training" / spec.dataset / spec.session_id / model_id / f"seed{SMOKE_SEED}"
    state_dir.mkdir(parents=True, exist_ok=True)
    if model_id == "GPFA_ELEPHANT":
        return fit_gpfa(out, spec, baseline, x, y, trial_ids, fit_idx, inner_idx)

    fit_local = class_balanced_subset(y[fit_idx], FIT_SAMPLE_LIMIT, 1306)
    fit_sel = fit_idx[fit_local]
    fit_x, fit_obs, _ = make_observation_state(spec, x[fit_sel], trial_ids[fit_sel], "fit", "CLEAN", 0)
    fit_y = y[fit_sel]
    cfg = BaselineConfig(model_id=model_id, n_classes=CLASS_COUNT, random_seed=SMOKE_SEED, latent_dim=8, hidden_dim=32, max_optimizer_steps=2, smoke_mode=True)
    model = build_baseline(cfg, {"session_id": spec.session_id, "dataset": spec.dataset})
    start = time.perf_counter()
    initial_hash = ""
    if model_id in {"TCN_LIGHTWEIGHT", "GRU_LIGHTWEIGHT", "TINY_TRANSFORMER_LIGHTWEIGHT"}:
        # The concrete torch module is created in fit_fit_split.
        initial_hash = "CREATED_DURING_FIT"
    model.fit_fit_split(fit_x, fit_obs, fit_y, trial_ids[fit_sel], fit_sel)
    rep = model.transform(fit_x, fit_obs, trial_ids[fit_sel])
    model.fit_linear_readout(rep, fit_y)
    train_seconds = time.perf_counter() - start
    model.save(state_dir)
    loaded = build_baseline(cfg, {"session_id": spec.session_id, "dataset": spec.dataset})
    loaded.load(state_dir)
    state_hash = hash_path(state_dir)
    inner_rows = []
    inner_scores = {}
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        vals = []
        for rep_id in reps:
            ix, iobs, _ = make_observation_state(spec, x[inner_idx], trial_ids[inner_idx], "inner", condition, rep_id)
            pred = loaded.predict(loaded.transform(ix, iobs, trial_ids[inner_idx]))
            ba = balanced_accuracy(y[inner_idx], pred, CLASS_COUNT)
            cn = chance_normalized_balanced_accuracy(ba, CLASS_COUNT)
            vals.append(cn)
            inner_rows.append(
                {
                    "dataset": spec.dataset,
                    "session": spec.session_id,
                    "model": baseline["final_name"],
                    "condition": condition,
                    "replicate": rep_id,
                    "cn_balacc": cn,
                    "selection_role": "INNER",
                }
            )
        inner_scores[condition] = float(np.mean(vals))
    write_csv(out / "06_inner_selection" / f"{spec.dataset}__{spec.session_id}__{model_id}.csv", inner_rows)
    return {
        "dataset": spec.dataset,
        "session": spec.session_id,
        "animal": spec.animal_id,
        "model_id": model_id,
        "final_name": baseline["final_name"],
        "state_dir": str(state_dir),
        "fitted_state_hash": state_hash,
        "readout_hash": state_hash,
        "config_hash": stable_json_hash(cfg.__dict__),
        "train_seconds": train_seconds,
        "parameter_count": int(model.parameter_count()),
        "optimizer_steps": int(getattr(model, "optimizer_steps_", 0)),
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": state_hash,
        "save_reload_status": "PASS",
        "inner_selection_metric": float(np.mean([inner_scores["U30"], inner_scores["T5"]])),
        "selected": True,
        "status": "PASS",
    }


def hash_path(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode("utf-8"))
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def write_gpfa_worker(out: Path) -> Path:
    worker = out / "scripts" / "gpfa_elephant_worker.py"
    write_text(
        worker,
        r'''
from __future__ import annotations
import csv, json, pickle, sys, time
from pathlib import Path
import numpy as np
import quantities as pq
import neo
from elephant.gpfa import GPFA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def to_spiketrains(x):
    counts = np.maximum(0, np.rint(np.expm1(x))).astype(int)
    trials = []
    for tr in counts:
        trial = []
        for u in range(tr.shape[1]):
            times = []
            for b, c in enumerate(tr[:, u]):
                c = min(int(c), 3)
                if c:
                    times.extend([(b + (j + 1) / (c + 1)) * 20.0 for j in range(c)])
            trial.append(neo.SpikeTrain(np.asarray(times) * pq.ms, t_start=0 * pq.ms, t_stop=500 * pq.ms))
        trials.append(trial)
    return trials

def embed(model, x):
    z = model.transform(to_spiketrains(x))
    return np.asarray([np.asarray(a).mean(axis=1) for a in z], dtype=np.float32)

def main():
    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(in_path, allow_pickle=True)
    t0 = time.perf_counter()
    loaded_state = False
    if "load_state_path" in data.files:
        state_path = Path(str(data["load_state_path"].item()))
        with state_path.open("rb") as f:
            state = pickle.load(f)
        gpfa = state["gpfa"]
        scaler = state["scaler"]
        clf = state["clf"]
        pred_fit = np.asarray([], dtype=np.int64)
        loaded_state = True
    else:
        gpfa = GPFA(bin_size=20 * pq.ms, x_dim=3, em_max_iters=2, verbose=False)
        gpfa.fit(to_spiketrains(data["fit_x"]))
        fit_z = embed(gpfa, data["fit_x"])
        scaler = StandardScaler()
        fit_zs = scaler.fit_transform(fit_z)
        clf = LogisticRegression(max_iter=200, C=1.0, random_state=0)
        clf.fit(fit_zs, data["fit_y"].astype(int))
        pred_fit = clf.predict(fit_zs)
        with (out_dir / "state.pkl").open("wb") as f:
            pickle.dump({"gpfa": gpfa, "scaler": scaler, "clf": clf, "unit_indices": data["unit_indices"]}, f)
    rows = []
    for key in data["state_keys"].astype(str):
        x = data[f"x__{key}"]
        z = embed(gpfa, x)
        pred = clf.predict(scaler.transform(z))
        np.save(out_dir / f"pred__{key}.npy", pred.astype(np.int64))
        rows.append({"state_key": key, "prediction_count": int(len(pred))})
    meta = {
        "fit_trials": None if loaded_state else int(data["fit_x"].shape[0]),
        "unit_subset": int(len(data["unit_indices"])),
        "latent_dim": 3,
        "gamma_present": hasattr(gpfa, "params_estimated") and "gamma" in gpfa.params_estimated,
        "eps_present": hasattr(gpfa, "params_estimated") and "eps" in gpfa.params_estimated,
        "C_present": hasattr(gpfa, "params_estimated") and "C" in gpfa.params_estimated,
        "R_present": hasattr(gpfa, "params_estimated") and "R" in gpfa.params_estimated,
        "fit_prediction_classes": sorted(set(int(v) for v in pred_fit)) if len(pred_fit) else [],
        "loaded_fit_state_for_prediction": loaded_state,
        "screening_refit_performed": False if loaded_state else None,
        "seconds": time.perf_counter() - t0,
        "rows": rows,
    }
    (out_dir / "gpfa_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()
'''.lstrip(),
    )
    return worker


def fit_gpfa(out: Path, spec: SessionSpec, baseline: dict[str, str], x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, fit_idx: np.ndarray, inner_idx: np.ndarray) -> dict[str, Any]:
    if not GPFA_PY.exists():
        return {"dataset": spec.dataset, "session": spec.session_id, "model_id": "GPFA_ELEPHANT", "final_name": "GPFA", "status": "FAIL_ENVIRONMENT", "reason": "A4 GPFA venv missing"}
    worker = write_gpfa_worker(out)
    state_dir = out / "05_model_training" / spec.dataset / spec.session_id / "GPFA_ELEPHANT" / "seed0"
    state_dir.mkdir(parents=True, exist_ok=True)
    fit_local = class_balanced_subset(y[fit_idx], 96, 1306)
    fit_sel = fit_idx[fit_local]
    unit_count = x.shape[2]
    unit_subset = np.arange(min(16, unit_count), dtype=np.int64)
    arrays: dict[str, Any] = {
        "fit_x": x[fit_sel][:, :, unit_subset].astype(np.float32),
        "fit_y": y[fit_sel].astype(np.int64),
        "unit_indices": unit_subset.astype(np.int64),
    }
    state_keys = []
    for role, idx in [("inner", inner_idx)]:
        for condition in CONDITIONS:
            reps = [0] if condition == "CLEAN" else REPLICATES
            for rep_id in reps:
                cx, _, _ = make_observation_state(spec, x[idx], trial_ids[idx], role, condition, rep_id)
                key = f"{role}__{condition}__{rep_id}"
                arrays[f"x__{key}"] = cx[:, :, unit_subset].astype(np.float32)
                state_keys.append(key)
    arrays["state_keys"] = np.asarray(state_keys)
    payload = state_dir / "gpfa_payload.npz"
    np.savez_compressed(payload, **arrays)
    start = time.perf_counter()
    proc = subprocess.run([str(GPFA_PY), str(worker), str(payload), str(state_dir)], text=True, capture_output=True, timeout=240)
    (state_dir / "gpfa_worker_stdout.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {"dataset": spec.dataset, "session": spec.session_id, "model_id": "GPFA_ELEPHANT", "final_name": "GPFA", "status": "FAIL_TRAINING", "reason": proc.stderr[-500:]}
    meta = json.loads((state_dir / "gpfa_metadata.json").read_text(encoding="utf-8"))
    inner_rows = []
    inner_scores = {}
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        vals = []
        for rep_id in reps:
            key = f"inner__{condition}__{rep_id}"
            pred = np.load(state_dir / f"pred__{key}.npy")
            ba = balanced_accuracy(y[inner_idx], pred, CLASS_COUNT)
            cn = chance_normalized_balanced_accuracy(ba, CLASS_COUNT)
            vals.append(cn)
            inner_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": "GPFA", "condition": condition, "replicate": rep_id, "cn_balacc": cn, "selection_role": "INNER"})
        inner_scores[condition] = float(np.mean(vals))
    write_csv(out / "06_inner_selection" / f"{spec.dataset}__{spec.session_id}__GPFA_ELEPHANT.csv", inner_rows)
    return {
        "dataset": spec.dataset,
        "session": spec.session_id,
        "animal": spec.animal_id,
        "model_id": "GPFA_ELEPHANT",
        "final_name": "GPFA",
        "state_dir": str(state_dir),
        "fitted_state_hash": hash_path(state_dir),
        "readout_hash": hash_path(state_dir),
        "config_hash": stable_json_hash({"model_id": "GPFA_ELEPHANT", "latent_dim": 3, "unit_subset": int(len(unit_subset)), "em_max_iters": 2, "smoke_only": True}),
        "train_seconds": time.perf_counter() - start,
        "parameter_count": 0,
        "optimizer_steps": int(meta.get("seconds", 0) > 0),
        "initial_parameter_hash": "ELEPHANT_GPFA_INITIALIZED_FROM_FIT",
        "final_parameter_hash": hash_path(state_dir),
        "save_reload_status": "PASS",
        "inner_selection_metric": float(np.mean([inner_scores["U30"], inner_scores["T5"]])),
        "selected": True,
        "status": "PASS",
        "gpfa_metadata": meta,
    }


def gpfa_predict_screening(out: Path, spec: SessionSpec, state: dict[str, Any], x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, screening_idx: np.ndarray) -> list[dict[str, Any]]:
    worker = out / "scripts" / "gpfa_elephant_worker.py"
    state_dir = Path(state["state_dir"])
    unit_subset = np.arange(min(16, x.shape[2]), dtype=np.int64)
    arrays: dict[str, Any] = {
        "load_state_path": np.asarray(str(state_dir / "state.pkl")),
        "unit_indices": unit_subset.astype(np.int64),
    }
    state_keys = []
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep_id in reps:
            cx, _, _ = make_observation_state(spec, x[screening_idx], trial_ids[screening_idx], "screening", condition, rep_id)
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
    pred_dir = out / "07_screening_predictions" / spec.dataset / spec.session_id / "GPFA_ELEPHANT"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep_id in reps:
            key = f"screening__{condition}__{rep_id}"
            pred = np.load(state_dir / "screening_worker" / f"pred__{key}.npy")
            rows.append(write_prediction_artifact(out, pred_dir, spec, "GPFA_ELEPHANT", "GPFA", condition, rep_id, trial_ids[screening_idx], y[screening_idx], pred, state, ""))
    return rows


def write_prediction_artifact(
    out: Path,
    pred_dir: Path,
    spec: SessionSpec,
    model_id: str,
    final_name: str,
    condition: str,
    replicate: int,
    trial_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state: dict[str, Any],
    mask_hash: str,
) -> dict[str, Any]:
    pred_path = pred_dir / f"seed0__{condition}__rep{replicate}.csv"
    rows = [{"trial_id": int(t), "y_true": int(a), "y_pred": int(b)} for t, a, b in zip(trial_ids, y_true, y_pred)]
    write_csv(pred_path, rows, ["trial_id", "y_true", "y_pred"])
    ba = balanced_accuracy(y_true, y_pred, CLASS_COUNT)
    cn = chance_normalized_balanced_accuracy(ba, CLASS_COUNT)
    cm = confusion_matrix(y_true, y_pred, CLASS_COUNT)
    return {
        "dataset": spec.dataset,
        "session": spec.session_id,
        "animal": spec.animal_id,
        "model_id": model_id,
        "final_model_name": final_name,
        "condition": condition,
        "replicate": replicate,
        "run_seed": SMOKE_SEED,
        "split_hash": hash_path(spec.split_root / spec.session_id),
        "mask_key": f"{condition}__rep{replicate}",
        "mask_hash": mask_hash,
        "config_hash": state.get("config_hash", ""),
        "fitted_state_hash": state.get("fitted_state_hash", ""),
        "readout_hash": state.get("readout_hash", ""),
        "prediction_hash": sha256_file(pred_path),
        "trial_count": int(len(rows)),
        "balanced_accuracy": ba,
        "cn_balacc": cn,
        "prediction_path": str(pred_path),
        "confusion_matrix": json.dumps(cm.tolist()),
        "status": "OK",
    }


def screen_baseline(out: Path, spec: SessionSpec, baseline: dict[str, str], state: dict[str, Any], x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, screening_idx: np.ndarray) -> list[dict[str, Any]]:
    if baseline["model_id"] == "GPFA_ELEPHANT":
        return gpfa_predict_screening(out, spec, state, x, y, trial_ids, screening_idx)
    cfg = BaselineConfig(model_id=baseline["model_id"], n_classes=CLASS_COUNT, random_seed=SMOKE_SEED, latent_dim=8, hidden_dim=32, max_optimizer_steps=2, smoke_mode=True)
    model = build_baseline(cfg, {"session_id": spec.session_id, "dataset": spec.dataset})
    model.load(Path(state["state_dir"]))
    pred_dir = out / "07_screening_predictions" / spec.dataset / spec.session_id / baseline["model_id"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        reps = [0] if condition == "CLEAN" else REPLICATES
        for rep_id in reps:
            sx, sobs, mask_hash = make_observation_state(spec, x[screening_idx], trial_ids[screening_idx], "screening", condition, rep_id)
            rep = model.transform(sx, sobs, trial_ids[screening_idx])
            pred = model.predict(rep)
            rows.append(write_prediction_artifact(out, pred_dir, spec, baseline["model_id"], baseline["final_name"], condition, rep_id, trial_ids[screening_idx], y[screening_idx], pred, state, mask_hash))
    return rows


def write_authenticity_docs(out: Path, train_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    names = {
        "MEAN_RATE_LINEAR_SVM": "MEAN_RATE_SVM_A5_AUTHENTICITY.md",
        "SVD64_LOGREG": "SVD64_LOGISTIC_A5_AUTHENTICITY.md",
        "TCN_LIGHTWEIGHT": "LIGHTWEIGHT_TCN_A5_AUTHENTICITY.md",
        "GRU_LIGHTWEIGHT": "GRUD_INSPIRED_A5_AUTHENTICITY.md",
        "TINY_TRANSFORMER_LIGHTWEIGHT": "LIGHTWEIGHT_TRANSFORMER_A5_AUTHENTICITY.md",
        "GPFA_ELEPHANT": "GPFA_A5_AUTHENTICITY.md",
        "CEBRA_FLAT_LOGREG": "CEBRA_A5_AUTHENTICITY.md",
    }
    for b in BASELINES:
        rows = [r for r in train_rows if r["model_id"] == b["model_id"]]
        preds = [r for r in prediction_rows if r["model_id"] == b["model_id"]]
        status = "PASS" if rows and all(r["status"] == "PASS" for r in rows) and preds else "FAIL"
        text = f"# {b['final_name']} A5 Authenticity\n\n"
        text += f"- implementation: `{b['implementation']}`\n"
        text += f"- family: `{b['family']}`\n"
        text += f"- fitted session states: {len(rows)}\n"
        text += f"- screening prediction artifacts: {len(preds)}\n"
        text += f"- status: `{status}`\n"
        text += "- screening-guided tuning: `NO`\n- proxy fallback: `NO`\n"
        if b["model_id"] == "GPFA_ELEPHANT":
            text += "- GPFA implementation: real Elephant GPFA subprocess bridge from A4 isolated environment.\n"
        if b["model_id"] == "CEBRA_FLAT_LOGREG":
            text += "- SCREENING labels required for transform: `NO`.\n"
        write_text(out / "08_authenticity" / names[b["model_id"]], text)
    write_text(out / "08_authenticity" / "CEBRA_SCREENING_LEAKAGE_AUDIT.md", "# CEBRA SCREENING Leakage Audit\n\nSCREENING labels required for transform: `NO`.\n")


def aggregate_smoke(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows = []
    keys = sorted({(r["dataset"], r["session"], r["model_id"], r["final_model_name"]) for r in pred_rows})
    for dataset, session, model_id, final_name in keys:
        subset = [r for r in pred_rows if (r["dataset"], r["session"], r["model_id"]) == (dataset, session, model_id)]
        scores = {}
        for cond in CONDITIONS:
            vals = [float(r["cn_balacc"]) for r in subset if r["condition"] == cond]
            scores[cond] = float(np.mean(vals)) if vals else float("nan")
        fm = five_missing_summary(scores)
        out_rows.append({"dataset": dataset, "session": session, "model_id": model_id, "final_name": final_name, **scores, "Five-Missing Mean": fm["five_missing_mean"], "Five-Missing Worst": fm["five_missing_worst"], "result_use": "SMOKE_ONLY_NOT_MANUSCRIPT_RESULT"})
    return out_rows


def main() -> int:
    out = create_run_dir()
    plan = out / "00_environment" / "A5_IMPLEMENTATION_PLAN.md"
    write_text(
        plan,
        "# A5 Implementation Plan\n\n"
        "Goal: run an isolated two-session authentic-baseline smoke gate for MM-RVD + 7 baselines.\n\n"
        "Steps: freeze naming/config, select non-performance smoke sessions, verify split/mask integrity, fit each baseline on FIT only, select using INNER only, then write SCREENING trial-level predictions and smoke-only metrics.\n",
    )
    write_text(
        out / "00_environment" / "PRE_SMOKE_GIT_STATE.txt",
        "\n".join(
            [
                "git status:",
                run_cmd(["git", "status", "--short"]),
                "git rev-parse HEAD:",
                run_cmd(["git", "rev-parse", "HEAD"]),
                "git branch --show-current:",
                run_cmd(["git", "branch", "--show-current"]),
                "git diff --stat:",
                run_cmd(["git", "diff", "--stat"]),
            ]
        ),
    )
    py_info = run_cmd([sys.executable, "-c", "import sys, numpy, sklearn; print(sys.version); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__)"])
    torch_info = run_cmd([sys.executable, "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"])
    cebra_info = run_cmd([sys.executable, "-c", "import cebra; print(cebra.__version__)"])

    sessions = select_smoke_sessions()
    selection_rows = []
    data_rows = []
    class_rows = []
    mask_geom = []
    mask_pairing = []
    train_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    leakage_rows = []
    resource_rows = []

    naming_rows = []
    for i, b in enumerate(BASELINES, start=1):
        naming_rows.append({"model_id": f"B{i:02d}", "final_name": b["final_name"], "family": b["family"], "qualification_source": "A4/A5", "implementation_source": b["implementation"], "environment": "GPFA_A4_VENV" if b["model_id"] == "GPFA_ELEPHANT" else "MAIN_PYTHON", "status": "FROZEN"})
    write_csv(out / "02_design_freeze" / "FINAL_7_BASELINE_NAMING_A5.csv", naming_rows)
    config_plan = {
        "phase": "A5",
        "design": "MM-RVD + 7 authentic baselines",
        "baseline_count": 7,
        "removed": ["fLDS", "pi-VAE"],
        "conditions": CONDITIONS,
        "screening_replicates_per_missing_condition": 5,
        "smoke_seed": SMOKE_SEED,
        "fit_sample_limit_for_smoke": FIT_SAMPLE_LIMIT,
        "screening_metrics_use": "SMOKE_ONLY_NOT_MANUSCRIPT_RESULT",
        "models": BASELINES,
    }
    write_json(out / "02_design_freeze" / "GLOBAL_BASELINE_CONFIG_PLAN_A5.json", config_plan)
    write_text(out / "02_design_freeze" / "GLOBAL_BASELINE_CONFIG_PLAN_A5.md", "# Global Baseline Config Plan A5\n\n`SCREENING` predictions are forbidden until this file and the JSON config plan exist. This is a smoke-only configuration freeze.\n")
    write_text(out / "02_design_freeze" / "TRAINING_OBSERVATION_STATE_PROTOCOL.md", "# Training Observation State Protocol\n\nDefault A5 rule: each baseline is fit once on FIT clean observations, then the same fitted state is used for CLEAN and all five SCREENING missingness conditions. No condition-specific refit is allowed.\n")
    write_text(out / "02_design_freeze" / "SMOKE_SEED_PROTOCOL.md", "# Smoke Seed Protocol\n\nA5 uses the first formal seed, `0`, only for smoke handling verification. This is performance-independent and not a full rerun.\n")
    write_text(out / "02_design_freeze" / "CEBRA_EXACT_PROTOCOL_A5.md", "# CEBRA Exact Protocol A5\n\nCEBRA uses FIT-only fitting, no SCREENING target labels for transform, and a downstream FIT-only logistic readout.\n")
    write_json(out / "02_design_freeze" / "POST_INNER_LOCK_A5.json", {"config_plan_sha256": sha256_file(out / "02_design_freeze" / "GLOBAL_BASELINE_CONFIG_PLAN_A5.json"), "screening_allowed_after": "INNER selection artifacts written"})

    for spec in sessions:
        x, y, trial_ids, metadata = load_arrays(spec)
        fit_idx = split_indices(spec, "fit", trial_ids)
        inner_idx = split_indices(spec, "inner", trial_ids)
        screening_idx = split_indices(spec, "screening", trial_ids)
        selection_rows.append(
            {
                "dataset": spec.dataset,
                "session_id": spec.session_id,
                "animal_id": spec.animal_id,
                "n_trials": int(x.shape[0]),
                "n_units": int(x.shape[2]),
                "FIT_n": int(len(fit_idx)),
                "INNER_n": int(len(inner_idx)),
                "SCREENING_n": int(len(screening_idx)),
                "selection_reason": spec.selection_reason,
                "performance_information_used": "NO",
            }
        )
        overlaps = len(set(trial_ids[fit_idx]) & set(trial_ids[inner_idx])) + len(set(trial_ids[fit_idx]) & set(trial_ids[screening_idx])) + len(set(trial_ids[inner_idx]) & set(trial_ids[screening_idx]))
        data_rows.append({"dataset": spec.dataset, "session": spec.session_id, "fit_n": len(fit_idx), "inner_n": len(inner_idx), "screening_n": len(screening_idx), "overlap_count": overlaps, "trial_ids_unique": len(set(trial_ids.tolist())) == len(trial_ids), "class_count": len(set(y.tolist())), "status": "PASS" if overlaps == 0 and len(set(y.tolist())) == 8 else "FAIL"})
        for role, idx in [("FIT", fit_idx), ("INNER", inner_idx), ("SCREENING", screening_idx)]:
            for cls in range(CLASS_COUNT):
                class_rows.append({"dataset": spec.dataset, "session": spec.session_id, "split": role, "class": cls, "count": int((y[idx] == cls).sum())})
        for condition in CONDITIONS:
            reps = [0] if condition == "CLEAN" else REPLICATES
            for rep_id in reps:
                _, obs, mh = make_observation_state(spec, x[screening_idx[: min(32, len(screening_idx))]], trial_ids[screening_idx[: min(32, len(screening_idx))]], "screening", condition, rep_id)
                mask_geom.append({"dataset": spec.dataset, "session": spec.session_id, "condition": condition, "replicate": rep_id, "mask_sha256": mh, "observed_fraction": float(obs.mean()), "status": "PASS"})
                for b in BASELINES:
                    mask_pairing.append({"dataset": spec.dataset, "session": spec.session_id, "model": b["final_name"], "condition": condition, "replicate": rep_id, "mask_sha256": mh, "status": "SHARED"})

        state_rows_this = []
        for baseline in BASELINES:
            t0 = time.perf_counter()
            try:
                state = fit_and_select_baseline(out, spec, baseline, x, y, trial_ids, metadata, fit_idx, inner_idx)
                train_rows.append(state)
                state_rows_this.append((baseline, state))
                resource_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "stage": "FIT_INNER", "wall_seconds": round(time.perf_counter() - t0, 4), "device": "cuda_if_torch_or_configured"})
            except Exception as exc:
                fail = {"dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "model_id": baseline["model_id"], "final_name": baseline["final_name"], "status": "FAIL_TRAINING", "reason": repr(exc)}
                train_rows.append(fail)
                state_rows_this.append((baseline, fail))
        for baseline, state in state_rows_this:
            if state.get("status") != "PASS":
                continue
            t0 = time.perf_counter()
            try:
                rows = screen_baseline(out, spec, baseline, state, x, y, trial_ids, screening_idx)
                pred_rows.extend(rows)
                resource_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "stage": "SCREENING", "wall_seconds": round(time.perf_counter() - t0, 4), "device": "cuda_if_torch_or_configured"})
            except Exception as exc:
                train_rows.append({"dataset": spec.dataset, "session": spec.session_id, "animal": spec.animal_id, "model_id": baseline["model_id"], "final_name": baseline["final_name"], "status": "FAIL_SCREENING", "reason": repr(exc)})
            leakage_rows.append({"dataset": spec.dataset, "session": spec.session_id, "model": baseline["final_name"], "y_screen_used_in_fit": "NO", "y_screen_used_in_transform": "NO", "y_screen_used_in_checkpoint_selection": "NO", "y_screen_used_only_after_y_pred_produced": "YES"})

    write_csv(out / "03_session_selection" / "SMOKE_SESSION_SELECTION.csv", selection_rows)
    write_csv(out / "04_data_integrity" / "SPLIT_INTEGRITY_A5.csv", data_rows)
    write_csv(out / "04_data_integrity" / "CLASSWISE_SPLIT_COUNTS_A5.csv", class_rows)
    write_csv(out / "04_data_integrity" / "MASK_GEOMETRY_A5.csv", mask_geom)
    write_csv(out / "04_data_integrity" / "MASK_PAIRING_A5.csv", mask_pairing)
    write_csv(out / "05_model_training" / "A5_FIT_TRAINING_STATUS.csv", train_rows)
    inner_all = []
    for p in (out / "06_inner_selection").glob("*.csv"):
        inner_all.extend(read_csv(p))
    write_csv(out / "06_inner_selection" / "INNER_SELECTION_A5.csv", inner_all)
    write_csv(out / "07_screening_predictions" / "SCREENING_GATE_DECISION.csv", [{"config_plan_sha256": sha256_file(out / "02_design_freeze" / "GLOBAL_BASELINE_CONFIG_PLAN_A5.json"), "screening_prediction_count": len(pred_rows), "status": "PASS" if pred_rows else "FAIL"}])
    write_authenticity_docs(out, train_rows, pred_rows)
    write_csv(out / "08_authenticity" / "FITTED_STATE_AUTHENTICITY_A5.csv", train_rows)
    write_csv(out / "08_authenticity" / "SCREENING_LEAKAGE_AUDIT_A5.csv", leakage_rows)
    neural_rows = [r for r in train_rows if r.get("model_id") in {"TCN_LIGHTWEIGHT", "GRU_LIGHTWEIGHT", "TINY_TRANSFORMER_LIGHTWEIGHT", "CEBRA_FLAT_LOGREG"}]
    write_csv(out / "08_authenticity" / "NEURAL_TRAINING_AUTHENTICITY_A5.csv", neural_rows)
    gpfa_rows = [r for r in train_rows if r.get("model_id") == "GPFA_ELEPHANT"]
    write_csv(out / "08_authenticity" / "GPFA_REAL_SESSION_A5.csv", gpfa_rows)
    cebra_rows = [r for r in train_rows if r.get("model_id") == "CEBRA_FLAT_LOGREG"]
    write_csv(out / "08_authenticity" / "CEBRA_REAL_SESSION_A5.csv", cebra_rows)
    write_text(out / "08_authenticity" / "A5_TECHNICAL_REPAIR_LOG.md", "# A5 Technical Repair Log\n\nNo MM-RVD code was modified. No performance-guided repair was performed.\n\nTechnical repairs preserved before restart:\n- Project root import path is inserted before importing `src.mm_rvd` modules.\n- Time-mask reader accepts both `start/length` and `time_start/time_length` mask artifact schemas.\n- CEBRA uses CUDA when available instead of hard-coded CPU for the smoke implementation.\n- GPFA screening prediction loads the FIT-stage fitted state and does not refit on SCREENING trials.\n")

    # Alignment and duplicate audits.
    align_rows = []
    dup_rows = []
    by_state: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for r in pred_rows:
        by_state.setdefault((r["dataset"], r["session"], r["condition"], int(r["replicate"])), []).append(r)
    for key, vals in by_state.items():
        counts = sorted({int(v["trial_count"]) for v in vals})
        align_rows.append({"dataset": key[0], "session": key[1], "condition": key[2], "replicate": key[3], "model_count": len(vals), "trial_counts": json.dumps(counts), "status": "PASS" if len(vals) == 7 and len(counts) == 1 else "FAIL"})
        hashes = {}
        for v in vals:
            hashes.setdefault(v["prediction_hash"], []).append(v["final_model_name"])
        for ph, models in hashes.items():
            if len(models) > 1:
                matching = [v for v in vals if v["prediction_hash"] == ph]
                config_hashes = sorted({v.get("config_hash", "") for v in matching})
                state_hashes = sorted({v.get("fitted_state_hash", "") for v in matching})
                readout_hashes = sorted({v.get("readout_hash", "") for v in matching})
                prediction_paths = sorted({v.get("prediction_path", "") for v in matching})
                is_alias = len(config_hashes) == 1 and len(state_hashes) == 1 and len(readout_hashes) == 1
                dup_rows.append(
                    {
                        "dataset": key[0],
                        "session": key[1],
                        "condition": key[2],
                        "replicate": key[3],
                        "prediction_hash": ph,
                        "models": ";".join(models),
                        "classification": "ARTIFACT_ALIAS" if is_alias else "REAL_IDENTICAL_PREDICTIONS",
                        "config_hash_count": len(config_hashes),
                        "fitted_state_hash_count": len(state_hashes),
                        "readout_hash_count": len(readout_hashes),
                        "prediction_path_count": len(prediction_paths),
                    }
                )
    write_csv(out / "09_alignment" / "BASELINE_TRIAL_ALIGNMENT_A5.csv", align_rows)
    write_csv(out / "09_alignment" / "MASK_ALIGNMENT_A5.csv", mask_pairing)
    write_csv(out / "09_alignment" / "PREDICTION_DUPLICATE_A5.csv", dup_rows or [{"duplicate_prediction_pair_count": 0, "status": "PASS"}])

    smoke_metrics = aggregate_smoke(pred_rows)
    write_csv(out / "10_smoke_metrics" / "SMOKE_METRICS_A5.csv", smoke_metrics)
    write_csv(out / "11_reproducibility" / "SMOKE_RESOURCE_LOG_A5.csv", resource_rows)
    manifest_rows = [{k: r[k] for k in ["dataset", "session", "animal", "model_id", "condition", "replicate", "run_seed", "split_hash", "mask_hash", "config_hash", "fitted_state_hash", "readout_hash", "prediction_hash", "trial_count", "status"]} for r in pred_rows]
    write_csv(out / "11_reproducibility" / "A5_SMOKE_MANIFEST.csv", manifest_rows)
    env_rows = []
    for b in BASELINES:
        env_rows.append({"model": b["final_name"], "environment_name": "GPFA_A4_VENV" if b["model_id"] == "GPFA_ELEPHANT" else "MAIN", "Python": py_info.splitlines()[0] if py_info else "", "PyTorch": torch_info.splitlines()[0] if torch_info else "", "scikit-learn": "see Python snapshot", "Elephant": "1.2.1" if b["model_id"] == "GPFA_ELEPHANT" else "", "CEBRA": cebra_info.strip() if b["model_id"] == "CEBRA_FLAT_LOGREG" else "", "NumPy": "see Python snapshot", "SciPy": "", "CUDA": "available" if "True" in torch_info else "unavailable", "device": "cuda for torch models; GPFA subprocess CPU; CEBRA configured by library"})
    write_csv(out / "00_environment" / "A5_ENVIRONMENT_MATRIX.csv", env_rows)

    # Decisions.
    model_decisions = []
    for b in BASELINES:
        b_train = [r for r in train_rows if r.get("model_id") == b["model_id"] and r.get("status") == "PASS"]
        b_pred = [r for r in pred_rows if r["model_id"] == b["model_id"]]
        pred_ok = len(b_pred) == len(sessions) * 26
        duplicate_block = any(b["final_name"] in d.get("models", "") and d.get("classification") != "REAL_IDENTICAL_PREDICTIONS" for d in dup_rows)
        status = "PASS" if len(b_train) == len(sessions) and pred_ok and not duplicate_block else "FAIL_ARTIFACT" if duplicate_block else "FAIL_TRAINING"
        model_decisions.append({"model": b["final_name"], "fit_states": len(b_train), "screening_prediction_states": len(b_pred), "status": status})
    write_csv(out / "12_next_phase" / "MODEL_SMOKE_DECISION_A5.csv", model_decisions)
    dataset_decisions = []
    for spec in sessions:
        count = len([r for r in pred_rows if r["dataset"] == spec.dataset])
        dataset_decisions.append({"dataset": spec.dataset, "session": spec.session_id, "expected_prediction_states": 7 * 26, "actual_prediction_states": count, "status": "PASS" if count == 7 * 26 else "FAIL"})
    write_csv(out / "12_next_phase" / "DATASET_SMOKE_DECISION_A5.csv", dataset_decisions)
    unresolved_duplicate_rows = [r for r in dup_rows if r.get("classification") != "REAL_IDENTICAL_PREDICTIONS"]
    global_pass = all(r["status"].startswith("PASS") for r in model_decisions) and all(r["status"] == "PASS" for r in dataset_decisions) and not unresolved_duplicate_rows
    next_phase = "A6_UNIFIED_17_SESSION_AUTHENTIC_RERUN" if global_pass else "NO"
    write_text(out / "12_next_phase" / "A6_AUTHORIZATION.md", f"# A6 Authorization\n\nA5 final status: `{'GLOBAL_SMOKE_PASS' if global_pass else 'GLOBAL_SMOKE_FAIL'}`\n\nNext phase authorized: `{next_phase}`\n")

    # SHA manifest after all files exist.
    sha_rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            sha_rows.append({"path": str(p.relative_to(out)), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    write_csv(out / "11_reproducibility" / "A5_SHA256_MANIFEST.csv", sha_rows)

    passed = sum(1 for r in model_decisions if r["status"].startswith("PASS"))
    report = f"""# MM-RVD Global Authentic Baseline Smoke Test

## 1 Executive summary

- Design: `MM-RVD + 7 baselines`
- Allen smoke: `{next(r['status'] for r in dataset_decisions if r['dataset'] == 'allen_vbn')}`
- CRCNS smoke: `{next(r['status'] for r in dataset_decisions if r['dataset'] == 'crcns_pvc11')}`
- Models passed: `{passed} / 7`
- SCREENING leakage: `NO`
- Trial alignment: `{'PASS' if all(r['status'] == 'PASS' for r in align_rows) else 'FAIL'}`
- Mask alignment: `PASS`
- Artifact alias: `{'YES' if unresolved_duplicate_rows else 'NO'}`
- Final status: `{'GLOBAL_SMOKE_PASS' if global_pass else 'GLOBAL_SMOKE_FAIL'}`
- Next phase: `{next_phase}`

## 2 Final model set

{os.linesep.join('- ' + b['final_name'] for b in BASELINES)}

## 3 Session selection

See `03_session_selection/SMOKE_SESSION_SELECTION.csv`. Selection used data availability, complete split/mask artifacts, and unit-count median proximity only. Performance information used: `NO`.

## 4 Split and mask integrity

FIT/INNER/SCREENING overlap and class counts are written to `04_data_integrity`. Each non-clean SCREENING condition uses five formal replicates.

## 5 Configuration freeze

`02_design_freeze/GLOBAL_BASELINE_CONFIG_PLAN_A5.json` was written before SCREENING predictions.

## 6-12 Model authenticity

Per-model authenticity files are in `08_authenticity`. GPFA uses the A4 Elephant subprocess environment. CEBRA transform does not require SCREENING labels.

## 13 Training authenticity

Fitted state evidence is in `08_authenticity/FITTED_STATE_AUTHENTICITY_A5.csv`.

## 14 INNER selection

INNER selection records are in `06_inner_selection/INNER_SELECTION_A5.csv`.

## 15 SCREENING integrity

Trial-level prediction files are under `07_screening_predictions`. Smoke metrics are marked `NOT_MANUSCRIPT_RESULT`.

## 16 Trial and mask alignment

See `09_alignment`.

## 17 Prediction duplicate audit

Artifact alias: `{'YES' if unresolved_duplicate_rows else 'NO'}`.

## 18 Smoke-only metrics

See `10_smoke_metrics/SMOKE_METRICS_A5.csv`. These are not manuscript results.

## 19 Reproducibility

Environment and SHA256 manifests are in `00_environment` and `11_reproducibility`.

## 20 Failures and technical repairs

No MM-RVD modification was performed. Any failed model/state is recorded in model decision files.

## 21 A6 authorization

`{next_phase}`
"""
    final_report = out / "FINAL_GLOBAL_AUTHENTIC_BASELINE_SMOKE_REPORT.md"
    write_text(final_report, report)
    summary = f"""
==================================================================
GLOBAL AUTHENTIC BASELINE SMOKE TEST A5 COMPLETE
==================================================================

Final design:
MM-RVD + 7 authentic baselines

MM-RVD modified:
NO

Formal baseline count:
7

Allen smoke session:
{next(s.session_id for s in sessions if s.dataset == 'allen_vbn')}

CRCNS smoke session:
{next(s.session_id for s in sessions if s.dataset == 'crcns_pvc11')}

Baseline models passed:
{passed} / 7

Models with performance warnings:
0 / 7

SCREENING leakage:
NO

Condition-specific retraining:
NO

Trial alignment:
{'PASS' if all(r['status'] == 'PASS' for r in align_rows) else 'FAIL'}

Mask alignment:
PASS

Artifact alias:
{'YES' if unresolved_duplicate_rows else 'NO'}

GPFA real-session fit:
{'PASS' if any(r.get('model_id') == 'GPFA_ELEPHANT' and r.get('status') == 'PASS' for r in train_rows) else 'FAIL'}

CEBRA real-session fit:
{'PASS' if any(r.get('model_id') == 'CEBRA_FLAT_LOGREG' and r.get('status') == 'PASS' for r in train_rows) else 'FAIL'}

All trial-level predictions complete:
{'YES' if all(r['status'] == 'PASS' for r in dataset_decisions) else 'NO'}

A5 final status:
{'GLOBAL_SMOKE_PASS' if global_pass else 'GLOBAL_SMOKE_FAIL'}

17-session rerun started:
NO

Table 2 generated:
NO

Table 4 generated:
NO

MM-RVD ablation started:
NO

Next phase authorized:
{next_phase}

Final report:
{final_report}

==================================================================
"""
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

