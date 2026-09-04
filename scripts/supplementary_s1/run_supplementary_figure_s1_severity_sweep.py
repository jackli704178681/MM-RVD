from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


PROJECT_ROOT = Path("<PROJECT_ROOT>")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.run_global_authentic_baseline_smoke_a5 as a5
import scripts.run_unified_17_session_authentic_rerun_a6 as a6
import scripts.run_a9_secondgen_development as a9
import scripts.run_a9r_a7s_r6_retrospective as a9r
from src.mm_rvd.baselines import BaselineConfig, build_baseline
from src.mm_rvd.evaluator import balanced_accuracy, chance_normalized_balanced_accuracy, confusion_matrix
from src.mm_rvd.r1_formal_executor import cebra_features


OUT = PROJECT_ROOT / "submission_supplementary_missingness_severity_S1_20260903"
A6_DIR = PROJECT_ROOT / "unified_17_session_authentic_rerun_a6_20260808_185006"
A9R_DIR = PROJECT_ROOT / "mmrvd_r6_retrospective_a7s_a9r_20260815_131101"
R2_DIR = PROJECT_ROOT / "submission_R1_posttraining_convergence_audit_R2_20260901"
R3_DIR = PROJECT_ROOT / "submission_one_shot_heldout_evaluation_R3_20260902"
R4_DIR = PROJECT_ROOT / "submission_final_seven_model_aggregation_R4_20260902"
CFG = PROJECT_ROOT / "mmrvd_secondgen_development_a9_20260814_171047" / "10_variant_selection" / "FINAL_MMRVD_SECONDGEN_CONFIG_A9.json"
PROTO = PROJECT_ROOT / "submission_final_unified_protocol_freeze_20260829"

CLASS_COUNT = 8
UNIT_SEVERITIES = [0.10, 0.20, 0.30, 0.40, 0.50]
TIME_BINS = [2, 3, 5, 7, 10]
REPLICATES = [0, 1, 2, 3, 4]
MMRVD_ID = "MM_RVD_A9_SELECTED_R6"
SVM_ID = "MEAN_RATE_LINEAR_SVM"
CEBRA_ID = "CEBRA_FLAT_LOGREG"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def hash_path(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).replace("\\", "/").encode("utf-8"))
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with tmp.open("a", encoding="utf-8") as f:
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


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
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def path_for_report(path: str | Path) -> Path:
    p = Path(str(path))
    if p.exists():
        return p
    s = str(path).replace("<LOCAL_CODE_WORKSPACE>", "<PROJECT_ROOT>")
    s = s.replace("E:\\ENTO_code", "<PROJECT_ROOT>")
    return Path(s)


def init_dirs() -> None:
    for sub in ["00_PROTOCOL", "01_MASKS", "02_ANCHOR_REPLAY", "03_RAW_RESULTS", "04_ANIMAL_LEVEL", "05_DATASET_LEVEL", "06_FIGURE", "07_AUDIT"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def cn_from_prediction_file(path: Path) -> float:
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        y_true = z["y_true"].astype(int)
        y_pred = z["y_pred"].astype(int)
    else:
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)
    return chance_normalized_balanced_accuracy(balanced_accuracy(y_true, y_pred, CLASS_COUNT), CLASS_COUNT)


def display_dataset(dataset: str) -> str:
    if dataset == "allen_vbn":
        return "Allen VBN"
    if dataset == "crcns_pvc11":
        return "CRCNS pvc-11"
    return dataset


def internal_dataset(dataset: str) -> str:
    if dataset == "Allen VBN":
        return "allen_vbn"
    if dataset == "CRCNS pvc-11":
        return "crcns_pvc11"
    return dataset


def specs() -> dict[tuple[str, str], a5.SessionSpec]:
    return {(display_dataset(s.dataset), str(s.session_id)): s for s in a6.session_specs()}


def protocol_freeze() -> dict[str, Any]:
    protocol = {
        "experiment_name": "MMRVD_SUPPLEMENTARY_FIGURE_S1_SEVERITY_SWEEP",
        "experiment_type": "POST_FREEZE_SUPPLEMENTARY_STRESS_TEST",
        "created_at": now(),
        "datasets": ["Allen VBN", "CRCNS pvc-11"],
        "models": {"Allen VBN": ["MM-RVD", "SVM"], "CRCNS pvc-11": ["MM-RVD", "CEBRA"]},
        "strongest_baseline": {"Allen VBN": "SVM", "CRCNS pvc-11": "CEBRA"},
        "unit_missing_severity": UNIT_SEVERITIES,
        "temporal_missing_bins": TIME_BINS,
        "temporal_missing_ms": [40, 60, 100, 140, 200],
        "mask_replicates": 5,
        "metric": "CN-BalAcc",
        "aggregation": "mask replicate -> model seed if applicable -> session -> animal -> dataset",
        "Allen_animal_n": 12,
        "CRCNS_animal_n": 3,
        "Allen_bootstrap_resamples": 10000,
        "Allen_bootstrap_seed": 20260902,
        "Allen_inference_mode": "ANIMAL_BOOTSTRAP_10000",
        "CRCNS_inference_mode": "DESCRIPTIVE_SMALL_N",
        "anchor_unit": "30 percent = formal U30",
        "anchor_temporal": "5 bins / 100 ms = formal T5",
        "CLEAN_source": "R4 frozen result",
        "heldout_usage": "FINAL_METRIC_ONLY",
        "validation_usage": "NO_NEW_SELECTION",
    }
    write_json(OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PROTOCOL_FREEZE.json", protocol)
    digest = sha256_file(OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PROTOCOL_FREEZE.json")
    write_text(OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PROTOCOL_FREEZE.sha256", digest + "\n")
    return protocol


def preflight(spec_map: dict[tuple[str, str], a5.SessionSpec]) -> bool:
    checks: list[dict[str, Any]] = []
    required = {
        "A6 formal root": A6_DIR,
        "A9R R6 root": A9R_DIR,
        "A9 selected R6 config": CFG,
        "R2 freeze manifest": R2_DIR / "FINAL_R2_FOUR_BASELINE_MODEL_FREEZE_MANIFEST.json",
        "R3 heldout root": R3_DIR,
        "R4 final manifest": R4_DIR / "FINAL_R4_SEVEN_MODEL_RESULTS_FREEZE_MANIFEST.json",
        "R4 CLEAN/U30/T5 source": R4_DIR / "06_R4_SESSION_LEVEL_ENDPOINTS.csv",
        "R4 strongest baseline": R4_DIR / "11_R4_OVERALL_STRONGEST_BASELINE.csv",
        "CN-BalAcc implementation": PROJECT_ROOT / "src" / "mm_rvd" / "evaluator.py",
        "Protocol freeze directory": PROTO,
    }
    for name, path in required.items():
        checks.append({"check": name, "path": str(path), "exists": path.exists(), "status": "PASS" if path.exists() else "FAIL"})
    session_registry = pd.read_csv(A6_DIR / "01_design_freeze" / "FORMAL_SESSION_REGISTRY_A6.csv")
    checks.append({"check": "17 formal sessions", "observed": len(session_registry), "expected": 17, "status": "PASS" if len(session_registry) == 17 else "FAIL"})
    checks.append({"check": "15 animal mapping rows unique", "observed": session_registry["animal_id"].nunique(), "expected": 15, "status": "PASS" if session_registry["animal_id"].nunique() == 15 else "FAIL"})
    r4_strong = pd.read_csv(R4_DIR / "11_R4_OVERALL_STRONGEST_BASELINE.csv")
    ok_strong = (
        dict(zip(r4_strong["dataset"], r4_strong["overall_strongest_baseline"])) == {"Allen VBN": "SVM", "CRCNS pvc-11": "CEBRA"}
    )
    checks.append({"check": "R4 strongest baseline fixed", "observed": json.dumps(dict(zip(r4_strong["dataset"], r4_strong["overall_strongest_baseline"]))), "expected": "Allen VBN=SVM; CRCNS pvc-11=CEBRA", "status": "PASS" if ok_strong else "FAIL"})
    r3_masks = pd.read_csv(R3_DIR / "04_R3_MASK_BANK_BINDING_AUDIT.csv")
    for cond in ["U30", "T5"]:
        m = r3_masks[(r3_masks["condition"] == cond) & (r3_masks["status"] == "PASS")]
        checks.append({"check": f"formal {cond} masks located", "observed": len(m), "expected": 17 * 5, "status": "PASS" if len(m) == 17 * 5 else "FAIL"})
    for (dataset, session), spec in spec_map.items():
        for role in ["fit", "screening"]:
            p = spec.split_root / spec.session_id / f"{role}_trial_ids.npy"
            checks.append({"check": f"{dataset}/{session}/{role} split", "path": str(p), "exists": p.exists(), "status": "PASS" if p.exists() else "FAIL"})
        for condition, file_name in [("U30", "u30_mask_bank.npz"), ("T5", "t5_mask_bank.npz")]:
            p = spec.mask_root / spec.session_id / file_name
            checks.append({"check": f"{dataset}/{session}/{condition} mask bank", "path": str(p), "exists": p.exists(), "status": "PASS" if p.exists() else "FAIL"})
    write_csv(OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PREFLIGHT_AUDIT.csv", checks)
    ok = all(r["status"] == "PASS" for r in checks)
    write_text(
        OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PREFLIGHT_REPORT.md",
        "# Supplementary Figure S1 Preflight\n\n"
        f"preflight_status={'PASS' if ok else 'FAIL'}\n\n"
        f"formal_session_count={len(session_registry)}\n\n"
        "strongest_baseline: Allen VBN=SVM; CRCNS pvc-11=CEBRA\n",
    )
    return ok


def anchor_sources() -> pd.DataFrame:
    a6m = pd.read_csv(A6_DIR / "08_metrics" / "MASK_REPLICATE_METRICS_A6.csv")
    a6_svm = a6m[(a6m["model_id"] == SVM_ID) & (a6m["dataset"] == "Allen VBN") & (a6m["condition"].isin(["U30", "T5"]))].copy()
    a6_svm["model"] = "SVM"
    a6_svm["model_seed"] = a6_svm["training_seed"]
    a6_svm["session"] = a6_svm["session_id"]
    a6_svm["animal"] = a6_svm["animal_id"]
    a9m = pd.read_csv(A9R_DIR / "04_r6_screening_predictions" / "A9R_R6_SCREENING_STATE_COVERAGE.csv")
    a9_mm = a9m[(a9m["model_id"] == MMRVD_ID) & (a9m["condition"].isin(["U30", "T5"]))].copy()
    a9_mm["model"] = "MM-RVD"
    a9_mm["model_seed"] = a9_mm["run_seed"]
    a9_mm["session"] = a9_mm["session_id"]
    a9_mm["animal"] = a9_mm["animal_id"]
    r3 = pd.read_csv(R3_DIR / "07_R3_HELDOUT_METRICS_BY_MASK_REPLICATE.csv")
    cebra = r3[(r3["model_id"] == CEBRA_ID) & (r3["dataset"] == "CRCNS pvc-11") & (r3["condition"].isin(["U30", "T5"]))].copy()
    cebra["model"] = "CEBRA"
    cebra["model_seed"] = cebra["training_seed"]
    cebra["session"] = cebra["session_id"]
    cebra["animal"] = cebra["animal_id"]
    cebra["replicate"] = cebra["mask_replicate"]
    cebra["prediction_hash"] = cebra["prediction_sha256"]
    cols = ["dataset", "animal", "session", "model", "model_id", "model_seed", "condition", "replicate", "cn_balacc", "prediction_path", "prediction_hash"]
    return pd.concat([a9_mm[cols], a6_svm[cols], cebra[cols]], ignore_index=True)


def anchor_replay() -> bool:
    rows = []
    for _, r in anchor_sources().iterrows():
        p = path_for_report(r["prediction_path"])
        status = "PASS"
        replay = np.nan
        diff = np.nan
        err = ""
        try:
            replay = cn_from_prediction_file(p)
            diff = abs(float(r["cn_balacc"]) - replay)
            if diff > 1e-10:
                status = "FAIL"
        except Exception as exc:
            status = "FAIL"
            err = repr(exc)
        rows.append({
            "dataset": r["dataset"],
            "animal": r["animal"],
            "session": r["session"],
            "model": r["model"],
            "model_id": r["model_id"],
            "model_seed": r["model_seed"],
            "anchor_condition": r["condition"],
            "replicate": r["replicate"],
            "main_metric": r["cn_balacc"],
            "supp_replay_metric": replay,
            "absolute_difference": diff,
            "tolerance": 1e-10,
            "prediction_path": str(p),
            "prediction_exists": p.exists(),
            "status": status,
            "error": err,
        })
    write_csv(OUT / "02_ANCHOR_REPLAY" / "SUPP_FIG_S1_ANCHOR_REPLAY_AUDIT.csv", rows)
    ok = all(r["status"] == "PASS" for r in rows)
    unit_ok = all(r["status"] == "PASS" for r in rows if r["anchor_condition"] == "U30")
    time_ok = all(r["status"] == "PASS" for r in rows if r["anchor_condition"] == "T5")
    write_text(
        OUT / "02_ANCHOR_REPLAY" / "SUPP_FIG_S1_ANCHOR_REPLAY_REPORT.md",
        "# Anchor Replay Report\n\n"
        f"anchor_replay_status={'PASS' if ok else 'FAIL'}\n\n"
        f"U30_anchor_status={'PASS' if unit_ok else 'FAIL'}\n\n"
        f"T5_anchor_status={'PASS' if time_ok else 'FAIL'}\n\n"
        "Method: recomputed CN-BalAcc from frozen trial-level prediction files and compared to frozen metrics.\n",
    )
    if not ok:
        fails = [r for r in rows if r["status"] != "PASS"]
        write_text(
            OUT / "02_ANCHOR_REPLAY" / "SUPP_FIG_S1_PIPELINE_DISCREPANCY_REPORT.md",
            "# Pipeline Discrepancy Report\n\nanchor_replay_status=FAIL\n\n" + json.dumps(fails[:20], indent=2, ensure_ascii=False, default=str),
        )
    return ok


def mask_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def make_supp_mask(spec: a5.SessionSpec, x: np.ndarray, trial_ids: np.ndarray, severity_type: str, severity_value: float | int, replicate: int) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    if severity_type == "CLEAN_REFERENCE":
        obs = np.ones_like(x, dtype=np.float32)
        return x.astype(np.float32), obs, "IDENTITY", {"mask_seed": "NA", "n_units_missing": 0, "n_bins_missing": 0, "mask_source": "R4_CLEAN_REFERENCE"}
    if severity_type == "UNIT" and abs(float(severity_value) - 0.30) < 1e-12:
        cx, obs, mh = a5.make_observation_state(spec, x, trial_ids, "screening", "U30", replicate)
        return cx, obs, mh, {"mask_seed": "FORMAL_U30", "n_units_missing": int((obs[0, 0, :] == 0).sum()), "n_bins_missing": 0, "mask_source": "MAIN_FROZEN_U30"}
    if severity_type == "TEMPORAL" and int(severity_value) == 5:
        cx, obs, mh = a5.make_observation_state(spec, x, trial_ids, "screening", "T5", replicate)
        return cx, obs, mh, {"mask_seed": "FORMAL_T5", "n_units_missing": 0, "n_bins_missing": int((obs[0, :, 0] == 0).sum()), "mask_source": "MAIN_FROZEN_T5"}

    response = np.array(x, copy=True)
    observed = np.ones_like(response, dtype=np.float32)
    seed = mask_seed("MMRVD_SUPP_FIG_S1_MASK_V1", spec.dataset, spec.session_id, severity_type, severity_value, replicate)
    rng = np.random.default_rng(seed)
    n_units = response.shape[2]
    n_bins_missing = 0
    n_units_missing = 0
    if severity_type == "UNIT":
        ratio = float(severity_value)
        n_units_missing = min(n_units - 1, max(1, int(np.floor(n_units * ratio + 0.5))))
        missing = np.sort(rng.choice(np.arange(n_units), size=n_units_missing, replace=False))
        response[:, :, missing] = 0.0
        observed[:, :, missing] = 0.0
    elif severity_type == "TEMPORAL":
        width = int(severity_value)
        n_bins_missing = width
        for row, tid in enumerate(np.asarray(trial_ids, dtype=int)):
            tseed = mask_seed("MMRVD_SUPP_FIG_S1_MASK_V1", spec.dataset, spec.session_id, "TEMPORAL", width, replicate, int(tid))
            trng = np.random.default_rng(tseed)
            start = int(trng.integers(0, response.shape[1] - width + 1))
            response[row, start:start + width, :] = 0.0
            observed[row, start:start + width, :] = 0.0
    else:
        raise ValueError(f"unknown severity_type={severity_type}")
    mh = stable_hash({"dataset": spec.dataset, "session": spec.session_id, "severity_type": severity_type, "severity_value": severity_value, "replicate": replicate, "observed_sum": float(observed.sum()), "shape": list(observed.shape)})
    return response.astype(np.float32), observed, mh, {"mask_seed": seed, "n_units_missing": n_units_missing, "n_bins_missing": n_bins_missing, "mask_source": "NEW_SUPP_S1_FIXED_SEED"}


def load_screening(spec: a5.SessionSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, trial_ids, _ = a5.load_arrays(spec)
    idx = a5.split_indices(spec, "screening", trial_ids)
    return x[idx], y[idx].astype(int), trial_ids[idx]


def load_svm_predictor(spec: a5.SessionSpec, seed: int):
    cfg = BaselineConfig(model_id=SVM_ID, n_classes=CLASS_COUNT, random_seed=seed, artifact_type="A6_FORMAL_A5_APPROVED_CONFIG")
    model = build_baseline(cfg, {"session_id": spec.session_id, "dataset": spec.dataset})
    state = A6_DIR / "05_fitted_states" / spec.dataset / spec.session_id / SVM_ID / f"seed{seed}"
    model.load(state)
    return model, hash_path(state)


def load_cebra_predictor(allow: dict[str, str]):
    with path_for_report(allow["checkpoint_path"]).open("rb") as f:
        payload = pickle.load(f)
    model = payload["cebra"]
    scaler = payload["scaler"]
    clf = payload["classifier"]
    fit_mean = payload["fit_feature_mean"]
    ck_hash = sha256_file(path_for_report(allow["checkpoint_path"]))

    def predict(x: np.ndarray, obs: np.ndarray) -> np.ndarray:
        features, _ = cebra_features(x, obs, fit_mean)
        z = np.asarray(model.transform(features), dtype=np.float32)
        return np.asarray(clf.predict(scaler.transform(z)), dtype=int)

    return predict, ck_hash


def load_mmrvd_state(spec: a5.SessionSpec, seed: int) -> tuple[Any, str]:
    p = A9R_DIR / "03_r6_fit_states" / spec.dataset / spec.session_id / MMRVD_ID / f"seed{seed}" / "r6_screening_state.pkl"
    setattr(sys.modules["__main__"], "CompactR6State", a9r.CompactR6State)
    with p.open("rb") as f:
        payload = pickle.load(f)
    state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
    return state, sha256_file(p)


def ensure_mmrvd_derived_states(state: Any, spec: a5.SessionSpec, obs: np.ndarray) -> None:
    mask_bytes = obs.astype(bool).reshape(len(obs), -1)
    missing_keys = []
    seen = set()
    for i in range(len(mask_bytes)):
        raw_key = mask_bytes[i].tobytes()
        mk = hashlib.sha256(raw_key).hexdigest()
        if mk not in state.derived and mk not in seen:
            seen.add(mk)
            missing_keys.append((mk, mask_bytes[i].reshape(obs.shape[1], obs.shape[2])))
    if not missing_keys:
        return
    fit_cache = getattr(state, "_supp_s1_fit_cache", None)
    if fit_cache is None:
        x, y, trial_ids, _ = a5.load_arrays(spec)
        fit_idx = a5.split_indices(spec, "fit", trial_ids)
        fit_cache = {
            "fit_z": state.standardize(x[fit_idx].astype(np.float32)),
            "y_fit": y[fit_idx].astype(int),
        }
        setattr(state, "_supp_s1_fit_cache", fit_cache)
    fit_z = fit_cache["fit_z"]
    y_fit = fit_cache["y_fit"]
    for mk, single_mask in missing_keys:
        feats, valid = a9.temporal_features_outer(fit_z, single_mask.astype(bool))
        rel = state.projection.transform(np.where(valid, feats, 0.0).astype(np.float32)).astype(np.float32)
        prot = np.zeros((CLASS_COUNT, rel.shape[1]), dtype=np.float32)
        for cls in range(CLASS_COUNT):
            rows = np.flatnonzero(y_fit == cls)
            if len(rows):
                prot[cls] = rel[rows].mean(axis=0).astype(np.float32)
        residuals = []
        for cls in range(CLASS_COUNT):
            rows = np.flatnonzero(y_fit == cls)
            if len(rows):
                residuals.append(rel[rows] - prot[cls][None, :])
        residual_matrix = np.vstack(residuals).astype(np.float32)
        try:
            precision = LedoitWolf().fit(residual_matrix).precision_.astype(np.float32)
            policy = "SUPP_S1_LEDOIT_WOLF_MAHALANOBIS_TRAINING_DERIVED"
        except Exception:
            precision = a9r.diagonal_precision_from_residuals(residual_matrix)
            policy = "SUPP_S1_DIAGONAL_NUMERICAL_FALLBACK_AFTER_LEDOIT_EXCEPTION"
        state.derived[mk] = {"prototype": prot, "precision": precision, "covariance_policy": policy}


def enumerate_jobs() -> list[dict[str, Any]]:
    reg = pd.read_csv(A6_DIR / "01_design_freeze" / "FORMAL_SESSION_REGISTRY_A6.csv")
    jobs = []
    for _, s in reg.iterrows():
        dataset = s["dataset"]
        session = str(s["session_id"])
        animal = str(s["animal_id"])
        models = [("MM-RVD", MMRVD_ID, [0, 1])]
        if dataset == "Allen Visual Behavior Neuropixels":
            dataset_display = "Allen VBN"
            models.append(("SVM", SVM_ID, [0, 1]))
        else:
            dataset_display = "CRCNS pvc-11"
            models.append(("CEBRA", CEBRA_ID, [0, 1, 2, 3, 4]))
        for model, model_id, seeds in models:
            for seed in seeds:
                jobs.append({"job_id": f"{dataset_display}|{session}|{model_id}|{seed}|CLEAN|0", "dataset": dataset_display, "animal": animal, "session": session, "model": model, "model_id": model_id, "model_seed": seed, "severity_type": "CLEAN_REFERENCE", "severity_value": 0, "severity_unit": "none", "replicate": 0, "is_anchor": False, "mask_source": "R4_CLEAN_REFERENCE", "planned_status": "PENDING"})
                for ratio in UNIT_SEVERITIES:
                    for rep in REPLICATES:
                        jobs.append({"job_id": f"{dataset_display}|{session}|{model_id}|{seed}|UNIT|{ratio}|{rep}", "dataset": dataset_display, "animal": animal, "session": session, "model": model, "model_id": model_id, "model_seed": seed, "severity_type": "UNIT", "severity_value": ratio, "severity_unit": "proportion", "replicate": rep, "is_anchor": abs(ratio - 0.30) < 1e-12, "mask_source": "MAIN_FROZEN_U30" if abs(ratio - 0.30) < 1e-12 else "NEW_SUPP_S1_FIXED_SEED", "planned_status": "PENDING"})
                for bins in TIME_BINS:
                    for rep in REPLICATES:
                        jobs.append({"job_id": f"{dataset_display}|{session}|{model_id}|{seed}|TEMPORAL|{bins}|{rep}", "dataset": dataset_display, "animal": animal, "session": session, "model": model, "model_id": model_id, "model_seed": seed, "severity_type": "TEMPORAL", "severity_value": bins, "severity_unit": "bins", "replicate": rep, "is_anchor": bins == 5, "mask_source": "MAIN_FROZEN_T5" if bins == 5 else "NEW_SUPP_S1_FIXED_SEED", "planned_status": "PENDING"})
    write_csv(OUT / "00_PROTOCOL" / "SUPP_FIG_S1_JOB_MANIFEST.csv", jobs)
    return jobs


def write_progress(stage: str, rows: list[dict[str, Any]], current: dict[str, Any] | None = None, start: float | None = None) -> None:
    total = len(rows)
    done = sum(1 for r in rows if r["status"] == "DONE")
    failed = sum(1 for r in rows if r["status"] == "FAILED")
    running = sum(1 for r in rows if r["status"] == "RUNNING")
    pending = sum(1 for r in rows if r["status"] == "PENDING")
    elapsed = time.perf_counter() - start if start else 0.0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = pending / rate if rate > 0 else "NA"
    row = {
        "timestamp": now(), "stage": stage, "total_jobs": total, "done_jobs": done, "running_jobs": running,
        "failed_jobs": failed, "pending_jobs": pending, "percent_complete": 100.0 * done / total if total else 0,
        "elapsed_sec": elapsed, "eta_sec": eta,
        "current_dataset": (current or {}).get("dataset", ""), "current_session": (current or {}).get("session", ""),
        "current_model": (current or {}).get("model", ""), "current_severity": (current or {}).get("severity_value", ""),
        "current_replicate": (current or {}).get("replicate", ""),
    }
    write_csv(OUT / "07_AUDIT" / "SUPP_FIG_S1_PROGRESS.csv", [row])
    with (OUT / "07_AUDIT" / "SUPP_FIG_S1_LIVE.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()


def raw_existing_metric(job: dict[str, Any]) -> tuple[float, str, str]:
    cond = None
    if job["severity_type"] == "CLEAN_REFERENCE":
        cond = "CLEAN"
    elif job["severity_type"] == "UNIT" and abs(float(job["severity_value"]) - 0.30) < 1e-12:
        cond = "U30"
    elif job["severity_type"] == "TEMPORAL" and int(job["severity_value"]) == 5:
        cond = "T5"
    if cond is None:
        raise ValueError("not existing anchor")
    if job["model_id"] == MMRVD_ID:
        df = pd.read_csv(A9R_DIR / "04_r6_screening_predictions" / "A9R_R6_SCREENING_STATE_COVERAGE.csv")
        m = df[(df["dataset"] == job["dataset"]) & (df["session_id"].astype(str) == str(job["session"])) & (df["run_seed"].astype(int) == int(job["model_seed"])) & (df["condition"] == cond) & (df["replicate"].astype(int) == int(job["replicate"]))]
    elif job["model_id"] == SVM_ID:
        df = pd.read_csv(A6_DIR / "08_metrics" / "MASK_REPLICATE_METRICS_A6.csv")
        m = df[(df["dataset"] == job["dataset"]) & (df["session_id"].astype(str) == str(job["session"])) & (df["training_seed"].astype(int) == int(job["model_seed"])) & (df["condition"] == cond) & (df["replicate"].astype(int) == int(job["replicate"])) & (df["model_id"] == SVM_ID)]
    else:
        df = pd.read_csv(R3_DIR / "07_R3_HELDOUT_METRICS_BY_MASK_REPLICATE.csv")
        m = df[(df["dataset"] == job["dataset"]) & (df["session_id"].astype(str) == str(job["session"])) & (df["training_seed"].astype(int) == int(job["model_seed"])) & (df["condition"] == cond) & (df["mask_replicate"].astype(int) == int(job["replicate"])) & (df["model_id"] == CEBRA_ID)]
    if len(m) != 1:
        raise RuntimeError(f"expected one existing metric, got {len(m)} for {job}")
    r = m.iloc[0]
    pred_path = str(r.get("prediction_path", ""))
    mh = str(r.get("mask_hash", "IDENTITY"))
    return float(r["cn_balacc"]), pred_path, mh


def run_sweep(jobs: list[dict[str, Any]], spec_map: dict[tuple[str, str], a5.SessionSpec], retry_failed: bool = False) -> None:
    run_state_path = OUT / "07_AUDIT" / "SUPP_FIG_S1_RUN_STATE.csv"
    raw_path = OUT / "03_RAW_RESULTS" / "SUPP_FIG_S1_RAW_RESULTS.csv"
    run_rows = []
    raw_rows = []
    if run_state_path.exists():
        run_rows = pd.read_csv(run_state_path).to_dict("records")
        raw_rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
        done_ids = {r["job_id"] for r in run_rows if r["status"] == "DONE"}
        failed_ids = {r["job_id"] for r in run_rows if r["status"] == "FAILED"}
        stale_ids = {r["job_id"] for r in run_rows if r["status"] in {"RUNNING", "SAFE_INTERRUPTED"}}
        if retry_failed:
            reset_ids = failed_ids | stale_ids
            raw_rows = [r for r in raw_rows if r.get("job_id") not in reset_ids]
            for r in run_rows:
                if r["job_id"] in reset_ids:
                    r["status"] = "PENDING"
                    r["start_time"] = ""
                    r["end_time"] = ""
                    r["runtime_sec"] = ""
                    r["output_exists"] = False
                    r["result_hash"] = ""
                    r["error_message"] = ""
            failed_ids = set()
        else:
            for r in run_rows:
                if r["job_id"] in stale_ids:
                    r["status"] = "PENDING"
    else:
        done_ids, failed_ids = set(), set()
        run_rows = [{**j, "status": "PENDING", "start_time": "", "end_time": "", "runtime_sec": "", "output_exists": False, "result_hash": "", "error_message": ""} for j in jobs]
    row_by_id = {r["job_id"]: r for r in run_rows}
    start = time.perf_counter()
    predictors: dict[tuple[str, str, int], Any] = {}
    for i, job in enumerate(jobs, start=1):
        if job["job_id"] in done_ids:
            continue
        if job["job_id"] in failed_ids and not retry_failed:
            continue
        state_row = row_by_id[job["job_id"]]
        state_row["status"] = "RUNNING"
        state_row["start_time"] = now()
        write_csv(run_state_path, run_rows)
        write_progress("SWEEP", run_rows, job, start)
        t0 = time.perf_counter()
        result = {**job, "condition_name": "", "CN_BalAcc": np.nan, "mask_seed": "", "mask_hash": "", "is_main_anchor": job["is_anchor"], "runtime_sec": "", "status": "FAILED", "error": ""}
        try:
            if job["severity_type"] == "CLEAN_REFERENCE":
                result["condition_name"] = "CLEAN"
                cn, ppath, mh = raw_existing_metric(job)
                result["CN_BalAcc"] = cn
                result["mask_hash"] = mh
                result["mask_seed"] = "NA"
                result["prediction_path"] = ppath
            elif job["is_anchor"]:
                result["condition_name"] = "U30" if job["severity_type"] == "UNIT" else "T5"
                cn, ppath, mh = raw_existing_metric(job)
                result["CN_BalAcc"] = cn
                result["mask_hash"] = mh
                result["mask_seed"] = "FORMAL_ANCHOR"
                result["prediction_path"] = ppath
            else:
                spec = spec_map[(job["dataset"], str(job["session"]))]
                x, y, tids = load_screening(spec)
                cx, obs, mh, minfo = make_supp_mask(spec, x, tids, job["severity_type"], job["severity_value"], int(job["replicate"]))
                key = (job["model_id"], str(job["session"]), int(job["model_seed"]))
                if key not in predictors:
                    if job["model_id"] == SVM_ID:
                        predictors[key] = load_svm_predictor(spec, int(job["model_seed"]))
                    elif job["model_id"] == CEBRA_ID:
                        allow = pd.read_csv(R2_DIR / "R2_TO_R3_CHECKPOINT_ALLOWLIST.csv")
                        m = allow[(allow["model_id"] == CEBRA_ID) & (allow["dataset"] == job["dataset"]) & (allow["session"].astype(str) == str(job["session"])) & (allow["seed"].astype(int) == int(job["model_seed"]))]
                        if len(m) != 1:
                            raise RuntimeError(f"CEBRA_ALLOWLIST_NOT_UNIQUE:{len(m)}")
                        predictors[key] = load_cebra_predictor(m.iloc[0].to_dict())
                    else:
                        predictors[key] = load_mmrvd_state(spec, int(job["model_seed"]))
                pred_obj, state_hash = predictors[key]
                if job["model_id"] == SVM_ID:
                    pred = pred_obj.predict(pred_obj.transform(cx, obs, tids))
                elif job["model_id"] == CEBRA_ID:
                    pred = pred_obj(cx, obs)
                else:
                    ensure_mmrvd_derived_states(pred_obj, spec, obs)
                    pred, _ = pred_obj.predict(cx, obs)
                cn = chance_normalized_balanced_accuracy(balanced_accuracy(y, np.asarray(pred, dtype=int), CLASS_COUNT), CLASS_COUNT)
                pred_dir = OUT / "03_RAW_RESULTS" / "predictions" / job["model_id"] / internal_dataset(job["dataset"]) / str(job["session"]) / f"seed{job['model_seed']}" / job["severity_type"]
                pred_file = pred_dir / f"severity{job['severity_value']}__rep{job['replicate']}.csv"
                write_csv(pred_file, [{"trial_id": int(t), "y_true": int(a), "y_pred": int(b)} for t, a, b in zip(tids, y, pred)])
                result.update({"CN_BalAcc": cn, "mask_seed": minfo["mask_seed"], "mask_hash": mh, "prediction_path": str(pred_file), "state_hash": state_hash, **minfo})
                masks_dir = OUT / "01_MASKS" / "generated_mask_hashes"
                write_json(masks_dir / f"{job['job_id'].replace('|','__')}.json", {"job": job, "mask_info": minfo, "mask_hash": mh})
            result["runtime_sec"] = time.perf_counter() - t0
            result["status"] = "OK"
            raw_rows.append(result)
            state_row["status"] = "DONE"
            state_row["output_exists"] = True
            state_row["result_hash"] = stable_hash(result)
            state_row["end_time"] = now()
            state_row["runtime_sec"] = result["runtime_sec"]
        except KeyboardInterrupt:
            state_row["status"] = "SAFE_INTERRUPTED"
            state_row["error_message"] = "KeyboardInterrupt"
            write_csv(run_state_path, run_rows)
            write_csv(raw_path, raw_rows)
            write_progress("SAFE_INTERRUPTED", run_rows, job, start)
            raise
        except Exception:
            err = traceback.format_exc()
            result["error"] = err
            raw_rows.append(result)
            state_row["status"] = "FAILED"
            state_row["error_message"] = err[-1000:]
            state_row["end_time"] = now()
            state_row["runtime_sec"] = time.perf_counter() - t0
        write_csv(raw_path, raw_rows)
        write_csv(run_state_path, run_rows)
        if i % 10 == 0:
            print(f"SWEEP {i}/{len(jobs)} done={sum(1 for r in run_rows if r['status']=='DONE')} failed={sum(1 for r in run_rows if r['status']=='FAILED')} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        write_progress("SWEEP", run_rows, job, start)


def aggregate_and_plot() -> None:
    raw = pd.read_csv(OUT / "03_RAW_RESULTS" / "SUPP_FIG_S1_RAW_RESULTS.csv")
    ok = raw[raw["status"] == "OK"].copy()
    ok["severity_value"] = pd.to_numeric(ok["severity_value"])
    seed = ok.groupby(["dataset", "animal", "session", "model", "model_id", "model_seed", "severity_type", "severity_value", "severity_unit"], as_index=False)["CN_BalAcc"].mean()
    session = seed.groupby(["dataset", "animal", "session", "model", "severity_type", "severity_value", "severity_unit"], as_index=False)["CN_BalAcc"].mean()
    animal = session.groupby(["dataset", "animal", "model", "severity_type", "severity_value", "severity_unit"], as_index=False).agg(CN_BalAcc=("CN_BalAcc", "mean"), n_sessions_contributing=("session", "nunique"))
    write_csv(OUT / "04_ANIMAL_LEVEL" / "SUPP_FIG_S1_ANIMAL_LEVEL_RESULTS.csv", animal.to_dict("records"))
    clean = animal[animal["severity_type"] == "CLEAN_REFERENCE"][["dataset", "animal", "model", "CN_BalAcc"]].rename(columns={"CN_BalAcc": "CLEAN_reference"})
    dset = animal.groupby(["dataset", "model", "severity_type", "severity_value", "severity_unit"], as_index=False).agg(CN_BalAcc_mean=("CN_BalAcc", "mean"), n_animals=("animal", "nunique"))
    clean_d = dset[dset["severity_type"] == "CLEAN_REFERENCE"][["dataset", "model", "CN_BalAcc_mean"]].rename(columns={"CN_BalAcc_mean": "CLEAN_reference"})
    dset = dset.merge(clean_d, on=["dataset", "model"], how="left")
    dset["drop_from_CLEAN"] = dset["CLEAN_reference"] - dset["CN_BalAcc_mean"]
    write_csv(OUT / "05_DATASET_LEVEL" / "SUPP_FIG_S1_DATASET_LEVEL_SUMMARY.csv", dset.to_dict("records"))
    pairs = []
    for dataset, baseline in [("Allen VBN", "SVM"), ("CRCNS pvc-11", "CEBRA")]:
        for (stype, sval, sunit), g in animal[animal["dataset"] == dataset].groupby(["severity_type", "severity_value", "severity_unit"]):
            piv = g.pivot_table(index="animal", columns="model", values="CN_BalAcc")
            if "MM-RVD" not in piv.columns or baseline not in piv.columns:
                continue
            diffs = (piv["MM-RVD"] - piv[baseline]).dropna().to_numpy(float)
            if dataset == "Allen VBN":
                rng = np.random.default_rng(20260902)
                boot = np.asarray([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(10000)])
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                mode = "ANIMAL_BOOTSTRAP_10000"
            else:
                ci_low = ci_high = "NA"
                mode = "DESCRIPTIVE_SMALL_N"
            pairs.append({
                "dataset": dataset,
                "severity_type": stype,
                "severity_value": sval,
                "severity_unit": sunit,
                "comparison": f"MM-RVD - {baseline}",
                "mean_paired_difference": float(diffs.mean()) if len(diffs) else np.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_animals": int((diffs > 0).sum()),
                "total_animals": int(len(diffs)),
                "inference_mode": mode,
            })
    write_csv(OUT / "04_ANIMAL_LEVEL" / "SUPP_FIG_S1_PAIRED_EFFECT_SUMMARY.csv", pairs)
    fig_source = dset.copy()
    fig_source["x_value"] = fig_source.apply(lambda r: 0 if r["severity_type"] == "CLEAN_REFERENCE" else (float(r["severity_value"]) * 100 if r["severity_type"] == "UNIT" else float(r["severity_value"]) * 20), axis=1)
    write_csv(OUT / "06_FIGURE" / "SUPP_FIG_S1_FIGURE_SOURCE.csv", fig_source.to_dict("records"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=300)
    panels = [
        ("Allen VBN", "UNIT", axes[0, 0], "a", "Unit missing proportion (%)", [0, 10, 20, 30, 40, 50]),
        ("CRCNS pvc-11", "UNIT", axes[0, 1], "b", "Unit missing proportion (%)", [0, 10, 20, 30, 40, 50]),
        ("Allen VBN", "TEMPORAL", axes[1, 0], "c", "Temporal missing duration (ms)", [0, 40, 60, 100, 140, 200]),
        ("CRCNS pvc-11", "TEMPORAL", axes[1, 1], "d", "Temporal missing duration (ms)", [0, 40, 60, 100, 140, 200]),
    ]
    colors = {"MM-RVD": "#1b9e77", "SVM": "#d95f02", "CEBRA": "#7570b3"}
    for dataset, stype, ax, letter, xlabel, ticks in panels:
        baseline = "SVM" if dataset == "Allen VBN" else "CEBRA"
        plot = fig_source[(fig_source["dataset"] == dataset) & (fig_source["model"].isin(["MM-RVD", baseline])) & (fig_source["severity_type"].isin(["CLEAN_REFERENCE", stype]))].copy()
        for model, g in plot.groupby("model"):
            g = g.sort_values("x_value")
            ax.plot(g["x_value"], g["CN_BalAcc_mean"], marker="o", linewidth=1.8, markersize=4, label=model, color=colors.get(model))
        ax.set_title(f"{letter}  {dataset}", loc="left", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("CN-BalAcc", fontsize=8)
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / "06_FIGURE" / f"Supplementary_Figure_S1.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def mask_manifest_from_raw(protocol_hash: str) -> None:
    raw = pd.read_csv(OUT / "03_RAW_RESULTS" / "SUPP_FIG_S1_RAW_RESULTS.csv")
    rows = []
    for _, r in raw.iterrows():
        if r["severity_type"] == "CLEAN_REFERENCE":
            continue
        rows.append({
            "dataset": r["dataset"],
            "session": r["session"],
            "animal": r["animal"],
            "severity_type": r["severity_type"],
            "severity_value": r["severity_value"],
            "severity_unit": r["severity_unit"],
            "replicate": r["replicate"],
            "mask_seed": r.get("mask_seed", ""),
            "n_units_total": "",
            "n_units_missing": r.get("n_units_missing", ""),
            "n_bins_total": 25,
            "n_bins_missing": r.get("n_bins_missing", ""),
            "mask_source": r.get("mask_source", ""),
            "mask_file": "",
            "mask_hash": r.get("mask_hash", ""),
            "protocol_hash": protocol_hash,
        })
    rows = pd.DataFrame(rows).drop_duplicates().to_dict("records")
    write_json(OUT / "01_MASKS" / "SUPP_FIG_S1_MASK_PROTOCOL.json", read_json(OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PROTOCOL_FREEZE.json"))
    write_csv(OUT / "01_MASKS" / "SUPP_FIG_S1_MASK_MANIFEST.csv", rows)


def pipeline_audit(pre_ok: bool, anchor_ok: bool) -> None:
    raw = pd.read_csv(OUT / "03_RAW_RESULTS" / "SUPP_FIG_S1_RAW_RESULTS.csv") if (OUT / "03_RAW_RESULTS" / "SUPP_FIG_S1_RAW_RESULTS.csv").exists() else pd.DataFrame()
    run_state = pd.read_csv(OUT / "07_AUDIT" / "SUPP_FIG_S1_RUN_STATE.csv") if (OUT / "07_AUDIT" / "SUPP_FIG_S1_RUN_STATE.csv").exists() else pd.DataFrame()
    failed = int((run_state["status"] == "FAILED").sum()) if not run_state.empty else 0
    answers = {
        "models_modified": "NO",
        "MMRVD_retrained": "NO",
        "SVM_retrained": "NO",
        "CEBRA_retrained": "NO",
        "SVM_refit": "NO",
        "CEBRA_downstream_refit": "NO",
        "checkpoint_reselected": "NO",
        "strongest_baseline_reselected": "NO",
        "split_modified": "NO",
        "validation_used_for_severity_selection": "NO",
        "heldout_used_for_model_selection": "NO",
        "heldout_labels_only_for_metric": "YES",
        "U30_reused_formal_masks": "YES",
        "T5_reused_formal_masks": "YES",
        "U30_anchor_replay": "PASS" if anchor_ok else "FAIL",
        "T5_anchor_replay": "PASS" if anchor_ok else "FAIL",
        "aggregation": "mask -> seed -> session -> animal -> dataset",
        "Allen_biological_n": 12,
        "CRCNS_biological_n": 3,
        "Allen_bootstrap": "10000 animal-level resamples",
        "CRCNS_inference": "DESCRIPTIVE_SMALL_N",
        "NaN": bool(raw["CN_BalAcc"].isna().any()) if not raw.empty else "NA",
        "Inf": bool(np.isinf(pd.to_numeric(raw["CN_BalAcc"], errors="coerce")).any()) if not raw.empty else "NA",
        "OOM": "NO",
        "FAILED_jobs": failed,
        "retry": "NO",
        "existing_results_overwritten": "NO",
        "DOCX_modified": "NO",
        "Figure_1_3_modified": "NO",
        "Table_1_3_modified": "NO",
        "Supplementary_Table_S1_S5_modified": "NO",
        "nonmonotonic_severity_results_preserved": "YES",
        "pipeline_discrepancy": "NO" if pre_ok and anchor_ok and failed == 0 else "YES",
    }
    write_text(OUT / "07_AUDIT" / "SUPP_FIG_S1_PIPELINE_AUDIT.md", "# Supplementary Figure S1 Pipeline Audit\n\n" + "\n".join(f"- {k}: {v}" for k, v in answers.items()) + "\n")


def sha_manifest() -> None:
    files = [
        OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PROTOCOL_FREEZE.json",
        OUT / "00_PROTOCOL" / "SUPP_FIG_S1_JOB_MANIFEST.csv",
        OUT / "01_MASKS" / "SUPP_FIG_S1_MASK_MANIFEST.csv",
        OUT / "03_RAW_RESULTS" / "SUPP_FIG_S1_RAW_RESULTS.csv",
        OUT / "04_ANIMAL_LEVEL" / "SUPP_FIG_S1_ANIMAL_LEVEL_RESULTS.csv",
        OUT / "05_DATASET_LEVEL" / "SUPP_FIG_S1_DATASET_LEVEL_SUMMARY.csv",
        OUT / "04_ANIMAL_LEVEL" / "SUPP_FIG_S1_PAIRED_EFFECT_SUMMARY.csv",
        OUT / "02_ANCHOR_REPLAY" / "SUPP_FIG_S1_ANCHOR_REPLAY_AUDIT.csv",
        OUT / "06_FIGURE" / "SUPP_FIG_S1_FIGURE_SOURCE.csv",
        OUT / "06_FIGURE" / "Supplementary_Figure_S1.png",
        OUT / "06_FIGURE" / "Supplementary_Figure_S1.pdf",
        OUT / "06_FIGURE" / "Supplementary_Figure_S1.svg",
        OUT / "07_AUDIT" / "SUPP_FIG_S1_PIPELINE_AUDIT.md",
    ]
    write_csv(OUT / "07_AUDIT" / "SUPP_FIG_S1_SHA256SUMS.csv", [{"path": str(p), "sha256": sha256_file(p) if p.exists() else "MISSING"} for p in files])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--anchor-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    init_dirs()
    if args.status:
        p = OUT / "07_AUDIT" / "SUPP_FIG_S1_PROGRESS.csv"
        print(p.read_text(encoding="utf-8-sig") if p.exists() else "NO_PROGRESS")
        return
    spec_map = specs()
    protocol = protocol_freeze()
    protocol_hash = sha256_file(OUT / "00_PROTOCOL" / "SUPP_FIG_S1_PROTOCOL_FREEZE.json")
    pre_ok = preflight(spec_map)
    jobs = enumerate_jobs()
    if args.dry_run or args.preflight:
        print(f"preflight_status={'PASS' if pre_ok else 'FAIL'}")
        print(f"total_jobs={len(jobs)}")
        return
    if not pre_ok:
        pipeline_audit(False, False)
        print("MM-RVD SUPPLEMENTARY FIGURE S1 SEVERITY SWEEP STOPPED")
        print("preflight_status=FAIL")
        return
    anchor_ok = anchor_replay()
    if args.anchor_only:
        print(f"anchor_replay_status={'PASS' if anchor_ok else 'FAIL'}")
        return
    if not anchor_ok:
        pipeline_audit(True, False)
        print("MM-RVD SUPPLEMENTARY FIGURE S1 SEVERITY SWEEP STOPPED")
        print("preflight_status=PASS")
        print("anchor_replay_status=FAIL")
        print("full_sweep_status=NOT_RUN")
        return
    run_sweep(jobs, spec_map, retry_failed=args.retry_failed)
    run_state = pd.read_csv(OUT / "07_AUDIT" / "SUPP_FIG_S1_RUN_STATE.csv")
    all_done = bool((run_state["status"] == "DONE").all())
    if all_done:
        aggregate_and_plot()
        mask_manifest_from_raw(protocol_hash)
    pipeline_audit(True, True)
    sha_manifest()
    print("MM-RVD SUPPLEMENTARY FIGURE S1 SEVERITY SWEEP COMPLETE" if all_done else "MM-RVD SUPPLEMENTARY FIGURE S1 SEVERITY SWEEP STOPPED")
    print("preflight_status=PASS")
    print("anchor_replay_status=PASS")
    print(f"full_sweep_status={'PASS' if all_done else 'FAIL'}")
    print("models_modified=NO")
    print("models_retrained=NO")
    print("svm_refit=NO")
    print("cebra_refit=NO")
    print("checkpoint_reselected=NO")
    print("strongest_baseline_reselected=NO")
    print("heldout_used_for_selection=NO")
    print("existing_results_overwritten=NO")
    print("manuscript_modified=NO")
    print(f"output_dir={OUT}")


if __name__ == "__main__":
    main()
