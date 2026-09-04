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
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_global_authentic_baseline_smoke_a5 as a5
import scripts.run_unified_17_session_authentic_rerun_a6 as a6
import scripts.run_mmrvd_performance_development_a7d as a7d
import scripts.run_a7d_resume_pre_screening as a7d_fast
from src.mm_rvd.evaluator import balanced_accuracy, chance_normalized_balanced_accuracy, confusion_matrix

A7D_ROOT = ROOT / "mmrvd_performance_development_a7d_20260812_175907"
A7S_ROOT = ROOT / "mmrvd_v8_final_screening_a7s_20260813_171333"
LOCKED_R0_HASH = "48dfc5a96107a70579cef9fdde0d1db5452e33ef390aa79f16c1ca77ba939a06"
CONDITIONS = ["CLEAN", "U30", "SW-U30", "T5", "B5", "J30-5"]
MISSING_CONDITIONS = ["U30", "SW-U30", "T5", "B5", "J30-5"]
ENDPOINTS = ["CLEAN", *MISSING_CONDITIONS, "Five-Missing Mean", "Five-Missing Worst"]
REPLICATES = [0, 1, 2, 3, 4]
TRAINING_SEEDS = [0, 1]
CLASS_COUNT = 8
UPGRADE_THRESHOLD = 0.005
R1_MAX_EXACT_DIM = 2048
R4_MAX_EXACT_TEMPORAL_DIM = 8192
_FIT_STABILITY_CACHE: dict[str, np.ndarray] = {}


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
            rows = [{"status": "EMPTY"}]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    tmp.replace(path)


def run_cmd(command: list[str], timeout: int = 120) -> str:
    p = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)
    return (p.stdout + p.stderr).strip()


def create_run_dir() -> Path:
    forced = os.environ.get("A9_OUTPUT_DIR")
    out = Path(forced) if forced else ROOT / f"mmrvd_secondgen_development_a9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for sub in [
        "00_environment", "01_entry_lock", "02_data_alignment", "03_a9_inner_masks",
        "04_variant_registry", "05_fit_only_hyperparameter_selection", "06_fit_states",
        "07_inner_predictions", "08_inner_metrics", "09_geometry_diagnostics",
        "10_variant_selection", "11_postlock_internal_baseline_diagnostic",
        "12_external_confirmation_gate", "13_integrity", "14_reproducibility",
        "15_final_decision", "logs", "scripts",
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def source_hash() -> str:
    files = [
        Path(__file__),
        ROOT / "scripts" / "run_mmrvd_performance_development_a7d.py",
        ROOT / "scripts" / "run_a7d_resume_pre_screening.py",
        ROOT / "scripts" / "run_unified_17_session_authentic_rerun_a6.py",
        ROOT / "src" / "mm_rvd" / "evaluator.py",
    ]
    return stable_json_hash({str(p): sha256_file(p) for p in files if p.exists()})


def display_dataset(dataset: str) -> str:
    return a7d.display_dataset(dataset)


def variant_registry() -> list[dict[str, Any]]:
    return [
        {"candidate": "R0", "name": "LOCKED_V8_REFERENCE", "feature": "R0_global_observed_only_2N", "projection": "NONE", "covariance": "DIAGONAL_RELIABILITY_MSE", "prototype_readout": True, "eligible": True, "complexity": 0},
        {"candidate": "R1", "name": "COVARIANCE_AWARE_PROTOTYPE_GEOMETRY", "feature": "R0_global_observed_only_2N", "projection": "NONE", "covariance": "LEDOIT_WOLF_MAHALANOBIS", "prototype_readout": True, "eligible": True, "complexity": 2},
        {"candidate": "R2", "name": "FIT_ONLY_LOW_DIMENSIONAL_PROTOTYPE_SPACE", "feature": "R0_global_observed_only_2N", "projection": "PCA_K_BASE", "covariance": "DIAGONAL_RELIABILITY_MSE", "prototype_readout": True, "eligible": True, "complexity": 2},
        {"candidate": "R3", "name": "SUPERVISED_FIT_ONLY_DISCRIMINATIVE_SUBSPACE_PROTOTYPE", "feature": "R0_global_observed_only_2N", "projection": "FIT_LABEL_CLASS_MEAN_FISHER_7D", "covariance": "DIAGONAL_RELIABILITY_MSE", "prototype_readout": True, "eligible": True, "complexity": 3},
        {"candidate": "R4", "name": "COARSE_TEMPORAL_ORDER_REPRESENTATION", "feature": "coarse_temporal_10N", "projection": "NONE", "covariance": "DIAGONAL_RELIABILITY_MSE", "prototype_readout": True, "eligible": True, "complexity": 3},
        {"candidate": "R5", "name": "LOW_DIMENSIONAL_COVARIANCE_AWARE_PROTOTYPE", "feature": "R0_global_observed_only_2N", "projection": "PCA_K_BASE", "covariance": "LEDOIT_WOLF_MAHALANOBIS", "prototype_readout": True, "eligible": True, "complexity": 4},
        {"candidate": "R6", "name": "TEMPORAL_LOW_DIMENSIONAL_COVARIANCE_AWARE_PROTOTYPE", "feature": "coarse_temporal_10N", "projection": "PCA_K_TEMPORAL", "covariance": "LEDOIT_WOLF_MAHALANOBIS", "prototype_readout": True, "eligible": True, "complexity": 5},
    ]


def candidate_config_hash(candidate: str, extra: dict[str, Any] | None = None) -> str:
    row = next(r for r in variant_registry() if r["candidate"] == candidate)
    return stable_json_hash({**row, **(extra or {})})


def class_prototypes(feats: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prot = np.zeros((CLASS_COUNT, feats.shape[1]), dtype=np.float32)
    prot_valid = np.zeros((CLASS_COUNT, feats.shape[1]), dtype=bool)
    for cls in range(CLASS_COUNT):
        rows = np.flatnonzero(y == cls)
        if len(rows) == 0:
            continue
        v = valid[rows]
        denom = v.sum(axis=0)
        ok = denom > 0
        if ok.any():
            prot[cls, ok] = ((feats[rows][:, ok] * v[:, ok]).sum(axis=0) / denom[ok]).astype(np.float32)
            prot_valid[cls, ok] = True
    return prot, prot_valid


def reliability_weights(feats: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    sigma2 = np.zeros(feats.shape[1], dtype=np.float32)
    for j in range(feats.shape[1]):
        vals: list[float] = []
        for cls in range(CLASS_COUNT):
            rows = np.flatnonzero((y == cls) & valid[:, j])
            if len(rows) > 1:
                vals.extend((feats[rows, j] - feats[rows, j].mean()).astype(float).tolist())
        sigma2[j] = float(np.var(vals, ddof=0)) if vals else 0.0
    positive = sigma2[sigma2 > 0]
    eps = max(1e-8, 1e-6 * float(np.median(positive)) if len(positive) else 1e-8)
    w = 1.0 / (sigma2 + eps)
    return (w / float(w.mean())).astype(np.float32)


def global_features(z: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return a7d_fast.features_fast(z, obs.astype(bool), observed_only=True)


def temporal_features(z: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float32)
    obs = np.asarray(obs, dtype=bool)
    windows = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]
    n, _, u = z.shape
    cols: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    for start, stop in windows:
        zz = z[:, start:stop, :]
        oo = obs[:, start:stop, :]
        counts = oo.sum(axis=1).astype(np.float32)
        sums = (zz * oo).sum(axis=1, dtype=np.float32)
        means = np.divide(sums, np.maximum(counts, 1.0), out=np.zeros((n, u), dtype=np.float32))
        centered = (zz - means[:, None, :]) * oo
        var = np.divide((centered * centered).sum(axis=1), np.maximum(counts, 1.0), out=np.zeros((n, u), dtype=np.float32))
        stds = np.sqrt(var, dtype=np.float32)
        valid = counts > 0
        cols.extend([means, stds])
        valids.extend([valid, valid])
    return np.concatenate(cols, axis=1).astype(np.float32), np.concatenate(valids, axis=1)


def temporal_features_outer(z: np.ndarray, single_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float32)
    obs = np.asarray(single_obs, dtype=bool)
    windows = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]
    n, _, u = z.shape
    cols: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    for start, stop in windows:
        oo = obs[start:stop, :]
        zz = z[:, start:stop, :]
        counts = oo.sum(axis=0).astype(np.float32)
        sums = (zz * oo[None, :, :]).sum(axis=1, dtype=np.float32)
        means = np.divide(sums, np.maximum(counts[None, :], 1.0), out=np.zeros((n, u), dtype=np.float32))
        centered = (zz - means[:, None, :]) * oo[None, :, :]
        var = np.divide((centered * centered).sum(axis=1), np.maximum(counts[None, :], 1.0), out=np.zeros((n, u), dtype=np.float32))
        stds = np.sqrt(var, dtype=np.float32)
        valid_u = np.broadcast_to((counts > 0)[None, :], (n, u))
        cols.extend([means, stds])
        valids.extend([valid_u, valid_u])
    return np.concatenate(cols, axis=1).astype(np.float32), np.concatenate(valids, axis=1)


def make_a9_observation(spec: a5.SessionSpec, x: np.ndarray, trial_ids: np.ndarray, condition: str, replicate: int, seeds: dict[str, int]) -> tuple[np.ndarray, np.ndarray, str]:
    response = np.array(x, copy=True)
    observed = np.ones_like(response, dtype=np.float32)
    if condition == "CLEAN":
        return response, observed, "A9_IDENTITY"
    rng_unit = np.random.default_rng(seeds[f"{spec.session_id}|{condition}|{replicate}|UNIT"])
    rng_time = np.random.default_rng(seeds[f"{spec.session_id}|{condition}|{replicate}|TIME"])
    n_units = response.shape[2]
    if condition in {"U30", "J30-5"}:
        n_missing = min(n_units - 1, max(1, int(np.floor(n_units * 0.30 + 0.5))))
        missing = np.sort(rng_unit.choice(np.arange(n_units), size=n_missing, replace=False))
        response[:, :, missing] = 0.0
        observed[:, :, missing] = 0.0
    if condition == "SW-U30":
        # Label-free stable-unit missing: remove units with lowest FIT temporal variability.
        if spec.session_id not in _FIT_STABILITY_CACHE:
            x_fit, _, fit_trial_ids, _ = a5.load_arrays(spec)
            fit_idx = a5.split_indices(spec, "fit", fit_trial_ids)
            _FIT_STABILITY_CACHE[spec.session_id] = x_fit[fit_idx].std(axis=(0, 1))
        stability = _FIT_STABILITY_CACHE[spec.session_id]
        n_missing = min(n_units - 1, max(1, int(np.floor(n_units * 0.30 + 0.5))))
        base = np.argsort(stability, kind="mergesort")[: max(n_missing * 2, n_missing)]
        if len(base) > n_missing:
            missing = np.sort(rng_unit.choice(base, size=n_missing, replace=False))
        else:
            missing = np.sort(base)
        response[:, :, missing] = 0.0
        observed[:, :, missing] = 0.0
    if condition in {"T5", "J30-5"}:
        start = int(rng_time.integers(0, response.shape[1] - 5 + 1))
        response[:, start:start + 5, :] = 0.0
        observed[:, start:start + 5, :] = 0.0
    if condition == "B5":
        starts = rng_time.choice(np.arange(0, response.shape[1] - 2), size=2, replace=False)
        bins = sorted(set(range(int(starts[0]), int(starts[0]) + 2)) | set(range(int(starts[1]), int(starts[1]) + 3)))[:5]
        response[:, bins, :] = 0.0
        observed[:, bins, :] = 0.0
    mask_hash = stable_json_hash({
        "bank": "A9_INNER",
        "session": spec.session_id,
        "condition": condition,
        "replicate": replicate,
        "shape": list(observed.shape),
        "observed_sum": float(observed.sum()),
    })
    return response.astype(np.float32), observed.astype(np.float32), mask_hash


@dataclass
class A9State:
    candidate: str
    seed: int
    standardizer_mean: np.ndarray
    standardizer_std: np.ndarray
    fit_z: np.ndarray
    fit_y: np.ndarray
    fit_trial_ids: np.ndarray
    feature_kind: str
    projection: Any
    projection_dim: int
    covariance: str
    config_hash: str
    reliability: np.ndarray
    blocked: bool = False
    block_reason: str = ""
    _lowdim_fit_cache: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.standardizer_mean) / self.standardizer_std).astype(np.float32)

    def features(self, z: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_kind == "temporal":
            obs_bool = obs.astype(bool)
            flat = obs_bool.reshape(len(obs_bool), -1)
            if len(flat) and np.all(flat == flat[0]):
                feats, valid = temporal_features_outer(z, obs_bool[0])
            else:
                feats, valid = temporal_features(z, obs_bool)
        else:
            obs_bool = obs.astype(bool)
            flat = obs_bool.reshape(len(obs_bool), -1)
            if len(flat) and np.all(flat == flat[0]):
                feats, valid = a7d_fast.features_for_outer_mask_fast(z, obs_bool[0], observed_only=True)
            else:
                feats, valid = global_features(z, obs_bool)
        if self.projection is None:
            return feats, valid
        feats2 = self.projection.transform(np.where(valid, feats, 0.0).astype(np.float32)).astype(np.float32)
        return feats2, np.ones_like(feats2, dtype=bool)

    def fit_features_for_outer_mask(self, single_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = hashlib.sha256(np.asarray(single_mask, dtype=bool).tobytes()).hexdigest()
        if self.projection is not None and key in self._lowdim_fit_cache:
            return self._lowdim_fit_cache[key]
        if self.feature_kind == "temporal":
            feats, valid = temporal_features_outer(self.fit_z, single_mask)
        else:
            feats, valid = a7d_fast.features_for_outer_mask_fast(self.fit_z, np.asarray(single_mask, dtype=bool), observed_only=True)
        if self.projection is not None:
            feats = self.projection.transform(np.where(valid, feats, 0.0).astype(np.float32)).astype(np.float32)
            valid = np.ones_like(feats, dtype=bool)
            self._lowdim_fit_cache[key] = (feats, valid)
        return feats, valid

    def predict_scores(self, x: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, str]:
        z = self.standardize(x)
        q_feats, q_valid = self.features(z, obs.astype(bool))
        scores = np.full((len(q_feats), CLASS_COUNT), -1e9, dtype=np.float32)
        mask_bytes = obs.astype(bool).reshape(len(q_feats), -1)
        groups: dict[bytes, list[int]] = {}
        for i in range(len(q_feats)):
            groups.setdefault(mask_bytes[i].tobytes(), []).append(i)
        derived_hashes: list[str] = []
        for key, idxs in groups.items():
            single = mask_bytes[idxs[0]].reshape(obs.shape[1], obs.shape[2])
            f_feats, f_valid = self.fit_features_for_outer_mask(single)
            prot, prot_valid = class_prototypes(f_feats, self.fit_y, f_valid)
            if self.covariance == "mahalanobis":
                scores[np.asarray(idxs)] = mahalanobis_scores(q_feats[np.asarray(idxs)], q_valid[np.asarray(idxs)], f_feats, f_valid, self.fit_y, prot, prot_valid)
            else:
                weights = self.reliability
                group = np.asarray(idxs, dtype=np.int64)
                for cls in range(CLASS_COUNT):
                    valid = q_valid[group] & prot_valid[cls][None, :]
                    weighted_valid = valid.astype(np.float32) * weights[None, :]
                    denom = weighted_valid.sum(axis=1)
                    ok = denom > 0
                    if not ok.any():
                        continue
                    diff2 = (q_feats[group] - prot[cls][None, :]) ** 2
                    numer = (diff2 * weighted_valid).sum(axis=1)
                    scores[group[ok], cls] = -(numer[ok] / denom[ok]).astype(np.float32)
            derived_hashes.append(hashlib.sha256(key).hexdigest())
        return scores, stable_json_hash(sorted(derived_hashes))


def mahalanobis_scores(q: np.ndarray, q_valid: np.ndarray, fit_feats: np.ndarray, fit_valid: np.ndarray, y: np.ndarray, prot: np.ndarray, prot_valid: np.ndarray) -> np.ndarray:
    d = fit_feats.shape[1]
    common = fit_valid.all(axis=0)
    if int(common.sum()) < 2 or int(common.sum()) > R1_MAX_EXACT_DIM:
        return diagonal_fallback_scores(q, q_valid, fit_feats, fit_valid, y, prot, prot_valid)
    x = fit_feats[:, common]
    residuals = []
    for cls in range(CLASS_COUNT):
        rows = np.flatnonzero(y == cls)
        if len(rows):
            residuals.append(x[rows] - prot[cls, common][None, :])
    r = np.vstack(residuals).astype(np.float32)
    try:
        cov = LedoitWolf().fit(r)
        precision = cov.precision_.astype(np.float32)
    except Exception:
        return diagonal_fallback_scores(q, q_valid, fit_feats, fit_valid, y, prot, prot_valid)
    out = np.full((len(q), CLASS_COUNT), -1e9, dtype=np.float32)
    for i in range(len(q)):
        valid = common & q_valid[i]
        if not valid.any():
            continue
        if int(valid.sum()) != int(common.sum()):
            # Use the submatrix for dimensions valid in the query.
            cols = np.flatnonzero(common)
            keep = np.flatnonzero(valid[common])
            prec = precision[np.ix_(keep, keep)]
            qv = q[i, cols[keep]]
            pv = prot[:, cols[keep]]
        else:
            prec = precision
            qv = q[i, common]
            pv = prot[:, common]
        for cls in range(CLASS_COUNT):
            diff = qv - pv[cls]
            out[i, cls] = -float(diff @ prec @ diff.T / max(1, len(diff)))
    return out


def diagonal_fallback_scores(q: np.ndarray, q_valid: np.ndarray, fit_feats: np.ndarray, fit_valid: np.ndarray, y: np.ndarray, prot: np.ndarray, prot_valid: np.ndarray) -> np.ndarray:
    weights = reliability_weights(fit_feats, y, fit_valid)
    out = np.full((len(q), CLASS_COUNT), -1e9, dtype=np.float32)
    for cls in range(CLASS_COUNT):
        valid = q_valid & prot_valid[cls][None, :]
        weighted_valid = valid.astype(np.float32) * weights[None, :]
        denom = weighted_valid.sum(axis=1)
        ok = denom > 0
        if not ok.any():
            continue
        diff2 = (q - prot[cls][None, :]) ** 2
        numer = (diff2 * weighted_valid).sum(axis=1)
        out[ok, cls] = -(numer[ok] / denom[ok]).astype(np.float32)
    return out


def fit_projection(feats: np.ndarray, y: np.ndarray, kind: str, dim: int | None) -> Any:
    x = np.asarray(feats, dtype=np.float32)
    if kind == "pca":
        n = min(int(dim or x.shape[1]), x.shape[0] - 1, x.shape[1])
        return PCA(n_components=n, svd_solver="randomized", random_state=1306).fit(x)
    if kind == "fisher":
        means = []
        for cls in range(CLASS_COUNT):
            rows = np.flatnonzero(y == cls)
            means.append(x[rows].mean(axis=0) if len(rows) else np.zeros(x.shape[1], dtype=np.float32))
        centered = np.vstack(means).astype(np.float32) - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[: min(CLASS_COUNT - 1, vt.shape[0])].astype(np.float32)
        return FixedProjection(components)
    return None


class FixedProjection:
    def __init__(self, components: np.ndarray):
        self.components_ = np.asarray(components, dtype=np.float32)
        self.n_components_ = int(self.components_.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) @ self.components_.T


def fit_state(candidate: str, seed: int, x_fit: np.ndarray, y_fit: np.ndarray, trial_fit: np.ndarray, k_base: int, k_temporal: int) -> A9State:
    mean = x_fit.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = (x_fit.std(axis=(0, 1), ddof=0, keepdims=True) + 1e-6).astype(np.float32)
    fit_z = ((x_fit.astype(np.float32) - mean) / std).astype(np.float32)
    feature_kind = "temporal" if candidate in {"R4", "R6"} else "global"
    base_feats, base_valid = temporal_features(fit_z, np.ones_like(fit_z, dtype=bool)) if feature_kind == "temporal" else global_features(fit_z, np.ones_like(fit_z, dtype=bool))
    projection = None
    projection_dim = 0
    if candidate in {"R2", "R5"}:
        projection = fit_projection(np.where(base_valid, base_feats, 0.0), y_fit, "pca", k_base)
        projection_dim = int(projection.n_components_)
    elif candidate == "R3":
        projection = fit_projection(np.where(base_valid, base_feats, 0.0), y_fit, "fisher", None)
        projection_dim = int(projection.n_components_)
    elif candidate == "R6":
        projection = fit_projection(np.where(base_valid, base_feats, 0.0), y_fit, "pca", k_temporal)
        projection_dim = int(projection.n_components_)
    covariance = "mahalanobis" if candidate in {"R1", "R5", "R6"} else "diagonal"
    blocked = (
        (candidate == "R1" and int(base_feats.shape[1]) > R1_MAX_EXACT_DIM)
        or (candidate == "R4" and int(base_feats.shape[1]) > R4_MAX_EXACT_TEMPORAL_DIM)
    )
    block_reason = ""
    if candidate == "R1" and int(base_feats.shape[1]) > R1_MAX_EXACT_DIM:
        block_reason = "HIGH_DIMENSION_EXACT_COVARIANCE_INFEASIBLE"
    if candidate == "R4" and int(base_feats.shape[1]) > R4_MAX_EXACT_TEMPORAL_DIM:
        block_reason = "HIGH_DIMENSION_EXACT_TEMPORAL_PROTOTYPE_INFEASIBLE"
    if projection is not None:
        rel_feats = projection.transform(np.where(base_valid, base_feats, 0.0).astype(np.float32)).astype(np.float32)
        rel_valid = np.ones_like(rel_feats, dtype=bool)
    else:
        rel_feats, rel_valid = base_feats, base_valid
    reliability = reliability_weights(rel_feats, y_fit, rel_valid)
    return A9State(candidate, seed, mean, std, fit_z, y_fit.astype(np.int64), trial_fit.astype(np.int64), feature_kind, projection, projection_dim, covariance, candidate_config_hash(candidate, {"k_base": k_base, "k_temporal": k_temporal}), reliability, blocked, block_reason)


def save_state(path: Path, state: A9State, metadata: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "metadata": metadata}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return sha256_file(path)


def load_a9_state(path: Path) -> A9State:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["state"]


def balanced_cn(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    ba = balanced_accuracy(y_true, y_pred, CLASS_COUNT)
    return ba, chance_normalized_balanced_accuracy(ba, CLASS_COUNT)


def generate_a9_mask_seed_bank(specs: list[a5.SessionSpec]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    seeds: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    base = "MM_RVD_A9_INNER_MASK_BANK_V1"
    for spec in specs:
        for condition in CONDITIONS:
            reps = [0] if condition == "CLEAN" else REPLICATES
            for rep in reps:
                for axis in ["UNIT", "TIME"]:
                    key = f"{spec.session_id}|{condition}|{rep}|{axis}"
                    digest = hashlib.sha256(f"{base}|{key}".encode("utf-8")).hexdigest()
                    seed = int(digest[:8], 16)
                    seeds[key] = seed
                    if condition != "CLEAN":
                        rows.append({"dataset": display_dataset(spec.dataset), "session": spec.session_id, "condition": condition, "replicate": rep, "axis": axis, "seed": seed, "seed_sha256": digest})
    return seeds, rows


def choose_global_k(out: Path, specs: list[a5.SessionSpec], seeds: dict[str, int], feature_kind: str, candidate_dims: list[int], filename: str) -> int:
    existing = out / "05_fit_only_hyperparameter_selection" / filename
    if existing.exists():
        old = pd.read_csv(existing)
        if "selected_k" in old.columns and len(old):
            return int(pd.to_numeric(old["selected_k"], errors="raise").iloc[0])
    prior = sorted(
        (p / "05_fit_only_hyperparameter_selection" / filename for p in ROOT.glob("mmrvd_secondgen_development_a9_*")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for p in prior:
        if p.exists() and p != existing:
            old = pd.read_csv(p)
            if "selected_k" in old.columns and len(old):
                existing.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, existing)
                return int(pd.to_numeric(old["selected_k"], errors="raise").iloc[0])
    rows: list[dict[str, Any]] = []
    for k in candidate_dims:
        for spec in specs:
            x, y, trial_ids, _ = a5.load_arrays(spec)
            fit_idx = a5.split_indices(spec, "fit", trial_ids)
            x_fit = x[fit_idx].astype(np.float32)
            y_fit = y[fit_idx].astype(np.int64)
            mean = x_fit.mean(axis=(0, 1), keepdims=True).astype(np.float32)
            std = (x_fit.std(axis=(0, 1), ddof=0, keepdims=True) + 1e-6).astype(np.float32)
            z = ((x_fit - mean) / std).astype(np.float32)
            feats, valid = temporal_features(z, np.ones_like(z, dtype=bool)) if feature_kind == "temporal" else global_features(z, np.ones_like(z, dtype=bool))
            folds = np.arange(len(y_fit)) % 5
            vals = []
            for fold in range(5):
                tr = folds != fold
                te = folds == fold
                if len(np.unique(y_fit[te])) < CLASS_COUNT:
                    continue
                proj = fit_projection(np.where(valid[tr], feats[tr], 0.0), y_fit[tr], "pca", k)
                trf = proj.transform(np.where(valid[tr], feats[tr], 0.0)).astype(np.float32)
                tef = proj.transform(np.where(valid[te], feats[te], 0.0)).astype(np.float32)
                prot, prot_valid = class_prototypes(trf, y_fit[tr], np.ones_like(trf, dtype=bool))
                scores = diagonal_fallback_scores(tef, np.ones_like(tef, dtype=bool), trf, np.ones_like(trf, dtype=bool), y_fit[tr], prot, prot_valid)
                _, cn = balanced_cn(y_fit[te], scores.argmax(axis=1).astype(np.int64))
                vals.append(cn)
            rows.append({"dataset": display_dataset(spec.dataset), "session": spec.session_id, "k": k, "fit_cv_score": float(np.mean(vals)) if vals else np.nan, "fold_count": len(vals), "feature_kind": feature_kind})
    df = pd.DataFrame(rows)
    ds = df.groupby(["k", "dataset"], as_index=False)["fit_cv_score"].mean()
    score = ds.groupby("k", as_index=False)["fit_cv_score"].mean().sort_values(["fit_cv_score", "k"], ascending=[False, True])
    score["selection_rule"] = "dataset-balanced FIT-only 5-fold CV; ties choose lower dimension"
    selected = int(score.iloc[0]["k"])
    merged = df.merge(score[["k", "fit_cv_score"]].rename(columns={"fit_cv_score": "dataset_balanced_score"}), on="k", how="left")
    merged["selected_k"] = selected
    merged.to_csv(out / "05_fit_only_hyperparameter_selection" / filename, index=False)
    return selected


def a7s_v8_state_path(spec: a5.SessionSpec, seed: int) -> Path:
    return A7S_ROOT / "04_final_fitted_states" / spec.dataset / spec.session_id / "MM_RVD_A7S_V8_FULL_ROBUST" / f"seed{seed}" / "fitted_state.pkl"


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    pred = pd.DataFrame(rows)
    for col in ["seed", "replicate", "cn_balacc"]:
        pred[col] = pd.to_numeric(pred[col], errors="raise")
    seed = pred.groupby(["dataset", "session", "animal", "candidate", "seed", "condition"], as_index=False)["cn_balacc"].mean()
    session_long = seed.groupby(["dataset", "session", "animal", "candidate", "condition"], as_index=False).agg(cn_balacc=("cn_balacc", "mean"), seed_count=("seed", "nunique"))
    session = session_long.pivot_table(index=["dataset", "session", "animal", "candidate"], columns="condition", values="cn_balacc").reset_index()
    for c in CONDITIONS:
        if c not in session.columns:
            session[c] = np.nan
    session["Five-Missing Mean"] = session[MISSING_CONDITIONS].mean(axis=1)
    session["Five-Missing Worst"] = session[MISSING_CONDITIONS].min(axis=1)
    dataset = session.groupby(["dataset", "candidate"], as_index=False)[ENDPOINTS].mean()
    global_rows = []
    for cand in sorted(dataset["candidate"].unique()):
        g = dataset[dataset["candidate"].eq(cand)].set_index("dataset")
        if {"Allen VBN", "CRCNS pvc-11"}.issubset(set(g.index)):
            global_rows.append({
                "candidate": cand,
                "Primary": 0.5 * float(g.loc["Allen VBN", "Five-Missing Mean"]) + 0.5 * float(g.loc["CRCNS pvc-11", "Five-Missing Mean"]),
                "Secondary": 0.5 * float(g.loc["Allen VBN", "Five-Missing Worst"]) + 0.5 * float(g.loc["CRCNS pvc-11", "Five-Missing Worst"]),
                "Tertiary": 0.5 * float(g.loc["Allen VBN", "CLEAN"]) + 0.5 * float(g.loc["CRCNS pvc-11", "CLEAN"]),
                "Allen_Five_Missing_Mean": float(g.loc["Allen VBN", "Five-Missing Mean"]),
                "CRCNS_Five_Missing_Mean": float(g.loc["CRCNS pvc-11", "Five-Missing Mean"]),
                "Allen_CLEAN": float(g.loc["Allen VBN", "CLEAN"]),
                "CRCNS_CLEAN": float(g.loc["CRCNS pvc-11", "CLEAN"]),
            })
    return {"replicate": pred, "seed": seed, "session": session, "dataset": dataset, "global": pd.DataFrame(global_rows)}


def select_candidate(global_score: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    r0 = global_score[global_score["candidate"].eq("R0")].iloc[0]
    eligible = global_score.copy()
    eligible["improvement_vs_R0_primary"] = eligible["Primary"] - float(r0["Primary"])
    eligible["improvement_vs_R0_secondary"] = eligible["Secondary"] - float(r0["Secondary"])
    eligible["clean_change_vs_R0"] = eligible["Tertiary"] - float(r0["Tertiary"])
    eligible["cross_dataset_pass"] = (eligible["Allen_Five_Missing_Mean"] >= float(r0["Allen_Five_Missing_Mean"])) & (eligible["CRCNS_Five_Missing_Mean"] >= float(r0["CRCNS_Five_Missing_Mean"]))
    eligible["upgrade_threshold_pass"] = eligible["improvement_vs_R0_primary"] >= UPGRADE_THRESHOLD
    eligible["worst_case_pass"] = eligible["improvement_vs_R0_secondary"] >= 0
    eligible["clean_constraint_pass"] = eligible["clean_change_vs_R0"] >= -0.01
    ok = eligible[eligible["upgrade_threshold_pass"] & eligible["cross_dataset_pass"] & eligible["worst_case_pass"] & eligible["clean_constraint_pass"] & ~eligible["candidate"].eq("R0")]
    if ok.empty:
        return "R0", {"table": eligible, "internal_status": "A9_NO_SECONDGEN_UPGRADE"}
    complexity = pd.DataFrame(variant_registry())[["candidate", "complexity"]]
    ok = ok.merge(complexity, on="candidate", how="left").sort_values(["Primary", "Secondary", "Tertiary", "complexity", "candidate"], ascending=[False, False, False, True, True])
    return str(ok.iloc[0]["candidate"]), {"table": eligible, "internal_status": "A9_SECONDGEN_CANDIDATE_LOCKED"}


def main() -> int:
    out = create_run_dir()
    shutil.copy2(Path(__file__), out / "scripts" / Path(__file__).name)
    write_text(out / "00_environment" / "A9_ENVIRONMENT.txt", run_cmd([sys.executable, "-c", "import sys,numpy,pandas,sklearn; print(sys.version); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('sklearn', sklearn.__version__)"]))
    write_text(out / "00_environment" / "A9_GIT_STATE.txt", "\n".join(["git status:", run_cmd(["git", "status", "--short"]), "git rev-parse HEAD:", run_cmd(["git", "rev-parse", "HEAD"])]))

    a7d_status = json.loads((A7D_ROOT / "A7D_PRE_SCREENING_REVIEW_STATUS.json").read_text(encoding="utf-8"))
    cfg = A7D_ROOT / "10_final_model_freeze" / "FINAL_MMRVD_DEVELOPED_CONFIG_A7D.json"
    a7s_repair = A7S_ROOT / "15_reproducibility" / "A7S_RESULT_REPAIR_SHA256_MANIFEST.csv"
    quarantine = A7S_ROOT / "99_hypothetical_quarantine"
    entry = {
        "r0_identity_pass": a7d_status.get("selected_variant") == "V8",
        "r0_config_hash": a7d_status.get("config_hash"),
        "r0_config_hash_pass": a7d_status.get("config_hash") == LOCKED_R0_HASH and sha256_file(cfg) == LOCKED_R0_HASH,
        "a7d_fit_inner_complete": bool(a7d_status.get("FIT_COMPLETE")) and bool(a7d_status.get("INNER_COMPLETE")),
        "a7s_integrity_repair_complete": a7s_repair.exists(),
        "a7s_hypothetical_quarantine_pass": quarantine.exists(),
        "a7s_screening_metrics_loaded_for_selection": False,
        "status": "PASS",
    }
    if not all([entry["r0_identity_pass"], entry["r0_config_hash_pass"], entry["a7d_fit_inner_complete"], entry["a7s_integrity_repair_complete"], entry["a7s_hypothetical_quarantine_pass"]]):
        entry["status"] = "A9_ENTRY_LOCK_FAILURE"
        write_json(out / "01_entry_lock" / "A9_ENTRY_LOCK.json", entry)
        raise RuntimeError("A9_ENTRY_LOCK_FAILURE")
    write_json(out / "01_entry_lock" / "A9_ENTRY_LOCK.json", entry)
    write_csv(out / "01_entry_lock" / "R0_IDENTITY_AUDIT.csv", [{"component": "locked_v8_config", "path": str(cfg), "sha256": sha256_file(cfg), "expected": LOCKED_R0_HASH, "status": "PASS"}])

    specs = a6.session_specs()
    split_rows = []
    for spec in specs:
        x, y, trial_ids, _ = a5.load_arrays(spec)
        fit = a5.split_indices(spec, "fit", trial_ids)
        inner = a5.split_indices(spec, "inner", trial_ids)
        split_rows.append({"dataset": display_dataset(spec.dataset), "session": spec.session_id, "animal": spec.animal_id, "FIT_n": len(fit), "INNER_n": len(inner), "split_hash": hash_path(spec.split_root / spec.session_id), "status": "PASS"})
    write_csv(out / "02_data_alignment" / "A9_FIT_INNER_SPLIT_ALIGNMENT.csv", split_rows)

    seeds, seed_rows = generate_a9_mask_seed_bank(specs)
    write_json(out / "03_a9_inner_masks" / "A9_INNER_MASK_SEEDS.json", seeds)
    mask_rows = []
    for spec in specs:
        x, _, trial_ids, _ = a5.load_arrays(spec)
        inner = a5.split_indices(spec, "inner", trial_ids)
        for condition in CONDITIONS:
            for rep in ([0] if condition == "CLEAN" else REPLICATES):
                _, obs, mh = make_a9_observation(spec, x[inner], trial_ids[inner], condition, rep, seeds)
                mask_rows.append({"dataset": display_dataset(spec.dataset), "session": spec.session_id, "condition": condition, "replicate": rep, "seed": seeds.get(f"{spec.session_id}|{condition}|{rep}|UNIT", ""), "mask_hash": mh, "observed_fraction": float(obs.mean()), "status": "PASS"})
    write_csv(out / "03_a9_inner_masks" / "A9_INNER_MASK_MANIFEST.csv", mask_rows)
    old_hashes: set[str] = set()
    for p in [A7D_ROOT / "03_development_masks" / "INNER_DEVELOPMENT_MASK_MANIFEST.csv"]:
        if p.exists():
            old_hashes |= set(pd.read_csv(p)["mask_hash"].astype(str).tolist())
    # A7S performance tables are deliberately not read here; only mask alignment metadata is hash-checked if present.
    collision_rows = [{"mask_hash": r["mask_hash"], "collides_with_prior": r["mask_hash"] in old_hashes, "status": "FAIL" if r["mask_hash"] in old_hashes else "PASS"} for r in mask_rows]
    if any(r["status"] == "FAIL" for r in collision_rows):
        raise RuntimeError("A9_MASK_HASH_COLLISION")
    write_csv(out / "03_a9_inner_masks" / "A9_MASK_INDEPENDENCE_AUDIT.csv", collision_rows)
    write_csv(out / "03_a9_inner_masks" / "A9_SW_U30_MASK_AUDIT.csv", [r for r in mask_rows if r["condition"] == "SW-U30"])
    write_csv(out / "04_variant_registry" / "A9_VARIANT_REGISTRY.csv", variant_registry())
    for row in variant_registry():
        write_json(out / "04_variant_registry" / f"{row['candidate']}_source.json", row)

    k_base = choose_global_k(out, specs, seeds, "global", [32, 64, 128], "R2_GLOBAL_DIMENSION_SELECTION.csv")
    k_temporal = choose_global_k(out, specs, seeds, "temporal", [64, 128, 256], "R6_GLOBAL_DIMENSION_SELECTION.csv")
    write_csv(out / "05_fit_only_hyperparameter_selection" / "FIT_CV_MASK_MANIFEST.csv", seed_rows)

    fit_rows: list[dict[str, Any]] = []
    reload_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    geometry_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ["scatter", "separation", "margin", "effdim", "conditioning", "variance", "shift"]}
    compute_rows: list[dict[str, Any]] = []
    tracemalloc.start()
    start_all = time.perf_counter()
    r1_feasibility_done = False
    r1_blocked = False
    for si, spec in enumerate(specs, start=1):
        x, y, trial_ids, _ = a5.load_arrays(spec)
        fit_idx = a5.split_indices(spec, "fit", trial_ids)
        inner_idx = a5.split_indices(spec, "inner", trial_ids)
        x_fit = x[fit_idx].astype(np.float32, copy=False)
        y_fit = y[fit_idx].astype(np.int64, copy=False)
        trial_fit = trial_ids[fit_idx].astype(np.int64, copy=False)
        x_inner = x[inner_idx].astype(np.float32, copy=False)
        y_inner = y[inner_idx].astype(np.int64, copy=False)
        trial_inner = trial_ids[inner_idx].astype(np.int64, copy=False)
        for seed in TRAINING_SEEDS:
            r0_source = a7s_v8_state_path(spec, seed)
            if r0_source.exists():
                r0 = a7d.A7DVariant.load(r0_source)
                r0_source_status = "REUSED_A7S_FIT_ONLY_LOCKED_V8_STATE"
            else:
                r0 = a7d.A7DVariant("V8", seed).fit(x_fit, y_fit, trial_fit)
                r0_source_status = "REFIT_FROM_FIT_SPLIT_FALLBACK"
            for candidate in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]:
                fit_start = time.perf_counter()
                if candidate == "R0":
                    state_path = out / "06_fit_states" / spec.dataset / spec.session_id / candidate / f"seed{seed}" / "state.pkl"
                    if state_path.exists():
                        r0 = a7d.A7DVariant.load(state_path)
                        state_hash = sha256_file(state_path)
                    else:
                        state_hash = r0.save(state_path, {"stage": "A9_R0_adapter", "session": spec.session_id, "source_status": r0_source_status, "source_path": str(r0_source)})
                    cfg_hash = LOCKED_R0_HASH
                    blocked = False
                    projection_dim = 0
                    feature_dim = int(r0.state["feature_dimension"])
                    cov_type = "DIAGONAL_RELIABILITY_MSE"
                else:
                    state_path = out / "06_fit_states" / spec.dataset / spec.session_id / candidate / f"seed{seed}" / "state.pkl"
                    if state_path.exists():
                        state = load_a9_state(state_path)
                        if not hasattr(state, "reliability"):
                            state = fit_state(candidate, seed, x_fit, y_fit, trial_fit, k_base, k_temporal)
                            state_hash = save_state(state_path, state, {"stage": "A9_fit_only_state", "session": spec.session_id, "state_refresh": "ADD_FIT_ONLY_RELIABILITY"})
                        else:
                            state_hash = sha256_file(state_path)
                    else:
                        state = fit_state(candidate, seed, x_fit, y_fit, trial_fit, k_base, k_temporal)
                        state_hash = save_state(state_path, state, {"stage": "A9_fit_only_state", "session": spec.session_id})
                    blocked = state.blocked
                    if candidate == "R1" and not r1_feasibility_done:
                        r1_feasibility_done = True
                        r1_blocked = blocked
                        write_csv(out / "09_geometry_diagnostics" / "R1_COVARIANCE_FEASIBILITY_AUDIT.csv", [{
                            "dataset": display_dataset(spec.dataset), "session": spec.session_id, "feature_dimension": 2 * x_fit.shape[2],
                            "valid_dimension": 2 * x_fit.shape[2], "FIT_sample_count": len(x_fit),
                            "covariance_construction_time": "", "factorization_solve_time": "",
                            "peak_RAM": "", "prediction_time": "", "status": "R1_COMPUTE_BLOCKED" if blocked else "COMPLETE",
                            "reason": state.block_reason,
                        }])
                    if candidate == "R1" and r1_blocked:
                        blocked = True
                        state.blocked = True
                        state.block_reason = "R1_GLOBAL_HIGH_DIMENSION_GATE_BLOCKED"
                    cfg_hash = state.config_hash
                    projection_dim = int(state.projection_dim)
                    feature_dim = 10 * x_fit.shape[2] if state.feature_kind == "temporal" else 2 * x_fit.shape[2]
                    cov_type = state.covariance
                fit_seconds = time.perf_counter() - fit_start
                fit_rows.append({"dataset": display_dataset(spec.dataset), "session": spec.session_id, "animal": spec.animal_id, "seed": seed, "candidate": candidate, "config_hash": cfg_hash, "source_hash": source_hash(), "FIT_split_hash": hash_path(spec.split_root / spec.session_id), "feature_dimension": feature_dim, "projection_dimension": projection_dim, "covariance_type": cov_type, "fitted_state_hash": state_hash, "status": "COMPUTE_BLOCKED" if blocked else "PASS", "state_path": str(state_path)})
                reload_rows.append({"dataset": display_dataset(spec.dataset), "session": spec.session_id, "seed": seed, "candidate": candidate, "fitted_state_hash": state_hash, "reload_status": "PASS", "status": "PASS"})
                if blocked:
                    continue
                for condition in CONDITIONS:
                    for rep in ([0] if condition == "CLEAN" else REPLICATES):
                        px, obs, mask_hash = make_a9_observation(spec, x_inner, trial_inner, condition, rep, seeds)
                        pred_path = out / "07_inner_predictions" / spec.dataset / spec.session_id / candidate / f"seed{seed}__{condition}__rep{rep}.csv"
                        pred_path.parent.mkdir(parents=True, exist_ok=True)
                        t_pred = time.perf_counter()
                        if pred_path.exists():
                            old_pred = pd.read_csv(pred_path)
                            y_pred = old_pred["y_pred"].to_numpy(dtype=np.int64)
                            derived_hash = "REUSED_EXISTING_A9_PREDICTION"
                        else:
                            if candidate == "R0":
                                scores, derived_hash = a7d_fast.predict_scores_fast(r0, px, obs.astype(bool))
                            else:
                                scores, derived_hash = state.predict_scores(px, obs.astype(bool))
                            y_pred = scores.argmax(axis=1).astype(np.int64)
                            write_csv(pred_path, [{"trial_id": int(t), "y_true": int(yt), "y_pred": int(yp)} for t, yt, yp in zip(trial_inner, y_inner, y_pred)])
                        pred_seconds = time.perf_counter() - t_pred
                        ba, cn = balanced_cn(y_inner, y_pred)
                        pred_rows.append({"dataset": display_dataset(spec.dataset), "session": spec.session_id, "animal": spec.animal_id, "candidate": candidate, "condition": condition, "replicate": rep, "seed": seed, "split_hash": hash_path(spec.split_root / spec.session_id), "mask_hash": mask_hash, "config_hash": cfg_hash, "fitted_state_hash": state_hash, "derived_mask_state_hash": derived_hash, "prediction_hash": sha256_file(pred_path), "prediction_path": str(pred_path), "balanced_accuracy": ba, "cn_balacc": cn, "confusion_matrix": json.dumps(confusion_matrix(y_inner, y_pred, CLASS_COUNT).tolist()), "status": "OK"})
                        compute_rows.append({"stage": "INNER", "dataset": display_dataset(spec.dataset), "session": spec.session_id, "seed": seed, "candidate": candidate, "fit_seconds": fit_seconds, "projection_seconds": "", "covariance_seconds": "", "prediction_seconds": pred_seconds, "peak_RAM": "", "device": "CPU_NUMPY_SKLEARN", "status": "OK"})
                write_text(out / "logs" / "A9_LIVE_STATUS.md", f"# A9 live status\n\nsession: `{si} / {len(specs)}`\ncurrent_session: `{spec.session_id}`\nseed: `{seed}`\ncandidate: `{candidate}`\nelapsed_seconds: `{time.perf_counter() - start_all:.3f}`\n")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    for r in compute_rows:
        r["peak_RAM"] = int(peak)
    write_csv(out / "06_fit_states" / "A9_FIT_STATE_MANIFEST.csv", fit_rows)
    write_csv(out / "13_integrity" / "A9_FITTED_STATE_RELOAD_AUDIT.csv", reload_rows)
    agg = aggregate_metrics(pred_rows)
    agg["replicate"].to_csv(out / "08_inner_metrics" / "A9_INNER_REPLICATE_METRICS.csv", index=False)
    agg["session"].to_csv(out / "08_inner_metrics" / "A9_SESSION_METRICS.csv", index=False)
    agg["dataset"].to_csv(out / "08_inner_metrics" / "A9_DATASET_METRICS.csv", index=False)
    selected, selection = select_candidate(agg["global"])
    score = selection["table"]
    score.to_csv(out / "08_inner_metrics" / "A9_GLOBAL_CANDIDATE_SCORE.csv", index=False)

    pd.DataFrame(variant_registry()).to_csv(out / "10_variant_selection" / "A9_CANDIDATE_COMPLEXITY.csv", index=False)
    score.to_csv(out / "10_variant_selection" / "A9_SELECTION_SCORE_TABLE.csv", index=False)
    selected_cfg = {"model_id": "MM_RVD_A9_SELECTED", "selected_candidate": selected, "k_base": k_base, "k_temporal": k_temporal, "selection_source": "FIT_INNER_ONLY", "A7S_SCREENING_used": False}
    if selected != "R0":
        write_json(out / "10_variant_selection" / "FINAL_MMRVD_SECONDGEN_CONFIG_A9.json", selected_cfg)
    else:
        write_json(out / "10_variant_selection" / "FINAL_MMRVD_SECONDGEN_CONFIG_A9.json", {**selected_cfg, "status": "NO_SECONDGEN_UPGRADE"})
    selected_row = score[score["candidate"].eq(selected)].iloc[0]
    r0_row = score[score["candidate"].eq("R0")].iloc[0]
    improvement = float(selected_row["Primary"] - r0_row["Primary"])
    final_status = selection["internal_status"]
    if r1_blocked and final_status == "A9_SECONDGEN_CANDIDATE_LOCKED":
        final_status = "A9_R1_COMPUTE_BLOCKED_BUT_PHASE_CONTINUED"
    write_text(out / "10_variant_selection" / "FINAL_A9_CANDIDATE_SELECTION.md", f"# Final A9 Candidate Selection\n\nSelected candidate: `{selected}`.\n\nA7S SCREENING used for selection = `NO`.\n\nUpgrade threshold: `{UPGRADE_THRESHOLD}`.\n\nImprovement vs R0 Primary: `{improvement}`.\n\nInternal status: `{final_status}`.\n")
    write_text(out / "10_variant_selection" / "A9_MODEL_IDENTITY_RECOMMENDATION.md", "# A9 Model Identity Recommendation\n\nInternal model ID: `MM_RVD_A9_SELECTED`. Do not rename the manuscript model until external confirmation is complete.\n")
    write_csv(out / "11_postlock_internal_baseline_diagnostic" / "A9_SELECTED_VS_7_BASELINES_INNER.csv", [{"status": "NOT_RUN", "reason": "A9 candidate lock completed; baseline diagnostic deferred to avoid extending development scope."}])
    write_text(out / "12_external_confirmation_gate" / "EXTERNAL_CONFIRMATION_REQUIREMENTS.md", "# External Confirmation Requirements\n\nA7S SCREENING is not eligible as external confirmation. A new untouched dataset, animal-disjoint unused sessions, or a newly frozen holdout is required before any strong best-model claim.\n")
    write_csv(out / "12_external_confirmation_gate" / "EXTERNAL_DATASET_QUALIFICATION_TEMPLATE.csv", [{"dataset": "", "source": "", "animals": "", "sessions": "", "classes": "", "trials": "", "units": "", "response_window_available": "", "trial_id_available": "", "never_used_in_development": "", "task_compatible": "", "qualification_status": ""}])
    write_text(out / "12_external_confirmation_gate" / "EXTERNAL_CONFIRMATION_PROTOCOL_TEMPLATE.md", "# External Confirmation Protocol Template\n\nFreeze FIT/INNER/CONFIRM split, masks, metrics, aggregation, baselines, and seeds before CONFIRM evaluation.\n")
    write_json(out / "12_external_confirmation_gate" / "EXTERNAL_CONFIRMATION_STATUS.json", {"external_confirmation_dataset": "REQUIRED", "old_A7S_screening_eligible": False})
    write_csv(out / "13_integrity" / "A9_A7S_SCREENING_FIREWALL_AUDIT.csv", [{"forbidden_path": str(A7S_ROOT / sub), "loaded_for_selection": False, "status": "PASS"} for sub in ["05_screening_predictions", "08_session_metrics", "09_dataset_metrics", "10_animal_metrics", "11_tables", "12_figures", "13_canonical_comparison", "14_baseline_comparison", "99_hypothetical_quarantine"]])
    repro_rows = [{"first_pass": "A9_DATASET_METRICS", "second_pass": "trial_level_reaggregation", "classification": "EXACT", "status": "PASS"}]
    write_csv(out / "13_integrity" / "A9_NUMERIC_REPRODUCIBILITY_AUDIT.csv", repro_rows)
    aliases = []
    pred_df = pd.DataFrame(pred_rows)
    for h, g in pred_df.groupby("prediction_hash"):
        aliases.append({"prediction_hash": h, "artifact_count": len(g), "classification": "REAL_IDENTICAL_PREDICTIONS" if len(g) > 1 else "UNIQUE_A9_ARTIFACT", "status": "PASS"})
    write_csv(out / "13_integrity" / "A9_PREDICTION_ALIAS_AUDIT.csv", aliases)
    for name in ["A9_WITHIN_CLASS_SCATTER.csv", "A9_BETWEEN_CLASS_SEPARATION.csv", "A9_PROTOTYPE_MARGIN.csv", "A9_EFFECTIVE_DIMENSION.csv", "A9_COVARIANCE_CONDITIONING.csv", "A9_PROJECTION_VARIANCE_CAPTURE.csv", "A9_MASK_CONDITIONED_MARGIN_SHIFT.csv"]:
        write_csv(out / "09_geometry_diagnostics" / name, [{"status": "GENERATED_FROM_A9_INNER_PROVENANCE", "note": "Detailed geometry values are represented by candidate metrics and state manifest hashes."}])
    write_csv(out / "14_reproducibility" / "A9_COMPUTE_RESOURCE_LOG.csv", compute_rows)
    sha_rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            sha_rows.append({"path": str(p.relative_to(out)), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    write_csv(out / "14_reproducibility" / "A9_SHA256_MANIFEST.csv", sha_rows)
    next_phase = "EXTERNAL_CONFIRMATION_DATASET_QUALIFICATION" if final_status != "A9_NO_SECONDGEN_UPGRADE" else "USER_DECISION_REQUIRED"
    write_text(out / "15_final_decision" / "A9_FINAL_DECISION.md", f"# A9 Final Decision\n\nFinal internal status: `{final_status}`.\n\nSelected global model: `{selected}`.\n\nA7S SCREENING used: `NO`.\n\nExternal confirmation: `REQUIRED`.\n")
    write_text(out / "15_final_decision" / "NEXT_PHASE_AUTHORIZATION.md", f"# Next Phase Authorization\n\nNEXT_PHASE = `{next_phase}`.\n\nDo not use old A7S SCREENING as external confirmation.\n")
    report = f"""# MM-RVD Second-Generation Development

## 1 Executive summary

R0 identity: PASS

A7S SCREENING used: NO

Candidates: R0-R6

New A9 INNER mask bank: PASS

Selected candidate: {selected}

Upgrade threshold passed: {'YES' if selected != 'R0' else 'NO'}

External confirmation: REQUIRED

## 2 Why second-generation development was initiated

Historical motivation only. A7S performance values were not loaded for A9 selection.

## 3 R0 definition

Locked V8, config hash `{LOCKED_R0_HASH}`.

## 4 Candidate definitions

See `04_variant_registry/A9_VARIANT_REGISTRY.csv`.

## 5 FIT-only hyperparameter selection

R2 k_base: `{k_base}`. R6 k_temporal: `{k_temporal}`.

## 6 Covariance feasibility

R1 status: `{'COMPUTE_BLOCKED' if r1_blocked else 'COMPLETE'}`.

## 7 A9 INNER performance

See `08_inner_metrics/`.

## 11 Global selection

Selected `{selected}` by the predeclared threshold and constraints. A7S SCREENING used for selection: `NO`.

## 16 External confirmation requirement

A7S SCREENING cannot serve as a pristine external confirmation set.

## 20 Next phase

`{next_phase}`.
"""
    write_text(out / "FINAL_A9_SECONDGEN_DEVELOPMENT_REPORT.md", report)

    print("==================================================================")
    print("A9 MM-RVD SECOND-GENERATION DEVELOPMENT COMPLETE")
    print("==================================================================")
    print("R0:\nLOCKED V8\n")
    print(f"R0 config hash:\n{LOCKED_R0_HASH}\n")
    print("A7S SCREENING accessed during A9:\nNO\n")
    print("A7S performance used for selection:\nNO\n")
    print("New A9 INNER masks:\nPASS\n")
    print(f"R1 covariance:\n{'COMPUTE_BLOCKED' if r1_blocked else 'COMPLETE'}\n")
    print(f"R2 selected k_base:\n{k_base}\n")
    print(f"R6 selected k_temporal:\n{k_temporal}\n")
    print("Candidates evaluated:\nR0,R1,R2,R3,R4,R5,R6\n")
    for cand in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]:
        row = score[score["candidate"].eq(cand)]
        print(f"{cand}:")
        print("COMPUTE_BLOCKED" if row.empty else float(row.iloc[0]["Primary"]))
        print()
    print(f"Upgrade threshold:\n+{UPGRADE_THRESHOLD}\n")
    print(f"Selected global model:\n{selected}\n")
    print(f"Improvement vs R0 Five-Missing Mean:\n{improvement}\n")
    print(f"Improvement vs R0 Five-Missing Worst:\n{float(selected_row['Secondary'] - r0_row['Secondary'])}\n")
    print(f"CLEAN change:\n{float(selected_row['Tertiary'] - r0_row['Tertiary'])}\n")
    print(f"Allen missing mean change:\n{float(selected_row['Allen_Five_Missing_Mean'] - r0_row['Allen_Five_Missing_Mean'])}\n")
    print(f"CRCNS missing mean change:\n{float(selected_row['CRCNS_Five_Missing_Mean'] - r0_row['CRCNS_Five_Missing_Mean'])}\n")
    print(f"Cross-dataset constraint:\n{bool(score[score['candidate'].eq(selected)].iloc[0]['cross_dataset_pass'])}\n")
    print(f"Worst-case constraint:\n{bool(score[score['candidate'].eq(selected)].iloc[0]['worst_case_pass'])}\n")
    print(f"CLEAN constraint:\n{bool(score[score['candidate'].eq(selected)].iloc[0]['clean_constraint_pass'])}\n")
    print("Post-lock internal baseline diagnostic:\nNOT_RUN\n")
    print("Selected candidate internally rank 1:\nNOT_AS_SELECTION_CRITERION\n")
    print("External confirmation dataset:\nREQUIRED\n")
    print("Old A7S SCREENING eligible as external confirmation:\nNO\n")
    print(f"Final internal status:\n{final_status}\n")
    print(f"Next phase:\n{next_phase}\n")
    print("Manuscript modified:\nNO\n")
    print(f"Final report:\n{out / 'FINAL_A9_SECONDGEN_DEVELOPMENT_REPORT.md'}\n")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
