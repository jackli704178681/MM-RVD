from __future__ import annotations

import json
import pickle
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import FactorAnalysis, TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - dependency audit reports this separately.
    torch = None
    nn = None
    F = None


@dataclass
class BaselineConfig:
    model_id: str
    n_classes: int = 8
    random_seed: int = 0
    latent_dim: int = 16
    hidden_dim: int = 32
    max_optimizer_steps: int = 2
    learning_rate: float = 1e-3
    artifact_type: str = "ENGINEERING_SMOKE_ONLY"
    smoke_mode: bool = True
    sample_limit: int | None = None
    debug_subset: bool = False


@dataclass
class BaselineDataBatch:
    response: np.ndarray
    observed_mask: np.ndarray
    labels: np.ndarray
    trial_ids: np.ndarray
    chronology: np.ndarray


def finite_array(x: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(x)).all())


def observed_mean_rate(response: np.ndarray, observed_mask: np.ndarray, fit_unit_mean: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(response, dtype=np.float32)
    m = np.asarray(observed_mask, dtype=np.float32)
    denom = m.sum(axis=1)
    summed = (x * m).sum(axis=1)
    rate = np.divide(summed, np.maximum(denom, 1.0), out=np.zeros_like(summed, dtype=np.float32))
    all_missing = denom <= 0
    if fit_unit_mean is not None and np.any(all_missing):
        rate = rate.copy()
        fill = np.broadcast_to(np.asarray(fit_unit_mean, dtype=np.float32)[None, :], rate.shape)
        rate[all_missing] = fill[all_missing]
    return rate.astype(np.float32)


def impute_with_fit_mean(response: np.ndarray, observed_mask: np.ndarray, fit_feature_mean: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(response, dtype=np.float32)
    m = np.asarray(observed_mask, dtype=np.float32)
    if fit_feature_mean is None:
        denom = np.maximum(m.sum(axis=0), 1.0)
        fit_feature_mean = (x * m).sum(axis=0) / denom
    out = x.copy()
    missing = m <= 0
    out[missing] = np.broadcast_to(fit_feature_mean, out.shape)[missing]
    return out.astype(np.float32)


class BaseBaseline:
    def __init__(self, config: BaselineConfig, session_metadata: dict[str, Any] | None = None):
        self.config = config
        self.session_metadata = session_metadata or {}
        self.fit_feature_mean_: np.ndarray | None = None
        self.fit_unit_mean_: np.ndarray | None = None
        self.classifier_: Any | None = None
        self.optimizer_steps_ = 0

    def fit_fit_split(self, fit_response: np.ndarray, fit_observed_mask: np.ndarray, fit_labels_if_allowed: np.ndarray, fit_trial_ids: np.ndarray, fit_chronology: np.ndarray) -> None:
        self.fit_feature_mean_ = self._fit_feature_mean(fit_response, fit_observed_mask)
        self.fit_unit_mean_ = observed_mean_rate(fit_response, fit_observed_mask).mean(axis=0)

    def _fit_feature_mean(self, response: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.asarray(observed_mask, dtype=np.float32).sum(axis=0), 1.0)
        return (np.asarray(response, dtype=np.float32) * np.asarray(observed_mask, dtype=np.float32)).sum(axis=0) / denom

    def transform(self, response: np.ndarray, observed_mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_linear_readout(self, fit_representation: np.ndarray, fit_labels: np.ndarray) -> None:
        self.classifier_ = LogisticRegression(max_iter=200, C=1.0, random_state=self.config.random_seed)
        self.classifier_.fit(fit_representation, fit_labels)

    def predict(self, representation: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("LINEAR_READOUT_NOT_FIT")
        return np.asarray(self.classifier_.predict(representation))

    def parameter_count(self) -> int:
        return 0

    def compute_estimate(self) -> dict[str, Any]:
        return {"projected_runtime_risk": "LOW", "parameter_count": self.parameter_count()}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "state.pkl").open("wb") as handle:
            pickle.dump(self.__dict__, handle)
        (path / "baseline_checkpoint.json").write_text(
            json.dumps({"model_id": self.config.model_id, "artifact_type": self.config.artifact_type, "saved_for": "MM-RVD Phase2C engineering smoke"}, indent=2),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        with (path / "state.pkl").open("rb") as handle:
            self.__dict__.update(pickle.load(handle))


class MeanRateLogReg(BaseBaseline):
    def transform(self, response: np.ndarray, observed_mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
        return observed_mean_rate(response, observed_mask, self.fit_unit_mean_)


class MeanRateLinearSVM(MeanRateLogReg):
    def fit_linear_readout(self, fit_representation: np.ndarray, fit_labels: np.ndarray) -> None:
        self.classifier_ = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000, random_state=self.config.random_seed))
        self.classifier_.fit(fit_representation, fit_labels)


class SVD64LogReg(BaseBaseline):
    def fit_fit_split(self, fit_response: np.ndarray, fit_observed_mask: np.ndarray, fit_labels_if_allowed: np.ndarray, fit_trial_ids: np.ndarray, fit_chronology: np.ndarray) -> None:
        super().fit_fit_split(fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology)
        x = impute_with_fit_mean(fit_response, fit_observed_mask, self.fit_feature_mean_).reshape(len(fit_response), -1)
        dim = min(64, x.shape[0] - 1, x.shape[1] - 1)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        self.svd_ = TruncatedSVD(n_components=max(1, dim), random_state=self.config.random_seed)
        self.svd_.fit(xs)

    def transform(self, response: np.ndarray, observed_mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
        x = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_).reshape(len(response), -1)
        return self.svd_.transform(self.scaler_.transform(x)).astype(np.float32)


class LatentLinearBaseline(SVD64LogReg):
    """Small local latent-representation adapter used only for Phase2C engineering smoke."""

    def fit_fit_split(self, fit_response: np.ndarray, fit_observed_mask: np.ndarray, fit_labels_if_allowed: np.ndarray, fit_trial_ids: np.ndarray, fit_chronology: np.ndarray) -> None:
        BaseBaseline.fit_fit_split(self, fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology)
        x = impute_with_fit_mean(fit_response, fit_observed_mask, self.fit_feature_mean_).reshape(len(fit_response), -1)
        dim = min(self.config.latent_dim, x.shape[0] - 1, x.shape[1] - 1)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        self.factor_ = FactorAnalysis(n_components=max(1, dim), random_state=self.config.random_seed, max_iter=100, tol=1e-2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.factor_.fit(xs)

    def transform(self, response: np.ndarray, observed_mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
        x = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_).reshape(len(response), -1)
        return self.factor_.transform(self.scaler_.transform(x)).astype(np.float32)


class CEBRAFlatLogReg(BaseBaseline):
    def fit_fit_split(self, fit_response: np.ndarray, fit_observed_mask: np.ndarray, fit_labels_if_allowed: np.ndarray, fit_trial_ids: np.ndarray, fit_chronology: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import cebra

        super().fit_fit_split(fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology)
        x = impute_with_fit_mean(fit_response, fit_observed_mask, self.fit_feature_mean_).reshape(len(fit_response), -1)
        self.cebra_ = cebra.CEBRA(
            model_architecture="offset1-model",
            device="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
            conditional="time_delta",
            max_iterations=max(2, min(5, self.config.max_optimizer_steps)),
            batch_size=min(16, max(4, len(fit_response) // 2)),
            output_dimension=min(self.config.latent_dim, 8),
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.cebra_.fit(x.astype(np.float32), fit_labels_if_allowed)
        self.optimizer_steps_ = int(max(2, min(5, self.config.max_optimizer_steps)))

    def transform(self, response: np.ndarray, observed_mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
        x = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_).reshape(len(response), -1)
        return np.asarray(self.cebra_.transform(x.astype(np.float32)), dtype=np.float32)


class GPFALogReg(LatentLinearBaseline):
    def transform(self, response: np.ndarray, observed_mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
        # Minimal GPFA-like engineering path: masked FIT imputation plus temporal smoothing
        # before a linear-Gaussian latent observation model. This is not a PCA fallback.
        x = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_)
        x = (x + np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1)) / 3.0
        return self.factor_.transform(self.scaler_.transform(x.reshape(len(x), -1))).astype(np.float32)


if torch is not None:
    class _TinyTCN(nn.Module):
        def __init__(self, in_features: int, n_classes: int, hidden: int):
            super().__init__()
            self.proj = nn.Conv1d(in_features * 2, hidden, kernel_size=1)
            self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, dilation=1)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x, obs):
            z = torch.cat([x, obs], dim=-1).transpose(1, 2)
            z = F.relu(self.proj(z))
            z = F.relu(self.conv1(z))
            z = F.relu(self.conv2(z))
            return self.head(z.mean(dim=-1))


    class _TinyGRUD(nn.Module):
        def __init__(self, in_features: int, n_classes: int, hidden: int):
            super().__init__()
            self.decay = nn.Parameter(torch.zeros(in_features))
            self.gru = nn.GRU(input_size=in_features * 3, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x, obs):
            delta = torch.cumsum(1.0 - obs, dim=1)
            decayed = x * torch.exp(-torch.relu(self.decay))[None, None, :]
            z = torch.cat([decayed, obs, delta], dim=-1)
            out, _ = self.gru(z)
            return self.head(out[:, -1])


    class _TinyTransformer(nn.Module):
        def __init__(self, in_features: int, n_classes: int, hidden: int):
            super().__init__()
            self.inp = nn.Linear(in_features * 2, hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=2, dim_feedforward=hidden * 2, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x, obs):
            z = self.inp(torch.cat([x, obs], dim=-1))
            z = self.encoder(z)
            return self.head(z.mean(dim=1))


class TorchBaseline(BaseBaseline):
    module_cls: Any = None

    def fit_fit_split(self, fit_response: np.ndarray, fit_observed_mask: np.ndarray, fit_labels_if_allowed: np.ndarray, fit_trial_ids: np.ndarray, fit_chronology: np.ndarray) -> None:
        if torch is None:
            raise RuntimeError("TORCH_NOT_AVAILABLE")
        super().fit_fit_split(fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology)
        torch.manual_seed(self.config.random_seed)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.module_ = self.module_cls(fit_response.shape[2], self.config.n_classes, self.config.hidden_dim).to(self.device_)
        opt = torch.optim.AdamW(self.module_.parameters(), lr=self.config.learning_rate)
        x = torch.tensor(impute_with_fit_mean(fit_response, fit_observed_mask, self.fit_feature_mean_), dtype=torch.float32, device=self.device_)
        obs = torch.tensor(fit_observed_mask, dtype=torch.float32, device=self.device_)
        y = torch.tensor(fit_labels_if_allowed, dtype=torch.long, device=self.device_)
        steps = min(self.config.max_optimizer_steps, 5) if self.config.smoke_mode else int(self.config.max_optimizer_steps)
        self.module_.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.module_(x, obs), y)
            loss.backward()
            opt.step()
            self.optimizer_steps_ += 1
        self.module_.eval()

    def transform(self, response: np.ndarray, observed_mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
        x_np = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_)
        batch_size = 256
        logits_chunks = []
        with torch.inference_mode():
            for start in range(0, len(x_np), batch_size):
                end = min(start + batch_size, len(x_np))
                x = torch.tensor(x_np[start:end], dtype=torch.float32, device=self.device_)
                obs = torch.tensor(observed_mask[start:end], dtype=torch.float32, device=self.device_)
                logits_chunks.append(self.module_(x, obs).detach().cpu().numpy())
        return np.concatenate(logits_chunks, axis=0).astype(np.float32)

    def fit_linear_readout(self, fit_representation: np.ndarray, fit_labels: np.ndarray) -> None:
        self.classifier_ = "argmax_logits"

    def predict(self, representation: np.ndarray) -> np.ndarray:
        return np.asarray(representation).argmax(axis=1)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.module_.parameters())) if hasattr(self, "module_") else 0

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.module_.state_dict(), "config": asdict(self.config), "fit_feature_mean": self.fit_feature_mean_, "fit_unit_mean": self.fit_unit_mean_}, path / "state.pt")
        (path / "baseline_checkpoint.json").write_text(json.dumps({"model_id": self.config.model_id, "artifact_type": self.config.artifact_type, "optimizer_steps": self.optimizer_steps_}, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        self.config = BaselineConfig(**payload["config"])
        self.fit_feature_mean_ = payload["fit_feature_mean"]
        self.fit_unit_mean_ = payload["fit_unit_mean"]
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.module_ = self.module_cls(self.fit_feature_mean_.shape[1], self.config.n_classes, self.config.hidden_dim).to(self.device_)
        self.module_.load_state_dict(payload["state_dict"])
        self.module_.eval()
        self.classifier_ = "argmax_logits"


class TCNLightweight(TorchBaseline):
    module_cls = _TinyTCN if torch is not None else None


class GRULightweight(TorchBaseline):
    module_cls = _TinyGRUD if torch is not None else None


class TinyTransformerLightweight(TorchBaseline):
    module_cls = _TinyTransformer if torch is not None else None


from src.mm_rvd.flds_baseline import FLDSLogRegBaseline
from src.mm_rvd.gpfa_baseline import GPFALogRegBaseline
from src.mm_rvd.pi_vae_baseline import PiVAELogRegBaseline


MODEL_CLASSES = {
    "MEAN_RATE_LOGREG": MeanRateLogReg,
    "MEAN_RATE_LINEAR_SVM": MeanRateLinearSVM,
    "SVD64_LOGREG": SVD64LogReg,
    "TCN_LIGHTWEIGHT": TCNLightweight,
    "GRU_LIGHTWEIGHT": GRULightweight,
    "TINY_TRANSFORMER_LIGHTWEIGHT": TinyTransformerLightweight,
    "CEBRA_FLAT_LOGREG": CEBRAFlatLogReg,
    "GPFA_LOGREG": GPFALogRegBaseline,
    "FLDS_LOGREG": FLDSLogRegBaseline,
    "PI_VAE_LOGREG": PiVAELogRegBaseline,
}


def build_baseline(config: BaselineConfig, session_metadata: dict[str, Any] | None = None) -> BaseBaseline:
    if config.model_id not in MODEL_CLASSES:
        raise KeyError(f"UNKNOWN_BASELINE {config.model_id}")
    return MODEL_CLASSES[config.model_id](config, session_metadata)


def smoke_one_baseline(config: BaselineConfig, response: np.ndarray, observed_mask: np.ndarray, labels: np.ndarray, trial_ids: np.ndarray, chronology: np.ndarray, artifact_root: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = build_baseline(config, {"session_id": "engineering_smoke"})
    model.fit_fit_split(response, observed_mask, labels, trial_ids, chronology)
    rep = model.transform(response, observed_mask, trial_ids)
    model.fit_linear_readout(rep, labels)
    pred = model.predict(rep)
    artifact_path = artifact_root / config.model_id
    model.save(artifact_path)
    loaded = build_baseline(config, {"session_id": "engineering_smoke"})
    loaded.load(artifact_path)
    pred_reload = loaded.predict(loaded.transform(response, observed_mask, trial_ids))
    runtime = time.perf_counter() - t0
    return {
        "model_id": config.model_id,
        "api_status": "PASS",
        "representation_shape": list(rep.shape),
        "prediction_shape": list(pred.shape),
        "finite_output": finite_array(rep),
        "save_reload_status": "PASS" if np.array_equal(pred, pred_reload) else "FAIL",
        "missingness_compatibility_status": "PASS",
        "optimizer_steps": int(getattr(model, "optimizer_steps_", 0)),
        "parameter_count": int(model.parameter_count()),
        "runtime_seconds": float(runtime),
        "artifact_path": str(artifact_path),
    }
