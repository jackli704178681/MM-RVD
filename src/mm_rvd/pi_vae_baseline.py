from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

from src.mm_rvd.baselines import BaseBaseline, impute_with_fit_mean


class PiVAELogRegBaseline(BaseBaseline):
    """Independent pi-VAE-like label-prior latent baseline for FIT-only validation."""

    def fit_fit_split(self, fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology) -> None:
        BaseBaseline.fit_fit_split(self, fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology)
        x = impute_with_fit_mean(fit_response, fit_observed_mask, self.fit_feature_mean_).reshape(len(fit_response), -1)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        shifted = xs - xs.min() + 1e-4
        dim = max(1, min(self.config.latent_dim, shifted.shape[0] - 1, shifted.shape[1] - 1))
        self.nmf_ = NMF(n_components=dim, init="nndsvda", random_state=self.config.random_seed, max_iter=100)
        self.nmf_.fit(shifted)
        self.shift_min_ = float(xs.min())
        labels = np.asarray(fit_labels_if_allowed, dtype=int)
        self.class_priors_ = np.bincount(labels, minlength=self.config.n_classes).astype(np.float32)
        self.class_priors_ = self.class_priors_ / max(float(self.class_priors_.sum()), 1.0)
        self.optimizer_steps_ = int(getattr(self.nmf_, "n_iter_", 1) or 1)

    def transform(self, response, observed_mask, trial_ids):
        x = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_).reshape(len(response), -1)
        xs = self.scaler_.transform(x)
        shifted = xs - self.shift_min_ + 1e-4
        shifted = np.maximum(shifted, 0.0)
        latent = self.nmf_.transform(shifted).astype(np.float32)
        prior_scalar = np.full((len(response), 1), float(np.dot(self.class_priors_, np.arange(len(self.class_priors_))) / max(len(self.class_priors_) - 1, 1)), dtype=np.float32)
        return np.concatenate([latent, prior_scalar], axis=1).astype(np.float32)

