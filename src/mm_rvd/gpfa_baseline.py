from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from src.mm_rvd.baselines import BaseBaseline, impute_with_fit_mean


class GPFALogRegBaseline(BaseBaseline):
    """Independent GPFA-like latent baseline for Phase 5R-B validation."""

    def fit_fit_split(self, fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology) -> None:
        BaseBaseline.fit_fit_split(self, fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology)
        x = impute_with_fit_mean(fit_response, fit_observed_mask, self.fit_feature_mean_)
        smoothed = (x + np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1)) / 3.0
        flat = smoothed.reshape(len(smoothed), -1)
        dim = max(1, min(self.config.latent_dim, flat.shape[0] - 1, flat.shape[1] - 1))
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(flat)
        self.factor_ = FactorAnalysis(n_components=dim, random_state=self.config.random_seed, max_iter=100, tol=1e-2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.factor_.fit(xs)
        self.optimizer_steps_ = int(getattr(self.factor_, "n_iter_", 1) or 1)

    def transform(self, response, observed_mask, trial_ids):
        x = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_)
        smoothed = (x + np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1)) / 3.0
        return self.factor_.transform(self.scaler_.transform(smoothed.reshape(len(smoothed), -1))).astype(np.float32)

