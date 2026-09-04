from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.mm_rvd.baselines import BaseBaseline, impute_with_fit_mean


class FLDSLogRegBaseline(BaseBaseline):
    """Independent fLDS-like baseline using temporal dynamics features."""

    def fit_fit_split(self, fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology) -> None:
        BaseBaseline.fit_fit_split(self, fit_response, fit_observed_mask, fit_labels_if_allowed, fit_trial_ids, fit_chronology)
        x = impute_with_fit_mean(fit_response, fit_observed_mask, self.fit_feature_mean_)
        dynamics = np.diff(x, axis=1, prepend=x[:, :1, :])
        features = np.concatenate([x.mean(axis=1), dynamics.mean(axis=1), dynamics.std(axis=1)], axis=1)
        dim = max(1, min(self.config.latent_dim, features.shape[0] - 1, features.shape[1] - 1))
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(features)
        self.svd_ = TruncatedSVD(n_components=dim, random_state=self.config.random_seed)
        self.svd_.fit(xs)
        self.optimizer_steps_ = 1

    def transform(self, response, observed_mask, trial_ids):
        x = impute_with_fit_mean(response, observed_mask, self.fit_feature_mean_)
        dynamics = np.diff(x, axis=1, prepend=x[:, :1, :])
        features = np.concatenate([x.mean(axis=1), dynamics.mean(axis=1), dynamics.std(axis=1)], axis=1)
        return self.svd_.transform(self.scaler_.transform(features)).astype(np.float32)

