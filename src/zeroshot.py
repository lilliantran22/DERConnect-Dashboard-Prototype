"""Zero-shot detector: trained on OTHER homes, applied to an unseen home with no refitting.

Linear autoregression on first differences (the AR baseline in Si et al.), fit by ridge
regression on pooled, per-series normalised data. Scoring a new series only needs its own
median/MAD for normalisation (no gradient steps, no fine-tuning).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class ZeroShotAR:
    name = "Zero-shot (AR)"

    def __init__(self, lags=16, ridge=1e-3):
        self.lags, self.ridge = lags, ridge

    @staticmethod
    def _norm(x):
        med = np.nanmedian(x)
        return med, max(1.4826 * np.nanmedian(np.abs(x - med)), 1e-3)

    def fit(self, train_df):
        Xs, ys = [], []
        for _, g in train_df.groupby(["building_id", "submeter"]):
            x = g.sort_values("timestamp")["kw"].to_numpy(float)
            x = x[~np.isnan(x)]
            med, sc = self._norm(x)
            d = np.diff((x - med) / sc)
            if len(d) <= self.lags:
                continue
            Xs.append(sliding_window_view(d, self.lags)[:-1])
            ys.append(d[self.lags:])
        X, y = np.vstack(Xs), np.concatenate(ys)
        self.coef_ = np.linalg.solve(X.T @ X + self.ridge * np.eye(self.lags), X.T @ y)
        self.n_series_ = len(Xs)
        return self

    def expected(self, s):
        """One-step-ahead forecast for each timestamp of series s."""
        x = s.interpolate(limit_direction="both").to_numpy(float)
        med, sc = self._norm(x)
        xn = (x - med) / sc
        out = np.full(len(x), np.nan)
        d = np.diff(xn)
        if len(d) > self.lags:
            pred = sliding_window_view(d, self.lags)[:-1] @ self.coef_
            idx = np.arange(self.lags + 1, len(x))
            out[idx] = (xn[idx - 1] + pred) * sc + med
        return pd.Series(out, index=s.index)
