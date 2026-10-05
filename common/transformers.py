"""Reference scikit-learn compatible transformers (fit / transform), used by several modules."""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class TargetEncoderOOF(BaseEstimator, TransformerMixin):
    """Target encoding with smoothing, with out-of-fold encoding on the training set.

    `fit_transform(X, y)` (used by Pipeline on the training set) encodes every row with the statistics computed WITHOUT that
    row (k-fold); `transform(X)` (validation, test, production) uses the statistics of the whole training set.
    It accepts one or more categorical columns (DataFrame); categories never seen get the global mean.

    Args:
        m: smoothing strength: a category with few rows is pulled toward the global mean as if it had `m` extra rows.
        n_splits: number of folds for the out-of-fold encoding.
        random_state: seed of the fold shuffling.
    """

    def __init__(self, m: float = 10.0, n_splits: int = 5, random_state: int = 0) -> None:
        self.m = m
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _estimate(col: pd.Series, y: np.ndarray, m: float) -> Tuple[Dict, float]:
        """Smoothed mean of `y` for each category of `col`, and the overall mean (used for unseen categories)."""
        d = pd.DataFrame({"c": np.asarray(col), "y": np.asarray(y, dtype=float)})
        g = d.groupby("c")["y"].agg(["sum", "count"])
        overall = d["y"].mean()
        return ((g["sum"] + m * overall) / (g["count"] + m)).to_dict(), overall

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TargetEncoderOOF":
        """Learn the smoothed category means on the whole training set."""
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.maps_ = {c: self._estimate(X[c], y, self.m) for c in self.columns_}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Encode new data with the statistics of the whole training set (unseen categories get the global mean)."""
        X = pd.DataFrame(X)
        return np.column_stack([X[c].map(self.maps_[c][0]).fillna(self.maps_[c][1]).to_numpy(dtype=float)
                                for c in self.columns_])

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Fit on the training set and return its out-of-fold encoding (each row encoded without its own target)."""
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        self.fit(X, y)
        out_of_fold = np.zeros((len(X), len(self.columns_)))
        for tr, va in KFold(self.n_splits, shuffle=True, random_state=self.random_state).split(X):
            for j, c in enumerate(self.columns_):
                mapping, overall = self._estimate(X.iloc[tr][c], y[tr], self.m)
                out_of_fold[va, j] = X.iloc[va][c].map(mapping).fillna(overall).to_numpy(dtype=float)
        return out_of_fold


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces every category with its relative frequency in the training set (0 for categories never seen)."""

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "FrequencyEncoder":
        """Learn the relative frequency of every category of every column (`y` is ignored)."""
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.freq_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self.columns_}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Replace each category with its training frequency (0.0 for categories never seen)."""
        X = pd.DataFrame(X)
        return np.column_stack([X[c].map(self.freq_[c]).fillna(0.0).to_numpy(dtype=float) for c in self.columns_])
