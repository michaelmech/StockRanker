from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence

import faiss
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


def _build_faiss_index(dim: int, use_gpu: bool = False):
    """
    Returns an index that
      • supports add / add_with_ids
      • supports remove_ids
      • can run on CPU **or** GPU
    """
    base = faiss.IndexFlatL2(dim)           # exact L2
    id_map = faiss.IndexIDMap(base)         # gives us add_with_ids / remove_ids

    if use_gpu:
        res = faiss.StandardGpuResources()
        id_map = faiss.index_cpu_to_gpu(res, 0, id_map)

    return id_map


def _aggregate(values: np.ndarray, agg_funcs: Sequence[str]) -> Dict[str, float]:
    out = {}
    if 'mean' in agg_funcs:
        out['mean'] = values.mean(axis=0)
    if 'std' in agg_funcs:
        out['std'] = values.std(axis=0, ddof=0)
    if 'median' in agg_funcs:
        out['median'] = np.median(values, axis=0)
    if 'max' in agg_funcs:
        out['max'] = values.max(axis=0)
    if 'min' in agg_funcs:
        out['min'] = values.min(axis=0)
    return out


class FaissKNNFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    sklearn-style FAISS KNN feature engineer.

    Notes
    -----
    - Works with MultiIndex (ticker, date) or flat columns containing date.
    - Neighbors are searched across the full cross-section (not per ticker).
    - Leakage-safe in fit_transform: each date is queried before being inserted
      into the index.
    - `lookback_window` is measured in number of unique historical dates.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        n_neighbors: int = 5,
        lookback_window: int | None = None,
        agg_funcs: Sequence[str] = ('mean',),
        use_gpu: bool = False,
        date_col: str = 'date',
    ):
        self.feature_cols = feature_cols
        self.n_neighbors = n_neighbors
        self.lookback_window = lookback_window
        self.agg_funcs = tuple(agg_funcs)
        self.use_gpu = use_gpu
        self.date_col = date_col

    def _prepare_polars(self, X: pd.DataFrame) -> pl.DataFrame:
        if isinstance(X.index, pd.MultiIndex):
            df = X.reset_index()
        else:
            df = X.copy()

        if self.date_col not in df.columns:
            raise ValueError(f"Input must contain '{self.date_col}' column")

        if self.feature_cols is None:
            excluded = {'ticker', self.date_col}
            self.feature_cols_ = [c for c in df.columns if c not in excluded]
        else:
            self.feature_cols_ = list(self.feature_cols)

        required = {self.date_col, *self.feature_cols_}
        if not required.issubset(df.columns):
            raise ValueError(
                f"df must contain '{self.date_col}' and all feature_cols"
            )

        return (
            pl.from_pandas(df)
            .with_columns(pl.col(self.date_col).str.to_date())
            .sort([self.date_col, 'ticker'] if 'ticker' in df.columns else [self.date_col])
        )

    def _initialize_state(self, dim: int):
        self.index_ = _build_faiss_index(dim, use_gpu=self.use_gpu)
        self.window_dates_ = deque()  # deque[(date, np.ndarray[row_ids])]
        self.fitted_ = True

    def _enforce_window(self):
        if self.lookback_window is None:
            return
        while len(self.window_dates_) > self.lookback_window:
            _, old_ids = self.window_dates_.popleft()
            if old_ids.size:
                self.index_.remove_ids(old_ids.astype(np.int64))

    def _query_rows(self, values: np.ndarray, row_ids: np.ndarray) -> List[Dict]:
        if self.index_.ntotal == 0:
            return []

        k = min(self.n_neighbors, self.index_.ntotal)
        _, idx = self.index_.search(values, k)
        out_rows: List[Dict] = []
        for local_i, neighbors in enumerate(idx):
            valid_idx = neighbors[neighbors >= 0]
            if valid_idx.size == 0:
                continue
            neigh_values = self.values_store_[valid_idx]
            aggs = _aggregate(neigh_values, self.agg_funcs)
            out_rows.append(
                {
                    "row_id": int(row_ids[local_i]),
                    **{
                        f"{col}_{agg}_{self.n_neighbors}": v[j]
                        for j, col in enumerate(self.feature_cols_)
                        for agg, v in aggs.items()
                    }
                }
            )
        return out_rows

    def fit(self, X: pd.DataFrame, y=None):
        df = self._prepare_polars(X)
        values = df.select(self.feature_cols_).to_numpy().astype('float32')
        self.values_store_ = values
        self._initialize_state(values.shape[1])
        row_ids = np.arange(len(values), dtype=np.int64)
        self.index_.add_with_ids(values, row_ids)
        if self.lookback_window is not None:
            date_series = df[self.date_col].to_numpy()
            for d in np.unique(date_series):
                ids = row_ids[date_series == d]
                self.window_dates_.append((d, ids))
                self._enforce_window()
        return self

    def partial_fit(self, X: pd.DataFrame, y=None):
        df = self._prepare_polars(X)
        values = df.select(self.feature_cols_).to_numpy().astype('float32')
        if not hasattr(self, 'fitted_'):
            self.values_store_ = values
            self._initialize_state(values.shape[1])
            base_id = 0
        else:
            base_id = len(self.values_store_)
            self.values_store_ = np.vstack([self.values_store_, values])

        row_ids = np.arange(base_id, base_id + len(values), dtype=np.int64)
        date_series = df[self.date_col].to_numpy()
        for d in np.unique(date_series):
            ids = row_ids[date_series == d]
            self.index_.add_with_ids(values[date_series == d], ids)
            self.window_dates_.append((d, ids))
            self._enforce_window()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'fitted_'):
            raise ValueError("Call fit or partial_fit before transform")

        df = self._prepare_polars(X)
        values = df.select(self.feature_cols_).to_numpy().astype('float32')
        row_ids = np.arange(len(values), dtype=np.int64)
        out_rows = self._query_rows(values, row_ids)

        feature_df = pl.DataFrame(out_rows) if out_rows else pl.DataFrame({'row_id': []})
        base_df = df.with_row_index('row_id')
        return base_df.join(feature_df, on='row_id', how='left').drop('row_id').to_pandas()

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        df = self._prepare_polars(X)
        values = df.select(self.feature_cols_).to_numpy().astype('float32')
        self.values_store_ = values
        self._initialize_state(values.shape[1])

        date_series = df[self.date_col].to_numpy()
        row_ids = np.arange(len(values), dtype=np.int64)
        out_rows: List[Dict] = []

        for d in np.unique(date_series):
            mask = (date_series == d)
            date_values = values[mask]
            date_ids = row_ids[mask]

            # query only against historical dates already in index
            out_rows.extend(self._query_rows(date_values, date_ids))

            # add current date rows after query to avoid same-date / future leakage
            self.index_.add_with_ids(date_values, date_ids)
            self.window_dates_.append((d, date_ids))
            self._enforce_window()

        feature_df = pl.DataFrame(out_rows) if out_rows else pl.DataFrame({'row_id': []})
        base_df = df.with_row_index('row_id')
        return base_df.join(feature_df, on='row_id', how='left').drop('row_id').to_pandas()
