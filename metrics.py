"""Metrics utilities for cross-sectional ranking tasks."""

from __future__ import annotations

import pandas as pd


def _validate_multiindex_series(
    values: pd.Series | pd.DataFrame,
    *,
    name: str,
    ticker_level: str,
    date_level: str,
) -> pd.Series:
    """Validate and normalize the input into a pandas Series with a required MultiIndex."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must be a Series or a single-column DataFrame.")
        values = values.iloc[:, 0]

    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series or single-column DataFrame.")

    if not isinstance(values.index, pd.MultiIndex):
        raise ValueError(f"{name} index must be a MultiIndex with '{ticker_level}' and '{date_level}' levels.")

    missing_levels = {ticker_level, date_level} - set(values.index.names)
    if missing_levels:
        raise ValueError(
            f"{name} index is missing required level(s): {sorted(missing_levels)}. "
            f"Found levels: {list(values.index.names)}"
        )

    return values


def sectional_spearman_ic(
    y_true: pd.Series | pd.DataFrame,
    y_pred: pd.Series | pd.DataFrame,
    *,
    ticker_level: str = "ticker",
    date_level: str = "date",
) -> float:
    """Compute mean cross-sectional Spearman rank correlation (IC) across dates.

    Parameters
    ----------
    y_true, y_pred
        Pandas Series (or single-column DataFrame) indexed by MultiIndex with
        at least `ticker_level` and `date_level`.
    ticker_level
        Name of ticker level in the MultiIndex.
    date_level
        Name of date level in the MultiIndex.

    Returns
    -------
    float
        Mean Information Coefficient (IC), i.e. average of per-date Spearman
        rank correlations between `y_true` and `y_pred` across tickers.
        Dates with fewer than two valid tickers are ignored.
    """
    true_s = _validate_multiindex_series(
        y_true,
        name="y_true",
        ticker_level=ticker_level,
        date_level=date_level,
    )
    pred_s = _validate_multiindex_series(
        y_pred,
        name="y_pred",
        ticker_level=ticker_level,
        date_level=date_level,
    )

    if not true_s.index.equals(pred_s.index):
        raise ValueError("y_true and y_pred must have identical MultiIndex values and ordering.")

    aligned = pd.concat([true_s.rename("y_true"), pred_s.rename("y_pred")], axis=1)

    per_date_ic = aligned.groupby(level=date_level).apply(
        lambda group: group["y_true"].corr(group["y_pred"], method="spearman")
    )

    mean_ic = per_date_ic.dropna().mean()
    return float(mean_ic)
