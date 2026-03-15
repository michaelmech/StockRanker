import pytest

pd = pytest.importorskip("pandas")

from metrics import sectional_spearman_ic


def _sample_index():
    tickers = ["AAA", "BBB", "CCC"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    return pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])


def test_sectional_spearman_ic_is_one_for_perfect_ranking():
    idx = _sample_index()

    y_true = pd.Series([1, 1, 2, 2, 3, 3], index=idx)
    y_pred = pd.Series([10, 10, 20, 20, 30, 30], index=idx)

    assert sectional_spearman_ic(y_true, y_pred) == pytest.approx(1.0)


def test_sectional_spearman_ic_is_negative_one_for_inverse_ranking():
    idx = _sample_index()

    y_true = pd.Series([1, 1, 2, 2, 3, 3], index=idx)
    y_pred = pd.Series([30, 30, 20, 20, 10, 10], index=idx)

    assert sectional_spearman_ic(y_true, y_pred) == pytest.approx(-1.0)


def test_sectional_spearman_ic_ignores_dates_with_too_few_valid_tickers():
    idx = _sample_index()

    y_true = pd.Series([1, 1, 2, 2, 3, 3], index=idx)
    y_pred = pd.Series([10, None, 20, None, 30, 999], index=idx)

    assert sectional_spearman_ic(y_true, y_pred) == pytest.approx(1.0)


def test_sectional_spearman_ic_requires_identical_indices():
    idx = _sample_index()

    y_true = pd.Series([1, 1, 2, 2, 3, 3], index=idx)
    y_pred = pd.Series([10, 10, 20, 20, 30, 30], index=idx[::-1])

    with pytest.raises(ValueError, match="identical MultiIndex"):
        sectional_spearman_ic(y_true, y_pred)
