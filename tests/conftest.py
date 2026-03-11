import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_multiindex_df():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    tickers = ["AAA", "BBB"]
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    idx = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    return pd.DataFrame(
        {
            "feature": np.arange(len(idx), dtype=float),
            "close": np.linspace(100, 115, len(idx)),
            "open": np.linspace(99, 114, len(idx)),
            "high": np.linspace(101, 116, len(idx)),
            "low": np.linspace(98, 113, len(idx)),
            "volume": np.arange(1, len(idx) + 1),
        },
        index=idx,
    )


@pytest.fixture
def tiny_rank_price_series(tiny_multiindex_df):
    rank = tiny_multiindex_df["feature"].copy()
    price = tiny_multiindex_df["close"].copy()
    return rank, price
