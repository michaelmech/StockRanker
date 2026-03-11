import importlib
import sys
import types
from enum import Enum

import pytest

pytest.importorskip("pandas")


def _load_transactions_with_stubs(monkeypatch):
    class _OrderSide(Enum):
        BUY = "buy"
        SELL = "sell"

    class _Dummy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    enums_mod = types.SimpleNamespace(
        PositionSide=object,
        OrderSide=_OrderSide,
        OrderType=object,
        TimeInForce=types.SimpleNamespace(DAY="day"),
        OrderClass=types.SimpleNamespace(BRACKET="bracket"),
    )

    monkeypatch.setitem(sys.modules, "alpaca", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", types.SimpleNamespace(TradingClient=object))
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", types.SimpleNamespace(MarketOrderRequest=_Dummy, OrderRequest=_Dummy))
    monkeypatch.setitem(sys.modules, "alpaca.data", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "alpaca.data.historical", types.SimpleNamespace(StockHistoricalDataClient=object))
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", types.SimpleNamespace(StockLatestQuoteRequest=_Dummy))

    sys.modules.pop("transactions", None)
    tx = importlib.import_module("transactions")
    return tx, _OrderSide


def test_pct_to_bracket_prices_long_short_sides(monkeypatch):
    tx, side = _load_transactions_with_stubs(monkeypatch)

    long_sl, long_tp = tx.pct_to_bracket_prices(100, side.BUY, stop_pct=0.1, tp_pct=0.2)
    short_sl, short_tp = tx.pct_to_bracket_prices(100, side.SELL, stop_pct=0.1, tp_pct=0.2)

    assert long_sl < 100 < long_tp
    assert short_tp < 100 < short_sl


def test_guardrails_correct_invalid_leg_placement(monkeypatch):
    tx, side = _load_transactions_with_stubs(monkeypatch)

    sl, tp = tx.apply_alpaca_bracket_guardrails(100, side.BUY, stop_price=101, tp_price=99)
    assert sl < 100 and tp > 100

    sl_s, tp_s = tx.apply_alpaca_bracket_guardrails(100, side.SELL, stop_price=99, tp_price=101)
    assert tp_s < 100 and sl_s > 100


def test_advprob_to_pcts_is_monotonic(monkeypatch):
    tx, _ = _load_transactions_with_stubs(monkeypatch)

    p0 = tx.advprob_to_pcts(0.0)
    p50 = tx.advprob_to_pcts(0.5)
    p1 = tx.advprob_to_pcts(1.0)

    assert p0[0] >= p50[0] >= p1[0]
    assert p0[1] >= p50[1] >= p1[1]
