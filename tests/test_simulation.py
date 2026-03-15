import importlib
import sys
import types

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


def _load_simulation_with_stubs(monkeypatch):
    calls = {}

    class DummyPortfolio:
        @staticmethod
        def from_signals(**kwargs):
            calls["kwargs"] = kwargs
            return {"ok": True}

    fake_plotly_io = types.SimpleNamespace(renderers=types.SimpleNamespace(default=None))
    fake_plotly = types.SimpleNamespace(io=fake_plotly_io)
    fake_vbt = types.SimpleNamespace(Portfolio=DummyPortfolio)

    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.io", fake_plotly_io)
    monkeypatch.setitem(sys.modules, "vectorbt", fake_vbt)

    sys.modules.pop("simulation", None)
    sim = importlib.import_module("simulation")
    return sim, calls


def test_simulation_normalizes_percent_stops_and_returns_consistent_masks(monkeypatch, tiny_rank_price_series):
    sim, calls = _load_simulation_with_stubs(monkeypatch)
    rank, price = tiny_rank_price_series

    pf, sel = sim.simulate_returns(
        rank_df=rank,
        price_df=price,
        n_long=1,
        n_short=1,
        sl_stop_pct=10,
        tp_stop_pct=20,
    )

    assert pf == {"ok": True}
    assert calls["kwargs"]["sl_stop"] == pytest.approx(0.10)
    assert calls["kwargs"]["tp_stop"] == pytest.approx(0.20)
    assert sel["long_mask"].shape == sel["short_mask"].shape
    assert not (sel["long_mask"] & sel["short_mask"]).any().any()


def test_simulation_applies_adversarial_scaling_to_stops(monkeypatch, tiny_rank_price_series):
    sim, calls = _load_simulation_with_stubs(monkeypatch)
    rank, price = tiny_rank_price_series
    adv = pd.Series(0.5, index=rank.index)

    sim.simulate_returns(
        rank_df=rank,
        price_df=price,
        n_long=1,
        n_short=1,
        sl_stop_pct=0.1,
        tp_stop_pct=0.2,
        adv_prob_df=adv,
        adv_stop_losses=True,
        adv_take_profits=True,
        adv_scale=1.0,
    )

    sl = calls["kwargs"]["sl_stop"]
    tp = calls["kwargs"]["tp_stop"]
    assert isinstance(sl, pd.DataFrame)
    assert isinstance(tp, pd.DataFrame)
    assert np.isfinite(sl.to_numpy()).all() and np.isfinite(tp.to_numpy()).all()
    assert (sl.to_numpy() >= 0.1).all()
    assert (tp.to_numpy() >= 0.2).all()


def test_calculate_smart_slippage_clips_and_handles_missing(monkeypatch):
    sim, _ = _load_simulation_with_stubs(monkeypatch)

    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = ["AAA", "BBB"]

    open_df = pd.DataFrame([[10.0, 20.0], [0.0, 40.0]], index=idx, columns=cols)
    high_df = pd.DataFrame([[11.0, 21.0], [10.0, 44.0]], index=idx, columns=cols)
    low_df = pd.DataFrame([[9.0, 19.0], [5.0, 36.0]], index=idx, columns=cols)
    close_df = pd.DataFrame([[10.0, 20.0], [10.0, 40.0]], index=idx, columns=cols)
    volume_df = pd.DataFrame([[1_000.0, 0.0], [2_000.0, 500.0]], index=idx, columns=cols)
    size_frac = pd.DataFrame([[0.10, 0.20], [0.30, 0.40]], index=idx, columns=cols)

    slippage = sim.calculate_smart_slippage(
        open_df=open_df,
        high_df=high_df,
        low_df=low_df,
        close_df=close_df,
        volume_df=volume_df,
        size_frac=size_frac,
        init_cash=100_000.0,
        base_spread=0.0002,
        vol_mult=0.1,
        impact_mult=0.1,
        min_slippage=0.0005,
        max_slippage=0.02,
    )

    assert slippage.shape == size_frac.shape
    assert list(slippage.columns) == cols
    assert (slippage.to_numpy() >= 0.0).all()
    assert (slippage.to_numpy() <= 0.02).all()
    assert slippage.loc[idx[0], "AAA"] == pytest.approx(0.02)
    assert slippage.loc[idx[0], "BBB"] == pytest.approx(0.0102)
    assert slippage.loc[idx[1], "AAA"] == pytest.approx(0.0)
    assert slippage.loc[idx[1], "BBB"] == pytest.approx(0.02)


def test_simulator_maps_predictions_to_entries_and_shifts_stops(monkeypatch, tiny_multiindex_df, capsys):
    sim, calls = _load_simulation_with_stubs(monkeypatch)
    df = tiny_multiindex_df

    market_open = df["open"]
    market_high = df["high"]
    market_low = df["low"]
    market_close = df["close"]
    market_volume = df["volume"]

    preds = pd.Series(0.0, index=df.index)
    aaa_dates = sorted(df.loc[("AAA", slice(None)), :].index.get_level_values(1))
    preds.loc[("AAA", aaa_dates[0])] = 1.0
    preds.loc[("AAA", aaa_dates[1])] = 1.0
    preds.loc[("AAA", aaa_dates[3])] = 1.0

    simulator = sim.Simulator(
        market_open,
        market_high,
        market_low,
        market_close,
        market_volume,
        tp=0.02,
        sl=0.01,
        max_trade_size=0.25,
        init_cash=100_000,
        frequency="1d",
        debug=True,
    )

    pf, diagnostics = simulator.run(preds)

    assert pf == {"ok": True}
    assert simulator.pf == {"ok": True}
    assert isinstance(simulator.trades_df, pd.DataFrame)
    assert simulator.trades_df.empty
    assert diagnostics["prediction_active_count"] == 3
    assert diagnostics["entry_count"] == 2
    assert diagnostics["mapping_gap"] == 1

    out = capsys.readouterr().out
    assert "Simulator diagnostics:" in out

    kwargs = calls["kwargs"]
    assert kwargs["cash_sharing"] is True
    assert kwargs["freq"] == "1d"

    sl_stop = kwargs["sl_stop"]
    tp_stop = kwargs["tp_stop"]
    assert np.isinf(sl_stop.iloc[0]).all()
    assert np.isinf(tp_stop.iloc[0]).all()
    assert np.allclose(sl_stop.iloc[1:].to_numpy(), 0.01)
    assert np.allclose(tp_stop.iloc[1:].to_numpy(), 0.02)
