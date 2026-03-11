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
