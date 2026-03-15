import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("flaml")

from sklearn.linear_model import LinearRegression, LogisticRegression

cvmod = pytest.importorskip("cross_validation")
modeling = pytest.importorskip("modeling")


def _future_prediction_set(columns):
    tickers = ["AAA", "BBB"]
    dates = pd.date_range("2024-01-09", periods=2, freq="D")
    idx = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    return pd.DataFrame(
        {col: np.linspace(0.1, 1.0, len(idx)) for col in columns},
        index=idx,
    )


def test_time_respecting_out_of_sample_predictor_returns_multiindex_predictions(tiny_multiindex_df):
    X = tiny_multiindex_df[["feature", "open", "high"]]
    y = pd.Series(np.linspace(0, 1, len(X)), index=X.index)

    cv = cvmod.SlidingWindowPurgedEmbargoCV(
        train_size_n_blocks=3,
        test_size_n_blocks=2,
        purge_gap_n_blocks=0,
        embargo_gap_n_blocks=0,
        step_n_blocks=1,
    )
    predictor = modeling.TimeRespectingOutOfSamplePredictor(
        primary_model=LinearRegression(),
        cv=cv,
    )

    pred_set = _future_prediction_set(X.columns)
    oof_pred, preds = predictor.fit_predict(X, y, [pred_set])

    assert isinstance(oof_pred, pd.Series)
    assert oof_pred.index.equals(X.index)
    assert isinstance(preds, list) and len(preds) == 1
    assert isinstance(preds[0], pd.Series)
    assert preds[0].index.equals(pred_set.index)
    assert preds[0].notna().all()


def test_meta_labeler_builds_meta_features_and_predicts(tiny_multiindex_df):
    X = tiny_multiindex_df[["feature", "open"]]
    y_meta = pd.Series((np.arange(len(X)) % 2).astype(int), index=X.index)

    pred_set = _future_prediction_set(X.columns)
    primary_predictions = pd.Series(
        np.linspace(0.0, 1.0, len(X) + len(pred_set)),
        index=X.index.append(pred_set.index),
    )
    profit_labels = pd.Series(np.ones(len(X), dtype=int), index=X.index)

    cv = cvmod.SlidingWindowPurgedEmbargoCV(
        train_size_n_blocks=3,
        test_size_n_blocks=2,
        purge_gap_n_blocks=0,
        embargo_gap_n_blocks=0,
        step_n_blocks=1,
    )
    meta = modeling.MetaLabeler(
        meta_model=LogisticRegression(max_iter=200),
        cv=cv,
        primary_predictions=primary_predictions,
        profit_labels=profit_labels,
    )

    oof_pred, preds = meta.fit_predict(X, y_meta, [pred_set])

    assert isinstance(oof_pred, pd.Series)
    assert oof_pred.index.equals(X.index)
    assert isinstance(preds, list) and len(preds) == 1
    assert preds[0].index.equals(pred_set.index)
    assert preds[0].notna().all()
