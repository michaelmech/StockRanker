import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression, LogisticRegression

cvmod = pytest.importorskip("cross_validation")
drift = pytest.importorskip("data_drift")


def test_smoke_cv_fit_and_spearman_score_runs(tiny_multiindex_df):
    X = tiny_multiindex_df[["feature", "open", "high"]]
    y = pd.Series(np.linspace(0, 1, len(X)), index=X.index)

    cv = cvmod.RollingPurgedKFold(n_splits=2, purge_gap=0, embargo_gap=0)
    mean_score, per_fold = cvmod.cross_val_spearman_score(
        estimator=LinearRegression(),
        X=X,
        y=y,
        cv=cv,
        n_jobs=1,
    )

    assert len(per_fold) > 0
    assert np.isfinite(mean_score)


def test_smoke_adv_cv_runs_with_time_aware_splitter(tiny_multiindex_df):
    X = tiny_multiindex_df[["feature", "open"]]
    cv = cvmod.RollingPurgedKFold(n_splits=2, purge_gap=0, embargo_gap=0)

    score = drift.adv_cv(
        model=LogisticRegression(max_iter=100, random_state=0),
        X=X,
        cv=cv,
    )

    assert np.isfinite(score)
