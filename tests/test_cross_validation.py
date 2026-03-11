import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cvmod = pytest.importorskip("cross_validation")


@pytest.mark.parametrize(
    "splitter",
    [
        pytest.param(
            cvmod.SlidingWindowPurgedEmbargoCV(
                train_size_n_blocks=3,
                test_size_n_blocks=2,
                purge_gap_n_blocks=1,
                embargo_gap_n_blocks=1,
                step_n_blocks=1,
            ),
            id="sliding",
        ),
        pytest.param(
            cvmod.CombinatorialPurgedEmbargoCV(
                n_blocks=4,
                test_size_n_blocks=1,
                purge_gap_n_blocks=1,
                embargo_gap_n_blocks=1,
            ),
            id="combinatorial",
        ),
        pytest.param(
            cvmod.RollingPurgedKFold(n_splits=2, purge_gap=1, embargo_gap=0),
            id="rolling",
        ),
    ],
)
def test_cv_splitters_return_valid_non_overlapping_indices(tiny_multiindex_df, splitter):
    splits = list(splitter.split(tiny_multiindex_df))
    assert splits, "Expected at least one split on tiny deterministic data"

    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert set(train_idx).isdisjoint(set(test_idx))
        assert train_idx.min() >= 0 and test_idx.min() >= 0
        assert train_idx.max() < len(tiny_multiindex_df)
        assert test_idx.max() < len(tiny_multiindex_df)

        train_dates = tiny_multiindex_df.index.get_level_values("date")[train_idx]
        test_dates = tiny_multiindex_df.index.get_level_values("date")[test_idx]
        assert pd.Timestamp(train_dates.max()) < pd.Timestamp(test_dates.min())


def test_sliding_window_respects_purge_gap_and_get_n_splits(tiny_multiindex_df):
    splitter = cvmod.SlidingWindowPurgedEmbargoCV(
        train_size_n_blocks=3,
        test_size_n_blocks=2,
        purge_gap_n_blocks=1,
        embargo_gap_n_blocks=0,
        step_n_blocks=1,
    )
    splits = list(splitter.split(tiny_multiindex_df))
    assert splitter.get_n_splits(tiny_multiindex_df) == len(splits)

    train_idx, test_idx = splits[0]
    dates = tiny_multiindex_df.index.get_level_values("date")
    last_train = pd.Timestamp(dates[train_idx].max())
    first_test = pd.Timestamp(dates[test_idx].min())
    assert (first_test - last_train).days >= 2  # one fully purged day in-between


def test_sliding_window_embargo_excludes_next_fold_training_dates(tiny_multiindex_df):
    splitter = cvmod.SlidingWindowPurgedEmbargoCV(
        train_size_n_blocks=3,
        test_size_n_blocks=1,
        purge_gap_n_blocks=0,
        embargo_gap_n_blocks=1,
        step_n_blocks=1,
    )
    splits = list(splitter.split(tiny_multiindex_df))
    assert len(splits) >= 2

    dates = tiny_multiindex_df.index.get_level_values("date")
    first_test_date = pd.Timestamp(dates[splits[0][1]].min())
    embargoed = first_test_date + pd.Timedelta(days=1)

    second_train_dates = pd.to_datetime(dates[splits[1][0]])
    assert embargoed not in set(second_train_dates)


def test_regime_cv_splitters_smoke(tiny_multiindex_df):
    regimes = [
        ("r1", "2024-01-01", "2024-01-03"),
        ("r2", "2024-01-04", "2024-01-06"),
        ("r3", "2024-01-07", "2024-01-08"),
    ]

    splitters = [
        cvmod.RegimePurgedEmbargoCV(regimes, purge_period=0, embargo_period=0, max_splits=2),
        cvmod.BalancedRegimePurgedEmbargoCV(regimes, purge_period=0, embargo_period=0, max_splits=2),
    ]

    for splitter in splitters:
        splits = list(splitter.split(tiny_multiindex_df))
        assert len(splits) == splitter.get_n_splits(tiny_multiindex_df)
        for train_idx, test_idx in splits:
            train_dates = tiny_multiindex_df.index.get_level_values("date")[train_idx]
            test_dates = tiny_multiindex_df.index.get_level_values("date")[test_idx]
            assert pd.Timestamp(train_dates.max()) < pd.Timestamp(test_dates.min())
