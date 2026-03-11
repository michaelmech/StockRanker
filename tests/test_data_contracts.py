import pytest

pd = pytest.importorskip("pandas")

data_generation = pytest.importorskip("data_generation")
check_format = data_generation.check_format
convert_df_index = data_generation.convert_df_index


@check_format
def _passthrough(df):
    return df.assign(out=1)


@check_format
def _returns_columns(df):
    return df.reset_index()[["ticker", "date"]].assign(v=1)


@check_format
def _bad_output(_df):
    return pd.DataFrame({"x": [1, 2]})


def test_check_format_accepts_multiindex_and_preserves_order(tiny_multiindex_df):
    out = _passthrough(tiny_multiindex_df)
    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.names == ["ticker", "date"]
    assert out.index.equals(tiny_multiindex_df.index)


def test_check_format_converts_ticker_date_columns_to_multiindex(tiny_multiindex_df):
    as_columns = tiny_multiindex_df.reset_index()
    out = _returns_columns(as_columns)
    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.names == ["ticker", "date"]
    assert "v" in out.columns


def test_check_format_rejects_malformed_input():
    bad = pd.DataFrame({"ticker": ["AAA"], "x": [1]})
    with pytest.raises(ValueError, match="ticker.*date"):
        _passthrough(bad)


def test_check_format_rejects_unusable_output(tiny_multiindex_df):
    with pytest.raises(ValueError, match="Returned DataFrame"):
        _bad_output(tiny_multiindex_df)


def test_convert_df_index_casts_date_level_to_string(tiny_multiindex_df):
    out = convert_df_index(tiny_multiindex_df.copy())
    dates = out.index.get_level_values("date")
    assert dates.dtype == object
    assert all(isinstance(x, str) for x in dates)
