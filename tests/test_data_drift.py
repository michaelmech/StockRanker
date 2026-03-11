import pytest

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression

drift = pytest.importorskip("data_drift")


def test_av_tts_accepts_1d_inputs_and_returns_expected_contract():
    x_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x_test = np.array([5.0, 6.0, 7.0, 8.0])

    out = drift.av_tts(
        x_train=x_train,
        x_test=x_test,
        model=LogisticRegression(max_iter=100, random_state=0),
        random_state=0,
    )

    assert {"avg_score", "val_probs", "val_preds", "models"}.issubset(out.keys())
    assert np.isfinite(out["avg_score"])
    assert out["val_probs"].ndim == 1
    assert out["val_preds"].ndim == 1
    assert len(out["val_probs"]) == len(out["val_preds"])
