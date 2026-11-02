import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dwpf.conformal import (aci_alphas, conformal_quantile, cqr_adjust, cqr_scores, rearrange,  # noqa: E402
                            rolling_conformal)
from dwpf.features import build_features, generic_power_curve  # noqa: E402
from dwpf.metrics import coverage, diebold_mariano, mean_pinball, pinball, winkler  # noqa: E402
from dwpf.models import QUANTILES, climatology, gbm_qr, rolling_backtest  # noqa: E402


def test_power_curve_shape():
    v = np.array([0, 2.9, 3.0, 8, 12.5, 20, 24.9, 25, 30])
    p = generic_power_curve(v)
    assert p[0] == p[1] == p[2] == 0 and p[4] == 1 and p[5] == 1 and p[6] == 1
    assert p[7] == 0 and p[8] == 0 and 0 < p[3] < 1
    assert np.all(np.diff(generic_power_curve(np.linspace(0, 12.5, 50))) >= 0)


def test_pinball_known_values():
    assert pinball(1.0, 0.0, 0.9) == pytest.approx(0.9)
    assert pinball(0.0, 1.0, 0.9) == pytest.approx(0.1)
    # median pinball = MAE / 2
    y, q = np.array([0.1, 0.5]), np.array([0.3, 0.2])
    assert pinball(y, q, 0.5).mean() == pytest.approx(np.abs(y - q).mean() / 2)


def test_mean_pinball_of_point_forecast_approximates_crps_as_mae():
    # a degenerate distribution (all quantiles equal) has CRPS = |y - x|; 2 x mean pinball over a
    # symmetric dense quantile grid approaches it
    taus = np.linspace(0.01, 0.99, 99)
    y, x = np.array([0.4]), 0.1
    Q = np.full((1, len(taus)), x)
    assert 2 * mean_pinball(y, Q, taus)[0] == pytest.approx(abs(0.4 - x), rel=0.02)


def test_rearrange_and_clip():
    q = rearrange(np.array([[0.3, 0.2, 1.2], [-0.1, 0.5, 0.4]]))
    assert np.all(np.diff(q, axis=1) >= 0) and q.min() >= 0 and q.max() <= 1


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 10, dtype=float)  # n = 9
    # k = ceil(0.9 * 10) = 9 -> 9th smallest
    assert conformal_quantile(s, 0.1) == 9.0
    assert conformal_quantile(s, 0.5) == 5.0
    assert conformal_quantile(np.arange(5.0), 0.1) == float("inf")  # too few points for 90 %


def test_cqr_reaches_target_coverage_on_iid_data():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(0, 1, n)
    lo, hi = np.full(n, -0.5), np.full(n, 0.5)   # badly undercovering raw interval
    cal, test = slice(0, 2000), slice(2000, n)
    qhat = conformal_quantile(cqr_scores(y[cal], lo[cal], hi[cal]), 0.1)
    a, b = cqr_adjust(lo[test], hi[test], qhat, -np.inf, np.inf)
    assert coverage(y[test], a, b) == pytest.approx(0.9, abs=0.02)


def test_rolling_conformal_is_ex_ante():
    # outcomes of day d must not influence the interval of day d (delay >= 1)
    day = np.repeat(np.arange(30), 24)
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, len(day))
    lo, hi = np.full(len(day), 0.3), np.full(len(day), 0.7)
    a1, b1, _ = rolling_conformal(y, lo, hi, day, 0.1, window_days=10, delay=2)
    y2 = y.copy()
    y2[day == 29] = 5.0  # wild outcomes on the last day only
    a2, b2, _ = rolling_conformal(y2, lo, hi, day, 0.1, window_days=10, delay=2)
    assert np.allclose(a1, a2) and np.allclose(b1, b2)


def test_aci_increases_alpha_after_overcoverage_and_decreases_after_misses():
    err = np.array([0.0] * 10 + [1.0] * 10)
    a = aci_alphas(err, alpha=0.1, gamma=0.05, delay=1)
    assert a[0] == 0.1 and a[9] > 0.1 and a[-1] < a[10]


def test_aci_targets_long_run_coverage_under_shift():
    rng = np.random.default_rng(3)
    days = 400
    day = np.repeat(np.arange(days), 24)
    scale = np.repeat(np.where(np.arange(days) < 200, 1.0, 2.0), 24)  # variance doubles mid-sample
    y = rng.normal(0, 1, len(day)) * scale
    lo, hi = np.full(len(day), -1.0), np.full(len(day), 1.0)
    a, b, _ = rolling_conformal(y, lo, hi, day, 0.1, window_days=60, gamma=0.02, delay=2, lo=-np.inf, hi=np.inf)
    late = day >= 260
    assert coverage(y[late], a[late], b[late]) == pytest.approx(0.9, abs=0.03)


def test_winkler_and_coverage():
    y = np.array([0.5, 0.0, 1.0])
    lo, hi = np.array([0.4, 0.2, 0.4]), np.array([0.6, 0.6, 0.6])
    w = winkler(y, lo, hi, 0.1)
    assert w[0] == pytest.approx(0.2) and w[1] == pytest.approx(0.4 + 20 * 0.2) and w[2] == pytest.approx(0.2 + 20 * 0.4)
    assert coverage(y, lo, hi) == pytest.approx(1 / 3)


def test_diebold_mariano_sign():
    rng = np.random.default_rng(5)
    l1 = rng.normal(1.0, 0.2, 300)
    l2 = l1 - 0.1 + rng.normal(0, 0.05, 300)
    stat, p = diebold_mariano(l1, l2, h=2)
    assert stat > 0 and p < 0.01


def _toy_frame(n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * n_days, freq="h", tz="UTC")
    base = 7 + 3 * np.sin(np.arange(len(idx)) / 37.0)
    df = pd.DataFrame({f"ws100_{n}": base + rng.normal(0, 0.5, len(idx)) for n in ("sh_a", "ni_b", "by_c")}, index=idx)
    df["ws10_mean"] = base * 0.7
    df["t2m_mean"] = 10.0
    df["cf"] = np.clip(generic_power_curve(base) * 0.6 + rng.normal(0, 0.03, len(idx)), 0, 1)
    df["tso_cf"] = df["cf"] + rng.normal(0, 0.02, len(idx))
    return df


def test_features_are_ex_ante_only():
    df = _toy_frame(20)
    X = build_features(df)
    assert "cf" not in X.columns and "tso_cf" not in X.columns
    X2 = build_features(df.assign(cf=0.0))  # changing the realised target must not change features
    pd.testing.assert_frame_equal(X, X2)
    assert "tso_cf" in build_features(df, with_tso=True).columns


def test_rolling_backtest_respects_training_cutoff():
    df = _toy_frame(120)
    X, y = build_features(df).dropna(), df["cf"]
    seen = []

    def spy(Xtr, ytr, Xte):
        seen.append((Xtr.index.max(), Xte.index.min()))
        return climatology(ytr, Xte.index)

    fc = rolling_backtest(X, y.reindex(X.index), "2024-03-01", "2024-04-01", spy, delay_days=1)
    assert len(seen) == 2 and list(fc.columns) == list(QUANTILES)
    for tr_end, te_start in seen:
        assert tr_end < te_start - pd.Timedelta(days=1) + pd.Timedelta(hours=1)


def test_gbm_quantiles_monotone_and_informative():
    df = _toy_frame(90)
    X, y = build_features(df).dropna(), df["cf"]
    y = y.reindex(X.index)
    n = len(X) * 3 // 4
    Q = gbm_qr(X.iloc[:n], y.iloc[:n], X.iloc[n:], max_iter=60)
    assert np.all(np.diff(Q, axis=1) >= 0)
    clim = climatology(y.iloc[:n], X.index[n:])
    yt = y.iloc[n:].to_numpy()
    assert mean_pinball(yt, Q, QUANTILES).mean() < mean_pinball(yt, clim, QUANTILES).mean()
