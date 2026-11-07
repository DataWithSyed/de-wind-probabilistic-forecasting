import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dwpf.recalibration import (expected_pit, pit_bins, ramp_flags, reliability_mad,  # noqa: E402
                                rolling_quantile_recalibration)

TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def _panel(n_days=200, per_day=24, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    return rng, day


def test_global_recalibration_fixes_bias_and_overconfidence():
    rng, day = _panel()
    y = rng.normal(0.0, 1.0, len(day))
    # forecast is biased (+0.5) and too narrow (sd 0.5)
    Q = 0.5 + 0.5 * stats.norm.ppf(TAUS)[None, :].repeat(len(day), axis=0)
    out = rolling_quantile_recalibration(y, Q, TAUS, day, window_days=60, delay=2,
                                         lo=-np.inf, hi=np.inf)
    late = day >= 100
    assert reliability_mad(y[late], Q[late], TAUS) > 0.10
    assert reliability_mad(y[late], out[late], TAUS) < 0.02


def test_recalibration_is_ex_ante():
    rng, day = _panel(n_days=40)
    y = rng.normal(size=len(day))
    Q = np.tile(stats.norm.ppf(TAUS), (len(day), 1))
    a = rolling_quantile_recalibration(y, Q, TAUS, day, window_days=10, delay=2,
                                       lo=-np.inf, hi=np.inf)
    y2 = y.copy()
    y2[(day == 30) | (day == 29)] += 5.0          # outcomes not yet known when forecasting day 30
    b = rolling_quantile_recalibration(y2, Q, TAUS, day, window_days=10, delay=2,
                                       lo=-np.inf, hi=np.inf)
    assert np.allclose(a[day == 30], b[day == 30])
    assert not np.allclose(a[day == 31], b[day == 31])   # day 29 becomes known for day 31
    assert np.allclose(a[day == 0], Q[day == 0])          # no history -> unchanged


def test_output_sorted_and_clipped():
    rng, day = _panel(n_days=30)
    y = rng.uniform(0, 1, len(day))
    Q = np.clip(rng.uniform(0, 1, (len(day), len(TAUS))), 0, 1)   # crossing on purpose
    out = rolling_quantile_recalibration(y, Q, TAUS, day, window_days=10)
    assert (np.diff(out, axis=1) >= 0).all() and out.min() >= 0 and out.max() <= 1


def test_mondrian_equalises_coverage_across_regimes():
    rng, day = _panel(n_days=240)
    level = rng.uniform(0, 1, len(day))
    y = level + rng.normal(0, 0.05 + 0.3 * level)      # noise grows with the level
    Q = level[:, None] + 0.15 * stats.norm.ppf(TAUS)[None, :]   # constant width
    kw = dict(window_days=90, delay=2, lo=-np.inf, hi=np.inf, min_group=100)
    g = rolling_quantile_recalibration(y, Q, TAUS, day, mondrian_bins=0, **kw)
    m = rolling_quantile_recalibration(y, Q, TAUS, day, mondrian_bins=3, **kw)
    late = day >= 120
    grp = np.digitize(level, [1 / 3, 2 / 3])

    def cov_spread(Qx):
        c = [np.mean((y[s] >= Qx[s, 0]) & (y[s] <= Qx[s, -1]))
             for s in [late & (grp == k) for k in range(3)]]
        return max(c) - min(c)

    assert cov_spread(m) < 0.5 * cov_spread(g)


def test_pit_bins_and_ramps():
    rng = np.random.default_rng(3)
    y = rng.normal(size=50_000)
    Q = np.tile(stats.norm.ppf(TAUS), (len(y), 1))
    pit = pit_bins(y, Q, TAUS)
    assert np.isclose(pit.sum(), 1.0)
    assert np.allclose(pit, expected_pit(TAUS), atol=0.01)
    walk = np.cumsum(rng.normal(size=10_000))
    assert 0.045 < ramp_flags(walk, step=3, q=0.95).mean() < 0.056
