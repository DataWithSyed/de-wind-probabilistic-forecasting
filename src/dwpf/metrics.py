"""Probabilistic forecast scores."""
from __future__ import annotations

import numpy as np
from scipy import stats


def pinball(y, q, tau) -> np.ndarray:
    """Pinball (quantile) loss per observation for quantile level ``tau``."""
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.maximum(tau * d, (tau - 1) * d)


def mean_pinball(y, Q, taus) -> np.ndarray:
    """Average pinball loss over quantile levels, per observation. 2x this approximates the CRPS."""
    Q = np.asarray(Q, dtype=float)
    return np.mean([pinball(y, Q[:, i], t) for i, t in enumerate(taus)], axis=0)


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def winkler(y, lo, hi, alpha: float) -> np.ndarray:
    """Interval (Winkler) score: width plus 2/alpha times the exceedance."""
    y, lo, hi = map(lambda a: np.asarray(a, dtype=float), (y, lo, hi))
    return (hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)


def diebold_mariano(l1, l2, h: int = 1) -> tuple[float, float]:
    """DM test on loss series (e.g. daily mean losses). Positive stat => model 2 better.
    HAC (Newey-West, Bartlett) variance with h-1 lags; two-sided p-value from Student t (n-1)."""
    d = np.asarray(l1, dtype=float) - np.asarray(l2, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, h):
        var += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)
