"""Full-distribution recalibration of quantile forecasts and diagnostics (PIT, regimes, ramps).

``rolling_quantile_recalibration`` shifts *every* quantile level, not only the interval end
points. For forecast day d and level tau the shift is the empirical tau-quantile of the
residuals y - q_tau over the previous ``window_days`` days whose outcomes are known at forecast
time (days <= d - ``delay``), i.e. a one-sided split-conformal correction per level (the
per-level analogue of CQR). Quantiles are re-sorted afterwards to remove crossing.

The *Mondrian* variant computes the shifts separately within bins of the ex-ante forecast
median (low / mid / high wind), with bin edges taken from the same history window, so the
correction can differ between calm and windy hours (Vovk et al. 2005, "Mondrian" conformal
predictors). Bins with fewer than ``min_group`` history points fall back to the global shift.
"""
from __future__ import annotations

import numpy as np


def _level_quantile(r: np.ndarray, tau: float) -> float:
    """Conservative empirical quantile: 'higher' order statistic above the median, 'lower' below."""
    return float(np.quantile(r, tau, method="higher" if tau >= 0.5 else "lower"))


def _shifts(R: np.ndarray, taus) -> np.ndarray:
    return np.array([_level_quantile(R[:, i], t) for i, t in enumerate(taus)])


def rolling_quantile_recalibration(y, Q, taus, day, window_days: int = 90, delay: int = 2,
                                   mondrian_bins: int = 0, min_group: int = 200,
                                   median_idx: int | None = None, lo: float = 0.0,
                                   hi: float = 1.0):
    """Sequential day-by-day per-level conformal shift of a quantile matrix ``Q`` (n x k).

    ``day``: non-decreasing integer day index per row. ``mondrian_bins`` > 1 enables the
    Mondrian variant with that many equal-frequency bins of the forecast median.
    Rows on days without any history are returned unchanged.
    Returns the recalibrated matrix (same shape).
    """
    y = np.asarray(y, dtype=float)
    Q = np.asarray(Q, dtype=float)
    day = np.asarray(day)
    taus = list(taus)
    k = len(taus)
    mi = median_idx if median_idx is not None else int(np.argmin(np.abs(np.array(taus) - 0.5)))
    out = Q.copy()
    known = ~np.isnan(y)
    for d in np.unique(day):
        m = day == d
        h = (day > d - delay - window_days) & (day <= d - delay) & known
        if h.sum() < 10:
            continue
        R = y[h, None] - Q[h]
        glob = _shifts(R, taus)
        if mondrian_bins and mondrian_bins > 1:
            med_h = Q[h, mi]
            edges = np.quantile(med_h, np.linspace(0, 1, mondrian_bins + 1)[1:-1])
            g_h = np.searchsorted(edges, med_h, side="right")
            g_d = np.searchsorted(edges, Q[m, mi], side="right")
            sh = np.empty((m.sum(), k))
            for g in range(mondrian_bins):
                sel = g_h == g
                s = _shifts(R[sel], taus) if sel.sum() >= min_group else glob
                sh[g_d == g] = s
            out[m] = Q[m] + sh
        else:
            out[m] = Q[m] + glob
    return np.clip(np.sort(out, axis=1), lo, hi)


def pit_bins(y, Q, taus) -> np.ndarray:
    """Observed probability mass in each of the k+1 bins delimited by the k quantiles.

    For a calibrated forecast the expected masses are the level gaps
    (tau_1, tau_2 - tau_1, ..., 1 - tau_k). This is a PIT histogram that needs no
    interpolation between quantiles.
    """
    y = np.asarray(y, dtype=float)
    Q = np.asarray(Q, dtype=float)
    idx = (y[:, None] > Q).sum(axis=1)      # number of quantiles strictly below y
    return np.bincount(idx, minlength=len(taus) + 1) / len(y)


def expected_pit(taus) -> np.ndarray:
    t = np.concatenate([[0.0], np.asarray(taus, dtype=float), [1.0]])
    return np.diff(t)


def reliability_mad(y, Q, taus) -> float:
    """Mean absolute deviation between observed frequency below each quantile and its level."""
    y = np.asarray(y, dtype=float)
    obs = (y[:, None] <= np.asarray(Q)).mean(axis=0)
    return float(np.mean(np.abs(obs - np.asarray(taus))))


def ramp_flags(y, step: int = 3, q: float = 0.95) -> np.ndarray:
    """Hours whose realised |y_t - y_{t-step}| is in the top (1 - q) share (evaluation only)."""
    y = np.asarray(y, dtype=float)
    d = np.full(len(y), np.nan)
    d[step:] = np.abs(y[step:] - y[:-step])
    thr = np.nanquantile(d, q)
    return np.nan_to_num(d, nan=0.0) >= thr
