"""Quantile post-processing and conformal calibration of prediction intervals.

* ``rearrange`` - sort quantiles to remove quantile crossing (Chernozhukov et al. 2010).
* ``cqr_scores`` / ``cqr_adjust`` - conformalized quantile regression (Romano, Patterson
  & Candes 2019): conformity score E = max(q_lo - y, y - q_hi); the interval is widened (or
  narrowed) by the finite-sample (1 - alpha)(1 + 1/n) empirical quantile of E.
* ``aci_alphas`` - adaptive conformal inference (Gibbs & Candes 2021):
  alpha_{t+1} = alpha_t + gamma (alpha - err_t), updated once per forecast day with the miss
  rate of a day whose outcome is already known at forecast time.

Hourly wind errors are strongly autocorrelated, so exchangeability does not hold and the
split-conformal guarantee is only approximate; ACI targets long-run coverage under drift.
"""
from __future__ import annotations

import numpy as np


def rearrange(q: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.clip(np.sort(np.asarray(q, dtype=float), axis=1), lo, hi)


def cqr_scores(y, q_lo, q_hi) -> np.ndarray:
    y, q_lo, q_hi = map(lambda a: np.asarray(a, dtype=float), (y, q_lo, q_hi))
    return np.maximum(q_lo - y, y - q_hi)


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample corrected (1-alpha) quantile of conformity scores."""
    s = np.sort(np.asarray(scores, dtype=float)[~np.isnan(scores)])
    n = len(s)
    if n == 0:
        return 0.0
    if alpha <= 0:
        return float("inf")
    if alpha >= 1:
        return float("-inf")
    k = int(np.ceil((1 - alpha) * (n + 1)))
    return float(s[min(max(k, 1), n) - 1]) if k <= n else float("inf")


def cqr_adjust(q_lo, q_hi, qhat: float, lo: float = 0.0, hi: float = 1.0):
    return np.clip(np.asarray(q_lo) - qhat, lo, hi), np.clip(np.asarray(q_hi) + qhat, lo, hi)


def aci_alphas(daily_err: np.ndarray, alpha: float, gamma: float, delay: int = 2) -> np.ndarray:
    """Per-day working alpha for ACI given the per-day miss rate series ``daily_err`` (NaN = unknown).

    Day d uses only errors up to day d - ``delay`` (ex-ante: at noon on D-1 the full outcome of
    D-2 is known). Returns alpha_t for every day.
    """
    n = len(daily_err)
    a = np.full(n, alpha, dtype=float)
    cur = alpha
    for d in range(n):
        a[d] = cur
        j = d - delay + 1  # error that becomes known in time for day d+1
        if 0 <= j < n and not np.isnan(daily_err[j]):
            cur = cur + gamma * (alpha - daily_err[j])
    return a


def rolling_conformal(y, q_lo, q_hi, day, alpha: float, window_days: int = 90,
                      gamma: float = 0.0, delay: int = 2, lo: float = 0.0, hi: float = 1.0):
    """Sequential day-by-day conformal interval adjustment on out-of-sample quantile forecasts.

    For forecast day d the CQR scores of the previous ``window_days`` days whose outcomes are
    known (days <= d - delay) set the margin. With ``gamma`` > 0 the working alpha is adapted
    by ACI using the realised miss rate of the conformalised interval on day d - delay.

    ``day`` is an integer day index per observation (non-decreasing). Returns (lo, hi, alpha_by_day).
    """
    y, q_lo, q_hi, day = (np.asarray(a) for a in (y, q_lo, q_hi, day))
    scores = cqr_scores(y, q_lo, q_hi)
    out_lo, out_hi = np.full(len(y), np.nan), np.full(len(y), np.nan)
    days = np.unique(day)
    a_t, alphas, miss = alpha, {}, {}
    for d in days:
        # ACI update with the newest day whose outcome is known at forecast time
        j = d - delay
        if gamma > 0 and j in miss and not np.isnan(miss[j]):
            a_t = a_t + gamma * (alpha - miss[j])
        alphas[d] = a_t
        m_hist = (day > d - delay - window_days) & (day <= d - delay)
        qhat = conformal_quantile(scores[m_hist], a_t) if m_hist.any() else 0.0
        if not np.isfinite(qhat):  # alpha_t <= 0 -> trivial full interval
            qhat = hi - lo
        m = day == d
        out_lo[m], out_hi[m] = cqr_adjust(q_lo[m], q_hi[m], qhat, lo, hi)
        ok = m & ~np.isnan(y)
        miss[d] = float(np.mean((y[ok] < out_lo[ok]) | (y[ok] > out_hi[ok]))) if ok.any() else np.nan
    return out_lo, out_hi, alphas
