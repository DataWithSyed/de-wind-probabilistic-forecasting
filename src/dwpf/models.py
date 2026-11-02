"""Quantile forecasting models and the rolling-origin (monthly refit) backtest."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

from .conformal import rearrange

QUANTILES = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
LQR_FEATURES = ["pc_mean", "pc_north", "pc_south", "pc_mean_roll6", "ws_std", "hour_sin", "hour_cos"]


def climatology(y_train: pd.Series, index: pd.DatetimeIndex, taus=QUANTILES) -> np.ndarray:
    """Empirical quantiles of the training capacity factor by calendar month (all hours)."""
    yt = y_train.dropna()
    tab = {m: np.quantile(g.to_numpy(), taus) for m, g in yt.groupby(yt.index.month)}
    allq = np.quantile(yt.to_numpy(), taus)
    return np.vstack([tab.get(m, allq) for m in index.month])


def linear_qr(X_train, y_train, X_test, taus=QUANTILES, max_rows: int = 8760) -> np.ndarray:
    """Linear quantile regression (HiGHS LP) on a small physically-motivated feature set,
    fitted on the most recent ``max_rows`` training hours."""
    Xtr, ytr = X_train[LQR_FEATURES].tail(max_rows), y_train.tail(max_rows)
    preds = []
    for t in taus:
        m = QuantileRegressor(quantile=float(t), alpha=0.0, solver="highs").fit(Xtr, ytr)
        preds.append(m.predict(X_test[LQR_FEATURES]))
    return rearrange(np.column_stack(preds))


def gbm_qr(X_train, y_train, X_test, taus=QUANTILES, seed: int = 0, **kw) -> np.ndarray:
    """One gradient-boosted tree model per quantile (sklearn HistGradientBoosting, pinball loss)."""
    params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, random_state=seed)
    params.update(kw)
    preds = []
    for t in taus:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=float(t), **params)
        m.fit(X_train, y_train)
        preds.append(m.predict(X_test))
    return rearrange(np.column_stack(preds))


def rolling_backtest(X: pd.DataFrame, y: pd.Series, first_month: str, last_month: str,
                     model, delay_days: int = 1, train_start: str | None = None) -> pd.DataFrame:
    """Refit at the start of each month; forecast every hour of that month.

    Training uses target hours strictly before (month start - ``delay_days``), i.e. outcomes up
    to the end of D-2 for the first forecast day D, which is what is known at a D-1 noon gate.
    Returns a DataFrame of quantile forecasts (columns = quantile levels) indexed like X.
    """
    out = []
    months = pd.date_range(first_month, last_month, freq="MS", tz="UTC")
    valid = X.notna().all(axis=1) & y.notna()
    for m in months:
        cutoff = m - pd.Timedelta(days=delay_days)
        tr = valid & (X.index < cutoff)
        if train_start is not None:
            tr &= X.index >= pd.Timestamp(train_start, tz="UTC")
        te = (X.index >= m) & (X.index < m + pd.offsets.MonthBegin(1)) & X.notna().all(axis=1)
        if not te.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = model(X[tr], y[tr], X[te])
        out.append(pd.DataFrame(q, index=X.index[te], columns=QUANTILES))
    return pd.concat(out)
