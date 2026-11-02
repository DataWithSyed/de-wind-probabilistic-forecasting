"""Ex-ante feature engineering from day-ahead NWP (ICON-EU previous-day run).

Every feature is a deterministic function of weather forecasts that were available
before the day-ahead gate closure, plus calendar terms; no realised generation is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

V_CUT_IN, V_RATED, V_CUT_OUT = 3.0, 12.5, 25.0


def generic_power_curve(v) -> np.ndarray:
    """Normalised generic turbine power curve (0..1): cubic between cut-in and rated,
    flat to cut-out. Used only as a physically-shaped feature, not as a turbine model."""
    v = np.asarray(v, dtype=float)
    x = np.clip((v**3 - V_CUT_IN**3) / (V_RATED**3 - V_CUT_IN**3), 0.0, 1.0)
    return np.where(v >= V_CUT_OUT, 0.0, x)


def site_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ws100_")]


def build_features(df: pd.DataFrame, with_tso: bool = False) -> pd.DataFrame:
    sites = site_columns(df)
    ws = df[sites]
    pc = pd.DataFrame(generic_power_curve(ws.to_numpy()), index=df.index, columns=[c.replace("ws100_", "pc_") for c in sites])
    X = pd.DataFrame(index=df.index)
    X["ws_mean"] = ws.mean(axis=1)
    X["ws_std"] = ws.std(axis=1)
    X["ws_max"] = ws.max(axis=1)
    X["pc_mean"] = pc.mean(axis=1)
    X["pc_std"] = pc.std(axis=1)
    # regional means (north vs. south) of the power-curve feature
    north = [c for c in pc.columns if c.split("_")[1] in ("sh", "ni", "mv", "bb", "st")]
    X["pc_north"] = pc[north].mean(axis=1)
    X["pc_south"] = pc.drop(columns=north).mean(axis=1)
    # forecast trajectory: neighbouring hours of the same (ex-ante) NWP run
    for lag in (-3, -1, 1, 3):
        X[f"pc_mean_shift{lag}"] = X["pc_mean"].shift(lag)
    X["pc_mean_roll6"] = X["pc_mean"].rolling(6, center=True, min_periods=1).mean()
    X["shear"] = X["ws_mean"] / df["ws10_mean"].clip(lower=0.5)
    X["t2m"] = df["t2m_mean"]
    X = pd.concat([X, pc], axis=1)
    hour = df.index.hour
    doy = df.index.dayofyear
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if with_tso:
        X["tso_cf"] = df["tso_cf"]
    return X
