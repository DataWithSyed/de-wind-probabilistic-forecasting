"""Download and cache the public data used by the project.

Sources (both CC BY 4.0):

* Energy-Charts.info (Fraunhofer ISE): realised German onshore wind generation
  (``public_power``), the TSOs' published day-ahead onshore wind forecast
  (``public_power_forecast``) and monthly installed onshore capacity
  (``installed_power``).
* Open-Meteo Previous Runs API: DWD ICON-EU numerical weather prediction (NWP)
  values *as forecast one day earlier* (``*_previous_day1``) at 20 grid points
  across Germany's main onshore wind regions. Using the previous-day run makes the
  weather inputs genuinely ex-ante (no reanalysis / observed-weather leakage).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

EC_API = "https://api.energy-charts.info"
OM_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
START = "2024-03-01"
END = "2026-10-31"

# (name, lat, lon) - grid points spread over the main onshore wind regions
SITES: list[tuple[str, float, float]] = [
    ("sh_west", 54.5, 9.2), ("sh_east", 54.0, 9.8),
    ("ni_eastfrisia", 53.4, 7.6), ("ni_weser", 53.0, 8.8), ("ni_heide", 52.6, 9.8),
    ("ni_south", 52.2, 8.0), ("mv_west", 53.8, 12.2), ("mv_east", 53.6, 13.4),
    ("bb_north", 52.8, 12.8), ("bb_east", 52.0, 14.0), ("bb_west", 52.2, 12.2),
    ("st_north", 52.0, 11.4), ("st_south", 51.6, 11.9), ("nw_muenster", 51.7, 7.0),
    ("nw_sauerland", 51.3, 8.5), ("th_north", 50.9, 10.6), ("he_vogelsberg", 50.6, 9.0),
    ("rp_hunsrueck", 49.9, 7.4), ("by_franken", 49.3, 10.6), ("bw_alb", 48.6, 9.5),
]
NWP_VARS = ("wind_speed_100m", "wind_speed_10m", "temperature_2m")


def _get(url: str, params: dict, max_tries: int = 12) -> dict | list:
    for _ in range(max_tries):
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "45")) + 2
            print(f"  429, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {url} {params}")


def _hourly(unix_seconds, values, name: str) -> pd.Series:
    idx = pd.to_datetime(unix_seconds, unit="s", utc=True)
    s = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
                  index=idx, name=name, dtype="float64")
    return s.resample("1h").mean()


def fetch_power(start: str = START, end: str = END) -> pd.DataFrame:
    """Realised onshore wind (MW) and TSO day-ahead onshore forecast (MW), hourly UTC."""
    years = range(int(start[:4]), int(end[:4]) + 1)
    act, fc = [], []
    for y in years:
        s = max(start, f"{y}-01-01")
        e = min(end, f"{y}-12-31")
        print(f"energy-charts {s}..{e}")
        p = _get(f"{EC_API}/public_power", {"country": "de", "start": s, "end": e})
        series = {pt["name"]: pt["data"] for pt in p["production_types"]}
        act.append(_hourly(p["unix_seconds"], series["Wind onshore"], "wind_onshore_mw"))
        f = _get(f"{EC_API}/public_power_forecast",
                 {"country": "de", "production_type": "wind_onshore",
                  "forecast_type": "day-ahead", "start": s, "end": e})
        fc.append(_hourly(f["unix_seconds"], f["forecast_values"], "tso_da_forecast_mw"))
    df = pd.concat([pd.concat(act), pd.concat(fc)], axis=1)
    return df[~df.index.duplicated()].sort_index()


def fetch_capacity() -> pd.Series:
    """Monthly installed onshore wind capacity (GW), indexed by month start (UTC)."""
    d = _get(f"{EC_API}/installed_power",
             {"country": "de", "time_step": "monthly", "installation_decommission": "false"})
    cap = next(p["data"] for p in d["production_types"] if p["name"] == "Wind onshore")
    idx = pd.to_datetime(d["time"], format="%m.%Y").tz_localize("UTC")
    return pd.Series(cap, index=idx, name="onshore_capacity_gw", dtype="float64")


def fetch_nwp(start: str = START, end: str = END, sleep: float = 8.0) -> pd.DataFrame:
    """ICON-EU previous-day-run NWP at each site; 100 m speed per site plus site means."""
    hourly = ",".join(f"{v}_previous_day1" for v in NWP_VARS)
    ws100, ws10, t2 = {}, [], []
    for name, lat, lon in SITES:
        print(f"open-meteo {name}")
        d = _get(OM_API, {"latitude": lat, "longitude": lon, "start_date": start,
                          "end_date": end, "hourly": hourly, "models": "icon_eu",
                          "wind_speed_unit": "ms", "timezone": "GMT"})
        h = d["hourly"]
        idx = pd.to_datetime(h["time"]).tz_localize("UTC")
        ws100[f"ws100_{name}"] = pd.Series(h["wind_speed_100m_previous_day1"], index=idx, dtype="float64")
        ws10.append(pd.Series(h["wind_speed_10m_previous_day1"], index=idx, dtype="float64"))
        t2.append(pd.Series(h["temperature_2m_previous_day1"], index=idx, dtype="float64"))
        time.sleep(sleep)
    df = pd.DataFrame(ws100)
    df["ws10_mean"] = pd.concat(ws10, axis=1).mean(axis=1)
    df["t2m_mean"] = pd.concat(t2, axis=1).mean(axis=1)
    return df


def build_dataset(out: Path | str = "data/de_wind_hourly.csv.gz") -> pd.DataFrame:
    power = fetch_power()
    cap = fetch_capacity()
    nwp = fetch_nwp()
    df = power.join(nwp, how="inner")
    # monthly capacity -> hourly by forward fill of the month value
    cap_h = cap.reindex(df.index.union(cap.index)).sort_index().ffill().reindex(df.index)
    df["onshore_capacity_mw"] = cap_h * 1000.0
    df.index.name = "time_utc"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(out, compression="gzip")
    return df


def load_dataset(path: Path | str = "data/de_wind_hourly.csv.gz") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Capacity factor target (realised MW / installed MW) and TSO benchmark in CF units."""
    out = df.copy()
    out["cf"] = out["wind_onshore_mw"] / out["onshore_capacity_mw"]
    out["tso_cf"] = out["tso_da_forecast_mw"] / out["onshore_capacity_mw"]
    return out.replace([np.inf, -np.inf], np.nan)
