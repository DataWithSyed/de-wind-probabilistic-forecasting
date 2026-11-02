"""Figures from results/ (run after run_backtest.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dwpf.data import add_target, load_dataset

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = {"climatology": "Climatology", "linear_qr": "Linear QR (NWP)", "gbm_nwp": "GBM-QR (NWP)",
         "gbm_nwp_tso": "GBM-QR (NWP + TSO fc)", "tso_point": "TSO point fc"}


def read(name):
    f = pd.read_csv(RES / name, index_col=0)
    f.index = pd.to_datetime(f.index, utc=True)
    return f


df = add_target(load_dataset(ROOT / "data/de_wind_hourly.csv.gz"))
metrics = pd.read_csv(RES / "metrics.csv").set_index("model")
fc = read("forecast_gbm_nwp_tso.csv.gz")
aci = read("interval90_aci_gbm_nwp_tso.csv.gz")
idx = aci.index
cap = df["onshore_capacity_mw"].reindex(idx)

# --- 1 headline: fan chart for a windy fortnight + CRPS bar ------------------
gw = (df["wind_onshore_mw"].reindex(idx) / 1000).rolling(24 * 14).std()
end = gw.idxmax()
win = (idx > end - pd.Timedelta(days=14)) & (idx <= end)
t = idx[win]
c = cap[win].to_numpy() / 1000
Q = fc.reindex(t).to_numpy() * c[:, None]
fig, ax = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [3, 1.1]})
ax[0].fill_between(t, aci.lo[win] * c, aci.hi[win] * c, color="#9ecae1", alpha=0.6, label="90 % interval (ACI-conformalised)")
ax[0].fill_between(t, Q[:, 1], Q[:, 9], color="#3182bd", alpha=0.5, label="80 % raw quantile band")
ax[0].plot(t, Q[:, 5], color="#08519c", lw=1.2, label="median forecast")
ax[0].plot(t, df["tso_cf"].reindex(t) * c, color="#ff7f0e", lw=0.9, ls="--", label="TSO day-ahead forecast")
ax[0].plot(t, df["wind_onshore_mw"].reindex(t) / 1000, color="k", lw=1.0, label="realised")
ax[0].set_ylabel("German onshore wind [GW]")
ax[0].set_title(f"Day-ahead probabilistic forecast, {t[0]:%d %b %Y} - {t[-1]:%d %b %Y} (out-of-sample)")
ax[0].legend(fontsize=8, loc="upper left")
order = ["climatology", "linear_qr", "gbm_nwp", "tso_point", "gbm_nwp_tso"]
vals = metrics.loc[order, "crps_mw"]
ax[1].barh([NAMES[o] for o in order], vals, color=["#bbbbbb", "#9ecae1", "#4292c6", "#ff7f0e", "#08519c"])
for i, v in enumerate(vals):
    ax[1].text(v, i, f" {v:,.0f}", va="center", fontsize=8)
ax[1].set_xlabel("mean CRPS [MW] (point fc: MAE)"); ax[1].set_title("Test year score (lower = better)")
ax[1].set_xlim(0, vals.max() * 1.25)
fig.tight_layout(); fig.savefig(FIG / "01_headline.png", dpi=130); plt.close(fig)

# --- 2 coverage by month -----------------------------------------------------
cm = pd.read_csv(RES / "coverage90_by_month.csv", index_col=0)
fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(cm))
for k, (col, lab, colr) in enumerate([("raw", "raw GBM quantiles", "#bbbbbb"), ("cqr", "rolling CQR (90 d)", "#6baed6"),
                                      ("aci", "rolling CQR + ACI", "#08519c")]):
    ax.bar(x + (k - 1) * 0.27, cm[col], 0.27, color=colr, label=lab)
ax.axhline(0.9, color="r", ls="--", lw=1, label="target 90 %")
ax.set_xticks(x); ax.set_xticklabels(cm.index, rotation=45, fontsize=8)
ax.set_ylim(0.6, 1.0); ax.set_ylabel("coverage of the 90 % interval"); ax.legend(fontsize=8, ncol=4, loc="lower left")
ax.set_title("Monthly coverage, GBM-QR (NWP + TSO), test period")
fig.tight_layout(); fig.savefig(FIG / "02_coverage_by_month.png", dpi=130); plt.close(fig)

# --- 3 reliability -------------------------------------------------------------
rel = pd.read_csv(RES / "reliability.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], "k--", lw=1)
for m in rel.columns:
    ax.plot(rel.index, rel[m], "o-", ms=4, label=NAMES[m])
ax.set_xlabel("nominal quantile level"); ax.set_ylabel("observed frequency y <= quantile")
ax.set_title("Reliability (test period, raw quantiles)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "03_reliability.png", dpi=130); plt.close(fig)

# --- 4 pinball by hour of day ---------------------------------------------------
from dwpf.metrics import mean_pinball  # noqa: E402
from dwpf.models import QUANTILES  # noqa: E402
y = df["cf"].reindex(idx).to_numpy()
fig, ax = plt.subplots(figsize=(8, 4))
for m in ("linear_qr", "gbm_nwp", "gbm_nwp_tso"):
    f = read(f"forecast_{m}.csv.gz").reindex(idx).to_numpy()
    crps = pd.Series(2 * mean_pinball(y, f, QUANTILES) * cap.to_numpy(), index=idx)
    ax.plot(crps.groupby(idx.hour).mean(), "o-", ms=3, label=NAMES[m],
            color={"linear_qr": "#9ecae1", "gbm_nwp": "#4292c6", "gbm_nwp_tso": "#08519c"}[m])
tso = pd.Series(np.abs(df["tso_cf"].reindex(idx).to_numpy() - y) * cap.to_numpy(), index=idx)
ax.plot(tso.groupby(idx.hour).mean(), "s--", ms=3, color="#ff7f0e", label=NAMES["tso_point"])
ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel("mean CRPS / MAE [MW]"); ax.legend(fontsize=8)
ax.set_title("Score by delivery hour (test period)")
fig.tight_layout(); fig.savefig(FIG / "04_crps_by_hour.png", dpi=130); plt.close(fig)
print("figures written")
