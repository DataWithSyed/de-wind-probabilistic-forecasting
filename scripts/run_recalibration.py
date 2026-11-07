"""Full-distribution recalibration (global and Mondrian per-level conformal shifts), PIT
histograms, and regime / ramp breakdowns on the stored out-of-sample backtest forecasts.

Uses only results/forecast_*.csv.gz from run_backtest.py and the cached data set; no refits.
"""
import json
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
from dwpf.metrics import coverage, diebold_mariano, mean_pinball, winkler
from dwpf.models import QUANTILES
from dwpf.recalibration import (expected_pit, pit_bins, ramp_flags, reliability_mad,
                                rolling_quantile_recalibration)

RES, FIG = ROOT / "results", ROOT / "figures"
TEST_START = "2025-11-01"
WINDOW_DAYS, DELAY, MONDRIAN_BINS = 90, 2, 3
MODELS = ["linear_qr", "gbm_nwp", "gbm_nwp_tso"]
I = {0.05: 0, 0.1: 1, 0.5: 5, 0.9: 9, 0.95: 10}


def main():
    df = add_target(load_dataset(ROOT / "data/de_wind_hourly.csv.gz"))
    df.loc[(df.cf < 0) | (df.cf > 1.05), "cf"] = np.nan
    y_all, cap_all = df["cf"], df["onshore_capacity_mw"]
    fc = {m: pd.read_csv(RES / f"forecast_{m}.csv.gz", index_col=0, parse_dates=True) for m in MODELS}
    # same evaluation index as run_backtest.py (all forecasts + target + TSO available, test year)
    idx = None
    for m in ["climatology"] + MODELS:
        f = pd.read_csv(RES / f"forecast_{m}.csv.gz", index_col=0, parse_dates=True)
        idx = f.index if idx is None else idx.intersection(f.index)
    idx = idx[(idx >= pd.Timestamp(TEST_START, tz="UTC")) & y_all.reindex(idx).notna().to_numpy()
              & df["tso_cf"].reindex(idx).notna().to_numpy()]
    yt, cap = y_all.reindex(idx).to_numpy(), cap_all.reindex(idx).to_numpy()
    print(f"evaluation hours: {len(idx)}  ({idx[0]} .. {idx[-1]})")

    variants, rows, daily, pit_rows = {}, [], {}, []
    for name, f in fc.items():
        full = f.index
        yf = y_all.reindex(full).to_numpy()
        dayf = ((full.floor("D") - full[0].floor("D")).days).to_numpy()
        Qf = f.to_numpy()
        for vname, bins in (("raw", None), ("global", 0), ("mondrian", MONDRIAN_BINS)):
            if bins is None:
                Qv = Qf
            else:
                Qv = rolling_quantile_recalibration(yf, Qf, QUANTILES, dayf, window_days=WINDOW_DAYS,
                                                    delay=DELAY, mondrian_bins=bins)
            Q = pd.DataFrame(Qv, index=full).reindex(idx).to_numpy()
            variants[(name, vname)] = Q
            pl = 2 * mean_pinball(yt, Q, QUANTILES)
            daily[(name, vname)] = pd.Series(pl, index=idx).groupby(idx.floor("D")).mean()
            row = dict(model=name, variant=vname, crps_mw=float(np.mean(pl * cap)),
                       mae_median_mw=float(np.mean(np.abs(Q[:, I[0.5]] - yt) * cap)),
                       reliability_mad_pp=100 * reliability_mad(yt, Q, QUANTILES))
            for lev, a, l, h in (("80", 0.2, 0.1, 0.9), ("90", 0.1, 0.05, 0.95)):
                lo_, hi_ = Q[:, I[l]], Q[:, I[h]]
                row[f"cov{lev}"] = coverage(yt, lo_, hi_)
                row[f"width{lev}_mw"] = float(np.mean((hi_ - lo_) * cap))
                row[f"winkler{lev}_mw"] = float(np.mean(winkler(yt, lo_, hi_, a) * cap))
            pit = pit_bins(yt, Q, QUANTILES)
            row["pit_max_abs_dev_pp"] = 100 * float(np.max(np.abs(pit - expected_pit(QUANTILES))))
            row["pit_tail_mass_pct"] = 100 * float(pit[0] + pit[-1])
            rows.append(row)
            pit_rows.append(dict(model=name, variant=vname, **{f"bin{i}": v for i, v in enumerate(pit)}))
    met = pd.DataFrame(rows)
    met.round(4).to_csv(RES / "recalibration_metrics.csv", index=False)
    pd.DataFrame(pit_rows).round(5).to_csv(RES / "pit_bins.csv", index=False)

    dm = []
    for name in MODELS:
        for a, b in (("raw", "global"), ("raw", "mondrian"), ("global", "mondrian")):
            s, p = diebold_mariano(daily[(name, a)], daily[(name, b)], h=2)
            dm.append(dict(model=name, variant_1=a, variant_2=b, dm_stat=s, p_value=p))
    dm = pd.DataFrame(dm)
    dm.round(6).to_csv(RES / "recalibration_dm_tests.csv", index=False)

    # regimes: terciles of the ex-ante raw forecast median of the best model (evaluation grouping)
    med = variants[("gbm_nwp_tso", "raw")][:, I[0.5]]
    edges = np.quantile(med, [1 / 3, 2 / 3])
    regime = np.array(["low", "mid", "high"])[np.searchsorted(edges, med, side="right")]
    ramp = ramp_flags(yt, step=3, q=0.95)
    seg = []
    for vname in ("raw", "global", "mondrian"):
        Q = variants[("gbm_nwp_tso", vname)]
        pl = 2 * mean_pinball(yt, Q, QUANTILES) * cap
        for gname, mask in ([(f"regime_{r}", regime == r) for r in ("low", "mid", "high")]
                            + [("ramp_top5pct", ramp), ("no_ramp", ~ramp), ("all", np.ones(len(yt), bool))]):
            lo_, hi_ = Q[mask, I[0.05]], Q[mask, I[0.95]]
            seg.append(dict(variant=vname, segment=gname, hours=int(mask.sum()),
                            crps_mw=float(pl[mask].mean()), cov90=coverage(yt[mask], lo_, hi_),
                            width90_mw=float(np.mean((hi_ - lo_) * cap[mask])),
                            below_q05=float(np.mean(yt[mask] < lo_)),
                            above_q95=float(np.mean(yt[mask] > hi_))))
    seg = pd.DataFrame(seg)
    seg.round(4).to_csv(RES / "regime_ramp_breakdown.csv", index=False)
    meta = dict(window_days=WINDOW_DAYS, delay_days=DELAY, mondrian_bins=MONDRIAN_BINS,
                regime_edges_cf=[float(e) for e in edges],
                ramp_threshold_cf_3h=float(np.nanquantile(np.abs(yt[3:] - yt[:-3]), 0.95)),
                n_hours=int(len(idx)))
    (RES / "recalibration_meta.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 220)
    print(met.round(3).to_string(index=False))
    print(dm.round(4).to_string(index=False))
    print(seg.round(3).to_string(index=False))
    print(meta)

    # ---- figures
    colors = {"raw": "#8c3a22", "global": "#1f5f8b", "mondrian": "#4c9a6a"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for vname in ("raw", "global", "mondrian"):
        Q = variants[("gbm_nwp_tso", vname)]
        obs = (yt[:, None] <= Q).mean(axis=0)
        r = met[(met.model == "gbm_nwp_tso") & (met.variant == vname)].iloc[0]
        ax[0].plot(QUANTILES, obs, "o-", color=colors[vname],
                   label=f"{vname} (MAD {r.reliability_mad_pp:.1f} pp, CRPS {r.crps_mw:.0f} MW)")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("nominal quantile level"); ax[0].set_ylabel("observed frequency y <= q")
    ax[0].set_title("Reliability, GBM-QR (NWP + TSO), test year", fontsize=9.5)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    exp = expected_pit(QUANTILES)
    xb = np.arange(len(exp))
    w = 0.27
    for j, vname in enumerate(("raw", "global", "mondrian")):
        pit = pit_bins(yt, variants[("gbm_nwp_tso", vname)], QUANTILES)
        ax[1].bar(xb + (j - 1) * w, pit / exp, w, color=colors[vname], label=vname)
    ax[1].axhline(1, color="k", lw=0.8)
    edges_lbl = ["<q05"] + [f"q{int(100*a):02d}-q{int(100*b):02d}" for a, b in
                            zip(QUANTILES[:-1], QUANTILES[1:])] + [">q95"]
    ax[1].set_xticks(xb); ax[1].set_xticklabels(edges_lbl, rotation=60, fontsize=6.5)
    ax[1].set_ylabel("observed / expected mass"); ax[1].set_title("PIT histogram (quantile bins)", fontsize=9.5)
    ax[1].legend(fontsize=7); ax[1].grid(axis="y", alpha=0.3)
    segs = ["regime_low", "regime_mid", "regime_high", "ramp_top5pct"]
    for j, vname in enumerate(("raw", "global", "mondrian")):
        s = seg[seg.variant == vname].set_index("segment").loc[segs]
        ax[2].bar(np.arange(len(segs)) + (j - 1) * w, 100 * s.cov90, w, color=colors[vname], label=vname)
    ax[2].axhline(90, color="k", ls="--", lw=0.8)
    ax[2].set_xticks(range(len(segs)))
    ax[2].set_xticklabels(["low wind", "mid wind", "high wind", "ramp hours\n(top 5 % |Δ3h|)"], fontsize=8)
    ax[2].set_ylim(50, 100); ax[2].set_ylabel("90 % interval coverage (%)")
    ax[2].set_title("Coverage by regime (ex-ante median terciles) and ramps", fontsize=9.5)
    ax[2].legend(fontsize=7); ax[2].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "05_recalibration_pit_regimes.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
