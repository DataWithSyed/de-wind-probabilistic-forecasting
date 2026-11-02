"""Rolling-origin backtest: climatology, linear QR, GBM-QR (NWP) and GBM-QR (NWP + TSO forecast),
with rolling CQR and ACI interval calibration. Writes results/ and prints a summary."""
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from dwpf.conformal import rolling_conformal
from dwpf.data import add_target, load_dataset
from dwpf.features import build_features
from dwpf.metrics import coverage, diebold_mariano, mean_pinball, pinball, winkler
from dwpf.models import QUANTILES, climatology, gbm_qr, linear_qr, rolling_backtest

RES = ROOT / "results"
FIRST_OOS, LAST_MONTH = "2025-07-01", "2026-10-01"   # OOS forecasts (first 4 months = conformal warm-up)
TEST_START = "2025-11-01"                             # evaluation window: 2025-11-01 .. end of data
ALPHAS = {"80": 0.20, "90": 0.10}
TAU_IDX = {0.05: 0, 0.1: 1, 0.5: 5, 0.9: 9, 0.95: 10}
GAMMA = 0.02            # ACI step per forecast day, fixed a priori (sensitivity reported separately)
WINDOW_DAYS = 90


def clim_model(Xtr, ytr, Xte):
    return climatology(ytr, Xte.index)


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    df = add_target(load_dataset(ROOT / "data/de_wind_hourly.csv.gz"))
    df.loc[(df.cf < 0) | (df.cf > 1.05), "cf"] = np.nan
    y = df["cf"]
    X_nwp, X_hyb = build_features(df), build_features(df, with_tso=True)
    models = {
        "climatology": (clim_model, X_nwp),
        "linear_qr": (linear_qr, X_nwp),
        "gbm_nwp": (partial(gbm_qr, seed=0), X_nwp),
        "gbm_nwp_tso": (partial(gbm_qr, seed=0), X_hyb),
    }
    fc = {}
    for name, (fn, X) in models.items():
        t1 = time.time()
        fc[name] = rolling_backtest(X, y, FIRST_OOS, LAST_MONTH, fn)
        fc[name].to_csv(RES / f"forecast_{name}.csv.gz", float_format="%.6f", compression="gzip")
        print(f"{name}: {len(fc[name])} h in {time.time()-t1:.0f}s", flush=True)

    # common evaluation index: all models + TSO + target available
    idx = fc["climatology"].index
    for f in fc.values():
        idx = idx.intersection(f.index)
    idx = idx[(idx >= pd.Timestamp(TEST_START, tz="UTC")) & y.reindex(idx).notna().to_numpy()
              & df["tso_cf"].reindex(idx).notna().to_numpy()]
    yt, cap = y.reindex(idx).to_numpy(), df["onshore_capacity_mw"].reindex(idx).to_numpy()
    tso = df["tso_cf"].reindex(idx).to_numpy()
    day = ((idx.floor("D") - idx[0].floor("D")).days).to_numpy()

    rows, daily_loss, intervals, iv_mem = [], {}, [], {}
    for name, f in fc.items():
        Q = f.reindex(idx).to_numpy()
        pl = mean_pinball(yt, Q, QUANTILES)
        daily_loss[name] = pd.Series(2 * pl, index=idx).groupby(idx.floor("D")).mean()
        med = Q[:, TAU_IDX[0.5]]
        row = dict(model=name, crps_cf=2 * pl.mean(), crps_mw=float(np.mean(2 * pl * cap)),
                   mae_median_mw=float(np.mean(np.abs(med - yt) * cap)),
                   rmse_median_mw=float(np.sqrt(np.mean(((med - yt) * cap) ** 2))))
        for lev, a in ALPHAS.items():
            lo, hi = Q[:, TAU_IDX[a / 2]], Q[:, TAU_IDX[1 - a / 2]]
            row[f"cov{lev}_raw"] = coverage(yt, lo, hi)
            row[f"width{lev}_raw_mw"] = float(np.mean((hi - lo) * cap))
            row[f"winkler{lev}_raw_mw"] = float(np.mean(winkler(yt, lo, hi, a) * cap))
        rows.append(row)

        # conformal calibration on the full OOS history (warm-up months feed the first windows)
        full = f.index[y.reindex(f.index).notna().to_numpy() | (f.index >= pd.Timestamp(TEST_START, tz="UTC"))]
        yf = y.reindex(full).to_numpy()
        dayf = ((full.floor("D") - full[0].floor("D")).days).to_numpy()
        Qf = f.reindex(full).to_numpy()
        for lev, a in ALPHAS.items():
            for method, g in (("cqr", 0.0), ("aci", GAMMA)):
                lo, hi, _ = rolling_conformal(yf, Qf[:, TAU_IDX[a / 2]], Qf[:, TAU_IDX[1 - a / 2]], dayf, a,
                                              window_days=WINDOW_DAYS, gamma=g)
                s_lo, s_hi = pd.Series(lo, index=full).reindex(idx).to_numpy(), pd.Series(hi, index=full).reindex(idx).to_numpy()
                intervals.append(dict(model=name, level=lev, method=method, cov=coverage(yt, s_lo, s_hi),
                                      width_mw=float(np.mean((s_hi - s_lo) * cap)),
                                      winkler_mw=float(np.mean(winkler(yt, s_lo, s_hi, a) * cap))))
                if name == "gbm_nwp_tso" and lev == "90":
                    iv_mem[method] = (s_lo, s_hi)
                    pd.DataFrame({"lo": s_lo, "hi": s_hi}, index=idx).to_csv(
                        RES / f"interval90_{method}_gbm_nwp_tso.csv.gz", float_format="%.6f", compression="gzip")
            intervals.append(dict(model=name, level=lev, method="raw", cov=row[f"cov{lev}_raw"],
                                  width_mw=row[f"width{lev}_raw_mw"], winkler_mw=row[f"winkler{lev}_raw_mw"]))

    # TSO day-ahead point forecast: its CRPS equals its absolute error
    tso_abs = np.abs(tso - yt)
    daily_loss["tso_point"] = pd.Series(tso_abs, index=idx).groupby(idx.floor("D")).mean()
    rows.append(dict(model="tso_point", crps_cf=tso_abs.mean(), crps_mw=float(np.mean(tso_abs * cap)),
                     mae_median_mw=float(np.mean(tso_abs * cap)),
                     rmse_median_mw=float(np.sqrt(np.mean(((tso - yt) * cap) ** 2)))))
    metrics = pd.DataFrame(rows)
    metrics["crps_skill_vs_clim"] = 1 - metrics.crps_cf / metrics.set_index("model").loc["climatology", "crps_cf"]
    metrics.round(5).to_csv(RES / "metrics.csv", index=False)
    iv = pd.DataFrame(intervals)
    iv.round(5).to_csv(RES / "intervals.csv", index=False)

    pairs = [("climatology", "linear_qr"), ("linear_qr", "gbm_nwp"), ("gbm_nwp", "gbm_nwp_tso"),
             ("tso_point", "gbm_nwp_tso"), ("tso_point", "gbm_nwp")]
    dm = pd.DataFrame([dict(model_1=a, model_2=b, dm_stat=s, p_value=p)
                       for a, b in pairs for s, p in [diebold_mariano(daily_loss[a], daily_loss[b], h=2)]])
    dm.round(6).to_csv(RES / "dm_tests.csv", index=False)

    # reliability: observed frequency below each quantile
    rel = pd.DataFrame({name: [(yt <= f.reindex(idx).to_numpy()[:, i]).mean() for i in range(len(QUANTILES))]
                        for name, f in fc.items()}, index=QUANTILES)
    rel.index.name = "tau"
    rel.round(4).to_csv(RES / "reliability.csv")

    # monthly coverage of the 90 % interval (best model)
    f = fc["gbm_nwp_tso"].reindex(idx).to_numpy()
    month = idx.strftime("%Y-%m")
    mon = pd.DataFrame({"y": yt, "raw_lo": f[:, 0], "raw_hi": f[:, 10]}, index=idx)
    for method in ("cqr", "aci"):  # in-memory (unrounded) intervals, identical to the ones scored above
        mon[f"{method}_lo"], mon[f"{method}_hi"] = iv_mem[method]
    cov_m = pd.DataFrame({m: mon.groupby(month).apply(lambda g, m=m: coverage(g.y, g[f"{m}_lo"], g[f"{m}_hi"]))
                          for m in ("raw", "cqr", "aci")})
    cov_m.index.name = "month"
    cov_m.round(4).to_csv(RES / "coverage90_by_month.csv")

    # ACI step-size sensitivity (gbm_nwp_tso, 90 %)
    full = fc["gbm_nwp_tso"].index
    yf = y.reindex(full).to_numpy(); dayf = ((full.floor("D") - full[0].floor("D")).days).to_numpy()
    Qf = fc["gbm_nwp_tso"].to_numpy()
    sens = []
    for g in (0.0, 0.005, 0.01, 0.02, 0.05):
        lo, hi, _ = rolling_conformal(yf, Qf[:, 0], Qf[:, 10], dayf, 0.10, window_days=WINDOW_DAYS, gamma=g)
        s = pd.DataFrame({"lo": lo, "hi": hi}, index=full).reindex(idx)
        cm = pd.Series((yt >= s.lo) & (yt <= s.hi), index=idx).groupby(month).mean()
        sens.append(dict(gamma=g, cov=coverage(yt, s.lo, s.hi), width_mw=float(np.mean((s.hi - s.lo) * cap)),
                         worst_month_cov=float(cm.min()), monthly_cov_abs_dev=float(np.mean(np.abs(cm - 0.9)))))
    pd.DataFrame(sens).round(5).to_csv(RES / "aci_gamma_sensitivity.csv", index=False)

    summary = dict(test_start=str(idx[0]), test_end=str(idx[-1]), n_hours=int(len(idx)),
                   mean_capacity_mw=float(cap.mean()),
                   metrics=metrics.set_index("model").round(5).to_dict(orient="index"),
                   intervals=iv.round(5).to_dict(orient="records"),
                   dm=dm.round(6).to_dict(orient="records"), runtime_s=round(time.time() - t0, 1))
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(metrics.round(4).to_string(index=False))
    print(iv.round(4).to_string(index=False))
    print(dm.round(5).to_string(index=False))
    print(cov_m.round(3).to_string())
    print(pd.DataFrame(sens).round(4).to_string(index=False))
    print(rel.round(3).to_string())
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
