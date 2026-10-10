"""The forecast log, and scoring old forecasts against what actually happened.

The log stores only forecasts as they were shown, never outcomes: outcomes are recomputed from prices each
time, so a data correction can never leave a stale score behind.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C

M_COLS = ["run_date", "key", "as_of", "px", "p_up", "p_crash", "pred_vol", "vol_now",
          "q05", "q25", "q50", "q75", "q95", "regime", "exposure", "p_up_raw", "p_crash_raw", "model_version"]
S_COLS = ["run_date", "ticker", "as_of", "px", "p", "p_rank", "flag", "region", "mkt_mult",
          "base_w", "final_w", "p_raw", "model_version"]


def append_log(path, new, keys):
    """Append new rows. Existing rows are never overwritten, and re-running the same data date adds nothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["run_date", "as_of"])
        new = pd.concat([old, new], ignore_index=True)       # old rows first so they are the ones kept
    new = new.drop_duplicates(keys, keep="first")
    new.to_csv(path, index=False)
    return new


def log_frames(mdash, ranking, run_date):
    mnew = mdash.assign(run_date=run_date, model_version=C.VERSION)[M_COLS]
    snew = ranking.rename(columns={"date": "as_of"}).assign(run_date=run_date, model_version=C.VERSION)[S_COLS]
    return mnew, snew


def score_markets(log, idx):
    out = []
    for _, r in log.iterrows():
        d = dict(actual_log_ret=np.nan, actual_vol=np.nan, actual_min=np.nan)
        if r["key"] in idx.columns:
            s = idx[r["key"]].dropna()
            pos = s.index.get_indexer([r["as_of"]])[0]
            if pos >= 0 and pos + C.H < len(s):
                seg = s.iloc[pos: pos + C.H + 1]
                d["actual_log_ret"] = float(np.log(seg.iloc[-1] / seg.iloc[0]))
                d["actual_vol"] = float(np.log(seg).diff().dropna().std() * np.sqrt(252))
                d["actual_min"] = float(seg.iloc[1:].min() / seg.iloc[0] - 1)
        out.append(d)
    return pd.concat([log.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def score_stocks(log, close):
    fwd = []
    d = C.ENTRY_DELAY
    for _, r in log.iterrows():
        f = np.nan
        if r["ticker"] in close.columns:
            s = close[r["ticker"]].dropna()
            pos = s.index.get_indexer([r["as_of"]])[0]
            if pos >= 0 and pos + d + C.SH < len(s):
                f = s.iloc[pos + d + C.SH] / s.iloc[pos + d] - 1
        fwd.append(f)
    df = log.reset_index(drop=True).assign(fwd_ret=fwd)
    df["excess"] = df["fwd_ret"] - df.groupby("run_date")["fwd_ret"].transform("mean")
    return df


def _bskill(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    den = float(np.mean((y - y.mean()) ** 2))
    return float(1 - np.mean((y - p) ** 2) / den) if den > 0 else None


def _turnover(w):
    return (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)


def build_scorecard(mlog, slog, ds):
    card = {"runs": int(mlog["run_date"].nunique()), "cost_bps": C.COST_BPS}

    # ---- markets
    ms = score_markets(mlog, ds.idx)
    sc = ms.dropna(subset=["actual_log_ret"])
    runs_scored = int(sc["run_date"].nunique()) if len(sc) else 0
    overlap = int(np.ceil(C.H / 5))                      # weekly runs, H-day horizon: forecasts overlap
    mk = {"scored": int(len(sc)), "logged": int(len(ms)), "horizon_days": C.H, "runs_scored": runs_scored,
          "effective_runs": round(runs_scored / overlap, 1)}
    if len(sc):
        up = (sc["actual_log_ret"] > 0).astype(int).values
        crash = (sc["actual_min"] < C.CRASH).astype(int).values
        av = np.log(np.clip(sc["actual_vol"].values, 1e-4, None))
        mk.update({
            "direction_accuracy": float(((sc["p_up"] > 0.5).astype(int).values == up).mean()),
            "always_up_accuracy": float(up.mean()),
            "direction_brier_skill": _bskill(up, sc["p_up"].values),
            "vol_error_model": float(np.sqrt(np.mean((np.log(sc["pred_vol"].values) - av) ** 2))),
            "vol_error_naive": float(np.sqrt(np.mean((np.log(np.clip(sc["vol_now"].values, 1e-4, None)) - av) ** 2))),
            "crash_brier_skill": _bskill(crash, sc["p_crash"].values),
            "crashes_seen": int(crash.sum()),
            "coverage_90": float(((sc["actual_log_ret"] >= sc["q05"]) & (sc["actual_log_ret"] <= sc["q95"])).mean()),
            "coverage_50": float(((sc["actual_log_ret"] >= sc["q25"]) & (sc["actual_log_ret"] <= sc["q75"])).mean()),
        })
    card["markets"] = mk

    # ---- stocks
    ss = score_stocks(slog, ds.close)
    s2 = ss.dropna(subset=["excess"]).copy()
    st = {"scored": int(len(s2)), "logged": int(len(ss)), "horizon_days": C.SH,
          "entry_delay": C.ENTRY_DELAY, "runs_scored": int(s2["run_date"].nunique()) if len(s2) else 0}
    comb = {"weeks": 0}
    if len(s2):
        s2["bucket"] = np.where(s2["p_rank"] >= 0.67, "top", np.where(s2["p_rank"] <= 0.33, "bottom", "middle"))
        st["avg_excess_pct"] = {k: float(100 * v) for k, v in s2.groupby("bucket")["excess"].mean().items()}

        spreads = []
        for _, g in s2.groupby("run_date"):
            t, b = g[g["bucket"] == "top"]["excess"], g[g["bucket"] == "bottom"]["excess"]
            if len(t) and len(b):
                spreads.append(float(t.mean() - b.mean()))
        if spreads:
            sp = np.array(spreads)
            st["spread_runs"] = int(len(sp))
            st["spread_mean_pct"] = float(100 * sp.mean())
            st["spread_hit_rate"] = float((sp > 0).mean())
            if len(sp) >= 8 and sp.std(ddof=1) > 0:
                st["spread_t"] = float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))))
        if len(s2) >= 100 and s2["excess"].gt(0).nunique() == 2:
            st["auc"] = float(roc_auc_score(s2["excess"] > 0, s2["p_raw"].fillna(s2["p"])))

        # Illustrative portfolios, net of trading costs charged on turnover between consecutive runs.
        w_s = slog.pivot_table(index="run_date", columns="ticker", values="final_w", aggfunc="sum", fill_value=0.0).sort_index()
        w_b = slog.pivot_table(index="run_date", columns="ticker", values="base_w", aggfunc="sum", fill_value=0.0).sort_index()
        t_s, t_b = _turnover(w_s), _turnover(w_b)
        cost = C.COST_BPS / 1e4
        rows = []
        for run, g in s2.groupby("run_date"):
            rows.append({
                "run_date": run,
                "scaled": float((g["final_w"] * g["fwd_ret"]).sum() - float(t_s.get(run, 0.0)) * cost),
                "unscaled": float((g["base_w"] * g["fwd_ret"]).sum() - float(t_b.get(run, 0.0)) * cost),
                "universe": float(g["fwd_ret"].mean()),
            })
        by_run = pd.DataFrame(rows).set_index("run_date").sort_index()
        curve = (1 + by_run).cumprod().tail(52)
        comb = {
            "weeks": int(len(by_run)),
            "scaled": float(curve["scaled"].iloc[-1]),
            "unscaled": float(curve["unscaled"].iloc[-1]),
            "universe": float(curve["universe"].iloc[-1]),
            "curve": [{"date": d.strftime("%Y-%m-%d"), "scaled": float(r["scaled"]),
                       "unscaled": float(r["unscaled"]), "universe": float(r["universe"])}
                      for d, r in curve.iterrows()],
        }
    card["stocks"] = st
    card["combined"] = comb
    return card
