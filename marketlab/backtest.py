"""Historical evidence for the risk scaler and the stock ranker.

Everything here is built from the walk-forward forecasts in health.py: each forecast was made by a model that
had only seen the past. On top of that, the calibration applied to each test year is itself estimated from
EARLIER years only ("honest" calibration), so none of the numbers below are flattered by hindsight.

Known simplifications, shown to the reader on the Trust tab:
  * the index trades are assumed to happen at the close the signal was computed on,
  * the stock universe is today's large companies (survivorship bias),
  * cash earns nothing, and there are no taxes, spreads or market impact beyond the flat cost.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import models

Q = ("q05", "q25", "q50", "q75", "q95")


def perf(r, ppy):
    """Annualised return, volatility, Sharpe and worst drawdown of a series of period returns."""
    r = pd.Series(np.asarray(r, dtype=float)).dropna()
    if len(r) < 2:
        return {"ann_return": None, "ann_vol": None, "sharpe": None, "max_drawdown": None}
    eq = (1 + r).cumprod()
    sd = float(r.std())
    return {
        "ann_return": float(eq.iloc[-1] ** (ppy / len(r)) - 1),
        "ann_vol": sd * np.sqrt(ppy),
        "sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else None,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }


def honest_calibrate(res, min_years=2):
    """Apply to each test year the calibration learned from earlier years only.

    Returns the rows of `res` for years that had at least `min_years` of earlier forecasts to learn from,
    with extra columns p_up_c, p_crash_c, pred_vol_c and q05_c .. q95_c.
    """
    from . import health                      # imported here: health imports this module when computing
    if res is None or len(res) == 0:
        return pd.DataFrame()
    res = res.sort_values("date").reset_index(drop=True)
    out = []
    for y in sorted(res["date"].dt.year.unique())[min_years:]:
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=2 * C.H)   # same gap as the walk-forward
        past = res[res["date"] < cutoff]
        cur = res[res["date"].dt.year == y].copy()
        if len(past) < 500 or cur.empty:
            continue
        f = health.factors(past)
        cur["p_up_c"] = models.shrink(cur["p_up"], cur["base_up"], f["lam_up"])
        cur["p_crash_c"] = models.shrink(cur["p_crash"], cur["base_crash"], f["lam_crash"])
        cur["pred_vol_c"] = cur["pred_vol"] * np.exp(f["vol_bias"])
        for c in Q:
            cur[c + "_c"] = cur["q50"] + f["k_range"] * (cur[c] - cur["q50"])
        out.append(cur)
    return pd.concat(out).reset_index(drop=True) if out else pd.DataFrame()


def honest_metrics(h):
    """Skill of the calibrated forecasts on the days where calibration used only earlier years."""
    from . import health
    yu, yc = h["y_up"].to_numpy(float), h["y_crash"].to_numpy(float)
    act = np.log(np.clip(h["fwd_vol"].to_numpy(float), 1e-4, None))

    def rmse(v):
        return float(np.sqrt(np.mean((np.log(np.clip(np.asarray(v, dtype=float), 1e-4, None)) - act) ** 2)))

    def inside(lo, hi):
        return float(((h["fwd_ret"] >= h[lo]) & (h["fwd_ret"] <= h[hi])).mean())

    rm, rn = rmse(h["pred_vol_c"]), rmse(h["vol_21"])
    return {
        "n_days": int(len(h)),
        "first": h["date"].min().strftime("%Y-%m-%d"),
        "last": h["date"].max().strftime("%Y-%m-%d"),
        "brier_skill_up": health._brier_skill(yu, h["p_up_c"], h["base_up"]),
        "brier_skill_crash": health._brier_skill(yc, h["p_crash_c"], h["base_crash"]),
        "vol_improve_pct": float(100 * (1 - rm / rn)) if rn > 0 else None,
        "cov90": inside("q05_c", "q95_c"),
        "cov50": inside("q25_c", "q75_c"),
    }


def overlay(h, cost_bps=None):
    """Buy-and-hold versus the exact live risk scaler, rebalanced every H trading days (no overlapping periods)."""
    cost = (C.COST_BPS_MARKET if cost_bps is None else cost_bps) / 1e4
    d = h.sort_values("date").iloc[:: C.H].reset_index(drop=True)
    if len(d) < 8:
        return None
    ret = d["fwd_simple"].to_numpy(float)
    _, _, _, e = models.exposure_rule(d["p_crash_c"], d["base_crash"], d["pred_vol_c"], d["regime"])
    prev = np.concatenate([[e[0]], e[:-1]])
    scaled = e * ret - np.abs(e - prev) * cost
    ppy = 252 / C.H
    bh, sc = perf(ret, ppy), perf(scaled, ppy)
    bh["avg_exposure"], sc["avg_exposure"] = 1.0, float(np.mean(e))
    return {"buy_hold": bh, "scaler": sc, "periods": int(len(d)), "cost_bps": C.COST_BPS_MARKET,
            "first": d["date"].min().strftime("%Y-%m-%d"), "last": d["date"].max().strftime("%Y-%m-%d")}


def stock_strategy(r):
    """Hold the top-K stocks by model score each week (inverse-volatility weights), net of trading costs.

    `r` needs columns date, ticker, p, fwd_ret, vol_20. fwd_ret already starts ENTRY_DELAY days after the
    signal, so this is the return you could actually have earned.
    """
    if not {"date", "ticker", "p", "fwd_ret", "vol_20"} <= set(r.columns):
        return None
    groups = dict(tuple(r.groupby("date")))
    cost = C.COST_BPS / 1e4
    prev, rets, bench, turns = {}, [], [], []
    for d in sorted(groups)[:: C.SH]:
        day = groups[d].dropna(subset=["fwd_ret", "vol_20"])
        if len(day) < C.TOP_K + 1:
            continue
        picks = day.nlargest(C.TOP_K, "p")
        w = 1 / (picks["vol_20"] * np.sqrt(252)).clip(lower=0.05)          # same weighting as the live portfolio
        w = w / w.sum()
        new = dict(zip(picks["ticker"], w))
        turn = sum(abs(new.get(t, 0.0) - prev.get(t, 0.0)) for t in set(new) | set(prev))
        rets.append(float((picks["fwd_ret"] * w).sum() - turn * cost))
        bench.append(float(day["fwd_ret"].mean()))
        turns.append(float(turn))
        prev = new
    if len(rets) < 8:
        return None
    ppy = 252 / C.SH
    return {"strategy": perf(rets, ppy), "universe": perf(bench, ppy), "periods": int(len(rets)),
            "avg_turnover": float(np.mean(turns)), "cost_bps": C.COST_BPS, "top_k": C.TOP_K}
