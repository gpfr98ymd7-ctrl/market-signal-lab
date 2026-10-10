"""This week's forecasts: markets (with the risk scaler) and the stock ranking."""
import numpy as np
import pandas as pd

from . import config as C
from . import models


def forecast_markets(ds, frames, health):
    """One row per market. Probabilities are shrunk, ranges widened and volatility de-biased using the health file."""
    market_health = (health or {}).get("markets") or {}
    rows = []
    for m, (f, d, F) in frames.items():
        mods = models.fit_heads(d, F)
        last = f.sort_values("date").iloc[[-1]]
        pr = models.predict_heads(mods, last[F]).iloc[0]
        base_up, base_crash = float(d["y_up"].mean()), float(d["y_crash"].mean())

        h = market_health.get(m)
        lam_up = h["lam_up"] if h else C.DEFAULT_LAM
        lam_cr = h["lam_crash"] if h else C.DEFAULT_LAM
        bias = h["vol_bias"] if h else 0.0
        k = h["k_range"] if h else 1.0

        p_up = float(models.shrink(pr["p_up"], base_up, lam_up))
        p_crash = float(models.shrink(pr["p_crash"], base_crash, lam_cr))
        pred_vol = float(pr["pred_vol"] * np.exp(bias))
        q50 = float(pr["q50"])
        q = {c: q50 + k * (float(pr[c]) - q50) for c in ("q05", "q25", "q50", "q75", "q95")}

        # Risk scaler: the lowest of three simple, transparent limits.
        regime = last["regime"].iloc[0]
        crash_mult, vol_mult, reg_cap, exposure = (float(x) for x in
                                                   models.exposure_rule(p_crash, base_crash, pred_vol, regime))

        rows.append({
            "key": m, "market": C.NAMES[m], "as_of": last["date"].iloc[0], "px": float(last["px"].iloc[0]),
            "regime": regime, "p_up": p_up, "p_up_raw": float(pr["p_up"]), "base_up": base_up,
            "p_crash": p_crash, "p_crash_raw": float(pr["p_crash"]), "base_crash": base_crash,
            "pred_vol": pred_vol, "pred_vol_raw": float(pr["pred_vol"]), "vol_now": float(last["vol_21"].iloc[0]),
            **q, "crash_mult": crash_mult, "vol_mult": vol_mult, "regime_cap": reg_cap,
            "exposure": exposure,
            "lam_up": lam_up, "lam_crash": lam_cr, "k_range": k,
        })
    return pd.DataFrame(rows)


def _flag(r):
    if r["p_rank"] >= 0.67 and r["ret_20_rk"] <= 0.33:
        return "rebound candidate"
    if r["p_rank"] >= 0.67 and r["ret_20_rk"] >= 0.67:
        return "momentum + model agree"
    if r["p_rank"] <= 0.33:
        return "model cautious"
    return ""


def forecast_stocks(ds, panel, mdash, health):
    """Rank stocks by the model's odds of beating the universe average, then size an illustrative portfolio."""
    sfull, sdata, SF = panel
    model = models.fit_stock_model(sdata[SF], sdata["up"])

    latest = sfull.sort_values("date").groupby("ticker").tail(1)
    latest = latest[latest["date"] >= latest["date"].max() - pd.Timedelta(days=7)].copy()
    latest["p_raw"] = model.predict_proba(latest[SF])[:, 1]
    base = float(sdata["up"].mean())
    sh = (health or {}).get("stocks")
    lam = sh["lam"] if sh else C.DEFAULT_LAM
    latest["p"] = models.shrink(latest["p_raw"].values, base, lam)       # displayed odds; ranking uses p_raw
    latest["p_rank"] = latest["p_raw"].rank(pct=True)
    latest["vol_ann"] = latest["vol_20"] * np.sqrt(252)
    latest["flag"] = latest.apply(_flag, axis=1)

    latest["region"] = [C.region_of(t) for t in latest["ticker"]]
    mult = dict(zip(mdash["key"], mdash["exposure"]))
    latest["mkt_mult"] = latest["region"].map(mult).fillna(1.0)

    top = latest.nlargest(C.TOP_K, "p_raw")
    w = 1 / top["vol_ann"].clip(lower=0.05)
    latest["base_w"] = 0.0
    latest.loc[top.index, "base_w"] = (w / w.sum()).values
    latest["final_w"] = latest["base_w"] * latest["mkt_mult"]            # the rest is cash
    latest["px"] = [float(ds.close[t].dropna().loc[d]) for t, d in zip(latest["ticker"], latest["date"])]

    ranking = latest.sort_values("p_raw", ascending=False).reset_index(drop=True)
    return ranking, {"base": base, "lam": lam}
