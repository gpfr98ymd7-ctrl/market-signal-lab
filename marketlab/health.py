"""Walk-forward health check: how much should the model's output be trusted?

Forecasts are produced out-of-sample, year by year (train on the past only, forecast the next year).
Those honest forecasts are then used to:
  1. shrink probabilities toward their base rate by exactly as much as the model has failed to earn,
  2. widen return ranges until they actually contain the outcome as often as they claim,
  3. correct any systematic bias in the volatility forecast, and
  4. report plainly how good each forecast head has been historically.
On data with no real signal the shrinkage factor goes to about zero, so the dashboard quotes base rates
instead of confident-looking noise.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import backtest, macro, models, opts, util

FORMAT = 2        # bump to force a recompute of the saved health file (it is independent of config.py on purpose)


def _auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _lam(p, y, base):
    """Least-squares shrinkage factor in [0, 1]: the share of the model's deviation from the base rate that is real."""
    p, y, base = (np.asarray(a, dtype=float) for a in (p, y, base))
    d = p - base
    den = float(np.sum(d * d))
    if den <= 0:
        return 0.0
    return float(np.clip(np.sum(d * (y - base)) / den, 0.0, 1.0))


def _brier_skill(y, p, base):
    y, p, base = (np.asarray(a, dtype=float) for a in (y, p, base))
    den = float(np.mean((y - base) ** 2))
    return float(1 - np.mean((y - p) ** 2) / den) if den > 0 else None


def market_walk_forward(data, FEATURES):
    years = sorted(data["date"].dt.year.unique())
    out = []
    for y in years[C.MIN_TRAIN_YEARS_MARKET:]:
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=2 * C.H)   # gap: targets look H days ahead
        train = data[data["date"] < cutoff]
        test = data[data["date"].dt.year == y]
        if len(test) == 0 or len(train) < 500:
            continue
        pr = models.predict_heads(models.fit_heads(train, FEATURES), test[FEATURES])
        part = test[["date", "y_up", "y_crash", "fwd_ret", "fwd_simple", "fwd_vol", "vol_21", "regime"]].join(pr)
        part["base_up"] = train["y_up"].mean()
        part["base_crash"] = train["y_crash"].mean()
        out.append(part)
    return pd.concat(out).reset_index(drop=True) if out else pd.DataFrame()


def factors(res):
    """The calibration numbers learned from a set of out-of-sample forecasts:
    how much of each probability to keep, the volatility bias, and the range-widening factor."""
    lam_up = _lam(res["p_up"].values, res["y_up"].values, res["base_up"].values)
    lam_cr = _lam(res["p_crash"].values, res["y_crash"].values, res["base_crash"].values)
    act = np.log(np.clip(res["fwd_vol"].values, 1e-4, None))
    bias = float(np.mean(act - np.log(res["pred_vol"].values)))
    fwd = res["fwd_ret"].values
    q50, q05, q95 = res["q50"].values, res["q05"].values, res["q95"].values
    ks = np.arange(0.8, 3.0001, 0.05)
    cov = [float(np.mean((fwd >= q50 + k * (q05 - q50)) & (fwd <= q50 + k * (q95 - q50)))) for k in ks]
    k_best = float(ks[int(np.argmin([abs(c - 0.90) for c in cov]))])
    return {"lam_up": lam_up, "lam_crash": lam_cr, "vol_bias": bias, "k_range": k_best}


def calibrate_market(res):
    """Turn out-of-sample forecasts into calibration factors and a report card."""
    if len(res) < 250:
        return None
    y_up, p_up, b_up = res["y_up"].values, res["p_up"].values, res["base_up"].values
    y_cr, p_cr, b_cr = res["y_crash"].values, res["p_crash"].values, res["base_crash"].values
    fac = factors(res)
    lam_up, lam_cr, bias, k_best = fac["lam_up"], fac["lam_crash"], fac["vol_bias"], fac["k_range"]

    act = np.log(np.clip(res["fwd_vol"].values, 1e-4, None))
    pred = np.log(res["pred_vol"].values)
    naive = np.log(np.clip(res["vol_21"].values, 1e-4, None))
    rmse_m = float(np.sqrt(np.mean((pred + bias - act) ** 2)))
    rmse_n = float(np.sqrt(np.mean((naive - act) ** 2)))

    fwd = res["fwd_ret"].values
    q = {c: res[c].values for c in ("q05", "q25", "q50", "q75", "q95")}

    def cover(k, lo, hi):
        return float(np.mean((fwd >= q["q50"] + k * (q[lo] - q["q50"])) & (fwd <= q["q50"] + k * (q[hi] - q["q50"]))))

    return {
        "n_days": int(len(res)),
        "first_test": res["date"].min().strftime("%Y-%m-%d"),
        "last_test": res["date"].max().strftime("%Y-%m-%d"),
        "auc_up": _auc(y_up, p_up),
        "lam_up": lam_up,
        "brier_skill_up": _brier_skill(y_up, b_up + lam_up * (p_up - b_up), b_up),
        "auc_crash": _auc(y_cr, p_cr),
        "lam_crash": lam_cr,
        "brier_skill_crash": _brier_skill(y_cr, b_cr + lam_cr * (p_cr - b_cr), b_cr),
        "vol_bias": bias,
        "vol_improve_pct": float(100 * (1 - rmse_m / rmse_n)) if rmse_n > 0 else None,
        "k_range": k_best,
        "cov90_raw": cover(1.0, "q05", "q95"),
        "cov90_cal": cover(k_best, "q05", "q95"),
        "cov50_raw": cover(1.0, "q25", "q75"),
        "cov50_cal": cover(k_best, "q25", "q75"),
    }


def stock_health(sdata, S_FEATURES):
    years = sorted(sdata["date"].dt.year.unique())
    res = []
    for y in years[C.MIN_TRAIN_YEARS_STOCK:]:
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=14)
        train = sdata[sdata["date"] < cutoff]
        test = sdata[sdata["date"].dt.year == y]
        if len(test) == 0 or len(train) < 2000:
            continue
        m = models.fit_stock_model(train[S_FEATURES], train["up"])
        part = test[["date", "ticker", "up", "excess", "fwd_ret", "vol_20"]].copy()
        part["p"] = m.predict_proba(test[S_FEATURES])[:, 1]
        part["base"] = train["up"].mean()
        res.append(part)
    if not res:
        return None
    r = pd.concat(res).reset_index(drop=True)

    by_year = {}
    for y, g in r.groupby(r["date"].dt.year):
        a = _auc(g["up"], g["p"])
        if a is not None:
            by_year[str(int(y))] = a

    r["rk"] = r.groupby("date")["p"].rank(pct=True)
    r["n"] = r.groupby("date")["p"].transform("size")
    r = r[r["n"] >= 9]
    top = r[r["rk"] >= 0.67].groupby("date")["excess"].mean()
    bot = r[r["rk"] <= 0.33].groupby("date")["excess"].mean()
    spread = (top - bot).dropna().sort_index()
    sp = spread.iloc[::C.SH]                      # roughly non-overlapping holding periods
    t = None
    if len(sp) > 8 and sp.std() > 0:
        t = float(sp.mean() / (sp.std() / np.sqrt(len(sp))))
    return {
        "n_rows": int(len(r)),
        "first_test": r["date"].min().strftime("%Y-%m-%d"),
        "last_test": r["date"].max().strftime("%Y-%m-%d"),
        "auc": _auc(r["up"], r["p"]),
        "auc_by_year": by_year,
        "years_above_half": int(sum(1 for a in by_year.values() if a > 0.5)),
        "n_years": int(len(by_year)),
        "spread_pct": float(100 * spread.mean()) if len(spread) else None,
        "spread_t": t,
        "lam": _lam(r["p"].values, r["up"].values, r["base"].values),
        "base": float(r["base"].mean()),
        "backtest": backtest.stock_strategy(r),
    }


# ---------------------------------------------------------------- do the extra global-driver inputs help?
def decide_stocks(base, ext):
    """Use the stock ranker with the extra inputs only if it did clearly better in most test years, and overall.
    `base` and `ext` are stock_health results for the two feature sets. Returns a small, plain record."""
    need_gain, need_share = opts.get("MACRO_MIN_AUC_GAIN"), opts.get("MACRO_MIN_YEARS_WON")
    rec = {"use": False, "auc_base": base.get("auc"), "auc_macro": ext.get("auc"),
           "spread_base": base.get("spread_pct"), "spread_macro": ext.get("spread_pct"),
           "need_gain": need_gain, "need_share": need_share}
    yb, ye = base.get("auc_by_year") or {}, ext.get("auc_by_year") or {}
    years = sorted(set(yb) & set(ye))
    rec["years"] = len(years)
    if len(years) < 5:
        rec["reason"] = "Too few test years to tell."
        return rec
    d = np.array([ye[y] - yb[y] for y in years])
    rec.update({"gain": float(d.mean()), "years_won": int((d > 0).sum())})
    ok_pooled = (ext.get("auc") or 0) > (base.get("auc") or 0)
    if d.mean() >= need_gain and (d > 0).mean() >= need_share and ok_pooled:
        rec["use"] = True
        rec["reason"] = "Clearly better in most test years."
    else:
        rec["reason"] = "No clear gain, so the original inputs are kept."
    return rec


def _gap(a, b, margin):
    """+1 if b beats a by more than `margin`, -1 if it is worse by more than `margin`, else 0 (None counts as a tie)."""
    if a is None or b is None:
        return 0
    d = b - a
    return 1 if d > margin else (-1 if d < -margin else 0)


def decide_market(base, ext):
    """Use a market's forecasts with the extra inputs only if they won on at least two of the three yardsticks
    (direction, crash, volatility) by a clear margin and lost on none."""
    g = {"up": _gap(base.get("auc_up"), ext.get("auc_up"), 0.005),
         "crash": _gap(base.get("auc_crash"), ext.get("auc_crash"), 0.005),
         "vol": _gap(base.get("vol_improve_pct"), ext.get("vol_improve_pct"), 0.5)}
    wins, losses = sum(v > 0 for v in g.values()), sum(v < 0 for v in g.values())
    use = wins >= 2 and losses == 0
    return {"use": use, "wins": wins, "losses": losses,
            "auc_up": [base.get("auc_up"), ext.get("auc_up")], "auc_crash": [base.get("auc_crash"), ext.get("auc_crash")],
            "vol_improve_pct": [base.get("vol_improve_pct"), ext.get("vol_improve_pct")],
            "reason": "Clearly better on at least two of three yardsticks." if use else "No clear gain, so the original inputs are kept."}


def compute(frames, panel, run_date, log=print, budget=None):
    """The full walk-forward health check. When the extra global-driver inputs are present, every model is checked
    twice (with and without them), the better setup is the one calibrated and used, and the comparison is recorded.
    `budget` (seconds) caps the extra work: when it would be exceeded the original inputs are used."""
    budget = opts.get("MACRO_TEST_BUDGET") if budget is None else budget
    health = {"format": FORMAT, "version": C.HEALTH_VERSION, "updated": pd.Timestamp(run_date).strftime("%Y-%m-%d"),
              "markets": {}, "stocks": None, "universe": [], "macro_test": {"stocks": None, "markets": {}, "ran": False,
                                                                           "out_of_time": False}}
    spent = 0.0
    mt = health["macro_test"]

    # ---- stocks first: they matter most to the Growth watch
    _, sdata, SF = panel
    health["universe"] = sorted(sdata["ticker"].unique().tolist())
    SF0 = macro.base_features(SF)
    t = time.time()
    st = stock_health(sdata, SF0)
    base_secs = time.time() - t
    chosen = st
    if st and len(SF0) < len(SF):
        if spent + base_secs <= budget:
            t = time.time()
            ext = stock_health(sdata, SF)
            spent += time.time() - t
            mt["ran"] = True
            if ext:
                rec = decide_stocks(st, ext)
                mt["stocks"] = rec
                if rec["use"]:
                    chosen = ext
                log(f"Global drivers, stock ranker: AUC {st['auc']:.3f} -> {ext['auc']:.3f}; "
                    f"{'USED' if rec['use'] else 'not used'} ({rec['reason']})")
        else:
            mt["out_of_time"] = True
            log("Global drivers: not enough time to test them on the stock ranker this month.")
    if chosen:
        chosen["features"] = "macro" if (chosen is not st) else "base"
        health["stocks"] = chosen
        log(f"Health stocks: AUC {chosen['auc']:.3f} | shrink {chosen['lam']:.2f} | inputs: {chosen['features']}")

    # ---- then each market
    for m, (f, d, F) in frames.items():
        F0 = macro.base_features(F)
        t = time.time()
        res = market_walk_forward(d, F0)
        base_secs = time.time() - t
        cal = calibrate_market(res)
        feats, res_used = "base", res
        if cal and len(F0) < len(F):
            if spent + base_secs <= budget:
                t = time.time()
                res_e = market_walk_forward(d, F)
                spent += time.time() - t
                mt["ran"] = True
                cal_e = calibrate_market(res_e)
                if cal_e:
                    rec = decide_market(cal, cal_e)
                    mt["markets"][m] = rec
                    if rec["use"]:
                        cal, res_used, feats = cal_e, res_e, "macro"
                    log(f"Global drivers, {C.NAMES[m]}: {'USED' if rec['use'] else 'not used'} "
                        f"(won {rec['wins']}, lost {rec['losses']} of 3 yardsticks)")
            else:
                mt["out_of_time"] = True
        if cal:
            h = backtest.honest_calibrate(res_used)
            if len(h) >= 500:
                cal["honest"] = backtest.honest_metrics(h)
                cal["overlay"] = backtest.overlay(h)
            cal["features"] = feats
            health["markets"][m] = cal
            log(f"Health {C.NAMES[m]}: AUC {cal['auc_up']} | shrink up {cal['lam_up']:.2f}, crash {cal['lam_crash']:.2f} "
                f"| vol {cal['vol_improve_pct']:.0f}% better than naive | range x{cal['k_range']:.2f} | inputs: {feats}")
    mt["seconds"] = round(spent)
    return health


def needs_refresh(h, today, force=False):
    if force or not h or h.get("version") != C.HEALTH_VERSION or h.get("format") != FORMAT or not h.get("markets"):
        return True
    try:
        age = (pd.Timestamp(today) - pd.Timestamp(h["updated"])).days
    except Exception:
        return True
    return age >= C.HEALTH_REFRESH_DAYS


def load(path):
    return util.read_json(path)


def save(path, health):
    util.write_json_atomic(path, health)
