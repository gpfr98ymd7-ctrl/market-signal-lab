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
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import backtest, models, util


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


def compute(frames, panel, run_date, log=print):
    health = {"version": C.HEALTH_VERSION, "updated": pd.Timestamp(run_date).strftime("%Y-%m-%d"),
              "markets": {}, "stocks": None}
    for m, (f, d, F) in frames.items():
        res = market_walk_forward(d, F)
        cal = calibrate_market(res)
        if cal:
            h = backtest.honest_calibrate(res)
            if len(h) >= 500:
                cal["honest"] = backtest.honest_metrics(h)
                cal["overlay"] = backtest.overlay(h)
            health["markets"][m] = cal
            log(f"Health {C.NAMES[m]}: AUC {cal['auc_up']} | shrink up {cal['lam_up']:.2f}, crash {cal['lam_crash']:.2f} "
                f"| vol {cal['vol_improve_pct']:.0f}% better than naive | range x{cal['k_range']:.2f}")
    _, sdata, SF = panel
    health["stocks"] = stock_health(sdata, SF)
    if health["stocks"]:
        log(f"Health stocks: AUC {health['stocks']['auc']:.3f} | shrink {health['stocks']['lam']:.2f}")
    return health


def needs_refresh(h, today, force=False):
    if force or not h or h.get("version") != C.HEALTH_VERSION or not h.get("markets"):
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
