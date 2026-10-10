"""Model fitting and prediction. Small, heavily regularised trees: index data is short and noisy."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C

QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)

MARKET_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=60,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                     random_state=7, verbose=-1)
STOCK_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                    random_state=7, verbose=-1)


class _ConstClassifier:
    """Used when a training window contains only one outcome (for example no crashes at all)."""

    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def fit_classifier(X, y, params):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return _ConstClassifier(float(y.mean()) if len(y) else 0.5)
    return lgb.LGBMClassifier(**params).fit(X, y)


def fit_stock_model(X, y):
    return fit_classifier(X, y, STOCK_PARAMS)


def fit_heads(d, FEATURES):
    """Four forecast heads for one market: direction, crash risk, volatility, return quantiles."""
    X = d[FEATURES]
    models = {
        "up": fit_classifier(X, d["y_up"], MARKET_PARAMS),
        "crash": fit_classifier(X, d["y_crash"], MARKET_PARAMS),
        "vol": lgb.LGBMRegressor(**MARKET_PARAMS).fit(X, np.log(np.clip(d["fwd_vol"], 1e-4, None))),
    }
    for a in QUANTS:
        models[a] = lgb.LGBMRegressor(objective="quantile", alpha=a, **MARKET_PARAMS).fit(X, d["fwd_ret"])
    return models


def predict_heads(models, X):
    out = pd.DataFrame(index=X.index)
    out["p_up"] = models["up"].predict_proba(X)[:, 1]
    out["p_crash"] = models["crash"].predict_proba(X)[:, 1]
    out["pred_vol"] = np.exp(models["vol"].predict(X))
    q = np.column_stack([models[a].predict(X) for a in QUANTS])
    q.sort(axis=1)                                      # prevents crossing quantiles
    for i, a in enumerate(QUANTS):
        out[f"q{int(a * 100):02d}"] = q[:, i]
    return out


def shrink(p, base, lam):
    """Pull a probability toward the long-run base rate. lam=0 returns the base rate, lam=1 the raw model."""
    return np.clip(base + lam * (np.asarray(p, dtype=float) - base), 0.0, 1.0)


def exposure_rule(p_crash, base_crash, pred_vol, regime):
    """The market-risk scaler: the lowest of three simple limits, between 0.25 and 1.0.

    One definition used by the live forecast AND the historical backtest, so they cannot drift apart.
    Accepts scalars or arrays. Returns (crash_mult, vol_mult, regime_cap, exposure).
    """
    pc = np.asarray(p_crash, dtype=float)
    bc = np.asarray(base_crash, dtype=float)
    pv = np.asarray(pred_vol, dtype=float)
    crash_mult = np.clip(1.5 - 0.5 * pc / np.maximum(bc, 1e-3), 0.25, 1.0)
    vol_mult = np.clip(C.VOL_TARGET / np.maximum(pv, 1e-6), 0.25, 1.0)
    cap = np.where(np.asarray(regime) == "Downtrend / stressed", 0.5, 1.0)
    return crash_mult, vol_mult, cap, np.minimum(np.minimum(crash_mult, vol_mult), cap)
