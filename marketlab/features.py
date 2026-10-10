"""Feature and target construction.

Rule for everything here: a feature on date t may use data up to and including t only. Targets look
forward and are used ONLY for training and scoring. tests/test_core.py checks this by truncating the
data and confirming that earlier features do not change.
"""
import numpy as np
import pandas as pd

from . import config as C

MARKET_NON_FEATURES = {"date", "px", "regime", "fwd_ret", "fwd_simple", "fwd_vol", "fmin_ret", "y_up", "y_crash"}
STOCK_TARGETS = ("date", "ticker", "fwd_ret", "excess", "up")


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def own_features(px):
    lp = np.log(px)
    r = lp.diff()
    df = pd.DataFrame(index=px.index)
    for n in (1, 5, 10, 21, 63, 126):
        df[f"ret_{n}"] = lp.diff(n)
    for n in (10, 21, 63):
        df[f"vol_{n}"] = r.rolling(n).std() * np.sqrt(252)
    df["vol_ratio"] = df["vol_10"] / df["vol_63"]
    df["downside_vol_21"] = r.clip(upper=0).rolling(21).std() * np.sqrt(252)
    for n in (50, 200):
        df[f"ma_ratio_{n}"] = px / px.rolling(n).mean() - 1
    df["dd_252"] = px / px.rolling(252).max() - 1
    df["rsi_14"] = rsi(px)
    macd = px.ewm(span=12).mean() - px.ewm(span=26).mean()
    df["macd_hist"] = (macd - macd.ewm(span=9).mean()) / px
    df["skew_63"] = r.rolling(63).skew()
    return df


def macro_features(macro):
    mac = pd.DataFrame(index=macro.index)
    vix = macro["vix"]
    mac["vix"] = vix
    mac["vix_chg_5"] = vix.pct_change(5)
    mac["vix_z"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    mac["curve_10y_3m"] = macro["us10y"] - macro["us3m"]
    mac["us10y_chg_21"] = macro["us10y"].diff(21)
    mac["dxy_ret_21"] = macro["dxy"].pct_change(21)
    mac["gold_ret_21"] = macro["gold"].pct_change(21)
    mac["oil_ret_21"] = macro["oil"].pct_change(21)
    mac["copper_gold_chg_21"] = (macro["copper"] / macro["gold"]).pct_change(21)
    credit = macro["hyg"] / macro["lqd"]              # falls when credit stress rises
    mac["credit_chg_21"] = credit.pct_change(21)
    mac["credit_rel_252"] = credit / credit.rolling(252).mean() - 1
    return mac


def market_frame(name, ds, mac):
    """Features and targets for one market index. Returns (full, training_rows, FEATURES).

    `full` keeps the most recent rows (no outcome yet) so they can be forecast; `training_rows`
    holds only rows whose outcome is already known.
    """
    px = ds.idx[name].dropna()
    own = own_features(px)

    oth = pd.DataFrame(index=ds.idx_ff.index)
    for m in ds.idx_ff.columns:
        if m == name:
            continue
        lp = np.log(ds.idx_ff[m])
        oth[f"{m}_ret_5"] = lp.diff(5)
        oth[f"{m}_ret_21"] = lp.diff(21)

    ctx = pd.concat([oth, mac], axis=1)
    # Lag the context so a market that closes later in the day cannot leak into this one.
    ctx = ctx.reindex(px.index.union(ctx.index)).ffill().shift(C.LAG).reindex(px.index)
    f = own.join(ctx)

    # Regime (past data only): trend versus the 200-day average, stress versus own 2-year median volatility.
    up = own["ma_ratio_200"] > 0
    vol_med = own["vol_21"].rolling(504, min_periods=252).median()
    calm = own["vol_21"] <= vol_med
    lab = np.where(up & calm, "Uptrend / calm",
          np.where(up, "Uptrend / stressed",
          np.where(calm, "Downtrend / calm", "Downtrend / stressed")))
    f["regime"] = np.where(vol_med.notna(), lab, "n/a")

    # Targets: the future.
    lp = np.log(px)
    f["fwd_ret"] = lp.shift(-C.H) - lp
    f["fwd_simple"] = px.shift(-C.H) / px - 1
    f["fwd_vol"] = lp.diff().rolling(C.H).std().shift(-C.H) * np.sqrt(252)
    fmin = px.iloc[::-1].rolling(C.H).min().iloc[::-1].shift(-1)      # lowest close over the next H days
    f["fmin_ret"] = fmin / px - 1
    f["y_up"] = np.where(f["fwd_ret"].notna(), (f["fwd_ret"] > 0).astype(float), np.nan)
    f["y_crash"] = np.where(f["fmin_ret"].notna(), (f["fmin_ret"] < C.CRASH).astype(float), np.nan)
    f["px"] = px

    f = f.replace([np.inf, -np.inf], np.nan)
    f.index.name = "date"
    f = f.reset_index()
    FEATURES = [c for c in f.columns if c not in MARKET_NON_FEATURES]
    f = f.dropna(subset=FEATURES)
    data = f.dropna(subset=["fwd_ret", "fwd_vol", "fmin_ret"]).copy()
    data[["y_up", "y_crash"]] = data[["y_up", "y_crash"]].astype(int)
    return f, data, FEATURES


def stock_features(c, v):
    r = c.pct_change()
    df = pd.DataFrame(index=c.index)
    for n in (1, 5, 10, 20, 60, 120):
        df[f"ret_{n}"] = c.pct_change(n)
    df["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    df["vol_20"] = r.rolling(20).std()
    df["vol_60"] = r.rolling(60).std()
    df["vol_ratio"] = df["vol_20"] / df["vol_60"]
    for n in (20, 50, 200):
        df[f"ma_ratio_{n}"] = c / c.rolling(n).mean() - 1
    df["rsi_14"] = rsi(c)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_pos"] = (c - m) / (2 * s)
    df["dd_252"] = c / c.rolling(252).max() - 1
    df["volume_ratio"] = v / v.rolling(20).mean()
    # Target: bought ENTRY_DELAY trading days after the signal date, held SH trading days.
    d = C.ENTRY_DELAY
    df["fwd_ret"] = c.shift(-(d + C.SH)) / c.shift(-d) - 1
    return df


def stock_context(ds):
    """S&P 500 and macro context for the stock model (same-date values)."""
    mkt = pd.DataFrame({"spx": ds.idx_ff["spx"], "vix": ds.macro["vix"], "us10y": ds.macro["us10y"],
                        "dxy": ds.macro["dxy"], "gold": ds.macro["gold"], "oil": ds.macro["oil"]}).ffill()
    mf = pd.DataFrame(index=mkt.index)
    mf["mkt_ret_5"] = mkt["spx"].pct_change(5)
    mf["mkt_ret_20"] = mkt["spx"].pct_change(20)
    mf["mkt_ret_60"] = mkt["spx"].pct_change(60)
    mf["mkt_vol_20"] = mkt["spx"].pct_change().rolling(20).std()
    mf["vix"] = mkt["vix"]
    mf["vix_chg_5"] = mkt["vix"].pct_change(5)
    mf["us10y_chg_20"] = mkt["us10y"].diff(20)
    mf["dxy_ret_20"] = mkt["dxy"].pct_change(20)
    mf["gold_ret_20"] = mkt["gold"].pct_change(20)
    mf["oil_ret_20"] = mkt["oil"].pct_change(20)
    return mf.ffill()


def stock_panel(ds):
    """Stack all stocks into one panel. Returns (full, training_rows, FEATURES)."""
    mf = stock_context(ds)
    rows = []
    for t in ds.stocks:
        c = ds.close[t].dropna()
        v = ds.volume[t].reindex(c.index)
        df = stock_features(c, v).join(mf.reindex(c.index))
        df["rs_20"] = df["ret_20"] - df["mkt_ret_20"]
        df["rs_60"] = df["ret_60"] - df["mkt_ret_60"]
        df.index.name = "date"
        df["ticker"] = t
        rows.append(df)
    sfull = pd.concat(rows).replace([np.inf, -np.inf], np.nan).reset_index()
    # Where does each stock sit versus the others on the same day?
    for col in ["ret_20", "ret_60", "vol_20", "rsi_14", "mom_12_1"]:
        sfull[f"{col}_rk"] = sfull.groupby("date")[col].rank(pct=True)
    sfull["excess"] = sfull["fwd_ret"] - sfull.groupby("date")["fwd_ret"].transform("mean")
    sfull["up"] = np.where(sfull["excess"].notna(), (sfull["excess"] > 0).astype(float), np.nan)
    FEATURES = [c for c in sfull.columns if c not in STOCK_TARGETS]
    sfull = sfull.dropna(subset=FEATURES)
    sdata = sfull.dropna(subset=["fwd_ret"]).copy()
    sdata["up"] = sdata["up"].astype(int)
    return sfull, sdata, FEATURES
