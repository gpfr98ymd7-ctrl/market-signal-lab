"""Global drivers: the market-wide forces that move stocks together (interest rates, the dollar, oil and other
commodities, credit conditions, market fear, how emerging markets, chips, small companies and the broad market are
doing against the S&P 500).

Three jobs, kept apart on purpose:

  1. READINGS (`drivers_view`): plain-English readings of each driver for the dashboard. Context only.
  2. FEATURES (`build_ext`): the same drivers turned into model inputs. Date-level ones (for example the slope of the
     long end of the yield curve) go to the market models and the stock ranker. Stock-level ones measure how sensitive
     each stock has been to a driver (its "beta", over the last 126 trading days) and how far that driver has moved
     lately (the "push"). Every input uses data up to and including its own date and nothing later.
  3. EVIDENCE (health.py): the extra inputs are only used in a live forecast if a walk-forward test (train on the past,
     forecast the next year, repeat) shows they beat the model without them. Otherwise they stay context. "More data"
     is not the same as "more accurate", and on noisy markets it is often not.

All extra input columns start with "mx_" so they can be told apart from the original ones.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PREFIX = "mx_"
WINDOW = 126                 # trading days of history behind each stock's sensitivity (about six months)
MOVE = 20                    # trading days over which a driver's recent move is measured

# The drivers a stock's sensitivity is measured against: key -> (label used in tags, what "up" means)
FACTORS = {
    "rates": "Rates",
    "dollar": "Dollar",
    "oil": "Oil",
    "credit": "Credit",
    "fear": "Market fear",
}


def is_extra(col):
    return str(col).startswith(PREFIX)


def base_features(features):
    """The original feature list without the extra global-driver inputs."""
    return [c for c in features if not is_extra(c)]


def _series(ds, key):
    """A driver's closes from the extra set, or from the core macro set, or None."""
    for frame in (getattr(ds, "extra", None), ds.macro):
        if frame is not None and key in frame.columns and frame[key].notna().sum() > 300:
            return frame[key]
    return None


def _oil(ds):
    s = _series(ds, "brent")
    return s if s is not None else _series(ds, "oil")


# ---------------------------------------------------------------- date-level inputs
def date_features(ds):
    """Market-wide extra inputs, one row per date. Columns that cannot be built are simply left out."""
    spx = ds.idx_ff["spx"]
    out = pd.DataFrame(index=ds.macro.index.union(spx.index))
    ff = lambda s: s.reindex(out.index).ffill()

    def add(name, s):
        if s is not None:
            out[PREFIX + name] = s

    g = lambda k: (lambda s: None if s is None else ff(s))(_series(ds, k))
    us5, us30, tlt = g("us5y"), g("us30y"), g("tlt")
    add("us5y_chg_20", None if us5 is None else us5.diff(20))
    add("us30y_chg_20", None if us30 is None else us30.diff(20))
    add("curve_30y_5y", None if (us5 is None or us30 is None) else us30 - us5)
    curve = ff(ds.macro["us10y"]) - ff(ds.macro["us3m"])
    add("curve_chg_60", curve.diff(60))
    add("tlt_ret_20", None if tlt is None else tlt.pct_change(20, fill_method=None))
    vix, vix3m = ff(ds.macro["vix"]), g("vix3m")
    add("vix_term", None if vix3m is None else vix / vix3m - 1)
    for k in ("eurusd", "gbpusd", "usdjpy", "brent", "natgas", "silver"):
        s = g(k)
        add(f"{k}_ret_20", None if s is None else s.pct_change(20, fill_method=None))
    spx_ret = ff(spx).pct_change(60, fill_method=None)
    for k, name in (("eem", "em"), ("china", "china"), ("semis", "semis"), ("smallcap", "smallcap"), ("equalw", "breadth")):
        s = g(k)
        add(f"{name}_rel_60", None if s is None else s.pct_change(60, fill_method=None) - spx_ret)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(axis=1, how="all")


# ---------------------------------------------------------------- stock-level inputs: sensitivity and push
def factor_shocks(ds):
    """Daily surprise in each driver, in units of that driver's own recent daily spread (so every driver is comparable
    and no unit conversion can matter). Positive = the driver rose; for credit, positive = credit conditions improved."""
    cal = ds.close.index
    cols = {}

    def prep(name, s, kind):
        if s is None:
            return
        s = s.reindex(cal.union(s.index)).ffill().reindex(cal)
        d = s.diff() if kind == "diff" else np.log(s.where(s > 0)).diff()
        sd = d.rolling(252, min_periods=126).std()
        cols[name] = (d / sd).clip(-6, 6)

    prep("rates", _series(ds, "us10y"), "diff")
    prep("dollar", _series(ds, "dxy"), "log")
    prep("oil", _oil(ds), "log")
    hyg, lqd = _series(ds, "hyg"), _series(ds, "lqd")
    if hyg is not None and lqd is not None:
        ratio = (hyg / lqd.reindex(hyg.index)).dropna()
        prep("credit", ratio, "log")
    prep("fear", _series(ds, "vix"), "log")
    return pd.DataFrame(cols, index=cal)


def exposures(ds, window=WINDOW, move=MOVE):
    """For each driver: every stock's sensitivity over the last `window` days (beta: its return relative to the
    S&P 500 per one-spread move in the driver) and the push (beta x the driver's moves over the last `move` days,
    as a fraction of the stock's price). Returns {"beta": {driver: frame}, "push": {driver: frame}}."""
    shocks = factor_shocks(ds)
    r = ds.close.pct_change(fill_method=None).clip(-0.25, 0.25)
    spx = ds.idx_ff["spx"].reindex(r.index).ffill().pct_change(fill_method=None)
    e = r.sub(spx.fillna(0.0), axis=0)
    half = window // 2
    m_e = e.rolling(window, min_periods=half).mean()
    beta, push = {}, {}
    for k in shocks.columns:
        z = shocks[k].fillna(0.0)
        m_ez = e.mul(z, axis=0).rolling(window, min_periods=half).mean()
        m_z = z.rolling(window, min_periods=half).mean()
        var = z.rolling(window, min_periods=half).var(ddof=0).where(lambda v: v > 1e-9)
        b = (m_ez - m_e.mul(m_z, axis=0)).div(var, axis=0).clip(-0.05, 0.05)      # beyond +/-5% a day per spread is noise
        beta[k] = b
        push[k] = b.mul(z.rolling(move, min_periods=move).sum(), axis=0)
    return {"beta": beta, "push": push}


@dataclass
class Ext:
    date: pd.DataFrame                              # market-wide inputs by date
    beta: dict = field(default_factory=dict)        # driver -> (dates x stocks)
    push: dict = field(default_factory=dict)

    def stock_frame(self, ticker, index):
        """The stock-level extra inputs for one stock, aligned to its own dates."""
        cols = {}
        total = None
        for k in self.beta:
            if ticker not in self.beta[k].columns:
                continue
            cols[f"{PREFIX}beta_{k}"] = self.beta[k][ticker]
            p = self.push[k][ticker]
            cols[f"{PREFIX}push_{k}"] = p
            total = p if total is None else total.add(p, fill_value=0.0)
        if total is not None:
            cols[PREFIX + "push_total"] = total
        return pd.DataFrame(cols).reindex(index)


def build_ext(ds):
    """All the extra inputs, or None when the extra series are not there (the model then runs exactly as before)."""
    extra = getattr(ds, "extra", None)
    if extra is None or extra.shape[1] == 0:
        return None
    date = date_features(ds)
    try:
        ex = exposures(ds)
    except Exception:                                    # sensitivities are a bonus; never lose the date-level inputs
        ex = {"beta": {}, "push": {}}
    if date.shape[1] == 0 and not ex["beta"]:
        return None
    return Ext(date=date, beta=ex["beta"], push=ex["push"])


def latest_push(ext, tickers, asof):
    """Each stock's push from each driver as of `asof`, in percent (a +2 means the driver's recent move has tended to
    lift this stock about 2% over the last month). None when there is nothing to show."""
    if ext is None or not ext.push:
        return None
    rows = {}
    for k, frame in ext.push.items():
        f = frame.loc[:asof]
        if f.empty:
            continue
        rows[k] = 100 * f.ffill().iloc[-1].reindex(list(tickers))
    if not rows:
        return None
    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return out.dropna(how="all")


# ---------------------------------------------------------------- readings for the dashboard
_REL = {"em": ("eem", "Emerging markets vs the S&P 500"), "china": ("china", "China (FXI) vs the S&P 500"),
        "semis": ("semis", "Chip stocks vs the S&P 500"), "smallcap": ("smallcap", "Small companies vs the S&P 500"),
        "breadth": ("equalw", "Average stock vs the S&P 500 (breadth)")}


def _spec_list(ds):
    """(key, label, group, series, kind, plain note on what a rise tends to mean)."""
    m = ds.macro
    g = lambda k: _series(ds, k)
    out = [
        ("us10y", "US 10-year yield", "Interest rates", m["us10y"], "pts", "Higher yields raise borrowing costs and tend to weigh on growth stocks."),
        ("curve", "Yield curve (10-year minus 3-month)", "Interest rates", m["us10y"] - m["us3m"], "pts", "Below zero means short rates sit above long ones, an old warning sign for slowdowns (with unreliable timing)."),
    ]
    if g("us30y") is not None:
        out.append(("us30y", "US 30-year yield", "Interest rates", g("us30y"), "pts", "The long end of the curve: what lenders want for locking money away for decades."))
    out.append(("vix", "VIX (market fear)", "Fear", m["vix"], "pts", "Higher means investors are paying more for protection; stocks usually fall when it jumps."))
    if g("vix3m") is not None:
        out.append(("vix_term", "VIX relative to 3-month VIX", "Fear", m["vix"] / g("vix3m").reindex(m.index).ffill(), "ratio", "Above 1 means near-term fear is greater than fear a few months out, a stress signal."))
    out.append(("dxy", "US dollar index", "Currencies", m["dxy"], "%", "A rising dollar tends to squeeze commodities, emerging markets and US exporters."))
    for k, lab in (("gbpusd", "Pound in dollars"), ("eurusd", "Euro in dollars"), ("usdjpy", "Dollar in yen")):
        if g(k) is not None:
            out.append((k, lab, "Currencies", g(k), "%", "Currency moves change what overseas earnings are worth at home."))
    out.append(("oil", "Oil (WTI)", "Commodities", m["oil"], "%", "Oil feeds into costs, inflation and energy company profits."))
    for k, lab in (("brent", "Brent crude"), ("natgas", "Natural gas"), ("silver", "Silver")):
        if g(k) is not None:
            out.append((k, lab, "Commodities", g(k), "%", "A key input and inflation signal."))
    out.append(("gold", "Gold", "Commodities", m["gold"], "%", "Often rises when investors want safety or doubt currencies."))
    out.append(("copper", "Copper", "Commodities", m["copper"], "%", "Used in nearly everything built, so a gauge of global industrial demand."))
    out.append(("credit", "Credit conditions (junk vs safe bonds)", "Credit", m["hyg"] / m["lqd"], "%", "A fall means lenders are demanding more to take risk, which has usually come with weaker markets."))
    if g("tlt") is not None:
        out.append(("tlt", "Long US government bonds (TLT)", "Credit", g("tlt"), "%", "Bond prices fall when yields rise."))
    spx = ds.idx_ff["spx"]
    for k, (src, lab) in _REL.items():
        s = g(src)
        if s is not None:
            rel = (s.pct_change(60, fill_method=None) - spx.reindex(s.index).ffill().pct_change(60, fill_method=None))
            out.append((k, lab, "Regions and style", rel, "rel", "Leadership from riskier or more cyclical corners usually goes with a healthy appetite for risk."))
    return out


def _fmt_change(s, kind):
    """The recent change in a driver, and the history of such changes (to say whether this one is unusual)."""
    s = s.dropna()
    if len(s) < 400:
        return None, None
    if kind == "pts":
        ch = s.diff(MOVE)
    elif kind in ("%", "ratio"):
        ch = s.pct_change(MOVE, fill_method=None) * 100
    else:                                               # "rel": already a 60-day relative return (a fraction)
        ch = s * 100
    return ch.dropna(), s


def drivers_view(ds):
    """The dashboard readings: one row per driver, plus a few plain notes about unusual ones."""
    rows, notes = [], []
    for key, label, group, s, kind, hint in _spec_list(ds):
        try:
            ch, lv = _fmt_change(s.replace([np.inf, -np.inf], np.nan), kind)
            if ch is None or len(ch) < 300:
                continue
            hist = ch.tail(750)
            sd = float(hist.std())
            z = float((ch.iloc[-1] - hist.mean()) / sd) if sd > 0 else 0.0
            last = float(lv.iloc[-1])
            if not (math.isfinite(last) and math.isfinite(float(ch.iloc[-1]))):
                continue
            rows.append({"key": key, "label": label, "group": group, "level": None if kind == "rel" else round(last, 3),
                         "change": round(float(ch.iloc[-1]), 2),
                         "unit": {"pts": "pts", "%": "%", "ratio": "%", "rel": "pts"}[kind],
                         "window": "60 days vs the S&P 500" if kind == "rel" else "20 days",
                         "z": round(z, 1), "state": "unusually high" if z >= 1.5 else ("unusually low" if z <= -1.5 else "normal"),
                         "hint": hint})
        except Exception:
            continue
    by = {r["key"]: r for r in rows}
    if "vix" in by:
        v = ds.macro["vix"].dropna().tail(750)
        pr = float((v < v.iloc[-1]).mean())
        if pr >= 0.85:
            notes.append(f"Market fear (VIX {v.iloc[-1]:.1f}) is higher than on about {pr:.0%} of days in the last three years.")
        elif pr <= 0.15:
            notes.append(f"Market fear (VIX {v.iloc[-1]:.1f}) is unusually low, which describes calm, not safety: calm markets can turn quickly.")
    if "vix_term" in by and by["vix_term"]["level"] > 1.0:
        notes.append("Near-term fear is above fear a few months out (VIX above its 3-month version), which has usually marked stress.")
    if "curve" in by and by["curve"]["level"] < 0:
        notes.append(f"The US yield curve is inverted ({by['curve']['level']:+.2f} points): short rates are above long rates.")
    for k, phrase in (("dxy", "The US dollar"), ("oil", "Oil"), ("credit", "Credit conditions"), ("us10y", "The US 10-year yield")):
        r = by.get(k)
        if r and r["state"] != "normal":
            notes.append(f"{phrase} has moved unusually {'far up' if r['z'] > 0 else 'far down'} over the last month ({r['change']:+.1f}{r['unit'] if r['unit'] != '%' else '%'}).")
    return {"available": bool(rows), "rows": rows, "notes": notes[:4], "n": len(rows)}
