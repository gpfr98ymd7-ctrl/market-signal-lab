"""Company data: weekly snapshots, a transparent score, earnings dates, and an honest check of whether it helps.

Why snapshots? Yahoo only gives today's ratios, not what they were years ago. Training a model on today's ratios
for past dates would quietly use the future. So the ratios are NOT fed to the model. Instead each week's values are
saved (a point-in-time history that builds from now on), shown next to the picks as context, and scored afterwards:
once enough weeks have passed, the scorecard reports whether the company-data score actually ranked later returns.

The score is deliberately simple: percentile ranks against the other tracked stocks (not sector-adjusted) for
value, quality and growth, averaged. It is shown for context and never changes a forecast, ranking or weight.
Earnings dates are forward-looking facts (known in advance), so flagging them is safe.
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import opts

NUM_FIELDS = ["trailingPE", "forwardPE", "priceToBook", "profitMargins", "operatingMargins", "returnOnEquity",
              "revenueGrowth", "earningsGrowth", "debtToEquity", "freeCashflow", "marketCap"]
TEXT_FIELDS = ["sector", "shortName"]
COLS = ["snap_date", "ticker", "next_earnings"] + NUM_FIELDS + TEXT_FIELDS
MAX_PE = 300.0             # above this a P/E is treated as missing (usually a data quirk or near-zero profit)
FCF_YIELD_RANGE = (-0.30, 0.40)   # a free-cash-flow yield outside this is treated as a data error


# ---------------------------------------------------------------- fetching
def yahoo_fetcher(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info or {}
    try:
        cal = t.calendar
    except Exception:
        cal = None
    return {"info": info, "calendar": cal}


def _num(x):
    if isinstance(x, bool) or x is None:
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _earnings_dates(cal, info):
    vals = []
    try:
        if isinstance(cal, dict):
            vals = cal.get("Earnings Date") or []
        elif cal is not None and hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
            vals = list(cal.loc["Earnings Date"].values)
        if not isinstance(vals, (list, tuple, np.ndarray, pd.Series)):
            vals = [vals]
    except Exception:
        vals = []
    out = []
    for v in vals:
        try:
            ts = pd.Timestamp(v)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            out.append(ts.normalize())
        except Exception:
            continue
    for k in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
        v = _num((info or {}).get(k))
        if math.isfinite(v) and v > 1e9:
            out.append(pd.to_datetime(v, unit="s").normalize())
    return out


def next_earnings(cal, info, today):
    """The earliest known earnings date that is today or later, or None."""
    today = pd.Timestamp(today).normalize()
    future = [d for d in _earnings_dates(cal, info) if d >= today]
    return min(future) if future else None


_CCY = {"GBp": "GBP", "GBX": "GBP", "ILA": "ILS", "ZAc": "ZAR"}      # pence and other minor units: same currency


def _ccy(x):
    c = str(x or "").strip()
    return _CCY.get(c, c.upper())


def clean_info(info):
    """A flat row of the fields we keep, or None if Yahoo gave too little to be useful.

    Two guards against Yahoo's known quirks (the ratios are only ever ranked, so a wrong scale would silently
    reorder stocks): free cash flow is reported in the company's reporting currency but market value in the
    trading currency, so a mismatch drops the cash-flow figure; and absurd P/E ratios are dropped."""
    info = info or {}
    row = {k: _num(info.get(k)) for k in NUM_FIELDS}
    trade, fin = _ccy(info.get("currency")), _ccy(info.get("financialCurrency"))
    if trade and fin and trade != fin:
        row["freeCashflow"] = float("nan")
    for k in ("trailingPE", "forwardPE"):
        if math.isfinite(row[k]) and row[k] > MAX_PE:
            row[k] = float("nan")
    if sum(math.isfinite(v) for v in row.values()) < 3:
        return None
    row["sector"] = str(info.get("sector") or "")[:40]
    row["shortName"] = str(info.get("shortName") or info.get("longName") or "")[:60]
    return row


def fetch_rows(tickers, snap_date, skip, fetcher, pause, budget, today, log=print):
    """Fetch what we can, politely. Gives up early when Yahoo is clearly refusing. Returns (rows, errors)."""
    rows, errors, consecutive = [], [], 0
    t0 = time.time()
    for t in tickers:
        if t in skip:
            continue
        if time.time() - t0 > budget:
            errors.append("time budget reached")
            break
        try:
            raw = fetcher(t) or {}
            row = clean_info(raw.get("info"))
            if row is None:
                raise ValueError("no company data")
            ne = next_earnings(raw.get("calendar"), raw.get("info"), today)
            row.update({"snap_date": pd.Timestamp(snap_date), "ticker": t, "next_earnings": ne})
            rows.append(row)
            consecutive = 0
        except Exception as e:
            consecutive += 1
            errors.append(f"{t}: {type(e).__name__}")
            if consecutive >= 6:
                errors.append("stopped after 6 failures in a row")
                log("Company data: Yahoo is refusing requests; stopping early.")
                break
            continue
        if pause > 0:
            time.sleep(pause)
    return rows, errors


# ---------------------------------------------------------------- storage
def read_snapshots(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    try:
        df = pd.read_csv(path, parse_dates=["snap_date", "next_earnings"], keep_default_na=True)
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLS]
    except Exception:
        return pd.DataFrame(columns=COLS)


def append_snapshots(path, snaps, rows):
    """Add new rows; an existing (date, ticker) is never overwritten."""
    if not rows:
        return snaps
    new = pd.concat([snaps, pd.DataFrame(rows, columns=COLS)], ignore_index=True) if len(snaps) else \
        pd.DataFrame(rows, columns=COLS)
    new = new.drop_duplicates(["snap_date", "ticker"], keep="first").sort_values(["snap_date", "ticker"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)
    return new.reset_index(drop=True)


# ---------------------------------------------------------------- the score
def _rank(s):
    return s.rank(pct=True)


def score_snapshot(df, min_names=8):
    """Add value, quality, growth and score (all 0-100) to one snapshot (one row per ticker)."""
    df = df.copy().reset_index(drop=True)
    for c in ("value", "quality", "growth", "score"):
        df[c] = np.nan
    if len(df) < min_names:
        return df
    pe, fpe = df["trailingPE"], df["forwardPE"]
    ey = _rank(1 / pe.where(pe > 0))
    ey = ey.where(~(pe <= 0), 0.0)                            # loss-makers rank worst on earnings yield
    fey = _rank(1 / fpe.where(fpe > 0))
    fy = df["freeCashflow"] / df["marketCap"].where(df["marketCap"] > 0)
    fcf = _rank(fy.where((fy >= FCF_YIELD_RANGE[0]) & (fy <= FCF_YIELD_RANGE[1])))
    value = pd.concat([ey, fey, fcf], axis=1).mean(axis=1, skipna=True)

    roe = _rank(df["returnOnEquity"].clip(-1, 1))
    opm = _rank(df["operatingMargins"].fillna(df["profitMargins"]))
    banks = df["sector"].fillna("").str.contains("Financial", case=False)
    lev = _rank(-df["debtToEquity"].where(df["debtToEquity"] >= 0)).where(~banks)   # banks are leveraged by nature
    quality = pd.concat([roe, opm, lev], axis=1).mean(axis=1, skipna=True)

    growth = pd.concat([_rank(df["revenueGrowth"].clip(-1, 1)), _rank(df["earningsGrowth"].clip(-1, 3))],
                       axis=1).mean(axis=1, skipna=True)

    comps = pd.concat([value, quality, growth], axis=1)
    score = comps.mean(axis=1, skipna=True).where(comps.notna().sum(axis=1) >= 2)
    df["value"], df["quality"], df["growth"], df["score"] = 100 * value, 100 * quality, 100 * growth, 100 * score
    return df


def stock_view(row, today, warn_days):
    def f(x, mult=1.0, nd=1):
        x = _num(x)
        return None if not math.isfinite(x) else round(x * mult, nd)

    ne = row.get("next_earnings")
    days = None
    if pd.notna(ne):
        days = int((pd.Timestamp(ne).normalize() - pd.Timestamp(today).normalize()).days)
        if days < 0:
            days = None
    return {
        "score": f(row.get("score"), nd=0), "value": f(row.get("value"), nd=0), "quality": f(row.get("quality"), nd=0),
        "growth": f(row.get("growth"), nd=0), "pe": f(row.get("trailingPE")), "fwd_pe": f(row.get("forwardPE")),
        "roe_pct": f(row.get("returnOnEquity"), 100, 0), "margin_pct": f(row.get("operatingMargins"), 100, 0),
        "rev_growth_pct": f(row.get("revenueGrowth"), 100, 0), "debt_eq": f(row.get("debtToEquity"), nd=0),
        "sector": row.get("sector") or None, "name": row.get("shortName") or None,
        "next_earnings": None if days is None else pd.Timestamp(ne).strftime("%Y-%m-%d"),
        "earnings_days": days, "earnings_soon": bool(days is not None and days <= warn_days),
    }


# ---------------------------------------------------------------- did it help? (scored afterwards, never in training)
def scorecard(snaps, close, horizon=None, min_weeks=None):
    horizon = horizon or opts.get("FUND_HORIZON")
    min_weeks = min_weeks or opts.get("FUND_MIN_WEEKS")
    ics, spreads, pending = [], [], 0
    if len(snaps):
        for date, g in snaps.groupby("snap_date"):
            g = g[g["ticker"].isin(close.columns)]
            pos = close.index.searchsorted(pd.Timestamp(date), side="right") - 1
            entry = pos + C.ENTRY_DELAY
            exit_ = entry + horizon
            if pos < 0 or exit_ >= len(close):
                pending += 1
                continue
            sc = score_snapshot(g)
            fwd = close.iloc[exit_] / close.iloc[entry] - 1
            d = pd.DataFrame({"score": sc.set_index("ticker")["score"], "fwd": fwd}).dropna()
            if len(d) < 12:
                continue
            ics.append(float(d["score"].corr(d["fwd"], method="spearman")))
            k = len(d) // 3
            ranked = d.sort_values("score")
            spreads.append(float(ranked["fwd"].iloc[-k:].mean() - ranked["fwd"].iloc[:k].mean()))
    n = len(ics)
    eff = n / math.ceil(horizon / 5)                          # weekly snapshots, multi-week horizon: they overlap
    out = {"weeks": n, "pending": pending, "horizon_days": int(horizon), "min_weeks": int(min_weeks),
           "effective_periods": round(eff, 1), "mean_ic": None, "hit_rate": None, "spread_pct": None, "t": None}
    if n:
        a = np.array(ics)
        out["mean_ic"] = float(a.mean())
        out["hit_rate"] = float((a > 0).mean())
        out["spread_pct"] = float(100 * np.mean(spreads))
        if eff >= 2 and a.std(ddof=1) > 0:
            out["t"] = float(a.mean() / (a.std(ddof=1) / math.sqrt(eff)))
    if n < min_weeks:
        out["verdict"] = "too early"
    elif out["t"] is not None and out["t"] >= 2:
        out["verdict"] = "positive so far"
    elif out["t"] is not None and out["t"] <= -2:
        out["verdict"] = "negative so far"
    else:
        out["verdict"] = "no clear edge"
    return out


# ---------------------------------------------------------------- one call for the weekly run
def update(log_dir, ds, now, fetcher=None, pause=None, budget=None, log=print):
    """Snapshot, score and summarise. Never raises for network problems. Returns the dashboard block."""
    path = Path(log_dir) / "fundamentals.csv"
    pause = opts.get("FUND_PAUSE") if pause is None else pause
    budget = opts.get("FUND_TIME_BUDGET") if budget is None else budget
    warn = opts.get("EARNINGS_WARN_DAYS")
    snaps = read_snapshots(path)
    snap_date = pd.Timestamp(ds.run_date)
    have = set(snaps.loc[snaps["snap_date"] == snap_date, "ticker"]) if len(snaps) else set()
    rows, errors = fetch_rows(ds.stocks, snap_date, have, fetcher or yahoo_fetcher, pause, budget, now, log=log)
    if rows:
        snaps = append_snapshots(path, snaps, rows)
        log(f"Company data: saved {len(rows)} new row(s) for {snap_date.date()}.")
    block = {"available": False, "errors": errors[:4], "by_ticker": {}}
    if len(snaps) == 0:
        block["reason"] = "No company data could be fetched yet."
        return block
    latest = snaps["snap_date"].max()
    if (pd.Timestamp(now).normalize() - latest).days > 21:
        block["reason"] = f"The latest company data is from {latest.strftime('%Y-%m-%d')} and is out of date."
        return block
    g = snaps[snaps["snap_date"] == latest]
    sc = score_snapshot(g)
    by = {r["ticker"]: stock_view(r, now, warn) for _, r in sc.iterrows() if r["ticker"] in ds.stocks}
    block.update({"available": True, "as_of": latest.strftime("%Y-%m-%d"), "fresh": bool(latest == snap_date),
                  "n": int(len(g)), "universe": int(len(ds.stocks)), "by_ticker": by,
                  "scorecard": scorecard(snaps, ds.close), "weeks_saved": int(snaps["snap_date"].nunique())})
    return block
