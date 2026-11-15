"""Growth watch: each week's stocks with the most upside potential over the next 5 to 7 trading days, and the paper
portfolio built from them.

This is a risk list, not a tested forecast. "Most upside potential" means a blend of four things, each turned into a
rank among the stocks the app follows:

    model odds        30%   the stock ranker's raw score (weak on its own; the Trust tab shows how weak)
    business growth   30%   revenue and earnings growth ranks from the weekly company-data snapshot
    weekly swing      25%   how far the stock typically moves in a week: bigger swing = bigger possible gain, and loss
    company quality   15%   returns, margins and debt, so the list is not only fast-moving junk

If a stock has no company data, its score uses the parts that exist. The top N (20 by default) are bought on paper
one trading day after the signal, weighted towards the higher scores (half split equally, half in proportion to how
far each score is above the lowest in the group, no stock above 10%), and swapped every week. The result is tracked
against the S&P 500 and the equal-weight basket, so the app measures for itself whether this idea has worked.

Global drivers (rates, dollar, oil, credit, market fear) only add plain-English tags here: they do not change a
score. Whether they help the model itself is decided by a walk-forward test (see macro.py), not assumed.
Every component score is logged each week so that, after enough weeks, the app can show which of the four parts
actually ranked stocks well (see `scorecard`).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import opts

LOG_COLS = ["run_date", "ticker", "as_of", "px", "score", "rank", "final_w", "c_model", "c_growth", "c_swing",
            "c_quality", "macro_push"]
WEIGHTS = {"model": 0.30, "growth": 0.30, "swing": 0.25, "quality": 0.15}
LABELS = {"model": "Model odds", "growth": "Business growth", "swing": "Weekly swing", "quality": "Company quality"}
OVERSOLD = -0.10            # 20-day return at or below this is labelled "down a lot"
EXTENDED = 0.10             # at or above this: "already up a lot"
CLUSTER_CORR = 0.6          # two stocks "move together" at or above this correlation of daily returns
CORR_DAYS = 126             # about six months of trading days
PUSH_MIN = 1.0              # a global driver's push has to be at least this many % points to be mentioned
PUSH_Z = 1.5                # ...and stand out this much from the other stocks


def _f(x, nd=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def _rank(s):
    return s.rank(pct=True)


def score(ranking, fund_by_ticker):
    """One row per stock with its 0-100 upside score, best first. `fund_by_ticker` may be empty or partial."""
    df = ranking[["ticker", "p_raw", "vol_ann"]].copy().reset_index(drop=True)
    fb = fund_by_ticker or {}
    get = lambda t, k: (fb.get(t) or {}).get(k)
    df["growth_pct"] = [get(t, "growth") for t in df["ticker"]]
    df["rev_pct"] = [get(t, "rev_growth_pct") for t in df["ticker"]]
    df["quality_pct"] = [get(t, "quality") for t in df["ticker"]]
    for c in ("growth_pct", "rev_pct", "quality_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    comp = pd.DataFrame(index=df.index)
    comp["model"] = _rank(df["p_raw"])
    comp["swing"] = _rank(df["vol_ann"])
    comp["growth"] = pd.concat([_rank(df["growth_pct"]), _rank(df["rev_pct"].clip(-10, 60))], axis=1).mean(axis=1, skipna=True)
    comp["quality"] = _rank(df["quality_pct"])
    w = pd.Series(WEIGHTS)
    avail = comp.notna()
    num = (comp.fillna(0) * w).sum(axis=1)
    den = (avail * w).sum(axis=1)
    df["score"] = 100 * num / den.where(den > 0)
    # a stock needs the two model-side parts to be scored at all
    df.loc[~(avail["model"] & avail["swing"]), "score"] = np.nan
    for k in WEIGHTS:
        df["c_" + k] = 100 * comp[k]
    df["parts"] = avail.sum(axis=1)
    out = df.dropna(subset=["score"]).sort_values(["score", "p_raw"], ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def weights(scores, tilt=None, max_w=None):
    """Portfolio weights for the top stocks: part equal, part in proportion to how far each score is above the lowest.
    No weight above `max_w` (the excess is shared among the rest). Always sums to 1."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n == 0:
        return s
    tilt = opts.get("GROWTH_TILT") if tilt is None else tilt
    max_w = opts.get("GROWTH_MAX_W") if max_w is None else max_w
    spread = s - s.min()
    t = spread / spread.sum() if spread.sum() > 0 else np.full(n, 1.0 / n)
    w = (1 - tilt) / n + tilt * t
    cap = max(max_w, 1.0 / n)
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if not free.any():
            break
        base = w[free]
        share = base / base.sum() if base.sum() > 0 else np.full(int(free.sum()), 1.0 / int(free.sum()))
        w[free] += excess * share
    return w / w.sum()


def concentration(close, tickers, days=None, thresh=None):
    """How much the picks move together, and the typical weekly swing of the equal-weight basket."""
    days = days or CORR_DAYS
    thresh = thresh or CLUSTER_CORR
    cols = [t for t in tickers if t in close.columns]
    if len(cols) < 3:
        return None
    rets = close[cols].pct_change(fill_method=None).tail(days)
    rets = rets.dropna(axis=1, thresh=int(days * 0.8)).fillna(0.0)
    cols = list(rets.columns)
    if len(cols) < 3 or len(rets) < 40:
        return None
    corr = rets.corr().values
    n = len(cols)
    iu = np.triu_indices(n, 1)
    avg = float(np.nanmean(corr[iu]))
    peers = [(int((corr[i] >= thresh).sum()), i) for i in range(n)]     # includes the stock itself
    size, best = max(peers)
    group = [cols[j] for j in range(n) if corr[best, j] >= thresh]
    cov = rets.cov().values
    w = np.full(n, 1.0 / n)
    day_sd = float(np.sqrt(w @ cov @ w))
    return {"avg_corr": _f(avg, 2), "cluster_size": int(size), "cluster": group, "n": int(n),
            "basket_swing_pct": _f(100 * day_sd * np.sqrt(5), 1), "threshold": thresh, "days": int(len(rets))}


def _tags(row, fund, regimes):
    tags = []
    r20 = row.get("ret_20")
    if r20 is not None and r20 <= OVERSOLD:
        tags.append("down a lot")
    elif r20 is not None and r20 >= EXTENDED:
        tags.append("already up a lot")
    d = (fund or {}).get("earnings_days")
    if d is not None and 0 <= d <= opts.get("EARNINGS_WARN_DAYS"):
        tags.append("reports in " + ("today" if d == 0 else f"{d}d"))
    reg = regimes.get(row.get("region"))
    if reg and "stressed" in str(reg).lower():
        tags.append("stressed market")
    if not fund:
        tags.append("no company data")
    return tags


def _macro_tags(push, labels, ticker):
    """Global drivers whose recent move gives this stock an unusual push, up or down. Context only."""
    if push is None or ticker not in push.index:
        return []
    row = push.loc[ticker]
    out = []
    for k in push.columns:
        col = push[k]
        sd = float(col.std())
        v = float(row[k])
        if not math.isfinite(v) or not math.isfinite(sd) or sd <= 0:
            continue
        z = (v - float(col.mean())) / sd
        if abs(v) >= PUSH_MIN and abs(z) >= PUSH_Z:
            out.append((abs(z), f"{labels.get(k, k)} {'tailwind' if v > 0 else 'headwind'}"))
    out.sort(reverse=True)
    return [t for _, t in out[:2]]


def build(ranking, fund_by_ticker, ds, mdash, top_n=None, push=None, push_labels=None):
    """The dashboard block and the frame to log. Pure computation: never touches the network.

    `push` (optional): per-stock table of how much each global driver's recent move is expected to have helped or
    hurt the stock (see macro.stock_push). It adds tags and a logged column, nothing more.
    """
    top_n = int(top_n or opts.get("GROWTH_TOP_N"))
    sc = score(ranking, fund_by_ticker)
    if len(sc) < max(top_n + 5, 10):
        return {"available": False, "reason": "Too few stocks could be scored this week to pick a top "
                                              f"{top_n} (the watchlist needs at least {max(top_n + 5, 10)})."}, None
    top = sc.head(top_n).copy()
    top["w"] = weights(top["score"].to_numpy())
    rk = ranking.set_index("ticker")
    regimes = dict(zip(mdash["key"], mdash["regime"])) if mdash is not None and len(mdash) else {}
    expo = dict(zip(mdash["key"], mdash["exposure"])) if mdash is not None and len(mdash) else {}
    fb = fund_by_ticker or {}
    labels = push_labels or {}

    picks = []
    for _, r in top.iterrows():
        t = r["ticker"]
        s = rk.loc[t]
        f = fb.get(t) or {}
        row = {"ticker": t, "ret_20": _f(s.get("ret_20")), "region": s.get("region")}
        picks.append({
            "rank": int(r["rank"]), "ticker": t, "name": f.get("name"), "sector": f.get("sector"),
            "w": _f(r["w"], 4), "score": _f(r["score"], 1), "p_raw": _f(s["p_raw"], 3), "ret_20": _f(s.get("ret_20"), 4),
            "vol_ann": _f(s["vol_ann"], 3), "swing_pct": _f(100 * s["vol_ann"] / math.sqrt(52), 1),
            "growth": _f(f.get("growth"), 0), "rev_growth_pct": _f(f.get("rev_growth_pct"), 0),
            "quality": _f(f.get("quality"), 0), "fwd_pe": _f(f.get("fwd_pe"), 1),
            "earnings_days": f.get("earnings_days"), "region": s.get("region"),
            "regime": regimes.get(s.get("region")), "tags": _tags(row, f, regimes),
            "macro_tags": _macro_tags(push, labels, t),
        })
    conc = concentration(ds.close, [p["ticker"] for p in picks])
    notes = []
    if conc and conc["cluster_size"] >= max(4, top_n // 2):
        notes.append(f"{conc['cluster_size']} of these {conc['n']} moved closely together over the last six months "
                     f"({', '.join(conc['cluster'][:8])}), so this is nearer one big bet than {top_n} separate ones.")
    if conc and conc["basket_swing_pct"] is not None:
        notes.append(f"A typical week moves this equal-weight basket by about {conc['basket_swing_pct']:.1f}% either way, "
                     "and a bad week can be much worse.")
    soon = [p["ticker"] + f" ({p['earnings_days']}d)" for p in picks
            if p["earnings_days"] is not None and 0 <= p["earnings_days"] <= opts.get("EARNINGS_WARN_DAYS")]
    if soon:
        notes.append("Reporting results within a week: " + ", ".join(soon) + ". Results can move a stock sharply either way.")
    down = [p["ticker"] for p in picks if "down a lot" in p["tags"]]
    if down:
        notes.append("Down a lot over 20 days: " + ", ".join(down) + ". A big drop can be a bargain or the start of a longer "
                     "slide, and the model cannot tell which. Check why each fell, especially after earnings.")
    no_data = [p["ticker"] for p in picks if "no company data" in p["tags"]]
    if no_data:
        notes.append("No company data this week for " + ", ".join(no_data) + ", so their score uses fewer parts.")

    scaler = float(sum(p["w"] * float(expo.get(p["region"], 1.0)) for p in picks)) if expo else None
    block = {"available": True, "n": int(len(picks)), "as_of": pd.Timestamp(ds.run_date).strftime("%Y-%m-%d"),
             "horizon": "5 to 7 trading days", "entry_delay": C.ENTRY_DELAY, "picks": picks, "concentration": conc,
             "notes": notes[:5], "weights": dict(WEIGHTS), "labels": dict(LABELS), "universe": int(len(sc)),
             "portfolio": {"tilt": opts.get("GROWTH_TILT"), "max_w": opts.get("GROWTH_MAX_W"),
                           "top_w": _f(top["w"].max(), 4), "low_w": _f(top["w"].min(), 4),
                           "scaler_exposure": _f(scaler, 3)}}

    log = sc[["ticker", "score", "rank", "c_model", "c_growth", "c_swing", "c_quality"]].copy()
    px = ranking.set_index("ticker")
    log["as_of"] = [px.loc[t, "date"] for t in log["ticker"]]
    log["px"] = [px.loc[t, "px"] for t in log["ticker"]]
    log["run_date"] = ds.run_date
    wmap = dict(zip(top["ticker"], top["w"]))
    log["final_w"] = [float(wmap.get(t, 0.0)) for t in log["ticker"]]
    log["macro_push"] = [float(push.loc[t].sum()) if (push is not None and t in push.index) else float("nan")
                         for t in log["ticker"]]
    return block, log[LOG_COLS]


def append_log(path, frame):
    """Append this run's rows. Existing rows are never changed; a re-run of the same data date adds nothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["run_date", "as_of"])
        for c in LOG_COLS:                                   # a log written by an older version lacks the new columns
            if c not in old.columns:
                old[c] = np.nan
        frame = pd.concat([old[LOG_COLS], frame], ignore_index=True)
    frame = frame.drop_duplicates(["run_date", "ticker"], keep="first")
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return frame


# ---------------------------------------------------------------- did the score work? (scored afterwards)
def scorecard(log, close, min_weeks=None):
    """Weekly rank correlation (IC) between each logged score and the stock's actual return over the holding period,
    for the combined score and for each of its four parts. Needs `min_weeks` weeks before it gives a verdict."""
    min_weeks = int(min_weeks or opts.get("PAPER_MIN_WEEKS"))
    out = {"weeks": 0, "pending": 0, "min_weeks": min_weeks, "verdict": "too early", "mean_ic": None, "hit_rate": None,
           "spread_pct": None, "t": None, "parts": []}
    if log is None or len(log) == 0:
        return out
    df = log.copy()
    df["run_date"] = pd.to_datetime(df["run_date"])
    df["as_of"] = pd.to_datetime(df["as_of"])
    d = C.ENTRY_DELAY
    cols = {"score": "Combined score", **{"c_" + k: LABELS[k] for k in WEIGHTS}}
    ics = {k: [] for k in cols}
    spreads, last = [], None
    for run, g in df.groupby("run_date"):
        a = g["as_of"].max()
        if last is not None and (a - last).days < opts.get("PAPER_MIN_GAP_DAYS"):
            continue
        last = a
        pos = int(close.index.searchsorted(a, side="right")) - 1
        entry = pos + d
        exit_ = entry + C.SH
        if pos < 0 or exit_ >= len(close):
            out["pending"] += 1
            continue
        fwd = (close.iloc[exit_] / close.iloc[entry] - 1)
        x = g.set_index("ticker")
        fw = fwd.reindex(x.index)
        ok = fw.notna()
        if ok.sum() < 20:
            continue
        for k in cols:
            v = pd.to_numeric(x[k], errors="coerce")[ok]
            ics[k].append(float(v.corr(fw[ok], method="spearman")) if v.notna().sum() >= 20 and v.std() > 0 else float("nan"))
        top = x[(x["final_w"] > 0) & ok]
        rest = x[(x["final_w"] <= 0) & ok]
        if len(top) and len(rest):
            spreads.append(float(fw[top.index].mean() - fw[rest.index].mean()))
    n = len(ics["score"])
    out["weeks"] = n
    if n:
        def stat(v):
            a = np.array([z for z in v if np.isfinite(z)])
            return a if len(a) else None
        a = stat(ics["score"])
        if a is not None:
            out["mean_ic"] = float(a.mean())
            out["hit_rate"] = float((a > 0).mean())
            if len(a) >= 8 and a.std(ddof=1) > 0:
                out["t"] = float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a))))
        if spreads:
            out["spread_pct"] = float(100 * np.mean(spreads))
        for k, lab in cols.items():
            b = stat(ics[k])
            if b is not None and k != "score":
                out["parts"].append({"key": k[2:], "label": lab, "mean_ic": float(b.mean()), "weeks": int(len(b))})
    if n < min_weeks:
        out["verdict"] = "too early"
    elif out["t"] is not None and out["t"] >= 2:
        out["verdict"] = "positive so far"
    elif out["t"] is not None and out["t"] <= -2:
        out["verdict"] = "negative so far"
    else:
        out["verdict"] = "no clear edge"
    return out
