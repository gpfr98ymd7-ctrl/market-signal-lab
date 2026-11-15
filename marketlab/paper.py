"""Paper trading: pretend portfolios, tracked honestly. Nothing here ever places a real order.

Every week the app logs the portfolio it would hold (the model's top picks, and the growth portfolio). This module
turns those logs into a track record:

  * each portfolio is "bought" at the close ENTRY_DELAY trading days after its signal date, and held until the next
    week's portfolio is bought, so every week is a real sell-and-buy;
  * trading costs (COST_BPS) are charged on everything that changes between one week's holdings and the next;
  * results are compared with two yardsticks over the same days: the S&P 500, and an equal-weight basket of every
    stock the app was following that week (the fair test of "did the picking help?");
  * a closed week is written to data/paper_<name>.csv once and never changed afterwards, so a later data
    correction cannot rewrite history; the week in progress is only marked to market;
  * no verdict is shown until PAPER_MIN_WEEKS weeks have closed, because a few weeks of results prove nothing.

The ledger is built from the forecast logs and from prices only, so it can be rebuilt, audited, or compared against
what the logs say at any time.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import opts

LEDGER_COLS = ["run_date", "entry_date", "exit_date", "ret_gross", "cost", "ret_net", "spx", "basket", "turnover",
               "n_held", "n_unpriced", "cash", "weights", "end_weights"]
LOGS = {"model": "log_stocks.csv", "growth": "log_growth.csv"}


# ---------------------------------------------------------------- reading and writing
def read_ledger(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    try:
        df = pd.read_csv(path, parse_dates=["run_date", "entry_date", "exit_date"])
        return df[LEDGER_COLS].sort_values("run_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=LEDGER_COLS)


def _write_ledger(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _prep(slog):
    """The weights log as a tidy frame with real dates, or None."""
    if slog is None or len(slog) == 0:
        return None
    df = slog.copy()
    df["run_date"] = pd.to_datetime(df["run_date"])
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["final_w"] = pd.to_numeric(df["final_w"], errors="coerce").fillna(0.0)
    return df


def held_tickers(log_dir):
    """Tickers the latest logged run of each paper portfolio holds. They are downloaded even if the watchlist no longer
    lists them, so the portfolio can still be priced when it is sold."""
    out = []
    for name, fname in LOGS.items():
        p = Path(log_dir) / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, usecols=["run_date", "ticker", "final_w"])
            last = df[df["run_date"] == df["run_date"].max()]
            out += last.loc[pd.to_numeric(last["final_w"], errors="coerce").fillna(0) > 0, "ticker"].astype(str).tolist()
        except Exception:
            continue
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- weeks and periods
def pick_weeks(slog, cal, min_gap_days):
    """One entry per week: the earliest logged run, skipping runs less than `min_gap_days` after the last kept one.
    Each has the run date, the data date (as_of) and the date the portfolio would actually be bought (or None if that
    trading day has not happened yet)."""
    df = _prep(slog)
    if df is None:
        return []
    asof = df.groupby("run_date")["as_of"].max().sort_index()
    out, last = [], None
    for run, a in asof.items():
        if last is not None and (a - last).days < min_gap_days:
            continue
        pos = int(cal.searchsorted(a, side="right")) - 1
        e = pos + C.ENTRY_DELAY
        out.append({"run_date": run, "as_of": a, "entry": cal[e] if (pos >= 0 and 0 <= e < len(cal)) else None})
        last = a
    return out


def _weights(df, run):
    g = df[df["run_date"] == run]
    w = g.groupby("ticker")["final_w"].sum()
    return {t: float(v) for t, v in w.items() if v > 1e-9}


def _universe(df, run):
    return df.loc[df["run_date"] == run, "ticker"].astype(str).unique().tolist()


def _ret(px, t, e0, e1):
    if t not in px.columns:
        return None
    a, b = px.at[e0, t], px.at[e1, t]
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return None
    return float(b / a - 1)


def _period(target, universe, e0, e1, px, spx, prev_end, cost_bps):
    """One holding period. `target` are the weights bought at e0 (after the previous week's drift `prev_end`)."""
    rets = {t: _ret(px, t, e0, e1) for t in set(target) | set(universe)}
    priced = {t: w for t, w in target.items() if rets.get(t) is not None}
    unpriced = [t for t in target if rets.get(t) is None]
    gross = float(sum(w * rets[t] for t, w in priced.items()))
    end = {t: w * (1 + rets[t]) / (1 + gross) for t, w in priced.items()}
    turn = float(sum(abs(priced.get(t, 0.0) - prev_end.get(t, 0.0)) for t in set(priced) | set(prev_end)))
    cost = turn * cost_bps / 1e4
    uni = [rets[t] for t in universe if rets.get(t) is not None]
    s0, s1 = spx.at[e0], spx.at[e1]
    return {"ret_gross": gross, "cost": cost, "ret_net": gross - cost, "turnover": turn,
            "spx": float(s1 / s0 - 1) if (s0 > 0 and np.isfinite(s0) and np.isfinite(s1)) else float("nan"),
            "basket": float(np.mean(uni)) if uni else float("nan"),
            "n_held": int(len(priced)), "n_unpriced": int(len(unpriced)), "cash": float(max(0.0, 1 - sum(priced.values()))),
            "weights": json.dumps({t: round(w, 6) for t, w in priced.items()}),
            "end_weights": json.dumps({t: round(w, 6) for t, w in end.items()})}


def _loads(s):
    try:
        d = json.loads(s) if isinstance(s, str) else {}
        return {str(k): float(v) for k, v in d.items()}
    except Exception:
        return {}


def closed_weeks(ledger, df, weeks, px, spx, cost_bps):
    """Append every week that can now be closed (its sell day has happened). Rows already in the ledger are kept as
    they are. Returns the new ledger."""
    have = set(pd.to_datetime(ledger["run_date"])) if len(ledger) else set()
    rows = [r for _, r in ledger.iterrows()]
    prev_end = _loads(rows[-1]["end_weights"]) if rows else {}
    new = []
    for k in range(len(weeks) - 1):
        wk, nxt = weeks[k], weeks[k + 1]
        if wk["run_date"] in have:
            continue
        e0, e1 = wk["entry"], nxt["entry"]
        if e0 is None or e1 is None or e0 not in px.index or e1 not in px.index or e1 <= e0:
            break
        res = _period(_weights(df, wk["run_date"]), _universe(df, wk["run_date"]), e0, e1, px, spx, prev_end, cost_bps)
        new.append({"run_date": wk["run_date"], "entry_date": e0, "exit_date": e1, **res})
        prev_end = _loads(res["end_weights"])
    if not new:
        return ledger
    out = pd.concat([ledger, pd.DataFrame(new)[LEDGER_COLS]], ignore_index=True) if len(ledger) else pd.DataFrame(new)[LEDGER_COLS]
    return out.sort_values("run_date").reset_index(drop=True)


def _mark(ledger, df, week, before, px, spx, last, cost_bps):
    """Mark one week's portfolio to market from its purchase day to the latest price day. `before` is the ledger row
    that precedes it (whose end weights the purchase is traded from), or None."""
    prev_end = _loads(before["end_weights"]) if before is not None else {}
    r = _period(_weights(df, week["run_date"]), _universe(df, week["run_date"]), week["entry"], last, px, spx, prev_end, cost_bps)
    return {"run_date": week["run_date"].strftime("%Y-%m-%d"), "since": week["entry"].strftime("%Y-%m-%d"),
            "through": last.strftime("%Y-%m-%d"), "ret_pct": 100 * r["ret_net"], "spx_pct": 100 * r["spx"],
            "basket_pct": 100 * r["basket"], "n": r["n_held"]}


def open_position(ledger, df, weeks, px, spx, cost_bps):
    """The latest week's portfolio: waiting to be bought (and then the week before it is the one still held), or bought
    and marked to market. `mark` is always the portfolio actually held right now, if there is one."""
    if not weeks:
        return None
    wk = weeks[-1]
    target = _weights(df, wk["run_date"])
    out = {"run_date": wk["run_date"].strftime("%Y-%m-%d"), "n": len(target),
           "top": [{"ticker": t, "w": round(w, 4)} for t, w in sorted(target.items(), key=lambda x: -x[1])[:5]]}
    last = px.index.max()
    rows = [r for _, r in ledger.iterrows()]
    if wk["entry"] is None or wk["entry"] > last:
        out.update({"status": "waiting to be bought", "buy_on": "the next trading day's close"})
        prev = weeks[-2] if len(weeks) > 1 else None
        if prev is not None and prev["entry"] is not None and prev["entry"] < last:
            before = rows[-1] if rows and rows[-1]["run_date"] < prev["run_date"] else None
            out["mark"] = _mark(ledger, df, prev, before, px, spx, last, cost_bps)
        return out
    before = rows[-1] if rows and rows[-1]["run_date"] < wk["run_date"] else None
    if wk["entry"] == last:
        out.update({"status": "bought at the latest close",
                    "mark": {"run_date": out["run_date"], "since": last.strftime("%Y-%m-%d"), "through": last.strftime("%Y-%m-%d"),
                             "ret_pct": 0.0, "spx_pct": 0.0, "basket_pct": 0.0, "n": len(target)}})
        return out
    out.update({"status": "held", "mark": _mark(ledger, df, wk, before, px, spx, last, cost_bps)})
    return out


# ---------------------------------------------------------------- this week's orders
def _drift(df, prev, px, until):
    """The previous week's holdings after drifting with prices from its purchase until `until`."""
    w = _weights(df, prev["run_date"])
    e0 = prev["entry"]
    if e0 is None or until <= e0:
        return w
    out = {}
    for t, x in w.items():
        r = _ret(px, t, e0, until)
        out[t] = x * (1 + (r if r is not None else 0.0))
    tot = sum(out.values()) + max(0.0, 1 - sum(w.values()))
    return {t: v / tot for t, v in out.items()} if tot > 0 else w


def trades(df, weeks, px, min_move=None):
    """The orders that turn last week's holdings into this week's target. Prices are the latest close for reference."""
    if not weeks:
        return None
    min_move = opts.get("PAPER_TRADE_MIN") if min_move is None else min_move
    cur = weeks[-1]
    target = _weights(df, cur["run_date"])
    until = cur["entry"] if (cur["entry"] is not None and cur["entry"] <= px.index.max()) else px.index.max()
    prev = _drift(df, weeks[-2], px, until) if len(weeks) > 1 else {}
    rows = []
    for t in sorted(set(target) | set(prev)):
        a, b = prev.get(t, 0.0), target.get(t, 0.0)
        d = b - a
        if a <= 1e-9 and b > 1e-9:
            act = "buy"
        elif b <= 1e-9 and a > 1e-9:
            act = "sell"
        elif abs(d) < min_move:
            act = "hold"
        else:
            act = "add" if d > 0 else "trim"
        last = px[t].dropna().iloc[-1] if (t in px.columns and px[t].notna().any()) else float("nan")
        rows.append({"ticker": t, "action": act, "from_w": round(a, 4), "to_w": round(b, 4), "change": round(d, 4),
                     "px": None if not np.isfinite(last) else round(float(last), 2)})
    order = {"sell": 0, "buy": 1, "trim": 2, "add": 3, "hold": 4}
    rows.sort(key=lambda r: (order[r["action"]], -abs(r["change"])))
    turn = float(sum(abs(r["change"]) for r in rows))
    return {"first": not prev, "rows": rows, "turnover": turn, "est_cost_pct": turn * C.COST_BPS / 1e4 * 100,
            "n_buy": sum(r["action"] == "buy" for r in rows), "n_sell": sum(r["action"] == "sell" for r in rows),
            "n_adjust": sum(r["action"] in ("add", "trim") for r in rows),
            "n_hold": sum(r["action"] == "hold" for r in rows), "buy_on": "the next trading day's close"
            if cur["entry"] is None or cur["entry"] > px.index.max() else "the latest close"}


# ---------------------------------------------------------------- the dashboard view
def _curve(ledger, open_pos, tail=60):
    if not len(ledger):
        return []
    r = ledger.sort_values("exit_date")
    pts = [{"date": r["entry_date"].iloc[0].strftime("%Y-%m-%d"), "paper": 100.0, "basket": 100.0, "spx": 100.0}]
    p = b = s = 100.0
    for _, x in r.iterrows():
        p *= 1 + x["ret_net"]
        b *= 1 + (0.0 if not np.isfinite(x["basket"]) else x["basket"])
        s *= 1 + (0.0 if not np.isfinite(x["spx"]) else x["spx"])
        pts.append({"date": x["exit_date"].strftime("%Y-%m-%d"), "paper": round(p, 3), "basket": round(b, 3), "spx": round(s, 3)})
    mk = (open_pos or {}).get("mark")
    if mk and mk.get("since") != mk.get("through"):
        pts.append({"date": mk["through"], "paper": round(p * (1 + mk["ret_pct"] / 100), 3),
                    "basket": round(b * (1 + mk["basket_pct"] / 100), 3),
                    "spx": round(s * (1 + mk["spx_pct"] / 100), 3), "open": True})
    return pts[-tail:]


def view(ledger, open_pos, min_weeks):
    n = int(len(ledger))
    out = {"weeks": n, "min_weeks": int(min_weeks), "open": open_pos}
    if n == 0:
        out.update({"curve": [], "recent": [], "verdict": "too early", "total_pct": None})
        return out
    r = ledger.sort_values("exit_date").reset_index(drop=True)
    net, bk, sp = r["ret_net"].astype(float), r["basket"].astype(float).fillna(0.0), r["spx"].astype(float).fillna(0.0)
    eq = (1 + net).cumprod()
    tot, tb, ts = float(eq.iloc[-1] - 1), float((1 + bk).prod() - 1), float((1 + sp).prod() - 1)
    ex = net - bk
    t = None
    if n >= 8 and float(ex.std(ddof=1)) > 0:
        t = float(ex.mean() / (ex.std(ddof=1) / math.sqrt(n)))
    dd = float((eq / eq.cummax() - 1).min())
    i_best, i_worst = int(net.idxmax()), int(net.idxmin())
    if n < min_weeks:
        verdict = "too early"
    elif t is not None and t >= 2 and tot > tb:
        verdict = "positive so far"
    elif t is not None and t <= -2:
        verdict = "negative so far"
    else:
        verdict = "no clear edge"
    out.update({
        "curve": _curve(ledger, open_pos),
        "recent": [{"entry": x["entry_date"].strftime("%Y-%m-%d"), "exit": x["exit_date"].strftime("%Y-%m-%d"),
                    "ret_pct": 100 * float(x["ret_net"]), "basket_pct": 100 * float(x["basket"]) if np.isfinite(x["basket"]) else None,
                    "spx_pct": 100 * float(x["spx"]) if np.isfinite(x["spx"]) else None, "n_held": int(x["n_held"]),
                    "turnover_pct": 100 * float(x["turnover"])} for _, x in r.tail(8).iloc[::-1].iterrows()],
        "total_pct": 100 * tot, "basket_total_pct": 100 * tb, "spx_total_pct": 100 * ts,
        "vs_basket_pts": 100 * (tot - tb), "vs_spx_pts": 100 * (tot - ts),
        "weeks_beat_basket": int((net > bk).sum()), "max_drawdown_pct": 100 * dd,
        "best": {"ret_pct": 100 * float(net[i_best]), "exit": r.loc[i_best, "exit_date"].strftime("%Y-%m-%d")},
        "worst": {"ret_pct": 100 * float(net[i_worst]), "exit": r.loc[i_worst, "exit_date"].strftime("%Y-%m-%d")},
        "avg_cash_pct": 100 * float(r["cash"].mean()), "avg_turnover_pct": 100 * float(r["turnover"].mean()),
        "t": t, "verdict": verdict,
        "start": r["entry_date"].iloc[0].strftime("%Y-%m-%d"), "end": r["exit_date"].iloc[-1].strftime("%Y-%m-%d"),
    })
    return out


def _notes(v, min_weeks):
    notes = []
    if v["weeks"] < min_weeks:
        notes.append(f"{v['weeks']} of {min_weeks} weeks closed. A handful of weeks cannot tell skill from luck, "
                     "so no verdict is given yet.")
    if v.get("open") and v["open"].get("n_unpriced"):
        notes.append("Some holdings had no price and were treated as cash.")
    return notes


def update(log_dir, slog, close_all, asof, log=print, name="model", label=None):
    """Close any finished weeks, mark the open one, and return the dashboard block for this paper portfolio.
    Never raises: a problem here must not stop the weekly run."""
    block = {"name": name, "label": label or name, "available": False, "weeks": 0,
             "min_weeks": int(opts.get("PAPER_MIN_WEEKS")), "cost_bps": C.COST_BPS}
    try:
        df = _prep(slog)
        if df is None:
            block["reason"] = "No portfolio has been logged yet."
            return block
        px = close_all.copy()
        px.index = pd.DatetimeIndex(px.index).normalize()
        px = px.sort_index()
        spx_t = C.MARKETS["spx"]
        if spx_t not in px.columns:
            block["reason"] = "No S&P 500 prices to compare against."
            return block
        cal = px[spx_t].dropna().index
        spx = px[spx_t].ffill()
        px = px.drop(columns=[spx_t]).ffill(limit=10).reindex(cal)
        spx = spx.reindex(cal)
        path = Path(log_dir) / f"paper_{name}.csv"
        weeks = pick_weeks(df, cal, opts.get("PAPER_MIN_GAP_DAYS"))
        ledger = read_ledger(path)
        grown = closed_weeks(ledger, df, weeks, px, spx, C.COST_BPS)
        if len(grown) != len(ledger):
            _write_ledger(path, grown)
            log(f"Paper portfolio '{name}': {len(grown) - len(ledger)} week(s) closed, {len(grown)} in total.")
        ledger = grown
        op = open_position(ledger, df, weeks, px, spx, C.COST_BPS)
        v = view(ledger, op, block["min_weeks"])
        block.update(v)
        block["available"] = True
        block["trades"] = trades(df, weeks, px)
        block["notes"] = _notes(v, block["min_weeks"])
        block["asof"] = pd.Timestamp(asof).strftime("%Y-%m-%d") if asof is not None else None
    except Exception as e:
        log(f"Paper portfolio '{name}' skipped ({type(e).__name__}: {e})")
        block["reason"] = "The paper portfolio could not be updated this time."
    return block
