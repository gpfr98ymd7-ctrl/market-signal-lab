"""Build the dashboard payload, validate it, and produce the phone-notification text."""
import math

import numpy as np
import pandas as pd

from . import config as C

SCHEMA = 2


# ---------------------------------------------------------------- health, in plain labels
def health_view(h):
    if not h:
        return {"available": False}
    auc, lam = h.get("auc_up"), h.get("lam_up") or 0.0
    direction = "none" if (auc is None or auc < 0.52 or lam < 0.1) else ("weak" if auc < 0.55 else "some")
    vi = h.get("vol_improve_pct")
    vol = "none" if vi is None or vi < 3 else ("modest" if vi < 10 else "useful")
    ac = h.get("auc_crash")
    crash = "none" if ac is None or ac < 0.55 else ("modest" if ac < 0.65 else "useful")
    keep = ("auc_up", "lam_up", "auc_crash", "lam_crash", "vol_improve_pct", "k_range", "cov90_raw", "cov90_cal",
            "cov50_raw", "cov50_cal", "first_test", "last_test", "n_days", "honest", "overlay")
    return {"available": True, "direction": direction, "vol": vol, "crash": crash, **{k: h.get(k) for k in keep}}


def stock_health_view(h):
    if not h:
        return None
    auc, lam = h.get("auc"), h.get("lam") or 0.0
    label = "none" if (auc is None or auc < 0.52 or lam < 0.1) else ("weak" if auc < 0.54 else "some")
    keep = ("auc", "years_above_half", "n_years", "spread_pct", "spread_t", "lam", "first_test", "last_test", "backtest")
    return {"label": label, **{k: h.get(k) for k in keep}}


# ---------------------------------------------------------------- checks before anything is saved
def validate_forecasts(mdash, ranking):
    """Raise before logging if this week's numbers are not sane, so a bad run never pollutes the log."""
    problems = []
    if len(mdash) != len(C.MARKETS):
        problems.append(f"expected {len(C.MARKETS)} markets, got {len(mdash)}")
    for _, r in mdash.iterrows():
        for c in ("p_up", "p_crash", "pred_vol", "vol_now", "q05", "q50", "q95", "exposure"):
            if not np.isfinite(r[c]):
                problems.append(f"{r['market']}: {c} is not a number")
        if not (0 <= r["p_up"] <= 1 and 0 <= r["p_crash"] <= 1):
            problems.append(f"{r['market']}: probability outside 0-1")
        if not (0.25 - 1e-9 <= r["exposure"] <= 1 + 1e-9):
            problems.append(f"{r['market']}: exposure outside 0.25-1")
        if not (r["q05"] <= r["q50"] <= r["q95"]):
            problems.append(f"{r['market']}: return range is out of order")
    if len(ranking) < C.TOP_K:
        problems.append("too few stocks ranked")
    else:
        if abs(float(ranking["base_w"].sum()) - 1.0) > 1e-6:
            problems.append("portfolio weights do not sum to 1")
        if float(ranking["final_w"].sum()) > 1 + 1e-9:
            problems.append("scaled weights exceed 100%")
        if not ranking["p"].between(0, 1).all() or not np.isfinite(ranking["p_raw"]).all():
            problems.append("stock probabilities invalid")
    if problems:
        raise ValueError("Forecast checks failed, nothing was saved: " + "; ".join(problems))


def validate_payload(p):
    if p.get("schema") != SCHEMA:
        raise ValueError("payload schema mismatch")
    if len(p.get("markets", [])) != len(C.MARKETS):
        raise ValueError("payload is missing markets")
    if len(p.get("stocks", [])) < C.TOP_K:
        raise ValueError("payload has too few stocks")
    for m in p["markets"]:
        for k in ("p_up", "p_crash", "pred_vol", "exposure"):
            v = m.get(k)
            if v is None or not math.isfinite(v):
                raise ValueError(f"{m.get('name')}: {k} is missing or not finite")
        for k in ("p5", "p50", "p95"):
            v = m["range_pct"].get(k)
            if v is None or not math.isfinite(v):
                raise ValueError(f"{m.get('name')}: range {k} is missing or not finite")
    for k in ("generated_utc", "data_through", "headline"):
        if not p.get(k):
            raise ValueError(f"payload field {k} is empty")


# ---------------------------------------------------------------- week-on-week
def changes_since_last(mdash, ranking, mlog, slog, run_date):
    empty = {"previous_run": None, "markets": [], "entered": [], "exited": [], "turnover": None, "est_cost_pct": None}
    before = mlog.loc[mlog["run_date"] < run_date, "run_date"]
    if before.empty:
        return empty
    prev = before.max()
    pm = mlog[mlog["run_date"] == prev].set_index("key")
    rows = []
    for _, r in mdash.iterrows():
        if r["key"] in pm.index and pd.notna(pm.loc[r["key"], "exposure"]):
            p = pm.loc[r["key"]]
            rows.append({"key": r["key"], "name": r["market"], "exposure_from": float(p["exposure"]),
                         "exposure_delta": float(r["exposure"] - p["exposure"]),
                         "p_crash_delta": float(r["p_crash"] - p["p_crash"]) if pd.notna(p["p_crash"]) else None,
                         "regime_from": p["regime"], "regime_to": r["regime"]})
    ps = slog[slog["run_date"] == prev]
    prev_port = set(ps.loc[ps["base_w"] > 0, "ticker"])
    cur_port = set(ranking.loc[ranking["base_w"] > 0, "ticker"])
    prev_w = ps.set_index("ticker")["final_w"].fillna(0.0)
    cur_w = ranking.set_index("ticker")["final_w"]
    turnover = float(sum(abs(float(cur_w.get(t, 0.0)) - float(prev_w.get(t, 0.0))) for t in set(prev_w.index) | set(cur_w.index)))
    return {"previous_run": prev.strftime("%Y-%m-%d"), "markets": rows,
            "entered": sorted(cur_port - prev_port), "exited": sorted(prev_port - cur_port),
            "turnover": turnover, "est_cost_pct": turnover * C.COST_BPS / 1e4 * 100}


def market_history(mlog, key, n=26):
    g = mlog[mlog["key"] == key].sort_values("run_date").tail(n)
    return [{"d": r["run_date"].strftime("%Y-%m-%d"), "p_up": r["p_up"], "p_crash": r["p_crash"],
             "exposure": r["exposure"], "pred_vol": r["pred_vol"]} for _, r in g.iterrows()]


# ---------------------------------------------------------------- alerts and headline
def build_alerts(mdash, changes, hviews):
    ch = {c["key"]: c for c in changes["markets"]}
    out = []
    for _, r in mdash.iterrows():
        c = ch.get(r["key"])
        if c and c["exposure_delta"] <= -0.2:
            out.append({"level": "alert", "text": f"{r['market']}: suggested exposure fell from {c['exposure_from']:.0%} to {r['exposure']:.0%}."})
        if r["regime"] == "Downtrend / stressed" and not (c and c["regime_from"] == r["regime"]):
            out.append({"level": "alert", "text": f"{r['market']} moved into a stressed downtrend."})
        if r["pred_vol"] >= 0.30:
            out.append({"level": "alert", "text": f"{r['market']}: expected volatility is {r['pred_vol']:.0%} a year."})
        hv = hviews.get(r["key"]) or {}
        if hv.get("crash") in ("modest", "useful") and r["p_crash"] > 1.5 * r["base_crash"]:
            out.append({"level": "alert", "text": f"{r['market']}: crash risk {r['p_crash']:.0%} is well above its normal {r['base_crash']:.0%}."})
    return out


def headline(mdash, ranking, hviews):
    avg = float(mdash["exposure"].mean())
    low = mdash.loc[mdash["exposure"].idxmin()]
    parts = [f"The risk scaler suggests {avg:.0%} exposure on average across the {len(mdash)} markets, lowest in {low['market']} ({low['exposure']:.0%})."]
    stressed = mdash.loc[mdash["regime"].str.contains("stressed"), "market"].tolist()
    parts.append(("Stressed conditions in " + ", ".join(stressed) + ".") if stressed else "No market is in a stressed regime.")
    picks = ranking.loc[ranking["base_w"] > 0, "ticker"].tolist()
    if picks:
        parts.append("Top stock picks: " + ", ".join(picks) + ".")
    avail = [h for h in hviews.values() if h.get("available")]
    if avail and sum(1 for h in avail if h["direction"] == "none") > len(avail) / 2:
        parts.append("Direction forecasts have shown no reliable edge historically, so the odds shown sit close to base rates. Lean on the volatility and range information instead.")
    return " ".join(parts)


# ---------------------------------------------------------------- payload
def build_payload(ds, mdash, ranking, stock_meta, health, scorecard, mlog, slog, run_date, outlook=None, demo=False,
                  headlines=None):
    mh = (health or {}).get("markets") or {}
    hviews = {m: health_view(mh.get(m)) for m in mdash["key"]}
    changes = changes_since_last(mdash, ranking, mlog, slog, run_date)
    alerts = build_alerts(mdash, changes, hviews)

    markets = []
    for _, r in mdash.iterrows():
        markets.append({
            "key": r["key"], "name": r["market"], "as_of": r["as_of"], "regime": r["regime"],
            "p_up": r["p_up"], "p_up_raw": r["p_up_raw"], "base_up": r["base_up"],
            "p_crash": r["p_crash"], "p_crash_raw": r["p_crash_raw"], "base_crash": r["base_crash"],
            "pred_vol": r["pred_vol"], "vol_now": r["vol_now"],
            "range_pct": {k: 100 * (np.exp(r[q]) - 1) for k, q in
                          (("p5", "q05"), ("p25", "q25"), ("p50", "q50"), ("p75", "q75"), ("p95", "q95"))},
            "exposure": r["exposure"], "crash_mult": r["crash_mult"], "vol_mult": r["vol_mult"],
            "regime_cap": r["regime_cap"], "health": hviews[r["key"]], "history": market_history(mlog, r["key"]),
            "px_hist": [round(float(x), 4) for x in ds.idx[r["key"]].dropna().tail(126)],
        })
    stocks = [{"ticker": r["ticker"], "p": r["p"], "p_raw": r["p_raw"], "p_rank": r["p_rank"], "flag": r["flag"],
               "region": r["region"], "ret_20": r["ret_20"], "rs_20": r["rs_20"], "vol_ann": r["vol_ann"],
               "mkt_mult": r["mkt_mult"], "base_w": r["base_w"], "final_w": r["final_w"]}
              for _, r in ranking.iterrows()]
    invested = float(ranking["final_w"].sum())

    payload = {
        "schema": SCHEMA, "version": C.VERSION, "demo": bool(demo),
        "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
        "data_through": ds.run_date,
        "horizons": {"market_days": C.H, "stock_days": C.SH, "entry_delay": C.ENTRY_DELAY},
        "headline": headline(mdash, ranking, hviews),
        "alerts": alerts,
        "data_warnings": list(ds.warnings),
        "changes": changes,
        "markets": markets,
        "stocks": stocks,
        "stock_model": {"base": stock_meta["base"], "lam": stock_meta["lam"],
                        "health": stock_health_view((health or {}).get("stocks"))},
        "portfolio": {"top_k": C.TOP_K, "invested": invested, "cash": 1 - invested,
                      "turnover": changes["turnover"], "est_cost_pct": changes["est_cost_pct"], "cost_bps": C.COST_BPS},
        "scorecard": scorecard,
        "health": {"updated": (health or {}).get("updated"), "available": bool(health and health.get("markets"))},
        "outlook": outlook,
        "headlines": list(headlines or []),
    }
    return payload


# ---------------------------------------------------------------- text for the notification and the AI prompt
def plain_text(payload):
    lines = []
    for m in payload["markets"]:
        lines.append(f"{m['name']}: regime {m['regime']}; P(up) {m['p_up']:.0%} (base {m['base_up']:.0%}); "
                     f"crash risk {m['p_crash']:.0%} (normal {m['base_crash']:.0%}); expected volatility {m['pred_vol']:.0%} "
                     f"(now {m['vol_now']:.0%}); 90% range {m['range_pct']['p5']:+.1f}% to {m['range_pct']['p95']:+.1f}%; "
                     f"suggested exposure x{m['exposure']:.2f}; historic direction skill: {m['health'].get('direction', 'unknown')}")
    picks = [s for s in payload["stocks"] if s["base_w"] > 0]
    lines.append("Top stock picks (weight after market-risk scaling): " +
                 ", ".join(f"{s['ticker']} {s['final_w']:.0%}" for s in picks) +
                 f"; cash {payload['portfolio']['cash']:.0%}")
    sh = (payload.get("stock_model") or {}).get("health")
    if sh:
        lines.append(f"Stock ranker historic skill: {sh['label']} (AUC {sh['auc']:.3f})")
    return "\n".join(lines)


def notification(payload):
    """(title, body, priority) for the optional ntfy phone alert."""
    alerts = payload["alerts"]
    title = "Market Lab: " + (f"{len(alerts)} alert(s)" if alerts else "weekly run")
    body = payload["headline"]
    if alerts:
        body += "\n\n" + "\n".join("- " + a["text"] for a in alerts[:4])
    if payload["data_warnings"]:
        body += f"\n\n{len(payload['data_warnings'])} data warning(s): see the dashboard."
    return title, body[:900], ("high" if alerts else "default")
