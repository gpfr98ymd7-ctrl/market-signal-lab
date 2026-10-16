"""News archive and news context.

None of this changes a forecast. It does four things:

1. Headline archive. Every headline from the free RSS feeds is saved (de-duplicated) with a crude word-count
   tone, so a text history builds up week by week.
2. Daily tone and attention series from GDELT (free, no key): the average tone of English-language coverage and
   its share of all coverage, for global markets and each index. They are merged into data/news_daily.csv every run,
   so the history keeps growing even though GDELT's live window is limited.
3. A news check for the week's top picks: tone this week against last week, and the latest headlines.
4. An experiment that tests, walk-forward and on identical rows, whether adding the news series to the market
   models would improve them. The result is only displayed; the forecasts do not use news.

Everything fails soft: if GDELT does not answer, the run carries on and the dashboard says so.
"""
import json
import math
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import models, opts, util


class NewsError(RuntimeError):
    """The news source did not give a usable answer."""


_DEADLINE = [None]          # wall-clock time after which no further news requests are made in this run


# ---------------------------------------------------------------- crude headline tone
POS = {"gain", "gains", "rise", "rises", "rising", "rally", "rallies", "surge", "surges", "jump", "jumps", "record",
       "beat", "beats", "growth", "profit", "profits", "upgrade", "upgrades", "recovery", "rebound", "rebounds",
       "boost", "boosts", "strong", "optimism", "soar", "soars", "climb", "climbs", "cool", "cools", "agreement",
       "deal", "approval", "approves", "wins", "relief"}
NEG = {"fall", "falls", "falling", "drop", "drops", "plunge", "plunges", "slump", "slumps", "crash", "crashes", "loss",
       "losses", "cut", "cuts", "downgrade", "downgrades", "fear", "fears", "recession", "war", "crisis", "default",
       "bankruptcy", "inflation", "tariff", "tariffs", "sanctions", "warns", "warning", "weak", "layoffs", "probe",
       "slides", "tumble", "tumbles", "shock", "attack", "strike", "strikes", "collapse", "debt", "worries"}


def tone_of(title):
    """Word-count tone in [-1, 1]: (positive - negative) / (positive + negative). Crude, and shown as such."""
    words = re.findall(r"[a-z']+", str(title).lower())
    p = sum(w in POS for w in words)
    n = sum(w in NEG for w in words)
    return round((p - n) / (p + n), 2) if (p + n) else 0.0


# ---------------------------------------------------------------- headline archive
ARCHIVE_COLS = ["first_seen", "published", "source", "title", "link", "tone"]


def read_archive(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ARCHIVE_COLS)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for c in ARCHIVE_COLS:
            if c not in df.columns:
                df[c] = ""
        return df[ARCHIVE_COLS]
    except Exception:
        return pd.DataFrame(columns=ARCHIVE_COLS)


def archive_headlines(path, items, now):
    """Append headlines not seen before. Returns (archive, number_added). Existing rows are never changed."""
    path = Path(path)
    old = read_archive(path)
    seen = set(zip(old["source"].str.lower(), old["title"].str.lower()))
    stamp = pd.Timestamp(now).strftime("%Y-%m-%d")
    rows = []
    for it in items or []:
        title = re.sub(r"\s+", " ", str(it.get("title") or "")).strip()[:200]
        if not title:
            continue
        key = (str(it.get("source") or "").lower(), title.lower())
        if key in seen:
            continue
        seen.add(key)
        pub = it.get("published")
        pub = pub.strftime("%Y-%m-%d %H:%M") if isinstance(pub, pd.Timestamp) else (str(pub) if pub else "")
        rows.append({"first_seen": stamp, "published": pub, "source": str(it.get("source") or ""),
                     "title": title, "link": it.get("link") or "", "tone": f"{tone_of(title):.2f}"})
    new = pd.concat([old, pd.DataFrame(rows, columns=ARCHIVE_COLS)], ignore_index=True) if rows else old
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        new.to_csv(tmp, index=False)
        tmp.replace(path)
    return new, len(rows)


def archive_view(arch):
    if arch is None or len(arch) == 0:
        return {"n": 0, "since": None, "tone_7": None}
    seen = pd.to_datetime(arch["first_seen"], errors="coerce")
    tone = pd.to_numeric(arch["tone"], errors="coerce")
    last = seen.max()
    recent = tone[seen >= last - pd.Timedelta(days=7)]
    return {"n": int(len(arch)), "since": seen.min().strftime("%Y-%m-%d"),
            "tone_7": float(recent.mean()) if recent.notna().any() else None}


# ---------------------------------------------------------------- GDELT
def _http_get(url, timeout=30):
    req = Request(url, headers={"User-Agent": "market-lab/2.1 (personal research tool)"})
    with urlopen(req, timeout=timeout) as r:
        return r.read(8_000_000)


def gdelt_url(query, mode, span=None, extra=None):
    params = {"query": f"{query} sourcelang:english", "mode": mode, "format": "json"}
    if span:
        params["timespan"] = span
    if extra:
        params.update(extra)
    return opts.get("GDELT_URL") + "?" + urlencode(params, quote_via=quote)


def get_json(url, fetcher=None, retries=1, wait=15.0):
    """Fetch and decode one GDELT reply. An empty reply means 'no matches'. Raises NewsError on anything else."""
    fetcher = fetcher or _http_get
    last = "no answer"
    for attempt in range(retries + 1):
        if _DEADLINE[0] is not None and time.time() > _DEADLINE[0]:
            raise NewsError("time budget reached")
        retry = False
        try:
            raw = fetcher(url)
        except HTTPError as e:
            last = f"HTTP {e.code}"
            retry = e.code in (429, 500, 502, 503, 504)
        except Exception as e:                      # timeouts, DNS, blocked
            last = type(e).__name__
            retry = True
        else:
            text = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)).strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except ValueError:
                last = "not JSON: " + re.sub(r"\s+", " ", text)[:90]
                retry = "limit" in text.lower()       # GDELT asks callers to slow down in plain text
        if not retry or attempt == retries:
            break
        time.sleep(wait)
    raise NewsError(last)


def parse_timeline(obj):
    """A GDELT timeline reply to a daily pandas Series (date -> value). Empty if there is no data."""
    tl = (obj or {}).get("timeline") if isinstance(obj, dict) else None
    rows = ((tl or [{}])[0] or {}).get("data") or []
    dates, vals = [], []
    for r in rows:
        try:
            d = pd.Timestamp(str(r["date"])[:8])
            v = float(r["value"])
        except Exception:
            continue
        if math.isfinite(v):
            dates.append(d)
            vals.append(v)
    if not vals:
        return pd.Series(dtype=float)
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), dtype=float)
    return s.groupby(level=0).mean().sort_index()           # hourly buckets become daily


def parse_articles(obj, limit=3):
    out, seen = [], set()
    for a in ((obj or {}).get("articles") or []) if isinstance(obj, dict) else []:
        title = re.sub(r"\s+", " ", str(a.get("title") or "")).strip()[:140]
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        url = str(a.get("url") or "").strip()
        out.append({"title": title, "url": url if re.match(r"^https?://", url, re.I) else None,
                    "domain": str(a.get("domain") or "")[:60], "seen": str(a.get("seendate") or "")[:8]})
        if len(out) >= limit:
            break
    return out


def series_keys():
    return ["global"] + list(C.MARKETS)


def read_daily(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col="date", parse_dates=["date"])
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def _write_daily(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    out = df.copy()
    out.index.name = "date"
    out.round(4).to_csv(tmp)
    tmp.replace(path)


def update_daily(path, now, fetcher=None, sleep=5.0, log=print, keys=None):
    """Fetch the daily tone and attention series, merge them into the archive and save. Returns (daily, status)."""
    path = Path(path)
    meta_path = path.with_name("news_meta.json")
    meta = util.read_json(meta_path) or {}
    old = read_daily(path)
    now = pd.Timestamp(now).normalize()
    queries = opts.get("NEWS_QUERIES")
    keys = [k for k in (keys or series_keys()) if k in queries]
    status = {"ok": False, "span": None, "requests": 0, "errors": []}
    wait = max(3 * sleep, 0.0)

    def pause():
        if sleep > 0:
            time.sleep(sleep)

    def fetch(key, mode, span):
        status["requests"] += 1
        return parse_timeline(get_json(gdelt_url(queries[key], mode, span), fetcher, wait=wait))

    # Which window to ask for? A long one while the archive is short (at most once a month), else the live window.
    have = int(old["global_tone"].notna().sum()) if "global_tone" in old.columns else 0
    spans = ["3months"]
    try:
        last_try = pd.Timestamp(meta.get("long_backfill_tried")) if meta.get("long_backfill_tried") else None
    except Exception:
        last_try = None
    retry_days = opts.get("NEWS_BACKFILL_RETRY_DAYS")
    if have < 700 and (last_try is None or (now - last_try).days >= retry_days):
        spans = list(opts.get("NEWS_BACKFILL_SPANS")) + spans
        meta["long_backfill_tried"] = now.strftime("%Y-%m-%d")

    first = {}
    chosen = None
    for span in spans:
        try:
            s = fetch("global", "timelinetone", span)
        except NewsError as e:
            status["errors"].append(f"{span}: {e}")
            log(f"News: window {span} failed ({e})")
            pause()
            continue
        if len(s) >= 20:
            chosen, first["global_tone"] = span, s
            break
        status["errors"].append(f"{span}: only {len(s)} days returned")
        pause()
    if chosen is None:
        util.write_json_atomic(meta_path, meta)
        log("News: GDELT gave no usable series; keeping the archive as it is.")
        return old, status
    status["span"] = chosen
    if chosen != "3months":
        meta["long_backfill_ok"] = chosen
    util.write_json_atomic(meta_path, meta)
    log(f"News: using the {chosen} window.")

    cols = {"global_tone": first["global_tone"]}
    consecutive = 0
    for key in keys:
        for mode, suffix in (("timelinetone", "tone"), ("timelinevol", "vol")):
            name = f"{key}_{suffix}"
            if name in cols:
                continue
            pause()
            try:
                s = fetch(key, mode, chosen)
                consecutive = 0
                if len(s):
                    cols[name] = s
            except NewsError as e:
                consecutive += 1
                status["errors"].append(f"{name}: {e}")
                log(f"News: {name} unavailable ({e})")
                if consecutive >= 3:
                    log("News: three failures in a row, stopping early.")
                    break
        if consecutive >= 3:
            break

    new = pd.DataFrame(cols)
    new.index = pd.DatetimeIndex(new.index)
    merged = new.combine_first(old) if len(old) else new
    merged = merged[merged.index < now].sort_index()           # today is incomplete: leave it out
    for key in keys:                                          # a day with no coverage has no tone
        t, v = f"{key}_tone", f"{key}_vol"
        if t in merged.columns and v in merged.columns:
            merged.loc[merged[v] <= 0, t] = np.nan
    merged = merged.loc[:, [c for c in merged.columns if merged[c].notna().any()]]
    _write_daily(path, merged)
    status["ok"] = "global_tone" in merged.columns
    return merged, status


# ---------------------------------------------------------------- views for the dashboard
def _z_latest(s, window=7, base_days=365, min_base=40):
    """Latest rolling mean, and how unusual it is against the same series' own earlier history."""
    s = s.dropna()
    if len(s) < 14:
        return None, None
    r = s.rolling(window, min_periods=max(3, window // 2)).mean().dropna()
    if r.empty:
        return None, None
    latest = float(r.iloc[-1])
    base = r[r.index <= r.index[-1] - pd.Timedelta(days=window)].tail(base_days)
    if len(base) < min_base or not base.std() > 0:
        return latest, None
    return latest, float((latest - base.mean()) / base.std())


def tone_label(z):
    if z is None:
        return None
    return "darker than usual" if z <= -1.0 else ("brighter than usual" if z >= 1.0 else "normal")


def news_view(daily):
    if daily is None or len(daily) == 0 or "global_tone" not in daily.columns:
        return {"available": False}
    last = daily.index.max()
    out = []
    names = {"global": "Global markets", **C.NAMES}
    for key in series_keys():
        t = daily.get(f"{key}_tone")
        if t is None or t.dropna().empty:
            continue
        tone7, tz = _z_latest(t)
        v = daily.get(f"{key}_vol")
        vol7, vz = _z_latest(v) if v is not None else (None, None)
        sp = t.dropna().rolling(3, min_periods=1).mean().tail(60)
        out.append({"key": key, "name": names.get(key, key), "tone_7": tone7, "tone_z": tz, "label": tone_label(tz),
                    "vol_z": vz, "heavy": bool(vz is not None and vz >= 1.5),
                    "spark": [round(float(x), 2) for x in sp.values]})
    return {"available": bool(out), "source": "GDELT", "as_of": last.strftime("%Y-%m-%d"),
            "days": int(daily["global_tone"].notna().sum()), "first": daily.index.min().strftime("%Y-%m-%d"),
            "series": out}


# ---------------------------------------------------------------- news check for the top picks
def stock_news(tickers, now, fetcher=None, sleep=5.0, log=print):
    queries = opts.get("NEWS_STOCK_QUERIES")
    now = pd.Timestamp(now).normalize()
    wait = max(3 * sleep, 0.0)
    out, errors, consecutive = [], [], 0
    for t in tickers:
        q = queries.get(t)
        if not q:
            continue
        try:
            if sleep > 0:
                time.sleep(sleep)
            ts = parse_timeline(get_json(gdelt_url(q, "timelinetone", "14days"), fetcher, wait=wait))
            if sleep > 0:
                time.sleep(sleep)
            arts = parse_articles(get_json(gdelt_url(q, "artlist", "7days", {"maxrecords": "10", "sort": "datedesc"}),
                                           fetcher, wait=wait), limit=3)
            consecutive = 0
        except NewsError as e:
            consecutive += 1
            errors.append(f"{t}: {e}")
            log(f"News check {t} unavailable ({e})")
            if consecutive >= 3:
                break
            continue
        ts = ts[ts.index < now]
        recent = ts[ts.index >= now - pd.Timedelta(days=7)]
        prev = ts[(ts.index < now - pd.Timedelta(days=7)) & (ts.index >= now - pd.Timedelta(days=14))]
        t7 = float(recent.mean()) if len(recent) >= 3 else None
        tp = float(prev.mean()) if len(prev) >= 3 else None
        delta = (t7 - tp) if (t7 is not None and tp is not None) else None
        label = None if delta is None else ("lower than last week" if delta <= -0.75 else
                                            "higher than last week" if delta >= 0.75 else "similar to last week")
        out.append({"ticker": t, "tone_7": t7, "tone_prev": tp, "delta": delta, "label": label, "headlines": arts})
    return out, errors


# ---------------------------------------------------------------- experiment: would news help the models?
def news_features(daily, key, dates):
    """Past-only news features for one market, one row per date in `dates`. A feature on date t uses news through t-1."""
    cal = pd.date_range(daily.index.min(), max(daily.index.max(), pd.Timestamp(dates.max())))
    d = daily.reindex(cal)
    out = pd.DataFrame(index=cal)
    for k in ("global", key):
        t, v = d.get(f"{k}_tone"), d.get(f"{k}_vol")
        if t is None or v is None:
            continue
        t7 = t.rolling(7, min_periods=4).mean()
        v7 = v.rolling(7, min_periods=4).mean()
        out[f"nw_{k}_tone_z"] = (t7 - t7.expanding(90).mean()) / t7.expanding(90).std()
        out[f"nw_{k}_tone_chg"] = t7 - t.rolling(28, min_periods=14).mean()
        out[f"nw_{k}_vol_z"] = (v7 - v7.expanding(90).mean()) / v7.expanding(90).std()
    out = out.replace([np.inf, -np.inf], np.nan).shift(1)
    return out.reindex(pd.DatetimeIndex(dates), method="ffill")


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _gain(y, pa, pb, kind):
    """How much better model b is than model a (positive = b better). Volatility: drop in error. Others: gain in AUC."""
    if kind == "vol":
        return _rmse(y, pa) - _rmse(y, pb)
    if len(np.unique(y)) < 2:
        return float("nan")
    return _auc(y, pb) - _auc(y, pa)


def _boot_interval(y, pa, pb, kind, n_boot, rng, block=21):
    """The gain from adding the news features, with a block-bootstrap 90% interval over the test rows."""
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    n = len(y)
    est = _gain(y, pa, pb, kind)
    nb = int(np.ceil(n / block))
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), nb)
        ix = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        v = _gain(y[ix], pa[ix], pb[ix], kind)
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < 20 or not np.isfinite(est):
        return None
    lo, hi = np.percentile(draws, [5, 95])
    return float(est), float(lo), float(hi)


def shifted(daily, offset):
    """The same news series moved `offset` days along the calendar, wrapping round. It keeps the series' own
    texture but cuts its link to the market's dates: a stand-in for 'news that means nothing'."""
    cal = pd.date_range(daily.index.min(), daily.index.max())
    d = daily.reindex(cal)
    return pd.DataFrame(np.roll(d.values, int(offset), axis=0), index=cal, columns=d.columns)


def placebo_offsets(n_days, k):
    lo = min(365, max(n_days // 4, 30))
    return [int(x) for x in np.linspace(lo, n_days - lo, k + 2)[1:-1]]


def run_experiment(frames, daily, run_date, n_boot=300, seed=0, min_days=None, placebos=None, budget=None, log=print):
    """Would adding the news features have improved the market models, walk-forward, on identical rows?

    Adding any extra columns changes a model's score a little by luck, and a bootstrap over test rows cannot see that
    (it only measures test-sample noise). So the real news is compared with several placebos: the same series moved
    to unrelated dates. A comparison only counts as 'helps' if the gain is clearly above zero AND bigger than every
    placebo's gain, which makes a lucky win from random columns roughly a 1-in-(placebos+1) event or rarer."""
    min_days = min_days or opts.get("NEWS_MIN_DAYS_EXPERIMENT")
    k = max(int(placebos if placebos is not None else opts.get("NEWS_PLACEBOS")), 4)
    budget = opts.get("NEWS_EXPERIMENT_BUDGET") if budget is None else budget
    if daily is None or len(daily) == 0 or "global_tone" not in daily.columns:
        return {"available": False, "reason": "No news history yet.", "have_days": 0, "need_days": int(min_days)}
    t0 = time.time()
    rng = np.random.default_rng(seed)
    offsets = placebo_offsets((daily.index.max() - daily.index.min()).days + 1, k)
    results, tested, skipped = [], [], []
    have_rows = 0
    for m, (f, d, F) in frames.items():
        if time.time() - t0 > budget:
            skipped.append(C.NAMES[m])
            continue
        nf = news_features(daily, m, d["date"])
        N = list(nf.columns)
        if not N:
            continue
        nf.index = d.index
        s = d.join(nf).dropna(subset=N)
        have_rows = max(have_rows, len(s))
        if len(s) < min_days:
            continue
        pcols = []
        for i, off in enumerate(offsets):                       # placebo columns, on the same rows
            pf = news_features(shifted(daily, off), m, d["date"]).reindex(columns=N)
            pf.index = d.index
            pf.columns = [f"p{i}_{c}" for c in N]
            s = s.join(pf)
            pcols.append(list(pf.columns))
        tags = {"a": list(F), "b": list(F) + N, **{f"p{i}": list(F) + pc for i, pc in enumerate(pcols)}}
        rows = {"date": [], "y_up": [], "y_crash": [], "lv": []}
        for tag in tags:
            for h in ("up", "cr", "lv"):
                rows[f"{h}_{tag}"] = []
        for y in sorted(s["date"].dt.year.unique()):
            cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=2 * C.H)
            train, test = s[s["date"] < cutoff], s[s["date"].dt.year == y]
            if len(train) < 500 or len(test) < 100:
                continue
            lvt = np.log(np.clip(train["fwd_vol"], 1e-4, None))
            for tag, cols in tags.items():
                up = models.fit_classifier(train[cols], train["y_up"], models.MARKET_PARAMS)
                cr = models.fit_classifier(train[cols], train["y_crash"], models.MARKET_PARAMS)
                vr = models.lgb.LGBMRegressor(**models.MARKET_PARAMS).fit(train[cols], lvt)
                rows[f"up_{tag}"] += list(up.predict_proba(test[cols])[:, 1])
                rows[f"cr_{tag}"] += list(cr.predict_proba(test[cols])[:, 1])
                rows[f"lv_{tag}"] += list(vr.predict(test[cols]))
            rows["date"] += list(test["date"])
            rows["y_up"] += list(test["y_up"])
            rows["y_crash"] += list(test["y_crash"])
            rows["lv"] += list(np.log(np.clip(test["fwd_vol"], 1e-4, None)))
        if len(rows["date"]) < 250:
            continue
        order = np.argsort(np.asarray(rows["date"], dtype="datetime64[ns]"))
        g = {key: np.asarray(v)[order] for key, v in rows.items()}

        def head(h, ykey, kind):
            y = g[ykey]
            base, real = g[f"{h}_a"], g[f"{h}_b"]
            out = {"a": _auc(y, base) if kind == "auc" else _rmse(y, base),
                   "b": _auc(y, real) if kind == "auc" else _rmse(y, real),
                   "diff": None, "lo": None, "hi": None, "null_lo": None, "null_hi": None,
                   "verdict": "no clear difference"}
            null = [_gain(y, base, g[f"{h}_p{i}"], kind) for i in range(k)]
            null = [v for v in null if np.isfinite(v)]
            iv = _boot_interval(y, base, real, kind, n_boot, rng)
            if iv is not None and len(null) >= 3:
                est, lo, hi = iv
                out.update({"diff": est, "lo": lo, "hi": hi, "null_lo": float(min(null)), "null_hi": float(max(null))})
                if lo > 0 and est > max(null):
                    out["verdict"] = "helps"
            return out

        heads = {"up": head("up", "y_up", "auc"), "crash": head("cr", "y_crash", "auc"), "vol": head("lv", "lv", "vol")}
        results.append({"key": m, "name": C.NAMES[m], "n_days": int(len(g["date"])), "heads": heads})
        tested.append((pd.Timestamp(g["date"].min()), pd.Timestamp(g["date"].max())))
        log(f"News experiment {C.NAMES[m]}: " + ", ".join(f"{key} {v['verdict']}" for key, v in heads.items()))

    if not results:
        need_more = max(int(min_days) - have_rows, 0)
        if skipped and not have_rows:
            reason = "The experiment ran out of time this run and will try again next time."
        else:
            reason = (f"Needs about {int(min_days)} trading days of news history; {have_rows} are usable so far" +
                      (f", roughly {int(np.ceil(need_more / 0.69 / 30))} more months of archiving." if need_more else "."))
        return {"available": False, "have_days": int(have_rows), "need_days": int(min_days),
                "first": daily.index.min().strftime("%Y-%m-%d"), "reason": reason}
    helps = sum(h["verdict"] == "helps" for r in results for h in r["heads"].values())
    total = 3 * len(results)
    expected = round(total / (k + 1), 1)                        # at most this many would beat every placebo by luck
    promising = helps >= max(3, int(np.ceil(3 * expected)))
    return {
        "available": True, "computed": pd.Timestamp(run_date).strftime("%Y-%m-%d"),
        "first": min(t[0] for t in tested).strftime("%Y-%m-%d"), "last": max(t[1] for t in tested).strftime("%Y-%m-%d"),
        "markets": results, "helps": int(helps), "comparisons": int(total), "placebos": int(k),
        "expected_by_chance": expected, "skipped": skipped,
        "verdict": "promising" if promising else "no evidence it helps",
    }


EXPERIMENT_DESIGN = 2        # bump when the experiment changes, so an old cached result is not reused


def experiment_cached(cache_path, frames, daily, run_date, force=False, log=print):
    cache_path = Path(cache_path)
    cached = util.read_json(cache_path) or {}
    res = cached.get("result")
    fresh = False
    try:
        age = (pd.Timestamp(run_date) - pd.Timestamp(cached.get("computed"))).days
        fresh = bool(res and res.get("available") and cached.get("design") == EXPERIMENT_DESIGN
                     and age < C.HEALTH_REFRESH_DAYS)
    except Exception:
        fresh = False
    if fresh and not force:
        return res
    res = run_experiment(frames, daily, run_date, log=log)
    util.write_json_atomic(cache_path, {"computed": pd.Timestamp(run_date).strftime("%Y-%m-%d"),
                                        "design": EXPERIMENT_DESIGN, "result": res})
    return res


# ---------------------------------------------------------------- one call for the weekly run
def update(log_dir, now, picks, frames, headline_items=None, fetcher=None, sleep=5.0, force_experiment=False,
           pick_check=None, log=print):
    """Everything news-related for one run. Returns the dashboard block. Never raises for network problems."""
    log_dir = Path(log_dir)
    now = pd.Timestamp(now)
    _DEADLINE[0] = time.time() + opts.get("NEWS_TIME_BUDGET")
    try:
        return _update(log_dir, now, picks, frames, headline_items, fetcher, sleep, force_experiment, pick_check, log)
    finally:
        _DEADLINE[0] = None


def _update(log_dir, now, picks, frames, headline_items, fetcher, sleep, force_experiment, pick_check, log):
    arch, added = archive_headlines(log_dir / "news_headlines.csv", headline_items, now)
    block = {"available": False, "source": "GDELT", "archive": archive_view(arch), "series": [], "picks": [],
             "experiment": None, "errors": []}
    if added:
        log(f"News archive: {added} new headline(s), {len(arch)} in total.")

    daily, status = update_daily(log_dir / "news_daily.csv", now, fetcher=fetcher, sleep=sleep, log=log)
    block["errors"] += status["errors"][:4]
    block.update({k: v for k, v in news_view(daily).items() if k != "available"})
    block["available"] = bool(block["series"])

    if pick_check is None:
        pick_check = opts.get("NEWS_PICK_CHECK")
    if pick_check and picks and status["ok"]:
        block["picks"], errs = stock_news(picks, now, fetcher=fetcher, sleep=sleep, log=log)
        block["errors"] += errs[:3]

    try:
        block["experiment"] = experiment_cached(log_dir / "news_experiment.json", frames, daily, now,
                                                force=force_experiment, log=log)
    except Exception as e:                                     # an experiment problem must never stop the run
        log(f"News experiment skipped ({type(e).__name__}: {e})")
        block["experiment"] = {"available": False, "reason": "The experiment could not run this time."}
    return block
