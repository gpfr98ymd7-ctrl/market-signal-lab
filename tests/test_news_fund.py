"""Tests for the news archive and company-data features: parsing, merging, no lookahead, honest experiments,
and above all that a failing news or data source can never break the weekly run. No network is used."""
import json
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from helpers import RSS, fake_company, fake_gdelt, make_ds, patched, silent
from marketlab import config as C
from marketlab import features, fundamentals, news, report, synthetic
import run_weekly

NOW = pd.Timestamp("2026-10-09 07:00")


# ---------------------------------------------------------------- GDELT parsing and errors
def test_timeline_parsing_is_tolerant():
    obj = {"timeline": [{"series": "Average Tone", "data": [
        {"date": "20261001T000000Z", "value": -2.0},
        {"date": "20261001T120000Z", "value": -4.0},            # hourly buckets average into one day
        {"date": "20261002T000000Z", "value": "nan"},            # not finite: ignored
        {"date": "garbage", "value": 1},
        {"date": "20261003T000000Z", "value": -1.5},
    ]}]}
    s = news.parse_timeline(obj)
    assert list(s.index) == [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-03")]
    assert s.iloc[0] == -3.0 and s.iloc[1] == -1.5
    assert news.parse_timeline({}).empty and news.parse_timeline(None).empty and news.parse_timeline({"timeline": []}).empty


def test_get_json_retries_slow_down_and_fails_clearly():
    replies = iter([b"Please limit requests to one every 5 seconds.", b'{"ok": 1}'])
    assert news.get_json("u", lambda url: next(replies), wait=0) == {"ok": 1}
    assert news.get_json("u", lambda url: b"  ", wait=0) == {}                       # empty reply = no matches
    for bad in (lambda url: b"Your query was too short", lambda url: (_ for _ in ()).throw(ConnectionError("x"))):
        try:
            news.get_json("u", bad, wait=0)
            assert False, "should have raised"
        except news.NewsError:
            pass
    state = {"n": 0}

    def flaky(url):
        state["n"] += 1
        if state["n"] == 1:
            raise HTTPError(url, 429, "slow down", {}, None)
        return b'{"a": 2}'
    assert news.get_json("u", flaky, wait=0) == {"a": 2}


def test_articles_are_cleaned_and_unsafe_links_dropped():
    arts = news.parse_articles({"articles": [
        {"title": "  A   headline ", "url": "https://x.com/1", "domain": "x.com", "seendate": "20261001T000000Z"},
        {"title": "a headline", "url": "https://x.com/dup"},
        {"title": "Bad link", "url": "javascript:alert(1)"},
        {"title": ""},
    ]})
    assert [a["title"] for a in arts] == ["A headline", "Bad link"]
    assert arts[0]["url"] == "https://x.com/1" and arts[1]["url"] is None


# ---------------------------------------------------------------- headline archive
def test_tone_lexicon_and_archive_dedupes(tmp_path):
    assert news.tone_of("Shares surge as profits beat forecasts") > 0
    assert news.tone_of("Markets plunge on recession fears") < 0
    assert news.tone_of("Committee meets on Tuesday") == 0
    path = tmp_path / "news_headlines.csv"
    items = [{"title": "Stocks surge", "source": "BBC", "link": "https://a", "published": pd.Timestamp("2026-10-08 09:00")},
             {"title": "Stocks surge", "source": "BBC", "link": "https://a", "published": None},
             {"title": "Stocks surge", "source": "Guardian", "link": "https://b", "published": "2026-10-08 10:00"},
             {"title": "   ", "source": "BBC"}]
    arch, added = news.archive_headlines(path, items, NOW)
    assert added == 2 and len(arch) == 2
    arch2, added2 = news.archive_headlines(path, items, NOW + pd.Timedelta(days=7))
    assert added2 == 0 and len(arch2) == 2
    view = news.archive_view(arch2)
    assert view["n"] == 2 and view["since"] == "2026-10-09" and view["tone_7"] > 0


# ---------------------------------------------------------------- daily series: backfill, merge, failure
def test_daily_backfill_then_live_window_and_merge(tmp_path):
    path = tmp_path / "news_daily.csv"
    calls = []
    daily, st = news.update_daily(path, NOW, fetcher=fake_gdelt("2026-10-09", calls), sleep=0, log=silent)
    assert st["ok"] and st["span"] == "96months"
    assert calls[0]["timespan"] == "96months"
    assert daily.index.max() == pd.Timestamp("2026-10-08")                 # today is incomplete: left out
    assert len(daily) > 2500 and {"global_tone", "global_vol", "spx_tone", "n225_vol"} <= set(daily.columns)
    assert (tmp_path / "news_meta.json").exists()

    calls.clear()                                                          # a week later: only the live window
    later = NOW + pd.Timedelta(days=7)
    daily2, st2 = news.update_daily(path, later, fetcher=fake_gdelt("2026-10-16", calls, tone_shift=1.0), sleep=0, log=silent)
    assert st2["span"] == "3months" and all(c["timespan"] == "3months" for c in calls)
    assert daily2.index.max() == pd.Timestamp("2026-10-15") and len(daily2) > len(daily)
    old_day, new_day = pd.Timestamp("2025-01-15"), pd.Timestamp("2026-10-01")
    assert daily2.loc[old_day, "global_tone"] == daily.loc[old_day, "global_tone"]          # history is kept
    assert abs(daily2.loc[new_day, "global_tone"] - daily.loc[new_day, "global_tone"] - 1.0) < 1e-3   # recent days refreshed


def test_long_window_failure_falls_back_and_is_not_retried_every_run(tmp_path):
    path = tmp_path / "news_daily.csv"
    calls = []
    long_fails = lambda qs: qs["timespan"] != "3months"
    daily, st = news.update_daily(path, NOW, fetcher=fake_gdelt("2026-10-09", calls, fail=long_fails), sleep=0, log=silent)
    assert st["ok"] and st["span"] == "3months" and 80 < len(daily) < 100
    assert [c["timespan"] for c in calls[:5]] == ["96months", "96months", "48months", "48months", "24months"] or \
        calls[0]["timespan"] == "96months"
    calls.clear()
    news.update_daily(path, NOW + pd.Timedelta(days=7), fetcher=fake_gdelt("2026-10-16", calls), sleep=0, log=silent)
    assert calls[0]["timespan"] == "3months"                                # not retried within a month


def test_a_dead_news_source_keeps_the_archive_and_never_raises(tmp_path):
    path = tmp_path / "news_daily.csv"
    news.update_daily(path, NOW, fetcher=fake_gdelt("2026-10-09"), sleep=0, log=silent)
    before = path.read_text()
    daily, st = news.update_daily(path, NOW + pd.Timedelta(days=7), fetcher=fake_gdelt(fail=lambda qs: True), sleep=0, log=silent)
    assert not st["ok"] and st["errors"] and path.read_text() == before and len(daily) > 100

    # three failures in a row stop the loop early instead of hammering the source
    calls = []
    fail_after_probe = lambda qs: len(calls) > 3
    news.update_daily(tmp_path / "x.csv", NOW, fetcher=fake_gdelt("2026-10-09", calls, fail=fail_after_probe), sleep=0, log=silent)
    assert len(calls) <= 12


def test_news_view_reads_unusual_tone_against_its_own_history():
    idx = pd.date_range("2025-09-01", "2026-10-08")
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"global_tone": -2 + rng.normal(0, 0.3, len(idx)), "global_vol": 1 + rng.normal(0, 0.1, len(idx))}, index=idx)
    calm = news.news_view(d)
    assert calm["available"] and calm["series"][0]["label"] == "normal"
    d.iloc[-7:, 0] -= 2.0
    d.iloc[-7:, 1] += 1.5
    s = news.news_view(d)["series"][0]
    assert s["label"] == "darker than usual" and s["tone_z"] < -1 and s["heavy"]
    assert not news.news_view(pd.DataFrame())["available"]


def test_pick_news_summarises_and_survives_errors():
    out, errs = news.stock_news(["AAPL", "NOPE", "MSFT"], NOW, fetcher=fake_gdelt("2026-10-08"), sleep=0, log=silent)
    assert [o["ticker"] for o in out] == ["AAPL", "MSFT"] and not errs        # "NOPE" has no search phrase
    assert out[0]["label"] in ("similar to last week", "lower than last week", "higher than last week")
    assert len(out[0]["headlines"]) == 3 and [h["url"] is None for h in out[0]["headlines"]] == [False, True, False]
    out2, errs2 = news.stock_news(["AAPL", "MSFT", "NVDA", "JPM"], NOW, fetcher=fake_gdelt(fail=lambda qs: True), sleep=0, log=silent)
    assert out2 == [] and 1 <= len(errs2) <= 3


# ---------------------------------------------------------------- the experiment: honest about signal and noise
def _frames(ds, markets=("spx", "ndx")):
    mac = features.macro_features(ds.macro)
    return {m: features.market_frame(m, ds, mac) for m in markets}


def test_news_features_use_only_the_past():
    idx = pd.date_range("2024-01-01", "2026-06-30")
    rng = np.random.default_rng(5)
    daily = pd.DataFrame({"global_tone": rng.normal(-2, 0.4, len(idx)), "global_vol": rng.normal(1, 0.2, len(idx)),
                          "spx_tone": rng.normal(-2, 0.4, len(idx)), "spx_vol": rng.normal(1, 0.2, len(idx))}, index=idx)
    dates = pd.bdate_range("2025-01-01", "2026-06-30")
    base = news.news_features(daily, "spx", dates)
    cut = pd.Timestamp("2026-03-02")
    changed = daily.copy()
    changed.loc[changed.index >= cut] += 5.0                                   # news from `cut` onward differs
    other = news.news_features(changed, "spx", dates)
    safe = dates <= cut                                                        # a feature on `cut` sees news through cut-1
    pd.testing.assert_frame_equal(base[safe], other[safe])
    assert not base[~safe].equals(other[~safe])


def test_placebo_news_is_the_same_series_moved_and_offsets_are_spread():
    idx = pd.date_range("2020-01-01", "2026-06-30")
    daily = pd.DataFrame({"global_tone": np.arange(len(idx), dtype=float)}, index=idx)
    sh = news.shifted(daily, 500)
    assert sh.index.equals(idx) and sorted(sh["global_tone"]) == sorted(daily["global_tone"])
    assert sh["global_tone"].iloc[500] == daily["global_tone"].iloc[0] and sh["global_tone"].iloc[0] != daily["global_tone"].iloc[0]
    offs = news.placebo_offsets(len(idx), 9)
    assert len(offs) == 9 and min(offs) >= 365 and max(offs) <= len(idx) - 365 and offs == sorted(set(offs))


def test_experiment_finds_real_signal_and_does_not_invent_it():
    ds = make_ds("2026-06-30")
    frames = _frames(ds, ("spx",))
    cal = pd.date_range("2019-01-01", "2026-06-30")
    rng = np.random.default_rng(8)

    # (a) news that carries information about the volatility that follows: the experiment should notice
    f, d, F = frames["spx"]
    fv = d.set_index("date")["fwd_vol"].reindex(cal).ffill().bfill()
    informative = pd.DataFrame({"global_tone": rng.normal(-2, 0.3, len(cal)), "global_vol": np.log(fv.values) + rng.normal(0, 0.05, len(cal)),
                                "spx_tone": rng.normal(-2, 0.3, len(cal)), "spx_vol": np.log(fv.values) + rng.normal(0, 0.05, len(cal))},
                               index=cal)
    res = news.run_experiment(frames, informative, ds.run_date, n_boot=40, min_days=600, placebos=4, log=silent)
    assert res["available"] and res["placebos"] == 4
    vol = res["markets"][0]["heads"]["vol"]
    assert vol["verdict"] == "helps" and vol["b"] < vol["a"] and vol["diff"] > vol["null_hi"] > vol["null_lo"] - 1e-9

    # (b) pure noise, several times over: it must never call the news promising, and rarely call any comparison a help
    helped = 0
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        noise = pd.DataFrame({c: r.normal(0, 1, len(cal)) for c in informative.columns}, index=cal)
        res2 = news.run_experiment(frames, noise, ds.run_date, n_boot=40, min_days=600, placebos=4, log=silent)
        assert res2["available"] and res2["verdict"] == "no evidence it helps", seed
        helped += res2["helps"]
    assert helped <= 3, helped                    # 15 comparisons; chance alone allows about 1 in 5 of them at most


def test_experiment_waits_when_history_is_short():
    ds = make_ds("2026-06-30")
    idx = pd.date_range("2026-03-01", "2026-06-29")
    daily = pd.DataFrame({"global_tone": 0.0, "global_vol": 1.0}, index=idx)
    res = news.run_experiment(_frames(ds, ("spx",)), daily, ds.run_date, log=silent)
    assert not res["available"] and "trading days" in res["reason"]
    assert not news.run_experiment(_frames(ds, ("spx",)), pd.DataFrame(), ds.run_date, log=silent)["available"]


# ---------------------------------------------------------------- company data
def test_earnings_dates_from_every_shape_yahoo_returns():
    today = pd.Timestamp("2026-10-09")
    d = pd.Timestamp("2026-10-14").date()
    assert fundamentals.next_earnings({"Earnings Date": [d]}, {}, today) == pd.Timestamp("2026-10-14")
    cal = pd.DataFrame({0: [pd.Timestamp("2026-10-20")]}, index=["Earnings Date"])
    assert fundamentals.next_earnings(cal, {}, today) == pd.Timestamp("2026-10-20")
    ts = pd.Timestamp("2026-10-30 13:00").timestamp()
    assert fundamentals.next_earnings(None, {"earningsTimestamp": ts}, today) == pd.Timestamp("2026-10-30")
    assert fundamentals.next_earnings({"Earnings Date": [pd.Timestamp("2026-09-01").date()]}, {}, today) is None   # already past
    assert fundamentals.next_earnings("rubbish", {}, today) is None


def _table(n=12):
    q = np.linspace(0.05, 0.95, n)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)], "trailingPE": 10 + 40 * (1 - q), "forwardPE": 9 + 30 * (1 - q),
        "returnOnEquity": 0.05 + 0.3 * q, "operatingMargins": 0.08 + 0.25 * q, "profitMargins": 0.05 + 0.2 * q,
        "revenueGrowth": -0.05 + 0.2 * q, "earningsGrowth": -0.1 + 0.4 * q, "debtToEquity": 200 * (1 - q) + 10,
        "freeCashflow": 1e9 * (0.2 + q), "marketCap": 5e10, "sector": "Technology", "priceToBook": 3.0})


def test_score_ranks_cheap_quality_growth_first_and_handles_awkward_data():
    df = _table()
    sc = fundamentals.score_snapshot(df)
    assert sc["score"].is_monotonic_increasing and sc["score"].iloc[-1] > 90 and sc["score"].iloc[0] < 10

    loss = df.copy()
    loss.loc[3, "trailingPE"] = -15.0                                          # loss-maker: worst earnings yield
    assert fundamentals.score_snapshot(loss).loc[3, "value"] < sc.loc[3, "value"]

    bank_a, bank_b = df.copy(), df.copy()
    bank_a["sector"] = bank_b["sector"] = "Financial Services"
    bank_b.loc[5, "debtToEquity"] = 5000.0                                     # banks are leveraged: leverage is ignored
    qa, qb = fundamentals.score_snapshot(bank_a).loc[5, "quality"], fundamentals.score_snapshot(bank_b).loc[5, "quality"]
    assert qa == qb

    sparse = df.copy()
    sparse.loc[:, ["trailingPE", "forwardPE", "freeCashflow"]] = np.nan        # no value data at all
    s2 = fundamentals.score_snapshot(sparse)
    assert s2["value"].isna().all() and s2["score"].notna().all()              # still scored on quality and growth
    assert fundamentals.score_snapshot(df.head(5))["score"].isna().all()       # too few names to rank


def test_yahoo_quirks_are_guarded():
    base = {"trailingPE": 12.0, "forwardPE": 11.0, "profitMargins": 0.1, "returnOnEquity": 0.2,
            "freeCashflow": 5e9, "marketCap": 1e11, "currency": "GBp", "financialCurrency": "USD"}
    assert np.isnan(fundamentals.clean_info(base)["freeCashflow"])                      # trades in pence, reports in dollars
    assert fundamentals.clean_info({**base, "financialCurrency": "GBP"})["freeCashflow"] == 5e9   # pence and pounds agree
    assert fundamentals.clean_info({**base, "currency": "", "financialCurrency": ""})["freeCashflow"] == 5e9   # unknown: keep
    assert np.isnan(fundamentals.clean_info({**base, "trailingPE": 5000.0})["trailingPE"])
    assert fundamentals.clean_info({**base, "trailingPE": -20.0})["trailingPE"] == -20.0   # a loss is real information
    df = _table(12)
    df["freeCashflow"] = df["marketCap"] * np.linspace(0.01, 0.2, 12)
    ok = fundamentals.score_snapshot(df)["value"]
    df.loc[5, "freeCashflow"] = df.loc[5, "marketCap"] * 80                              # a 100x scale error
    bad = fundamentals.score_snapshot(df)
    assert bad["value"].notna().all() and abs(bad.loc[5, "value"] - ok[5]) < 40         # not allowed to dominate the ranking


def test_fetching_is_polite_and_partial_results_are_kept():
    calls = []
    ok = fake_company(calls=calls)
    tickers = [f"T{i}" for i in range(10)]

    def refuses_after_three(t):
        if len(calls) >= 3:
            calls.append(t)
            raise ConnectionError("429")
        return ok(t)
    rows, errs = fundamentals.fetch_rows(tickers, "2026-10-09", set(), refuses_after_three, 0, 60, NOW, log=silent)
    assert len(rows) == 3 and any("stopped after 6" in e for e in errs) and len(calls) == 3 + 6
    calls.clear()
    rows, errs = fundamentals.fetch_rows(tickers, "2026-10-09", {"T0", "T1"}, ok, 0, 60, NOW, log=silent)
    assert len(rows) == 8 and "T0" not in calls
    rows, errs = fundamentals.fetch_rows(tickers, "2026-10-09", set(), ok, 0, -1, NOW, log=silent)
    assert rows == [] and "time budget reached" in errs
    thin = lambda t: {"info": {"trailingPE": 12.0}}
    assert fundamentals.fetch_rows(["A"], "2026-10-09", set(), thin, 0, 60, NOW, log=silent)[0] == []


def test_snapshots_accumulate_without_overwriting_and_fall_back_when_yahoo_is_down(tmp_path):
    ds = make_ds("2026-06-30")
    now = pd.Timestamp("2026-07-01 07:00")
    calls = []
    b1 = fundamentals.update(tmp_path, ds, now, fetcher=fake_company("2026-07-01", calls), pause=0, log=silent)
    assert b1["available"] and b1["fresh"] and b1["n"] == len(ds.stocks) == len(calls)
    t0 = ds.stocks[0]
    assert b1["by_ticker"][t0]["score"] is not None and b1["scorecard"]["verdict"] == "too early"

    n_calls = len(calls)
    b2 = fundamentals.update(tmp_path, ds, now, fetcher=fake_company("2026-07-01", calls), pause=0, log=silent)
    assert len(calls) == n_calls and b2["n"] == b1["n"]                        # same date: nothing is fetched twice
    assert len(fundamentals.read_snapshots(tmp_path / "fundamentals.csv")) == b1["n"]

    ds2 = make_ds("2026-07-08")
    b3 = fundamentals.update(tmp_path, ds2, pd.Timestamp("2026-07-09 07:00"), fetcher=fake_company(dead=True), pause=0, log=silent)
    assert b3["available"] and not b3["fresh"] and b3["as_of"] == "2026-06-30" and b3["errors"]   # last good snapshot, labelled
    b4 = fundamentals.update(tmp_path, ds2, pd.Timestamp("2026-08-15 07:00"), fetcher=fake_company(dead=True), pause=0, log=silent)
    assert not b4["available"] and "out of date" in b4["reason"]

    empty = fundamentals.update(tmp_path / "fresh", ds, now, fetcher=fake_company(dead=True), pause=0, log=silent)
    assert not empty["available"]


def test_scorecard_detects_a_real_link_waits_when_early_and_does_not_invent_one():
    rng = np.random.default_rng(4)
    n_t = 20
    q = np.linspace(0.05, 0.95, n_t)
    days = pd.bdate_range("2024-01-01", "2026-06-30")

    def prices(drift_scale):
        r = rng.normal(0, 0.012, (len(days), n_t)) + drift_scale * (q - 0.5)
        return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=days, columns=[f"T{i}" for i in range(n_t)])

    base = _table(n_t)
    fridays = pd.date_range("2024-03-01", "2026-06-26", freq="W-FRI")
    snaps = pd.concat([base.assign(snap_date=d) for d in fridays], ignore_index=True)
    snaps["shortName"] = ""
    snaps["next_earnings"] = pd.NaT

    good = fundamentals.scorecard(snaps, prices(0.002), horizon=21, min_weeks=24)
    assert good["weeks"] > 90 and good["mean_ic"] > 0.15 and good["verdict"] == "positive so far" and good["spread_pct"] > 0

    flat = fundamentals.scorecard(snaps, prices(0.0), horizon=21, min_weeks=24)
    assert flat["verdict"] != "positive so far" or flat["t"] < 3

    early = fundamentals.scorecard(snaps[snaps["snap_date"] <= "2024-05-31"], prices(0.002), horizon=21, min_weeks=24)
    assert early["verdict"] == "too early"
    recent = fundamentals.scorecard(snaps[snaps["snap_date"] >= "2026-06-12"], prices(0.0), horizon=21, min_weeks=24)
    assert recent["weeks"] == 0 and recent["pending"] > 0                      # not enough days have passed yet


# ---------------------------------------------------------------- notes
def test_notes_are_neutral_and_capped():
    stocks = [{"ticker": "AAA", "base_w": 0.5, "fund": {"earnings_soon": True, "earnings_days": 2}, "news": {"label": "lower than last week"}},
              {"ticker": "BBB", "base_w": 0.0, "fund": {"earnings_soon": True, "earnings_days": 1}}]
    nw = {"series": [{"name": "Nikkei 225", "label": "darker than usual", "heavy": True}]}
    notes = report.build_notes(stocks, nw, {})
    text = " ".join(notes)
    assert "AAA (in 2d)" in text and "BBB" not in text                          # only the picks are flagged
    assert "Nikkei 225" in text and "AAA is lower" in text.replace("for AAA", "AAA") or "lower than last week" in text
    assert report.build_notes([], None, None) == []


# ---------------------------------------------------------------- the whole weekly run with both features
def _small():
    return dict(STOCKS=C.STOCKS[:14], MIN_TRAIN_YEARS_MARKET=3, MIN_TRAIN_YEARS_STOCK=2, HEALTH_REFRESH_DAYS=10 ** 6,
                NEWS_MIN_DAYS_EXPERIMENT=10 ** 6)


def test_pipeline_with_news_and_company_data(tmp_path):
    with patched(**_small()):
        def go(**kw):
            return run_weekly.run(tmp_path, downloader=synthetic.make_downloader(end="2026-10-08"),
                                  now="2026-10-09 07:00", sleep=0, log=silent, use_outlook=False,
                                  headlines_fetcher=lambda url: RSS,
                                  news_fetcher=fake_gdelt("2026-10-09"), fund_fetcher=fake_company("2026-10-09"), **kw)
        p = go()
        n, f = p["news"], p["fundamentals"]
        assert n["available"] and n["series"] and n["archive"]["n"] > 0 and n["experiment"]["available"] is False
        assert "picks" not in n and "by_ticker" not in f and f["available"]
        picks = [s for s in p["stocks"] if s["base_w"] > 0]
        assert all("fund" in s for s in p["stocks"]) and all("news" in s for s in picks)
        assert any(h["url"] is None for s in picks for h in s["news"]["headlines"])           # unsafe link was dropped
        for name in ("news_daily.csv", "news_headlines.csv", "fundamentals.csv", "news_experiment.json"):
            assert (tmp_path / "data" / name).exists(), name

        before = {n_: (tmp_path / "data" / n_).read_text() for n_ in ("news_headlines.csv", "fundamentals.csv")}
        go()                                                                                   # same data date again
        assert {n_: (tmp_path / "data" / n_).read_text() for n_ in before} == before

        saved = json.loads((tmp_path / "docs" / "data" / "latest.json").read_text(),
                           parse_constant=lambda c: (_ for _ in ()).throw(ValueError("NaN in JSON")))
        assert saved["schema"] == report.SCHEMA and saved["news"]["available"] and saved["fundamentals"]["available"]
        assert isinstance(saved["notes"], list)


def test_pipeline_survives_dead_news_and_company_sources(tmp_path):
    with patched(**_small()):
        def broken_headlines(url):
            raise ConnectionError("down")
        p = run_weekly.run(tmp_path, downloader=synthetic.make_downloader(end="2026-10-08"), now="2026-10-09 07:00",
                           sleep=0, log=silent, use_outlook=False, headlines_fetcher=broken_headlines,
                           news_fetcher=fake_gdelt(fail=lambda qs: True), fund_fetcher=fake_company(dead=True))
        assert p["news"]["available"] is False and p["news"]["errors"]
        assert p["fundamentals"]["available"] is False
        assert all("fund" not in s for s in p["stocks"])
        assert (tmp_path / "docs" / "data" / "latest.json").exists()                           # the forecast was still published

        def explodes(*a, **k):
            raise RuntimeError("bug in an optional step")
        from marketlab import news as news_mod, fundamentals as fund_mod
        old = (news_mod.update, fund_mod.update)
        news_mod.update, fund_mod.update = explodes, explodes
        try:
            p2 = run_weekly.run(tmp_path / "b", downloader=synthetic.make_downloader(end="2026-10-08"), now="2026-10-09 07:00",
                                sleep=0, log=silent, use_outlook=False, use_headlines=False)
        finally:
            news_mod.update, fund_mod.update = old
        assert p2["news"] is None and p2["fundamentals"] is None and len(p2["markets"]) == len(C.MARKETS)

        p3 = run_weekly.run(tmp_path / "c", downloader=synthetic.make_downloader(end="2026-10-08"), now="2026-10-09 07:00",
                            sleep=0, log=silent, use_outlook=False, use_headlines=False, use_news=False, use_fundamentals=False)
        assert p3["news"] is None and p3["fundamentals"] is None


def test_old_settings_file_without_the_new_options_still_works():
    from marketlab import opts
    saved = {k: getattr(C, k) for k in ("NEWS_ENABLED", "FUND_ENABLED", "FUND_HORIZON") if hasattr(C, k)}
    try:
        for k in saved:
            delattr(C, k)
        assert opts.get("NEWS_ENABLED") is True and opts.get("FUND_HORIZON") == 21
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
