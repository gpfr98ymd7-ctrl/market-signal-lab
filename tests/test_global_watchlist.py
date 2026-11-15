"""Tests for the watchlist, the global drivers (no lookahead, planted sensitivities, honest decisions) and the whole
weekly pipeline with a long watchlist, the top-20 growth portfolio and both paper ledgers."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from helpers import fake_company, fake_gdelt, make_ds, patched, silent
from marketlab import config as C
from marketlab import (data, features, forecast, growth, health, macro, news, opts, paper, report, synthetic,
                       universe)
import run_weekly

SMALL = dict(MIN_TRAIN_YEARS_MARKET=3, MIN_TRAIN_YEARS_STOCK=2, HEALTH_REFRESH_DAYS=10 ** 6)


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    """These tests check wiring and decisions, not model quality, so use far smaller models to keep the run short."""
    from marketlab import models
    for d in (models.MARKET_PARAMS, models.STOCK_PARAMS):
        monkeypatch.setitem(d, "n_estimators", 25)
        monkeypatch.setitem(d, "learning_rate", 0.1)


# ---------------------------------------------------------------- the watchlist
def test_parse_handles_comments_separators_share_classes_and_junk():
    t, notes = universe.parse("aapl, msft  # a comment\nBRK.B;BARC.L\n7203.T\n  \nbad!!\nAAPL\n")
    assert t == ["AAPL", "MSFT", "BRK-B", "BARC.L", "7203.T"]
    assert any("BRK.B was read as BRK-B" in n for n in notes) and any("Ignored" in n for n in notes)
    assert any("duplicate" in n for n in notes)
    assert universe.parse("") == ([], []) and universe.parse(None)[0] == []


def test_london_and_tokyo_tickers_are_not_mistaken_for_share_classes():
    assert universe.parse("BA.L RR.L 6758.T")[0] == ["BA.L", "RR.L", "6758.T"]
    assert universe.parse("BF.A LEN.B")[0] == ["BF-A", "LEN-B"]


def test_regions_follow_the_ticker_suffix_and_use_restores_the_list():
    assert [universe.region_of(t) for t in ("AAPL", "BARC.L", "SAP.DE", "ASML.AS", "NESN.SW", "7203.T", "RY.TO")] == \
        ["spx", "ftse", "sx5e", "sx5e", "sx5e", "n225", "spx"]
    before = list(C.STOCKS)
    with universe.use(["A", "B"]):
        assert C.STOCKS == ["A", "B"]
    assert C.STOCKS == before
    try:
        with universe.use(["A"]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert C.STOCKS == before


def test_resolve_falls_back_safely(tmp_path):
    default = ["X%d" % i for i in range(20)]
    r = universe.resolve(tmp_path, default)
    assert r.source == "built-in list" and r.tickers == default and r.notes == []
    (tmp_path / "watchlist.txt").write_text("AAPL\nMSFT\n")                              # too short
    r = universe.resolve(tmp_path, default)
    assert r.source == "built-in list" and any("at least" in n for n in r.notes)
    (tmp_path / "watchlist.txt").write_bytes(b"\xff\xfe\x00junk")                          # unreadable text
    assert universe.resolve(tmp_path, default).source == "built-in list"
    (tmp_path / "watchlist.txt").write_text("\n".join(f"T{i}" for i in range(universe.MAX_STOCKS + 10)))
    r = universe.resolve(tmp_path, default)
    assert r.source == "watchlist.txt" and len(r.tickers) == universe.MAX_STOCKS and any("only the first" in n for n in r.notes)
    (tmp_path / "watchlist.txt").write_text("\n".join(f"T{i}" for i in range(30)) + "\nRY.TO\n")
    r = universe.resolve(tmp_path, default)
    assert any("No matching market index" in n and "RY.TO" in n for n in r.notes)


def test_list_changed_compares_with_the_list_the_health_check_used():
    d = ["A", "B", "C", "D", "E"]
    assert universe.list_changed({"universe": d}, d, d) is False
    assert universe.list_changed({"universe": d}, d + ["F"], d) is False                      # one stock added: 5 of 6 shared
    assert universe.list_changed({"universe": d}, d + ["F", "G"], d) is True                  # 5 of 7 shared is a big change
    assert universe.list_changed({}, d, d) is False
    assert universe.list_changed({"markets": {}}, ["Z"], d) is True                            # an old file means "the built-in list"


def test_shipped_watchlist_is_valid_and_wide_enough_for_a_top_20():
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    text = (root / "watchlist.txt").read_text()
    tickers, notes = universe.parse(text)
    assert len(tickers) >= 80 and len(set(tickers)) == len(tickers) and notes == []
    assert universe.MIN_STOCKS <= len(tickers) <= universe.MAX_STOCKS
    assert {"AAPL", "NVDA", "AZN.L", "ASML.AS", "7203.T"} <= set(tickers)
    assert universe.resolve(root, C.STOCKS).source == "watchlist.txt"


def test_company_name_search_phrases():
    assert news.name_query("NVIDIA Corporation") == '("NVIDIA")'
    assert news.name_query("ASML Holding N.V.") == '("ASML")' or news.name_query("ASML Holding N.V.") == '("ASML Holding")'
    assert news.name_query("A") is None and news.name_query(None) is None and news.name_query("") is None
    calls = []
    out, errs = news.stock_news(["ZZZ", "AAPL", "NOQUERY"], pd.Timestamp("2026-10-09"), fetcher=fake_gdelt(calls=calls), sleep=0,
                                log=silent, extra_queries={"ZZZ": '("Zed Corp")', "AAPL": '("not used")'})
    queried = {c["query"] for c in calls}
    assert any("Zed Corp" in q for q in queried)                                               # a name phrase is used when there is no hand-written one
    assert any("Apple Inc" in q for q in queried) and not any("not used" in q for q in queried)  # the hand-written phrase wins
    assert {o["ticker"] for o in out} == {"ZZZ", "AAPL"}                                       # NOQUERY has no phrase at all


# ---------------------------------------------------------------- global drivers
def _small_ds(end="2026-06-30", n=14):
    with patched(STOCKS=C.STOCKS[:n]):
        return make_ds(end)


def test_extra_inputs_do_not_use_the_future():
    full, short = _small_ds("2026-06-30"), _small_ds("2025-06-30")
    cut = pd.Timestamp("2025-06-30")
    ef, es = macro.build_ext(full), macro.build_ext(short)
    assert ef is not None and es is not None
    f1, _, F1 = features.market_frame("spx", full, features.macro_features(full.macro), ef)
    f2, _, F2 = features.market_frame("spx", short, features.macro_features(short.macro), es)
    assert F1 == F2 and any(macro.is_extra(c) for c in F1)
    a, b = f1[f1["date"] <= cut].set_index("date"), f2[f2["date"] <= cut].set_index("date")
    common = a.index.intersection(b.index)
    assert len(common) > 1000
    xcols = [c for c in F1 if macro.is_extra(c)]
    pd.testing.assert_frame_equal(a.loc[common, xcols], b.loc[common, xcols], rtol=1e-9, atol=1e-12)
    s1, _, SF1 = features.stock_panel(full, ef)
    s2, _, SF2 = features.stock_panel(short, es)
    assert SF1 == SF2 and any(macro.is_extra(c) for c in SF1)
    xs = [c for c in SF1 if macro.is_extra(c)]
    a2 = s1[s1["date"] <= cut].set_index(["date", "ticker"])[xs]
    b2 = s2[s2["date"] <= cut].set_index(["date", "ticker"])[xs]
    common2 = a2.index.intersection(b2.index)
    assert len(common2) > 5000
    pd.testing.assert_frame_equal(a2.loc[common2], b2.loc[common2], rtol=1e-9, atol=1e-12)


def test_extra_inputs_add_columns_but_never_change_rows_or_original_columns():
    ds = _small_ds()
    ext = macro.build_ext(ds)
    mac = features.macro_features(ds.macro)
    f0, d0, F0 = features.market_frame("spx", ds, mac)
    f1, d1, F1 = features.market_frame("spx", ds, mac, ext)
    assert macro.base_features(F1) == F0 and len(F1) > len(F0)
    assert len(f0) == len(f1) and len(d0) == len(d1) and (f0["date"].values == f1["date"].values).all()
    pd.testing.assert_frame_equal(f0[F0].reset_index(drop=True), f1[F0].reset_index(drop=True))
    s0, sd0, SF0 = features.stock_panel(ds)
    s1, sd1, SF1 = features.stock_panel(ds, ext)
    assert macro.base_features(SF1) == SF0 and len(s0) == len(s1) and len(sd0) == len(sd1)
    pd.testing.assert_frame_equal(s0[SF0].reset_index(drop=True), s1[SF0].reset_index(drop=True))


def _planted(n=900, seed=4):
    """Three stocks: one that rises with oil shocks, one that falls with them, one unrelated."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2022-01-03", periods=n)
    oil = 70 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    macro_df = pd.DataFrame({"us10y": 3 + np.cumsum(rng.normal(0, 0.02, n)), "dxy": 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))),
                             "oil": oil, "hyg": 80 * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
                             "lqd": 110 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), "vix": 18 * np.exp(np.cumsum(rng.normal(0, 0.03, n)) * 0.2),
                             "gold": 1800 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), "copper": 4 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                             "us3m": np.full(n, 2.0)}, index=cal)
    spx = pd.Series(4000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=cal)
    ds0 = SimpleNamespace(close=pd.DataFrame(index=cal), idx_ff={"spx": spx}, macro=macro_df, extra=pd.DataFrame(index=cal))
    z = macro.factor_shocks(ds0)["oil"].fillna(0.0)
    spx_r = spx.pct_change().fillna(0.0)
    close = pd.DataFrame({
        "UP": 100 * np.cumprod(1 + spx_r + 0.010 * z + rng.normal(0, 0.004, n)),
        "DOWN": 100 * np.cumprod(1 + spx_r - 0.010 * z + rng.normal(0, 0.004, n)),
        "FLAT": 100 * np.cumprod(1 + spx_r + rng.normal(0, 0.004, n))}, index=cal)
    ds0.close = close
    return ds0, z


def test_stock_sensitivity_recovers_a_planted_link_to_oil_and_the_push_has_the_right_sign():
    ds, z = _planted()
    ex = macro.exposures(ds)
    b = ex["beta"]["oil"].iloc[-1]
    assert abs(b["UP"] - 0.010) < 0.003 and abs(b["DOWN"] + 0.010) < 0.003 and abs(b["FLAT"]) < 0.003
    move = z.rolling(macro.MOVE).sum().iloc[-1]
    push = ex["push"]["oil"].iloc[-1]
    assert np.sign(push["UP"]) == np.sign(move) and np.sign(push["DOWN"]) == -np.sign(move) and abs(push["FLAT"]) < abs(push["UP"])
    lp = macro.latest_push(macro.Ext(date=pd.DataFrame(), beta=ex["beta"], push=ex["push"]), ["UP", "DOWN", "FLAT", "NOPE"], ds.close.index[-1])
    assert list(lp.columns) == list(ex["push"].keys()) and abs(lp.loc["UP", "oil"] - 100 * push["UP"]) < 1e-9
    assert pd.isna(lp.loc["NOPE", "oil"]) if "NOPE" in lp.index else True
    assert macro.latest_push(None, ["UP"], ds.close.index[-1]) is None


def test_readings_flag_unusual_moves_and_survive_missing_extras():
    ds = _small_ds()
    v = macro.drivers_view(ds)
    assert v["available"] and v["n"] >= 15 and {r["group"] for r in v["rows"]} >= {"Interest rates", "Fear", "Currencies", "Commodities"}
    assert all(r["state"] in ("normal", "unusually high", "unusually low") and np.isfinite(r["change"]) for r in v["rows"])
    assert all(r["level"] is None for r in v["rows"] if r["window"].startswith("60 days"))
    hot = SimpleNamespace(**{**ds.__dict__})
    hot.macro = ds.macro.copy()
    hot.macro.iloc[-5:, hot.macro.columns.get_loc("vix")] = hot.macro["vix"].iloc[-6] * 3.0
    v2 = macro.drivers_view(hot)
    vix = {r["key"]: r for r in v2["rows"]}["vix"]
    assert vix["state"] == "unusually high" and any("VIX" in n for n in v2["notes"])
    bare = SimpleNamespace(**{**ds.__dict__})
    bare.extra = pd.DataFrame(index=ds.macro.index)
    v3 = macro.drivers_view(bare)
    assert v3["available"] and v3["n"] < v["n"] and macro.build_ext(bare) is None


def test_missing_global_drivers_are_skipped_with_one_plain_warning():
    real = synthetic.make_downloader(end="2026-06-30")

    def dl(tickers, start):
        out = real(tickers, start)
        return out.drop(columns=[c for c in out.columns if c[1] in ("BZ=F", "EEM")])

    with patched(STOCKS=C.STOCKS[:14]):
        close, vol = data.fetch(data.all_tickers(), C.START, downloader=dl, sleep=0, log=silent)
        ds = data.build_dataset(close, vol, pd.Timestamp("2026-07-01 07:00"), log=silent)
    assert "brent" not in ds.extra.columns and "eem" not in ds.extra.columns and "tlt" in ds.extra.columns
    assert any("Global drivers not available" in w and "brent" in w and "eem" in w for w in ds.warnings)
    ext = macro.build_ext(ds)
    assert "mx_brent_ret_20" not in ext.date.columns and "mx_em_rel_60" not in ext.date.columns and ext.date.shape[1] > 8


def test_global_drivers_can_be_switched_off():
    with patched(STOCKS=C.STOCKS[:14]):
        with patched(MACRO_ENABLED=False):
            assert data.extra_map() == {}
            assert not any(t in data.all_tickers() for t in ("BZ=F", "^FVX"))
        assert "BZ=F" in data.all_tickers()
        assert "ZZZ" in data.all_tickers(also=["ZZZ"]) and data.all_tickers(also=["ZZZ"]).count("ZZZ") == 1


# ---------------------------------------------------------------- do the extra inputs help? (honest decisions)
def _hs(auc, years_gain):
    ys = list(range(2012, 2012 + len(years_gain)))
    return {"auc": auc, "spread_pct": 0.1, "auc_by_year": {str(y): 0.5 + g for y, g in zip(ys, years_gain)}}


def test_stock_decision_needs_a_clear_gain_in_most_years_and_overall():
    base = _hs(0.51, [0.0] * 10)
    good = _hs(0.52, [0.004] * 8 + [-0.001] * 2)
    assert health.decide_stocks(base, good)["use"] is True
    flat = _hs(0.511, [0.001] * 10)
    assert health.decide_stocks(base, flat)["use"] is False                                    # average gain under the bar
    lucky = _hs(0.52, [0.05] * 3 + [-0.002] * 7)
    assert health.decide_stocks(base, lucky)["use"] is False                                   # big in few years only
    worse = _hs(0.50, [-0.01] * 10)
    assert health.decide_stocks(base, worse)["use"] is False
    short = health.decide_stocks(_hs(0.51, [0.0] * 3), _hs(0.53, [0.02] * 3))
    assert short["use"] is False and "Too few" in short["reason"]
    pooled_down = _hs(0.505, [0.004] * 10)                                                      # better in each year, worse pooled
    assert health.decide_stocks(base, pooled_down)["use"] is False


def test_market_decision_needs_two_clear_wins_and_no_losses():
    b = {"auc_up": 0.52, "auc_crash": 0.60, "vol_improve_pct": 10.0}
    assert health.decide_market(b, {"auc_up": 0.53, "auc_crash": 0.61, "vol_improve_pct": 10.2})["use"] is True
    assert health.decide_market(b, {"auc_up": 0.53, "auc_crash": 0.60, "vol_improve_pct": 11.0})["use"] is True
    assert health.decide_market(b, {"auc_up": 0.53, "auc_crash": 0.60, "vol_improve_pct": 10.0})["use"] is False   # one win only
    assert health.decide_market(b, {"auc_up": 0.55, "auc_crash": 0.58, "vol_improve_pct": 13.0})["use"] is False   # a loss
    assert health.decide_market(b, {"auc_up": None, "auc_crash": None, "vol_improve_pct": 12.0})["use"] is False   # ties count as ties


def test_health_check_compares_both_setups_and_respects_the_time_budget(monkeypatch):
    ds = _small_ds("2026-06-30")
    ext = macro.build_ext(ds)
    mac = features.macro_features(ds.macro)
    frames = {m: features.market_frame(m, ds, mac, ext) for m in ("spx", "ftse")}
    panel = features.stock_panel(ds, ext)
    with patched(**SMALL):
        h = health.compute(frames, panel, ds.run_date, log=silent, budget=10 ** 6)
        mt = h["macro_test"]
        assert mt["ran"] and mt["stocks"]["years"] >= 5 and set(mt["markets"]) == {"spx", "ftse"}
        assert h["stocks"]["features"] in ("base", "macro") and all(h["markets"][m]["features"] in ("base", "macro") for m in frames)
        assert h["format"] == health.FORMAT and set(h["universe"]) == set(ds.stocks)
        assert (h["stocks"]["features"] == "macro") == mt["stocks"]["use"]
        starved = health.compute(frames, panel, ds.run_date, log=silent, budget=0)
        assert starved["macro_test"]["out_of_time"] is True and starved["macro_test"]["stocks"] is None
        assert starved["stocks"]["features"] == "base" and all(v["features"] == "base" for v in starved["markets"].values())
        # force the decision and see the forecasts follow it
        monkeypatch.setattr(health, "decide_stocks", lambda a, b: {"use": True, "reason": "forced", "years": 9})
        monkeypatch.setattr(health, "decide_market", lambda a, b: {"use": True, "wins": 3, "losses": 0, "reason": "forced"})
        forced = health.compute(frames, panel, ds.run_date, log=silent, budget=10 ** 6)
        assert forced["stocks"]["features"] == "macro" and all(v["features"] == "macro" for v in forced["markets"].values())
        mdash = forecast.forecast_markets(ds, frames, forced)
        ranking, meta = forecast.forecast_stocks(ds, panel, mdash, forced)
        assert meta["inputs"] == "macro" and set(mdash["inputs"]) == {"macro"}
        mdash0 = forecast.forecast_markets(ds, frames, starved)
        _, meta0 = forecast.forecast_stocks(ds, panel, mdash0, starved)
        assert meta0["inputs"] == "base" and set(mdash0["inputs"]) == {"base"}


def test_old_health_files_are_redone_once():
    h = {"version": C.HEALTH_VERSION, "updated": "2026-10-01", "markets": {"spx": {}}, "stocks": {}}
    assert health.needs_refresh(h, pd.Timestamp("2026-10-05")) is True                         # no "format": from before this version
    h["format"] = health.FORMAT
    assert health.needs_refresh(h, pd.Timestamp("2026-10-05")) is False


# ---------------------------------------------------------------- the whole thing: watchlist, growth, paper, drivers
def _write_watchlist(root, tickers, extra=""):
    (root / "watchlist.txt").write_text("# test list\n" + "\n".join(tickers) + "\n" + extra)


def test_pipeline_with_a_watchlist_growth_top_20_and_two_paper_ledgers(tmp_path):
    tickers, _ = universe.parse(open(__import__("pathlib").Path(__file__).resolve().parent.parent / "watchlist.txt").read())
    wl = tickers[:32]
    _write_watchlist(tmp_path, wl, "NOTAREALTICKER\n")
    msgs = []
    with patched(STOCKS=C.STOCKS[:14], **SMALL):
        def go(end, **kw):
            return run_weekly.run(tmp_path, downloader=synthetic.make_downloader(end=end),
                                  now=pd.Timestamp(end) + pd.Timedelta(days=1, hours=7), sleep=0, log=msgs.append,
                                  use_outlook=False, use_headlines=False, use_news=False, use_fundamentals=True,
                                  fund_fetcher=fake_company(now=end), **kw)

        p1 = go("2026-09-04")
        assert p1["watchlist"]["source"] == "watchlist.txt" and p1["watchlist"]["used"] == 32
        assert any("NOTAREALTICKER" in w for w in p1["data_warnings"]) or any("NOTAREALTICKER" in m for m in msgs)
        g = p1["growth"]
        assert g["available"] and g["n"] == 20 and len(g["picks"]) == 20
        assert abs(sum(x["w"] for x in g["picks"]) - 1.0) < 2e-3 and max(x["w"] for x in g["picks"]) <= opts.get("GROWTH_MAX_W") + 1e-6
        assert all(x["ticker"] in wl for x in g["picks"]) and any(x["name"] for x in g["picks"])
        assert p1["paper"]["growth"]["available"] and p1["paper"]["growth"]["trades"]["first"] is True
        assert p1["paper"]["growth"]["trades"]["n_buy"] == 20 and p1["global"]["available"] and p1["global"]["test"]["ran"]
        assert p1["stock_model"]["inputs"] in ("base", "macro")
        lg = pd.read_csv(tmp_path / "data" / "log_growth.csv")
        assert len(lg) == p1["growth"]["universe"] and set(growth.LOG_COLS) <= set(lg.columns)
        assert abs(lg["final_w"].sum() - 1.0) < 1e-6
        health1 = json.loads((tmp_path / "data" / "health.json").read_text())
        assert set(health1["universe"]) == set(p1["watchlist"]["tickers"])

        msgs.clear()
        p2 = go("2026-09-11")
        assert not any("walk-forward health check" in m for m in msgs)                             # no needless redo
        assert p2["paper"]["growth"]["trades"]["first"] is False
        p3 = go("2026-09-18")
        assert p3["paper"]["growth"]["weeks"] == 1 and p3["paper"]["model"]["weeks"] == 1       # the first week has now closed
        led = paper.read_ledger(tmp_path / "data" / "paper_growth.csv")
        assert len(led) == 1 and led.loc[0, "turnover"] > 0.99 and led.loc[0, "n_held"] == 20
        frozen = led.copy()
        go("2026-09-18")                                                                         # same data date again changes nothing
        pd.testing.assert_frame_equal(paper.read_ledger(tmp_path / "data" / "paper_growth.csv"), frozen)

        # remove a third of the watchlist: the model check is redone, the change is reported, history is untouched
        held_now = paper.held_tickers(tmp_path / "data")
        _write_watchlist(tmp_path, wl[:20])
        msgs.clear()
        p4 = go("2026-09-25")
        assert any("changed a lot" in m for m in msgs)
        assert set(p4["watchlist"]["removed"]) == set(wl[20:]) and p4["watchlist"]["used"] == 20
        assert p4["growth"]["available"] is False and "watchlist" in p4["growth"]["reason"]    # 20 is too few for a top 20
        assert p4["paper"]["model"]["weeks"] == 2                                              # the 09-18 picks are still open
        assert all(t in wl for t in held_now)
        pd.testing.assert_frame_equal(paper.read_ledger(tmp_path / "data" / "paper_growth.csv").iloc[:1], frozen)

        raw = (tmp_path / "docs" / "data" / "latest.json").read_text()
        json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("NaN in JSON")))


def test_pipeline_survives_every_global_driver_vanishing_and_the_switches(tmp_path):
    real = synthetic.make_downloader(end="2026-08-28")
    extras = set(opts.get("MACRO_EXTRA").values())

    def dl(tickers, start):
        out = real(tickers, start)
        return out.drop(columns=[c for c in out.columns if c[1] in extras])

    with patched(STOCKS=C.STOCKS[:14], **SMALL):
        p = run_weekly.run(tmp_path, downloader=dl, now="2026-08-29 07:00", sleep=0, log=silent, use_outlook=False,
                           use_headlines=False, use_news=False, use_fundamentals=False)
        assert p["stock_model"]["inputs"] == "base" and any("Global drivers not available" in w for w in p["data_warnings"])
        assert p["global"]["available"] and p["global"]["test"]["ran"] is False
        assert p["growth"]["available"] is False                                                # 14 stocks cannot give a top 20
        p2 = run_weekly.run(tmp_path / "b", downloader=synthetic.make_downloader(end="2026-08-28"), now="2026-08-29 07:00", sleep=0,
                            log=silent, use_outlook=False, use_headlines=False, use_news=False, use_fundamentals=False,
                            use_growth=False, use_paper=False, use_macro=False)
        assert p2["growth"] is None and p2["paper"] is None and p2["global"]["available"] is False and p2["global"]["test"]["ran"] is False
        assert not (tmp_path / "b" / "data" / "log_growth.csv").exists()
        assert not (tmp_path / "b" / "data" / "paper_model.csv").exists()


def test_a_bad_watchlist_cannot_stop_the_run_but_all_typos_say_so(tmp_path):
    (tmp_path / "watchlist.txt").write_text("\n".join(f"TYPO{i}" for i in range(20)))
    real = synthetic.make_downloader(end="2026-08-28")

    def dl(tickers, start):
        out = real([t for t in tickers if not t.startswith("TYPO")], start)
        return out

    with patched(STOCKS=C.STOCKS[:14], **SMALL):
        with pytest.raises(data.DataError) as e:
            run_weekly.run(tmp_path, downloader=dl, now="2026-08-29 07:00", sleep=0, log=silent, use_outlook=False,
                           use_headlines=False, use_news=False, use_fundamentals=False)
        assert "watchlist.txt" in str(e.value)
    assert not (tmp_path / "docs" / "data" / "latest.json").exists()
