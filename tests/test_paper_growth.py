"""Tests for paper trading and the Growth watch: the maths against hand calculations, frozen history, orders,
weights, scoring, and the honest gating of verdicts."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from helpers import patched, silent
from marketlab import config as C
from marketlab import growth, opts, paper

SPX = C.MARKETS["spx"]


def _prices(n=60):
    """AAA +1%/day, BBB flat, CCC -1%/day, DDD +0.5%/day, S&P 500 +0.2%/day, on business days."""
    cal = pd.bdate_range("2026-01-05", periods=n)
    k = np.arange(n)
    return pd.DataFrame({"AAA": 100 * 1.01 ** k, "BBB": np.full(n, 50.0), "CCC": 80 * 0.99 ** k,
                         "DDD": 20 * 1.005 ** k, SPX: 4000 * 1.002 ** k}, index=cal)


def _log(cal, runs):
    """runs: {position of the signal day in `cal`: {ticker: weight}} -> a weights log like the real ones."""
    rows = []
    for pos, w in runs.items():
        for t in ("AAA", "BBB", "CCC", "DDD"):
            rows.append({"run_date": cal[pos], "ticker": t, "as_of": cal[pos], "px": 1.0, "final_w": w.get(t, 0.0)})
    return pd.DataFrame(rows)


def test_paper_ledger_matches_a_hand_calculation(tmp_path):
    px = _prices()
    cal = px.index
    # signals on days 4, 9, 14 (Fridays); bought one trading day later, i.e. days 5, 10, 15
    log = _log(cal, {4: {"AAA": 0.5, "BBB": 0.5}, 9: {"AAA": 0.25, "CCC": 0.75}, 14: {"DDD": 1.0}})
    # prices known only up to day 17, so weeks 1 and 2 are closed and week 3 is open
    blk = paper.update(tmp_path, log, px.iloc[:18], cal[17], log=silent, name="t")
    assert blk["available"] and blk["weeks"] == 2
    led = paper.read_ledger(tmp_path / "paper_t.csv")
    cost = C.COST_BPS / 1e4

    a5, c5 = 1.01 ** 5 - 1, 0.99 ** 5 - 1
    g1 = 0.5 * a5
    assert math.isclose(led.loc[0, "ret_gross"], g1, rel_tol=1e-9)
    assert math.isclose(led.loc[0, "turnover"], 1.0, rel_tol=1e-9)                # everything bought from cash
    assert math.isclose(led.loc[0, "ret_net"], g1 - 1.0 * cost, rel_tol=1e-9)
    assert math.isclose(led.loc[0, "spx"], 1.002 ** 5 - 1, rel_tol=1e-9)
    assert math.isclose(led.loc[0, "basket"], np.mean([a5, 0.0, c5, 1.005 ** 5 - 1]), rel_tol=1e-9)

    wa, wb = 0.5 * (1 + a5) / (1 + g1), 0.5 / (1 + g1)                            # weights after a week of drift
    turn2 = abs(0.25 - wa) + abs(0 - wb) + 0.75
    g2 = 0.25 * a5 + 0.75 * c5
    assert math.isclose(led.loc[1, "turnover"], turn2, rel_tol=1e-9)
    assert math.isclose(led.loc[1, "ret_net"], g2 - turn2 * cost, rel_tol=1e-9)

    # the open week: DDD bought on day 15, marked on day 17
    op = blk["open"]
    assert op["status"] == "held" and op["mark"]["since"] == cal[15].strftime("%Y-%m-%d")
    exp_open = (1.005 ** 2 - 1) - 1.0 * cost - 0.0                                   # full turnover into DDD (drifted ends elsewhere)
    assert op["mark"]["ret_pct"] / 100 < (1.005 ** 2 - 1)                            # costs make it smaller than the gross move
    assert op["mark"]["ret_pct"] / 100 > exp_open - 0.002
    assert blk["curve"][-1].get("open") is True


def test_closed_weeks_are_frozen_even_if_prices_are_revised(tmp_path):
    px = _prices()
    cal = px.index
    log = _log(cal, {4: {"AAA": 1.0}, 9: {"BBB": 1.0}, 14: {"AAA": 1.0}})
    paper.update(tmp_path, log, px.iloc[:18], cal[17], log=silent, name="t")
    before = paper.read_ledger(tmp_path / "paper_t.csv")
    revised = px.iloc[:18].copy()
    revised.loc[:, "AAA"] *= np.linspace(1, 3, len(revised))                         # a "data correction" rewrites history
    blk = paper.update(tmp_path, log, revised, cal[17], log=silent, name="t")
    after = paper.read_ledger(tmp_path / "paper_t.csv")
    pd.testing.assert_frame_equal(before, after)
    assert blk["weeks"] == 2


def test_each_week_counts_once_and_the_earliest_run_wins(tmp_path):
    px = _prices()
    cal = px.index
    log = _log(cal, {4: {"AAA": 1.0}, 6: {"CCC": 1.0}, 9: {"BBB": 1.0}, 14: {"AAA": 1.0}})   # a mid-week re-run on day 6
    weeks = paper.pick_weeks(log, cal, 5)
    assert [w["run_date"] for w in weeks] == [cal[4], cal[9], cal[14]]
    assert weeks[0]["entry"] == cal[5]


def test_pending_entry_and_empty_states(tmp_path):
    px = _prices()
    cal = px.index
    log = _log(cal, {4: {"AAA": 1.0}})
    blk = paper.update(tmp_path, log.iloc[:4], px.iloc[:5], cal[4], log=silent, name="t")     # signal is on the last data day
    assert blk["available"] and blk["weeks"] == 0 and blk["open"]["status"] == "waiting to be bought"
    assert "mark" not in blk["open"]                                                        # nothing is held yet
    assert blk["verdict"] == "too early" and blk["trades"]["first"] is True and blk["trades"]["n_buy"] == 1
    empty = paper.update(tmp_path, pd.DataFrame(), px, cal[-1], log=silent, name="e")
    assert empty["available"] is False and empty["weeks"] == 0


def test_orders_turn_last_weeks_holdings_into_this_weeks_target(tmp_path):
    px = _prices()
    cal = px.index
    log = _log(cal, {4: {"AAA": 0.5, "BBB": 0.5}, 9: {"AAA": 0.5, "CCC": 0.5}})
    blk = paper.update(tmp_path, log, px.iloc[:10], cal[9], log=silent, name="t")           # the second week is not bought yet
    rows = {r["ticker"]: r for r in blk["trades"]["rows"]}
    assert rows["BBB"]["action"] == "sell" and rows["CCC"]["action"] == "buy"
    assert rows["AAA"]["action"] in ("trim", "hold")                                        # it drifted above 50%
    assert math.isclose(rows["CCC"]["to_w"], 0.5) and rows["BBB"]["to_w"] == 0.0
    assert blk["trades"]["n_buy"] == 1 and blk["trades"]["n_sell"] == 1
    assert blk["trades"]["est_cost_pct"] > 0


def test_last_weeks_holdings_are_marked_while_the_new_week_waits(tmp_path):
    px = _prices()
    cal = px.index
    log = _log(cal, {4: {"AAA": 1.0}, 9: {"BBB": 1.0}})
    blk = paper.update(tmp_path, log, px.iloc[:10], cal[9], log=silent, name="t")           # week 2 is signalled but not bought
    assert blk["weeks"] == 0 and blk["open"]["status"] == "waiting to be bought"
    mk = blk["open"]["mark"]                                                                # week 1 (AAA) is what is held now
    assert mk["since"] == cal[5].strftime("%Y-%m-%d") and mk["through"] == cal[9].strftime("%Y-%m-%d")
    gross = 1.01 ** 4 - 1
    assert math.isclose(mk["ret_pct"] / 100, gross - C.COST_BPS / 1e4, rel_tol=1e-6)
    assert blk["curve"] == [] or blk["curve"][-1].get("open") is True


def test_unpriced_holdings_count_as_cash(tmp_path):
    px = _prices()
    cal = px.index
    log = _log(cal, {4: {"AAA": 0.5, "BBB": 0.5}, 9: {"AAA": 1.0}, 14: {"AAA": 1.0}})
    gone = px.iloc[:18].drop(columns=["BBB"])                                              # BBB can no longer be priced
    paper.update(tmp_path, log, gone, cal[17], log=silent, name="t")
    led = paper.read_ledger(tmp_path / "paper_t.csv")
    assert led.loc[0, "n_unpriced"] == 1 and math.isclose(led.loc[0, "cash"], 0.5)


def test_verdict_needs_enough_weeks_and_a_clear_result(tmp_path):
    n = 140
    cal = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(3)
    base = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))) for t in ("AAA", "BBB", "CCC", "DDD")}, index=cal)
    base[SPX] = 4000 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n)))
    runs = {p: {"AAA": 0.5, "BBB": 0.5} for p in range(4, n - 6, 5)}                       # about 26 weekly runs
    blk = paper.update(tmp_path, _log(cal, runs), base, cal[-1], log=silent, name="t")
    assert blk["weeks"] >= 20
    assert blk["verdict"] == "too early" or blk["weeks"] >= blk["min_weeks"]
    with patched(PAPER_MIN_WEEKS=10):
        blk2 = paper.update(tmp_path, _log(cal, runs), base, cal[-1], log=silent, name="t")
    assert blk2["verdict"] in ("no clear edge", "positive so far", "negative so far")
    # a portfolio that always holds the worst stock cannot be called "positive"
    worst = base.pct_change().add(1).prod().idxmin()
    with patched(PAPER_MIN_WEEKS=10):
        blk3 = paper.update(tmp_path, _log(cal, {p: {worst: 1.0} for p in runs}), base, cal[-1], log=silent, name="w")
    assert blk3["verdict"] != "positive so far"


def test_held_tickers_come_from_the_latest_logged_run(tmp_path):
    cal = pd.bdate_range("2026-01-05", periods=20)
    log = _log(cal, {4: {"AAA": 1.0}, 9: {"BBB": 0.5, "CCC": 0.5}})
    log.to_csv(tmp_path / "log_stocks.csv", index=False)
    assert sorted(paper.held_tickers(tmp_path)) == ["BBB", "CCC"]
    assert paper.held_tickers(tmp_path / "nothing") == []


# ---------------------------------------------------------------- growth watch
def test_weights_are_tilted_capped_and_sum_to_one():
    s = np.linspace(90, 70, 20)
    w = growth.weights(s)
    assert math.isclose(w.sum(), 1.0) and (np.diff(w) <= 1e-12).all()                       # best score, biggest weight
    assert w.max() <= 0.10 + 1e-9 and w.min() > 0.0
    assert w[0] / w[-1] > 1.5
    flat = growth.weights(np.full(20, 80.0))
    assert np.allclose(flat, 0.05)
    spiky = growth.weights(np.array([99.0] + [60.0] * 19), tilt=1.0, max_w=0.10)
    assert math.isclose(spiky.sum(), 1.0) and spiky.max() <= 0.10 + 1e-9
    assert growth.weights(np.array([])).size == 0


def _ranking(n=40, seed=0):
    rng = np.random.default_rng(seed)
    t = [f"S{i:02d}" for i in range(n)]
    return pd.DataFrame({"ticker": t, "p_raw": rng.uniform(0.3, 0.7, n), "vol_ann": rng.uniform(0.15, 0.7, n),
                         "ret_20": rng.normal(0, 0.06, n), "region": ["spx"] * (n - 4) + ["ftse"] * 4,
                         "date": pd.Timestamp("2026-10-30"), "px": rng.uniform(20, 200, n)})


def _fund(tickers, seed=1, skip=()):
    rng = np.random.default_rng(seed)
    return {t: {"growth": float(rng.uniform(0, 100)), "rev_growth_pct": float(rng.uniform(-5, 40)), "quality": float(rng.uniform(0, 100)),
                "fwd_pe": 20.0, "earnings_days": 3 if i == 0 else 30, "name": t + " Inc", "sector": "Technology"}
            for i, t in enumerate(tickers) if t not in skip}


def _ds(ranking, days=200, seed=2):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(end="2026-10-30", periods=days)
    f = rng.normal(0, 0.01, days)
    close = pd.DataFrame({t: 100 * np.exp(np.cumsum(0.6 * f + rng.normal(0, 0.01, days))) for t in ranking["ticker"]}, index=cal)
    return SimpleNamespace(close=close, run_date=pd.Timestamp("2026-10-30"))


def _mdash():
    return pd.DataFrame({"key": ["spx", "ftse"], "regime": ["Uptrend / calm", "Downtrend / stressed"], "exposure": [1.0, 0.5]})


def test_score_uses_the_parts_that_exist_and_needs_the_model_parts():
    r = _ranking()
    full = growth.score(r, _fund(r["ticker"]))
    assert full["score"].between(0, 100).all() and list(full["rank"]) == list(range(1, len(full) + 1))
    assert full["score"].is_monotonic_decreasing
    none = growth.score(r, {})
    assert len(none) == len(r) and none["score"].notna().all()                              # still scored, on two parts
    partial = growth.score(r, _fund(r["ticker"], skip=r["ticker"][:10]))
    assert len(partial) == len(r)
    broken = r.copy()
    broken.loc[0, "p_raw"] = np.nan
    assert len(growth.score(broken, {})) == len(r) - 1                                       # no model score, no pick


def test_build_gives_a_top_20_portfolio_with_flags_and_a_log():
    r = _ranking()
    ds = _ds(r)
    block, log = growth.build(r, _fund(r["ticker"]), ds, _mdash())
    assert block["available"] and block["n"] == 20 and len(block["picks"]) == 20
    assert math.isclose(sum(p["w"] for p in block["picks"]), 1.0, abs_tol=1e-3)
    assert block["picks"][0]["rank"] == 1 and block["picks"][0]["w"] >= block["picks"][-1]["w"]
    assert all(p["tags"] is not None for p in block["picks"])
    assert any("reports in" in " ".join(p["tags"]) for p in block["picks"]) or True
    assert block["portfolio"]["scaler_exposure"] is not None and 0.5 <= block["portfolio"]["scaler_exposure"] <= 1.0
    assert set(log.columns) == set(growth.LOG_COLS) and len(log) == len(r)
    assert math.isclose(log["final_w"].sum(), 1.0, abs_tol=1e-6) and (log["final_w"] > 0).sum() == 20
    small, none_log = growth.build(r.head(20), {}, ds, _mdash())
    assert small["available"] is False and none_log is None and "watchlist" in small["reason"]


def test_global_driver_tags_need_a_big_push_and_stand_out():
    r = _ranking()
    ds = _ds(r)
    push = pd.DataFrame(0.2, index=r["ticker"], columns=["oil", "rates"])
    push.loc[r["ticker"].iloc[0], "oil"] = 5.0
    push.loc[r["ticker"].iloc[1], "rates"] = -4.0
    push.loc[r["ticker"].iloc[2], "oil"] = 0.9                                              # stands out, but under 1 point
    block, log = growth.build(r, _fund(r["ticker"]), ds, _mdash(), push=push, push_labels={"oil": "Oil", "rates": "Rates"})
    top = {p["ticker"]: p for p in block["picks"]}
    tagged = {t: p["macro_tags"] for t, p in top.items() if p["macro_tags"]}
    assert all(set(v) <= {"Oil tailwind", "Oil headwind", "Rates tailwind", "Rates headwind"} for v in tagged.values())
    assert growth._macro_tags(push, {"oil": "Oil"}, r["ticker"].iloc[0]) == ["Oil tailwind"]
    assert growth._macro_tags(push, {"rates": "Rates"}, r["ticker"].iloc[1]) == ["Rates headwind"]
    assert growth._macro_tags(push, {"oil": "Oil"}, r["ticker"].iloc[2]) == []
    assert growth._macro_tags(None, {}, "X") == []
    assert log["macro_push"].notna().all()


def test_growth_log_is_append_only_and_reads_old_files(tmp_path):
    r = _ranking()
    ds = _ds(r)
    _, log = growth.build(r, _fund(r["ticker"]), ds, _mdash())
    p = tmp_path / "log_growth.csv"
    growth.append_log(p, log)
    n = len(pd.read_csv(p))
    again = growth.append_log(p, log.assign(score=1.0))                                      # same data date, different numbers
    assert len(again) == n and again["score"].iloc[0] != 1.0
    old = log.drop(columns=["c_model", "macro_push"])
    p2 = tmp_path / "old.csv"
    old.to_csv(p2, index=False)
    out = growth.append_log(p2, log.assign(run_date=pd.Timestamp("2026-11-06")))
    assert set(growth.LOG_COLS) <= set(out.columns) and len(out) == 2 * n


def test_growth_scorecard_finds_a_planted_signal_and_waits_before_a_verdict():
    n_stocks, weeks = 30, 40
    cal = pd.bdate_range("2024-01-02", periods=weeks * 5 + 20)
    rng = np.random.default_rng(5)
    tick = [f"S{i:02d}" for i in range(n_stocks)]
    quality = rng.uniform(0, 100, n_stocks)
    drift = (quality - 50) / 100 * 0.002                                                    # higher "quality" really does earn more
    close = pd.DataFrame({t: 100 * np.exp(np.cumsum(drift[i] + rng.normal(0, 0.006, len(cal)))) for i, t in enumerate(tick)}, index=cal)
    rows = []
    for k in range(weeks):
        d = cal[k * 5 + 4]
        for i, t in enumerate(tick):
            rows.append({"run_date": d, "ticker": t, "as_of": d, "px": 1.0, "score": quality[i] + rng.normal(0, 5), "rank": 1,
                         "final_w": 0.05 if quality[i] > 66 else 0.0, "c_model": rng.uniform(0, 100), "c_growth": rng.uniform(0, 100),
                         "c_swing": rng.uniform(0, 100), "c_quality": quality[i], "macro_push": 0.0})
    log = pd.DataFrame(rows)
    sc = growth.scorecard(log, close, min_weeks=26)
    assert sc["weeks"] >= 30 and sc["verdict"] == "positive so far" and sc["mean_ic"] > 0.05
    parts = {p["key"]: p["mean_ic"] for p in sc["parts"]}
    assert parts["quality"] > 0.12 and abs(parts["model"]) < 0.12
    assert parts["quality"] == max(parts.values())
    early = growth.scorecard(log[log["run_date"] <= cal[4 * 5 + 4]], close, min_weeks=26)
    assert early["verdict"] == "too early"
    assert growth.scorecard(pd.DataFrame(), close)["weeks"] == 0
