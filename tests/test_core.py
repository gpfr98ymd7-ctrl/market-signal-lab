"""Tests for the parts that matter most: no lookahead, honest calibration, safe failure, and a working pipeline.

Run with:  pytest -q
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from helpers import make_ds, patched, silent
from marketlab import config as C
from marketlab import data, features, health, models, outlook, report, scoring, synthetic
import run_weekly


# ---------------------------------------------------------------- no lookahead
def test_market_features_do_not_use_the_future():
    full = make_ds("2026-06-30")
    short = make_ds("2025-06-30")
    cut = pd.Timestamp("2025-06-30")
    f1, _, F1 = features.market_frame("spx", full, features.macro_features(full.macro))
    f2, _, F2 = features.market_frame("spx", short, features.macro_features(short.macro))
    assert F1 == F2
    a = f1[f1["date"] <= cut].set_index("date")
    b = f2[f2["date"] <= cut].set_index("date")
    common = a.index.intersection(b.index)
    assert len(common) > 1000
    pd.testing.assert_frame_equal(a.loc[common, F1], b.loc[common, F1], rtol=1e-9, atol=1e-12)
    assert (a.loc[common, "regime"] == b.loc[common, "regime"]).all()


def test_stock_features_do_not_use_the_future():
    full = make_ds("2026-06-30")
    short = make_ds("2025-06-30")
    cut = pd.Timestamp("2025-06-30")
    s1, _, F1 = features.stock_panel(full)
    s2, _, F2 = features.stock_panel(short)
    assert F1 == F2
    a = s1[s1["date"] <= cut].set_index(["date", "ticker"])[F1]
    b = s2[s2["date"] <= cut].set_index(["date", "ticker"])[F1]
    common = a.index.intersection(b.index)
    assert len(common) > 10000
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common], rtol=1e-9, atol=1e-12)


def test_targets_are_aligned_with_the_documented_definitions():
    ds = make_ds("2026-06-30")
    sfull, sdata, _ = features.stock_panel(ds)
    row = sdata[(sdata["ticker"] == "MSFT")].iloc[500]
    s = ds.close["MSFT"].dropna()
    pos = s.index.get_loc(row["date"])
    expected = s.iloc[pos + C.ENTRY_DELAY + C.SH] / s.iloc[pos + C.ENTRY_DELAY] - 1
    assert math.isclose(row["fwd_ret"], expected, rel_tol=1e-9)

    f, d, _ = features.market_frame("spx", ds, features.macro_features(ds.macro))
    r = d.iloc[800]
    px = ds.idx["spx"].dropna()
    i = px.index.get_loc(r["date"])
    nxt = px.iloc[i + 1: i + C.H + 1]
    assert math.isclose(r["fmin_ret"], nxt.min() / px.iloc[i] - 1, rel_tol=1e-9)
    assert math.isclose(r["fwd_ret"], np.log(px.iloc[i + C.H] / px.iloc[i]), rel_tol=1e-9)
    rets = np.log(px).diff().iloc[i + 1: i + C.H + 1]
    assert math.isclose(r["fwd_vol"], rets.std() * np.sqrt(252), rel_tol=1e-9)


# ---------------------------------------------------------------- calibration
def test_shrinkage_goes_to_zero_on_noise_and_stays_on_real_signal():
    rng = np.random.default_rng(1)
    n = 6000
    p = rng.uniform(0.1, 0.9, n)
    base = np.full(n, 0.5)
    noise_y = rng.binomial(1, 0.5, n)
    assert health._lam(p, noise_y, base) < 0.12
    signal_y = rng.binomial(1, p)
    assert health._lam(p, signal_y, base) > 0.8
    shrunk = models.shrink(p, 0.5, 0.0)
    assert np.allclose(shrunk, 0.5)
    assert np.allclose(models.shrink(p, 0.5, 1.0), p)


def test_range_widening_reaches_the_stated_coverage():
    rng = np.random.default_rng(2)
    n = 1500
    z = {5: -1.645, 25: -0.674, 50: 0.0, 75: 0.674, 95: 1.645}
    res = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n)})
    res["fwd_ret"] = rng.normal(0, 2.0, n)                    # the world is twice as wide as the model thinks
    for k, v in z.items():
        res[f"q{k:02d}"] = v * 1.0
    res["y_up"] = (res["fwd_ret"] > 0).astype(int)
    res["y_crash"] = 0
    res["p_up"], res["p_crash"] = 0.5, 0.1
    res["base_up"], res["base_crash"] = 0.5, 0.1
    res["fwd_vol"], res["pred_vol"], res["vol_21"] = 0.2, 0.2, 0.2
    cal = health.calibrate_market(res)
    assert cal["cov90_raw"] < 0.6
    assert abs(cal["cov90_cal"] - 0.90) < 0.03
    assert 1.8 < cal["k_range"] < 2.2


# ---------------------------------------------------------------- data layer
def test_fetch_retries_then_succeeds_and_recovers_missing_tickers():
    real = synthetic.make_downloader(end="2026-06-30")
    calls = {"n": 0}

    def flaky(tickers, start):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionError("rate limited")
        out = real(tickers, start)
        if len(tickers) > 1:                                  # batch call silently omits one ticker
            out = out.drop(columns=[c for c in out.columns if c[1] == "AAPL"])
        return out

    close, vol = data.fetch(data.all_tickers(), C.START, downloader=flaky, sleep=0, log=silent)
    assert calls["n"] >= 4
    assert "AAPL" in close.columns and close["AAPL"].notna().sum() > 1000


def test_fetch_gives_up_clearly():
    def dead(tickers, start):
        raise ConnectionError("blocked")
    try:
        data.fetch(data.all_tickers(), C.START, downloader=dead, sleep=0, retries=2, log=silent)
        assert False, "should have raised"
    except data.DataError as e:
        assert "rate-limiting" in str(e)


def test_bad_data_is_dropped_or_stops_the_run():
    dl = synthetic.make_downloader(end="2026-06-30")
    close, vol = data.fetch(data.all_tickers(), C.START, downloader=dl, sleep=0, log=silent)
    now = pd.Timestamp("2026-07-01 07:00")

    stale = close.copy()
    stale.loc[stale.index[-30:], "KO"] = np.nan               # no price for a month
    ds = data.build_dataset(stale, vol, now, log=silent)
    assert "KO" not in ds.stocks and any("KO" in w for w in ds.warnings)

    try:
        data.build_dataset(close, vol, pd.Timestamp("2026-09-01 07:00"), log=silent)
        assert False, "stale S&P 500 should stop the run"
    except data.DataError as e:
        assert "old" in str(e)

    ds = data.build_dataset(close, vol, pd.Timestamp("2026-06-30 12:00"), log=silent)   # same day, before US close
    assert ds.run_date == pd.Timestamp("2026-06-29")


# ---------------------------------------------------------------- checks before saving
def _fake_forecasts():
    keys = list(C.MARKETS)
    m = pd.DataFrame({"key": keys, "market": [C.NAMES[k] for k in keys], "p_up": 0.45, "p_crash": 0.2,
                      "pred_vol": 0.15, "vol_now": 0.14, "q05": -0.08, "q50": 0.01, "q95": 0.07, "exposure": 0.9})
    t = [f"T{i}" for i in range(8)]
    r = pd.DataFrame({"ticker": t, "p": 0.5, "p_raw": 0.5,
                      "base_w": [0.2] * 5 + [0.0] * 3, "final_w": [0.18] * 5 + [0.0] * 3})
    return m, r


def test_validate_forecasts_accepts_good_and_rejects_bad():
    m, r = _fake_forecasts()
    report.validate_forecasts(m, r)
    bad = m.copy()
    bad.loc[0, "p_up"] = float("nan")
    for args in ((bad, r), (m.assign(exposure=1.7), r), (m, r.assign(final_w=0.5))):
        try:
            report.validate_forecasts(*args)
            assert False, "should have raised"
        except ValueError:
            pass


def test_alerts_fire_on_real_changes_only():
    m, _ = _fake_forecasts()
    m["regime"] = "Uptrend / calm"
    m["base_crash"] = 0.2
    changes = {"markets": [{"key": "spx", "name": "S&P 500", "exposure_from": 0.95, "exposure_delta": -0.05,
                            "regime_from": "Uptrend / calm", "regime_to": "Uptrend / calm"}]}
    assert report.build_alerts(m, changes, {}) == []
    changes["markets"][0]["exposure_delta"] = -0.3
    alerts = report.build_alerts(m, changes, {})
    assert len(alerts) == 1 and "S&P 500" in alerts[0]["text"]


def test_append_log_never_overwrites(tmp_path):
    p = tmp_path / "log.csv"
    a = pd.DataFrame({"run_date": [pd.Timestamp("2026-01-02")], "as_of": [pd.Timestamp("2026-01-02")], "key": ["spx"], "v": [1]})
    scoring.append_log(p, a, ["run_date", "key"])
    b = a.assign(v=2)
    out = scoring.append_log(p, b, ["run_date", "key"])
    assert len(out) == 1 and out["v"].iloc[0] == 1


# ---------------------------------------------------------------- the optional AI commentary
def test_outlook_is_optional_and_fails_quietly():
    assert outlook.generate("x", {}, api_key=None, log=silent) is None

    class Good:
        class messages:
            @staticmethod
            def create(**kw):
                assert "untrusted" in kw["messages"][0]["content"].lower()
                return SimpleNamespace(content=[SimpleNamespace(text="A calm week.")])

    class Broken:
        class messages:
            @staticmethod
            def create(**kw):
                raise RuntimeError("boom")

    assert outlook.generate("x", {"AAPL": ["Apple does a thing"]}, client=Good, log=silent) == "A calm week."
    assert outlook.generate("x", {}, client=Broken, log=silent) is None


# ---------------------------------------------------------------- the whole pipeline
def test_pipeline_two_weeks_logs_scores_and_dedupes(tmp_path):
    small = dict(STOCKS=C.STOCKS[:14], MIN_TRAIN_YEARS_MARKET=3, MIN_TRAIN_YEARS_STOCK=2,
                 HEALTH_REFRESH_DAYS=10 ** 6)
    with patched(**small):
        def go(end):
            return run_weekly.run(tmp_path, downloader=synthetic.make_downloader(end=end),
                                  now=pd.Timestamp(end) + pd.Timedelta(days=1, hours=7), sleep=0, log=silent,
                                  use_outlook=False, use_headlines=False)

        p1 = go("2026-08-28")
        assert p1["changes"]["previous_run"] is None and p1["scorecard"]["markets"]["scored"] == 0
        p2 = go("2026-10-08")
        assert p2["changes"]["previous_run"] == "2026-08-28"
        assert p2["scorecard"]["markets"]["scored"] == len(C.MARKETS)         # 21+ trading days have passed
        assert p2["scorecard"]["stocks"]["scored"] > 0
        assert p2["scorecard"]["combined"]["weeks"] == 1
        assert all(len(m["history"]) == 2 for m in p2["markets"])

        n_m = len(pd.read_csv(tmp_path / "data" / "log_markets.csv"))
        n_s = len(pd.read_csv(tmp_path / "data" / "log_stocks.csv"))
        go("2026-10-08")                                                       # same data date again
        assert len(pd.read_csv(tmp_path / "data" / "log_markets.csv")) == n_m
        assert len(pd.read_csv(tmp_path / "data" / "log_stocks.csv")) == n_s

        raw = (tmp_path / "docs" / "data" / "latest.json").read_text()
        saved = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("NaN in JSON")))
        assert saved["schema"] == report.SCHEMA and len(saved["markets"]) == len(C.MARKETS)
        assert (tmp_path / "summary.txt").exists() and (tmp_path / "alert_level.txt").read_text() in ("default", "high")

        bad = json.loads(raw)
        bad["markets"][0]["p_up"] = None
        try:
            report.validate_payload(bad)
            assert False, "should have raised"
        except ValueError:
            pass


def test_a_failed_check_saves_nothing(tmp_path):
    def broken(end="2026-08-28"):
        real = synthetic.make_downloader(end=end)

        def dl(tickers, start):
            out = real(tickers, start)
            return out.drop(columns=[c for c in out.columns if c[1] == "^VIX"])      # a core series vanishes
        return dl

    with patched(STOCKS=C.STOCKS[:14]):
        try:
            run_weekly.run(tmp_path, downloader=broken(), now="2026-08-29 07:00", sleep=0, log=silent, use_outlook=False)
            assert False, "should have stopped"
        except data.DataError as e:
            assert "^VIX" in str(e)
    assert not (tmp_path / "docs" / "data" / "latest.json").exists()
    assert not (tmp_path / "data" / "log_markets.csv").exists()
