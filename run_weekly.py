#!/usr/bin/env python3
"""
Market Lab weekly runner.

Downloads data, checks it, forecasts the markets and ranks the stocks, calibrates the output against the
model's own walk-forward track record, logs the forecasts, scores older ones, picks the week's Growth watch top 20,
runs the paper-trading ledgers, and writes the files the phone dashboard reads. Research tool only, not financial
advice, and it never places a real order.

    python run_weekly.py                    normal weekly run
    python run_weekly.py --refresh-health   also force the monthly walk-forward check now
    python run_weekly.py --no-outlook       skip the optional AI commentary
    python run_weekly.py --no-headlines     skip the free news headlines
    python run_weekly.py --no-news          skip the news archive and tone series
    python run_weekly.py --no-fundamentals  skip the company-data snapshot
    python run_weekly.py --no-growth        skip the Growth watch (top 20) and its paper portfolio
    python run_weekly.py --no-paper         skip the paper-trading ledgers
    python run_weekly.py --no-macro         skip the extra global-driver inputs
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from marketlab import config as C
from marketlab import (data, features, forecast, fundamentals, growth, headlines, health as health_mod, macro, news,
                       opts, outlook, paper, report, scoring, universe, util)


def _soft(name, fn, log):
    """Run an optional step. Whatever goes wrong, the weekly run carries on without it."""
    try:
        return fn()
    except Exception as e:
        log(f"{name} skipped ({type(e).__name__}: {e})")
        return None


def run(root, downloader=None, now=None, force_health=False, use_outlook=True, demo=False,
        sleep=5.0, log=print, outlook_client=None, api_key=None, use_headlines=True, headlines_fetcher=None,
        use_news=True, news_fetcher=None, use_fundamentals=True, fund_fetcher=None, use_paper=True, use_growth=True,
        use_macro=True):
    """One weekly run. Returns the payload that was written. Raises (and saves nothing) if the data or the
    forecasts fail their checks. The stocks come from watchlist.txt when it is usable, else the built-in list."""
    root = Path(root)
    default_universe = list(C.STOCKS)
    res = universe.resolve(root, default_universe)
    for n in res.notes:
        log("Watchlist: " + n)
    with universe.use(res.tickers):
        try:
            return _run(root, downloader, now, force_health, use_outlook, demo, sleep, log, outlook_client, api_key,
                        use_headlines, headlines_fetcher, use_news, news_fetcher, use_fundamentals, fund_fetcher,
                        use_paper, use_growth, use_macro, res, default_universe)
        except data.DataError as e:
            if "usable stocks remain" in str(e) and res.source == universe.FILE:
                raise data.DataError(f"{e} Check the spelling of the tickers in {universe.FILE}.") from e
            raise


def _run(root, downloader, now, force_health, use_outlook, demo, sleep, log, outlook_client, api_key, use_headlines,
         headlines_fetcher, use_news, news_fetcher, use_fundamentals, fund_fetcher, use_paper, use_growth, use_macro,
         watchlist, default_universe):
    log_dir = root / "data"
    web_path = root / "docs" / "data" / "latest.json"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    live = not demo

    # 1. data, with retries and checks. Shares a paper portfolio still holds are fetched even if they left the watchlist.
    held = paper.held_tickers(log_dir) if (use_paper or use_growth) else []
    close_all, volume_all = data.fetch(data.all_tickers(also=held), C.START, downloader=downloader, sleep=sleep, log=log)
    ds = data.build_dataset(close_all, volume_all, now, log=log)
    log(f"Data through {ds.run_date.date()} | {len(ds.stocks)} stocks | {len(ds.idx.columns)} markets | "
        f"{0 if ds.extra is None else ds.extra.shape[1]} global drivers")

    # 2. features (the extra global-driver inputs are built here but only used if the monthly test says they help)
    mac = features.macro_features(ds.macro)
    ext = None
    if use_macro and opts.get("MACRO_ENABLED"):
        ext = _soft("Global-driver inputs", lambda: macro.build_ext(ds), log)
    frames = {m: features.market_frame(m, ds, mac, ext) for m in ds.idx.columns}
    panel = features.stock_panel(ds, ext)

    # 3. model health (monthly, or sooner if the stock list changed a lot), used to calibrate everything below
    hpath = log_dir / "health.json"
    h = health_mod.load(hpath)
    redo = health_mod.needs_refresh(h, ds.run_date, force_health)
    if not redo and universe.list_changed(h, ds.stocks, default_universe):
        log("The stock list changed a lot since the last model check, so it is being redone.")
        redo = True
    if redo:
        log("Running the walk-forward health check (about once a month)...")
        try:
            h = health_mod.compute(frames, panel, ds.run_date, log=log)
            health_mod.save(hpath, h)
        except Exception as e:                       # keep going on the last good health file
            log(f"Health check failed ({e}); using the previous one" if h else f"Health check failed ({e}); using default shrinkage")
    if not h:
        log("No health file: forecasts use default shrinkage.")

    # 4. this week's forecasts, checked before anything is saved
    mdash = forecast.forecast_markets(ds, frames, h)
    ranking, stock_meta = forecast.forecast_stocks(ds, panel, mdash, h)
    report.validate_forecasts(mdash, ranking)

    # 5. log them, then score older ones
    previous_universe = universe.last_logged(log_dir / "log_stocks.csv")
    mnew, snew = scoring.log_frames(mdash, ranking, ds.run_date)
    mlog = scoring.append_log(log_dir / "log_markets.csv", mnew, ["run_date", "key"])
    slog = scoring.append_log(log_dir / "log_stocks.csv", snew, ["run_date", "ticker"])
    scorecard = scoring.build_scorecard(mlog, slog, ds)

    # 6. context that never changes a forecast: company data, global-driver readings, headlines
    hl_fetch = headlines.memoize(headlines_fetcher)
    hl = headlines.fetch(fetcher=hl_fetch, now=now, log=log) if (use_headlines and live) else []
    hl_all = headlines.fetch_all(fetcher=hl_fetch, now=now, log=log) if (use_headlines and live) else []
    fund_block = None
    if use_fundamentals and live and opts.get("FUND_ENABLED"):
        fund_block = _soft("Company data", lambda: fundamentals.update(log_dir, ds, now, fetcher=fund_fetcher,
                                                                       pause=0 if sleep == 0 else None, log=log), log)
    by_ticker = ((fund_block or {}).get("by_ticker")) or {}
    drivers = _soft("Global-driver readings", lambda: macro.drivers_view(ds), log) if (use_macro and opts.get("MACRO_ENABLED")) else None
    push = _soft("Global-driver sensitivities", lambda: macro.latest_push(ext, ds.stocks, ds.run_date), log) if ext is not None else None

    # 7. the Growth watch (top 20, weighted) and the paper-trading ledgers
    growth_block, growth_all = None, None
    if use_growth and opts.get("GROWTH_ENABLED"):
        def _growth():
            block, glog = growth.build(ranking, by_ticker, ds, mdash, push=push, push_labels=macro.FACTORS)
            allrows = growth.append_log(log_dir / "log_growth.csv", glog) if glog is not None else None
            if block.get("available") and allrows is not None:
                block["scorecard"] = growth.scorecard(allrows, ds.close)
            return block, allrows
        out = _soft("Growth watch", _growth, log)
        if out:
            growth_block, growth_all = out
    paper_blocks = None
    if use_paper and opts.get("PAPER_ENABLED"):
        paper_blocks = {}
        if growth_all is not None:
            paper_blocks["growth"] = paper.update(log_dir, growth_all, close_all, ds.run_date, log=log, name="growth",
                                                  label=f"Growth watch top {opts.get('GROWTH_TOP_N')}")
        paper_blocks["model"] = paper.update(log_dir, slog, close_all, ds.run_date, log=log, name="model",
                                             label=f"Model top {C.TOP_K}")

    # 8. the news archive and tone series; the pick checks cover the model's picks and the growth picks
    picks = ranking.loc[ranking["base_w"] > 0, "ticker"].tolist()
    if growth_block and growth_block.get("available"):
        picks += [p["ticker"] for p in growth_block["picks"] if p["ticker"] not in picks]
    news_block = None
    if use_news and live and opts.get("NEWS_ENABLED"):
        names = {t: q for t, q in ((t, news.name_query(n)) for t, n in
                                    {**fundamentals.known_names(log_dir),
                                     **{t: v.get("name") for t, v in by_ticker.items()}}.items()) if q}
        base_frames = {m: (f, d, macro.base_features(F)) for m, (f, d, F) in frames.items()}
        news_block = _soft("News", lambda: news.update(log_dir, now, picks, base_frames, headline_items=hl_all,
                                                       fetcher=news_fetcher, sleep=sleep, force_experiment=force_health,
                                                       log=log, name_queries=names), log)

    # 9. the payload without the optional commentary, validated
    wl_block = universe.block(watchlist, ds.stocks, previous_universe)
    payload = report.build_payload(ds, mdash, ranking, stock_meta, h, scorecard, mlog, slog, ds.run_date, demo=demo,
                                   headlines=hl, news=news_block, fundamentals=fund_block, paper=paper_blocks,
                                   watchlist=wl_block, growth=growth_block, drivers=drivers)
    report.validate_payload(util.clean(payload))

    # 10. optional AI commentary: never affects a forecast
    if use_outlook and not demo and (outlook_client is not None or api_key or os.environ.get("ANTHROPIC_API_KEY")):
        top = [s["ticker"] for s in payload["stocks"] if s["base_w"] > 0]
        text = outlook.generate(report.plain_text(payload), outlook.fetch_headlines(["^GSPC"] + top) if outlook_client is None else {},
                                api_key=api_key, client=outlook_client, log=log)
        if text:
            payload["outlook"] = {"text": text, "model": C.OUTLOOK_MODEL,
                                  "generated_utc": payload["generated_utc"]}

    # 11. write (atomically) only now that everything has passed
    util.write_json_atomic(web_path, payload)
    log(f"Wrote {web_path}")
    if not demo:
        title, body, priority = report.notification(util.clean(payload))
        (root / "summary.txt").write_text(f"{title}\n{body}")
        (root / "alert_level.txt").write_text(priority)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-health", action="store_true", help="force the walk-forward health check now")
    ap.add_argument("--no-outlook", action="store_true", help="skip the optional AI commentary")
    ap.add_argument("--no-headlines", action="store_true", help="skip the free news headlines")
    ap.add_argument("--no-news", action="store_true", help="skip the news archive and tone series")
    ap.add_argument("--no-fundamentals", action="store_true", help="skip the company-data snapshot")
    ap.add_argument("--no-growth", action="store_true", help="skip the Growth watch and its paper portfolio")
    ap.add_argument("--no-paper", action="store_true", help="skip the paper-trading ledgers")
    ap.add_argument("--no-macro", action="store_true", help="skip the extra global-driver inputs")
    args = ap.parse_args(argv)
    try:
        run(Path(__file__).resolve().parent, force_health=args.refresh_health, use_outlook=not args.no_outlook,
            use_headlines=not args.no_headlines, use_news=not args.no_news, use_fundamentals=not args.no_fundamentals,
            use_growth=not args.no_growth, use_paper=not args.no_paper, use_macro=not args.no_macro)
    except data.DataError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
