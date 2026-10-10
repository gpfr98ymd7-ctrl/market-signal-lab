#!/usr/bin/env python3
"""
Market Lab weekly runner.

Downloads data, checks it, forecasts the markets and ranks the stocks, calibrates the output against the
model's own walk-forward track record, logs the forecasts, scores older ones, and writes the files the phone
dashboard reads. Research tool only, not financial advice.

    python run_weekly.py                    normal weekly run
    python run_weekly.py --refresh-health   also force the monthly walk-forward check now
    python run_weekly.py --no-outlook       skip the optional AI commentary
    python run_weekly.py --no-headlines     skip the free news headlines
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from marketlab import config as C
from marketlab import data, features, forecast, headlines, health as health_mod, outlook, report, scoring, util


def run(root, downloader=None, now=None, force_health=False, use_outlook=True, demo=False,
        sleep=5.0, log=print, outlook_client=None, api_key=None, use_headlines=True, headlines_fetcher=None):
    """One weekly run. Returns the payload that was written. Raises (and saves nothing) if the data or the
    forecasts fail their checks."""
    root = Path(root)
    log_dir = root / "data"
    web_path = root / "docs" / "data" / "latest.json"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)

    # 1. data, with retries and checks
    close_all, volume_all = data.fetch(data.all_tickers(), C.START, downloader=downloader, sleep=sleep, log=log)
    ds = data.build_dataset(close_all, volume_all, now, log=log)
    log(f"Data through {ds.run_date.date()} | {len(ds.stocks)} stocks | {len(ds.idx.columns)} markets")

    # 2. features
    mac = features.macro_features(ds.macro)
    frames = {m: features.market_frame(m, ds, mac) for m in ds.idx.columns}
    panel = features.stock_panel(ds)

    # 3. model health (monthly), used to calibrate everything below
    hpath = log_dir / "health.json"
    h = health_mod.load(hpath)
    if health_mod.needs_refresh(h, ds.run_date, force_health):
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
    mnew, snew = scoring.log_frames(mdash, ranking, ds.run_date)
    mlog = scoring.append_log(log_dir / "log_markets.csv", mnew, ["run_date", "key"])
    slog = scoring.append_log(log_dir / "log_stocks.csv", snew, ["run_date", "ticker"])
    scorecard = scoring.build_scorecard(mlog, slog, ds)

    # 6. free headlines (context only), then the payload without the optional commentary, validated
    hl = headlines.fetch(fetcher=headlines_fetcher, now=now, log=log) if (use_headlines and not demo) else []
    payload = report.build_payload(ds, mdash, ranking, stock_meta, h, scorecard, mlog, slog, ds.run_date, demo=demo,
                                   headlines=hl)
    report.validate_payload(util.clean(payload))

    # 7. optional AI commentary: never affects a forecast
    if use_outlook and not demo and (outlook_client is not None or api_key or os.environ.get("ANTHROPIC_API_KEY")):
        picks = [s["ticker"] for s in payload["stocks"] if s["base_w"] > 0]
        text = outlook.generate(report.plain_text(payload), outlook.fetch_headlines(["^GSPC"] + picks) if outlook_client is None else {},
                                api_key=api_key, client=outlook_client, log=log)
        if text:
            payload["outlook"] = {"text": text, "model": C.OUTLOOK_MODEL,
                                  "generated_utc": payload["generated_utc"]}

    # 8. write (atomically) only now that everything has passed
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
    args = ap.parse_args(argv)
    try:
        run(Path(__file__).resolve().parent, force_health=args.refresh_health, use_outlook=not args.no_outlook,
            use_headlines=not args.no_headlines)
    except data.DataError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
