"""Downloading market data, with retries, and refusing to proceed on bad data."""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


class DataError(RuntimeError):
    """The data is too incomplete or stale to produce trustworthy forecasts."""


@dataclass
class Dataset:
    idx: pd.DataFrame          # market index closes, columns = keys (spx, ndx, ...)
    idx_ff: pd.DataFrame       # same, forward-filled across other markets' trading days
    macro: pd.DataFrame        # macro closes, columns = keys (vix, us10y, ...), forward-filled
    close: pd.DataFrame        # stock closes, columns = tickers
    volume: pd.DataFrame
    stocks: list
    run_date: pd.Timestamp
    warnings: list = field(default_factory=list)


def yahoo_downloader(tickers, start):
    import yfinance as yf
    return yf.download(list(tickers), start=start, auto_adjust=True, progress=False, threads=True)


def all_tickers():
    return list(dict.fromkeys(list(C.STOCKS) + list(C.MARKETS.values()) + list(C.MACRO.values())))


def _split(raw, tickers):
    """Return (close, volume) frames from a downloader result, or (None, None)."""
    if raw is None or len(raw) == 0:
        return None, None
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        lv0 = set(cols.get_level_values(0))
        lv1 = set(cols.get_level_values(1))
        if "Close" in lv0:
            close = raw["Close"]
            vol = raw["Volume"] if "Volume" in lv0 else None
        elif "Close" in lv1:
            close = raw.xs("Close", axis=1, level=1)
            vol = raw.xs("Volume", axis=1, level=1) if "Volume" in lv1 else None
        else:
            return None, None
    elif "Close" in cols and len(tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        vol = raw[["Volume"]].rename(columns={"Volume": tickers[0]}) if "Volume" in cols else None
    else:
        return None, None
    if vol is None:
        vol = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    return close, vol


def fetch(tickers, start, downloader=None, retries=3, sleep=5.0, log=print):
    """Download everything, retrying with a growing pause, then re-fetch any ticker that came back empty."""
    downloader = downloader or yahoo_downloader
    tickers = list(dict.fromkeys(tickers))
    close = volume = None
    for attempt in range(retries):
        try:
            c, v = _split(downloader(tickers, start), tickers)
            if c is not None and len(c) > 0:
                close, volume = c.copy(), v.copy()
                break
            log(f"Download attempt {attempt + 1}: no data returned.")
        except Exception as e:  # network errors, rate limits, library changes
            log(f"Download attempt {attempt + 1} failed: {e}")
        if attempt < retries - 1:
            time.sleep(sleep * (attempt + 1))
    if close is None:
        raise DataError("Could not download any price data. The data provider may be rate-limiting "
                        "cloud servers: re-run later.")

    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    for t in missing:
        for attempt in range(2):
            try:
                c, v = _split(downloader([t], start), [t])
                if c is not None and t in c.columns and c[t].notna().any():
                    close[t] = c[t]
                    volume[t] = v[t] if t in v.columns else np.nan
                    log(f"Recovered {t} on a second request.")
                    break
            except Exception as e:
                log(f"Retry for {t} failed: {e}")
            time.sleep(sleep)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    volume = volume[~volume.index.duplicated(keep="last")].sort_index()
    return close, volume


def build_dataset(close_all, volume_all, now, log=print):
    """Validate the raw download and arrange it for the model. Raises DataError when unsafe to continue."""
    close_all, volume_all = close_all.copy(), volume_all.copy()
    for df in (close_all, volume_all):
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.DatetimeIndex(df.index).normalize()
    now = pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)

    # A bar dated today, fetched before the US close, is unfinished: drop it.
    if len(close_all) and close_all.index.max() == now.normalize() and now.hour < 21:
        close_all, volume_all = close_all.iloc[:-1], volume_all.iloc[:-1]
        log("Dropped today's unfinished price bar.")

    spx_t = C.MARKETS["spx"]
    core = list(C.MARKETS.values()) + list(C.MACRO.values())
    bad = [t for t in core if t not in close_all.columns or close_all[t].dropna().shape[0] < 800]
    if bad:
        raise DataError(f"Missing or too-short market/macro data for: {', '.join(bad)}")
    ref_last = close_all[spx_t].dropna().index.max()
    age = (now.normalize() - ref_last).days
    if age > C.MAX_STALE_DAYS:
        raise DataError(f"The latest S&P 500 data is {age} days old, which is too stale to forecast from.")

    warnings = []
    keep = []
    for t in C.STOCKS:
        if t not in close_all.columns:
            warnings.append(f"{t}: no data returned, skipped.")
            continue
        s = close_all[t].dropna()
        if len(s) < 400:
            warnings.append(f"{t}: under 400 days of history, skipped.")
            continue
        if (ref_last - s.index.max()).days > 10:
            warnings.append(f"{t}: no price for {(ref_last - s.index.max()).days} days, skipped.")
            continue
        gaps = s.index[-252:].to_series().diff().dt.days.max()
        if gaps is not None and not np.isnan(gaps) and gaps > 10:
            warnings.append(f"{t}: a gap of {int(gaps)} days in the last year, skipped.")
            continue
        if s.pct_change().abs().tail(504).max() > 0.5:
            warnings.append(f"{t}: a daily move above 50% in the last two years. Kept, but check for bad data.")
        keep.append(t)
    if len(keep) < max(C.TOP_K + 4, 10):
        raise DataError(f"Only {len(keep)} usable stocks remain after data checks.")
    for w in warnings:
        log("Data warning: " + w)

    inv = {v: k for k, v in C.MARKETS.items()}
    idx = close_all[list(C.MARKETS.values())].rename(columns=inv)
    macro = close_all[list(C.MACRO.values())].rename(columns={v: k for k, v in C.MACRO.items()}).ffill()
    return Dataset(idx=idx, idx_ff=idx.ffill(), macro=macro, close=close_all[keep],
                   volume=volume_all[keep], stocks=keep, run_date=ref_last, warnings=warnings)
