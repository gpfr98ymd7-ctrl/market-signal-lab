"""Shared test helpers."""
import json
import re
import zlib
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from marketlab import config as C
from marketlab import data, synthetic

_MISSING = object()


def silent(*a, **k):
    pass


@contextmanager
def patched(**kw):
    """Temporarily override settings in marketlab.config (including settings the file does not define)."""
    old = {k: getattr(C, k, _MISSING) for k in kw}
    try:
        for k, v in kw.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                delattr(C, k)
            else:
                setattr(C, k, v)


def make_ds(end, now=None, downloader=None):
    """A validated Dataset of synthetic data ending on `end`."""
    now = now or (pd.Timestamp(end) + pd.Timedelta(days=1, hours=7))
    dl = downloader or synthetic.make_downloader(end=end)
    close, vol = data.fetch(data.all_tickers(), C.START, downloader=dl, sleep=0, log=silent)
    return data.build_dataset(close, vol, pd.Timestamp(now), log=silent)


# ---------------------------------------------------------------- fake news source (no network)
_NOISE = np.random.default_rng(11).normal(0, 0.5, 20000)


def _span_days(span):
    m = re.match(r"(\d+)(months|days)", span)
    n = int(m.group(1))
    return n * 30 if m.group(2) == "months" else n


def fake_gdelt(last="2026-10-08", calls=None, fail=None, tone_shift=0.0):
    """A stand-in for the GDELT DOC API. The same date always gives the same value, whatever window is asked for."""
    def fetch(url):
        qs = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        if calls is not None:
            calls.append(qs)
        if fail is not None and fail(qs):
            raise ConnectionError("down")
        mode = qs["mode"]
        if mode == "artlist":
            arts = [{"url": f"https://example.com/{i}", "title": f"Headline {i} about {qs['query'][:12]}",
                     "seendate": "20261007T120000Z", "domain": "example.com"} for i in range(4)]
            arts.insert(1, {"url": "javascript:alert(1)", "title": "Unsafe link", "seendate": "20261007T120000Z", "domain": "x"})
            return json.dumps({"articles": arts}).encode()
        n = _span_days(qs["timespan"])
        dates = pd.date_range(end=last, periods=n)
        salt = zlib.crc32(qs["query"].encode()) % 1000
        base = -2.0 + tone_shift if mode == "timelinetone" else 1.5
        vals = [base + _NOISE[(salt + (d - pd.Timestamp("2010-01-01")).days) % len(_NOISE)] for d in dates]
        if mode == "timelinevol":
            vals = [abs(v) for v in vals]
        rows = [{"date": d.strftime("%Y%m%dT000000Z"), "value": round(float(v), 4)} for d, v in zip(dates, vals)]
        return json.dumps({"timeline": [{"series": "x", "data": rows}]}).encode()
    return fetch


# ---------------------------------------------------------------- fake company data (no network)
def fake_company(now="2026-10-09", calls=None, dead=False):
    now = pd.Timestamp(now)

    def fetch(ticker):
        if calls is not None:
            calls.append(ticker)
        if dead:
            raise ConnectionError("refused")
        h = zlib.crc32(ticker.encode())
        q = (h % 100) / 100
        soon = h % 5 == 0
        return {
            "info": {"trailingPE": 10 + 40 * (1 - q), "forwardPE": 9 + 30 * (1 - q), "priceToBook": 1 + 5 * q,
                     "profitMargins": 0.05 + 0.2 * q, "operatingMargins": 0.08 + 0.25 * q,
                     "returnOnEquity": 0.05 + 0.3 * q, "revenueGrowth": -0.05 + 0.2 * q,
                     "earningsGrowth": -0.1 + 0.4 * q, "debtToEquity": 200 * (1 - q) + 10,
                     "freeCashflow": 1e9 * (0.2 + q), "marketCap": 5e10, "sector": "Financial Services" if h % 7 == 0 else "Technology",
                     "shortName": ticker + " Inc"},
            "calendar": {"Earnings Date": [(now + pd.Timedelta(days=3 if soon else 40)).date()]},
        }
    return fetch


RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Central bank holds rates as inflation cools</title><link>https://example.com/a</link><pubDate>Thu, 08 Oct 2026 09:00:00 GMT</pubDate></item>
<item><title>Oil slips on demand worries</title><link>javascript:alert(1)</link><pubDate>Thu, 08 Oct 2026 07:30:00 GMT</pubDate></item>
<item><title>Tech earnings season opens with mixed results</title><link>https://example.com/c</link><pubDate>Wed, 07 Oct 2026 16:00:00 GMT</pubDate></item>
</channel></rss>"""
