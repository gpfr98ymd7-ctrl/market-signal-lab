"""Free global headlines (public RSS feeds), shown on the dashboard as context.

They are never used by the model and never change a forecast: they exist so the page shows what is in the news
next to what the numbers say. Failures are quiet: no headlines is a normal outcome, never an error.
"""
import re
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen

import pandas as pd

from . import config as C

try:                                                    # protects against hostile XML if the package is installed
    from defusedxml import ElementTree as ET
except ImportError:                                     # pragma: no cover
    import xml.etree.ElementTree as ET

FEEDS = [
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Guardian Business", "https://www.theguardian.com/uk/business/rss"),
]
MAX_BYTES = 2_000_000
PER_SOURCE = 5


def _http_get(url, timeout=15):
    req = Request(url, headers={"User-Agent": "market-lab/2.0 (personal research tool)"})
    with urlopen(req, timeout=timeout) as r:
        return r.read(MAX_BYTES)


def _safe_link(u):
    """Only plain web links: never a javascript: or data: URL."""
    u = (u or "").strip()
    return u if re.match(r"^https?://", u, re.I) else None


def _when(text):
    try:
        t = parsedate_to_datetime(text)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return pd.Timestamp(t).tz_convert("UTC").tz_localize(None)
    except Exception:
        return None


def parse_rss(xml_bytes, source):
    """Items from one RSS document: dicts with title, link, source, published (UTC timestamp or None)."""
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.iter("item"):
        title = re.sub(r"\s+", " ", (it.findtext("title") or "")).strip()
        if not title:
            continue
        out.append({"title": title[:160], "link": _safe_link(it.findtext("link")), "source": source,
                    "published": _when(it.findtext("pubDate"))})
    return out


def memoize(fetcher=None):
    """Wrap a fetcher so each URL is requested once per run, however many times it is asked for."""
    inner, cache = fetcher or _http_get, {}

    def get(url):
        if url not in cache:
            cache[url] = inner(url)
        return cache[url]
    return get


def fetch_all(feeds=None, fetcher=None, now=None, log=print, max_age_days=7):
    """Every recent, de-duplicated item from the feeds, untrimmed and with real timestamps. Used for the archive."""
    fetcher = fetcher or _http_get
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    cutoff = now - pd.Timedelta(days=max_age_days)
    seen, out = set(), []
    for name, url in (feeds or FEEDS):
        try:
            items = parse_rss(fetcher(url), name)
        except Exception as e:
            log(f"Headlines archive: {name} unavailable ({type(e).__name__})")
            continue
        for it in items:
            key = (name, it["title"].lower())
            if key in seen or (it["published"] is not None and it["published"] < cutoff):
                continue
            seen.add(key)
            out.append(it)
    return out


def fetch(feeds=None, fetcher=None, now=None, log=print):
    """Recent, de-duplicated headlines, newest first, spread across sources. Returns [] on any problem."""
    fetcher = fetcher or _http_get
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    cutoff = now - pd.Timedelta(days=C.HEADLINES_MAX_AGE_DAYS)
    seen, per_src = set(), []
    for name, url in (feeds or FEEDS):
        try:
            items = parse_rss(fetcher(url), name)
        except Exception as e:
            log(f"Headlines: {name} unavailable ({type(e).__name__})")
            continue
        keep = []
        for it in items:
            key = it["title"].lower()
            pub = it["published"]
            if key in seen or (pub is not None and pub < cutoff):
                continue
            seen.add(key)
            keep.append(it)
        keep.sort(key=lambda x: x["published"] or pd.Timestamp.min, reverse=True)
        per_src.append(keep[:PER_SOURCE])
    merged = [it for group in per_src for it in group]
    merged.sort(key=lambda x: x["published"] or pd.Timestamp.min, reverse=True)
    return [{"title": it["title"], "link": it["link"], "source": it["source"],
             "published": it["published"].strftime("%Y-%m-%d %H:%M") if it["published"] is not None else None}
            for it in merged[:C.HEADLINES_MAX]]
