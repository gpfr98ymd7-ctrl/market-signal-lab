"""Which stocks the app follows.

You can list them in a plain text file called watchlist.txt in the repository (edit it from the GitHub app: open
the file, tap the pencil, change the lines, commit). If the file is missing, or unusable, the built-in list in
marketlab/config.py is used and the dashboard says so. A bad watchlist can never stop the weekly run.
"""
import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config as C

FILE = "watchlist.txt"
MIN_STOCKS = 14            # the stock ranker compares stocks with each other, so it needs a decent crowd
MAX_STOCKS = 120           # keeps the weekly run inside its time limit
REFRESH_BELOW_OVERLAP = 0.8   # the monthly model check is redone early if the list changed by more than about a fifth

_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$")
_CLASS_SHARE = re.compile(r"^([A-Z]{1,5})\.([AB])$")           # BRK.B is spelled BRK-B on Yahoo (but BARC.L is London)
EUROPE = (".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".HE", ".LS", ".VI", ".IR", ".SW", ".ST", ".CO", ".OL")


def region_of(ticker):
    """Which market's risk scaler applies to a stock: the index of the stock's home market, else the S&P 500."""
    t = ticker.upper()
    if t.endswith(".L"):
        return "ftse"
    if t.endswith(EUROPE):
        return "sx5e"
    if t.endswith(".T"):
        return "n225"
    return "spx"


def _has_own_scaler(ticker):
    """False for a share listed somewhere this app has no index for (for example Hong Kong or Toronto)."""
    t = ticker.upper()
    return "." not in t or region_of(t) != "spx"


def parse(text):
    """Text to (tickers, notes). Anything after a # is a comment; tickers can be on separate lines or separated by
    spaces or commas. Letters are upper-cased and duplicates removed."""
    tickers, notes, seen, dupes = [], [], set(), 0
    bad = []
    for line in str(text or "").splitlines():
        line = line.split("#", 1)[0]
        for tok in re.split(r"[,\s;]+", line.strip()):
            if not tok:
                continue
            t = tok.strip("'\"").upper()
            m = _CLASS_SHARE.match(t)
            if m:
                fixed = f"{m.group(1)}-{m.group(2)}"
                notes.append(f"{t} was read as {fixed} (Yahoo writes share classes with a dash).")
                t = fixed
            if not _TICKER.match(t):
                bad.append(tok[:20])
                continue
            if t in seen:
                dupes += 1
                continue
            seen.add(t)
            tickers.append(t)
    if bad:
        notes.append("Ignored: " + ", ".join(bad[:6]) + (" and more" if len(bad) > 6 else "") +
                     " (not a ticker; use letters, digits, dots and dashes, for example BARC.L).")
    if dupes:
        notes.append(f"{dupes} duplicate line(s) ignored.")
    return tickers, notes


@dataclass
class Resolved:
    tickers: list
    source: str                       # "watchlist.txt" or "built-in list"
    notes: list = field(default_factory=list)
    requested: int = 0


def resolve(root, default):
    """The stocks to follow this run, with plain-English notes about anything that was ignored."""
    default = list(default)
    path = Path(root) / FILE
    if not path.exists():
        return Resolved(default, "built-in list", [], len(default))
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as e:
        return Resolved(default, "built-in list", [f"{FILE} could not be read ({type(e).__name__}); using the built-in list."],
                        len(default))
    tickers, notes = parse(text)
    n = len(tickers)
    if n < MIN_STOCKS:
        notes.append(f"{FILE} has {n} usable ticker(s) but at least {MIN_STOCKS} are needed, so the built-in list "
                     f"of {len(default)} was used this week.")
        return Resolved(default, "built-in list", notes, n)
    if n > MAX_STOCKS:
        notes.append(f"{FILE} has {n} tickers; only the first {MAX_STOCKS} are used to keep the weekly run on time.")
        tickers = tickers[:MAX_STOCKS]
    odd = [t for t in tickers if not _has_own_scaler(t)]
    if odd:
        notes.append("No matching market index for " + ", ".join(odd[:5]) + (" and more" if len(odd) > 5 else "") +
                     ", so the S&P 500 risk scaler is used for them.")
    return Resolved(tickers, FILE, notes, n)


@contextlib.contextmanager
def use(tickers):
    """Make `tickers` the stock universe for everything inside the block, then put the old one back."""
    old = C.STOCKS
    C.STOCKS = list(tickers)
    try:
        yield
    finally:
        C.STOCKS = old


def overlap(a, b):
    """Share of tickers two lists have in common (1 = identical, 0 = nothing shared)."""
    sa, sb = set(a or []), set(b or [])
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def list_changed(health, current, default):
    """True if the stock list differs a lot from the one the saved model check was made on. A health file from before
    the watchlist existed is taken to have used the built-in list."""
    if not health:
        return False
    return overlap(health.get("universe") or default, current) < REFRESH_BELOW_OVERLAP


def last_logged(path):
    """The stocks that were ranked in the most recent logged run, or None if there is no log yet."""
    import pandas as pd
    try:
        df = pd.read_csv(path, usecols=["run_date", "ticker"])
        return df.loc[df["run_date"] == df["run_date"].max(), "ticker"].astype(str).tolist() or None
    except Exception:
        return None


def block(res, ds_stocks, previous=None):
    """The small summary the dashboard shows."""
    used = list(ds_stocks)
    out = {"source": res.source, "requested": int(res.requested), "used": len(used),
           "notes": list(res.notes)[:6], "tickers": used[:MAX_STOCKS]}
    if previous:
        out["added"] = [t for t in used if t not in previous][:12]
        out["removed"] = [t for t in previous if t not in used][:12]
    return out
