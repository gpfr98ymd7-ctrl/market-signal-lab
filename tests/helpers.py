"""Shared test helpers."""
from contextlib import contextmanager

import pandas as pd

from marketlab import config as C
from marketlab import data, synthetic


def silent(*a, **k):
    pass


@contextmanager
def patched(**kw):
    """Temporarily override settings in marketlab.config."""
    old = {k: getattr(C, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def make_ds(end, now=None, downloader=None):
    """A validated Dataset of synthetic data ending on `end`."""
    now = now or (pd.Timestamp(end) + pd.Timedelta(days=1, hours=7))
    dl = downloader or synthetic.make_downloader(end=end)
    close, vol = data.fetch(data.all_tickers(), C.START, downloader=dl, sleep=0, log=silent)
    return data.build_dataset(close, vol, pd.Timestamp(now), log=silent)
