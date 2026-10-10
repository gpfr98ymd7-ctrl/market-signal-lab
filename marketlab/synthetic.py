"""Synthetic market data for tests and the demo dashboard.

Not real prices. It has a two-state volatility regime (so volatility is forecastable, as in real
markets) but NO exploitable drift, so a sound model should find little or no direction skill.
Paths are deterministic and extend consistently: data ending later is the same path, just longer.
"""
import zlib

import numpy as np
import pandas as pd

from . import config as C


def _rng(key, seed=0):
    return np.random.default_rng((zlib.crc32(key.encode()) + 7919 * seed) % (2 ** 32))


class World:
    def __init__(self, start="2012-01-01", end="2027-03-01", seed=0):
        self.idx = pd.bdate_range(start, end)
        self.n = n = len(self.idx)
        self.seed = seed
        rng = _rng("__market__", seed)
        u = rng.random(n)
        state = np.zeros(n, dtype=int)
        for i in range(1, n):
            stay = 0.992 if state[i - 1] == 0 else 0.97
            state[i] = state[i - 1] if u[i] < stay else 1 - state[i - 1]
        self.vol = np.where(state == 0, 0.007, 0.018) * np.exp(rng.normal(0, 0.1, n))
        self.f = rng.normal(0.0004, 1.0, n) * self.vol      # common market factor
        self._cache = {}

    def series(self, t):
        """(close, volume) arrays over the full calendar for one ticker."""
        if t in self._cache:
            return self._cache[t]
        n, f, vol = self.n, self.f, self.vol
        rng = _rng(t, self.seed)
        if t in set(C.MARKETS.values()):
            load = 0.9 + 0.2 * rng.random()
            c = 1000 * (1 + 4 * rng.random()) * np.exp(np.cumsum(load * f + rng.normal(0, 0.003, n)))
        elif t == "^VIX":
            c = np.clip(100 * vol * np.sqrt(252) * np.exp(rng.normal(0, 0.08, n)), 9, 80)
        elif t == "^TNX":
            c = np.clip(2.5 + np.cumsum(rng.normal(0, 0.02, n)), 0.5, 6.0)
        elif t == "^IRX":
            c = np.clip(self.series("^TNX")[0] - 1.0 + 0.2 * np.cumsum(rng.normal(0, 0.01, n)), 0.05, 6.0)
        elif t == "DX-Y.NYB":
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        elif t == "GC=F":
            c = 1500 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
        elif t == "CL=F":
            c = 60 * np.exp(np.cumsum(rng.normal(0, 0.02, n) + 0.3 * f))
        elif t == "HG=F":
            c = 3 * np.exp(np.cumsum(0.3 * f + rng.normal(0, 0.012, n)))
        elif t == "HYG":
            c = 80 * np.exp(np.cumsum(0.25 * f + rng.normal(0, 0.002, n)))
        elif t == "LQD":
            c = 110 * np.exp(np.cumsum(rng.normal(0.00005, 0.003, n)))
        else:                                               # a stock
            beta = 0.7 + 0.6 * rng.random()
            idio = rng.normal(0, 0.011, n) * (0.8 + 0.4 * rng.random())
            c = (50 + 100 * rng.random()) * np.exp(np.cumsum(beta * f + idio + 0.0001))
        v = np.exp(rng.normal(14.5, 0.3, n)) * (1 + 5 * vol)
        self._cache[t] = (c, v)
        return c, v


def make_downloader(end="2026-10-08", seed=0, start="2012-01-01", world_end="2027-03-01"):
    """A stand-in for the Yahoo downloader, returning data up to (and including) `end`."""
    world = World(start, world_end, seed)
    end_ts = pd.Timestamp(end)

    def download(tickers, start_str):
        keep = (world.idx >= pd.Timestamp(start_str)) & (world.idx <= end_ts)
        idx = world.idx[keep]
        close = pd.DataFrame({t: world.series(t)[0][keep] for t in tickers}, index=idx)
        volume = pd.DataFrame({t: world.series(t)[1][keep] for t in tickers}, index=idx)
        return pd.concat({"Close": close, "Volume": volume}, axis=1)

    return download
