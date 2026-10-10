"""Settings you are most likely to want to change. Every other module reads from here."""

VERSION = "2.0"

# ---- forecasting setup -----------------------------------------------------
START = "2005-01-01"   # how far back to download history
H = 21                 # market forecast horizon, in trading days (about one month)
SH = 5                 # stock forecast horizon, in trading days (about one week)
ENTRY_DELAY = 1        # stocks are assumed bought this many trading days AFTER the signal date
CRASH = -0.05          # a market "crash" = closing more than 5% below today's level within H days
LAG = 1                # cross-market and macro inputs are lagged by this many rows
TOP_K = 5              # stocks in the illustrative portfolio
VOL_TARGET = 0.15      # annual volatility the market-risk scaler aims for
COST_BPS = 10          # stock trading cost charged on turnover, in the backtest and live scorecard (10 bps = 0.10%)
COST_BPS_MARKET = 5    # cost charged on changes in market exposure in the risk-scaler backtest

# ---- model health and calibration -------------------------------------------
HEALTH_VERSION = "3"        # bump to force a recompute of the cached health file
HEALTH_REFRESH_DAYS = 28    # the walk-forward check is re-run about monthly
MIN_TRAIN_YEARS_MARKET = 5
MIN_TRAIN_YEARS_STOCK = 4
DEFAULT_LAM = 0.5           # shrinkage used before any health check exists (0 = base rate only, 1 = raw model)

# ---- free headlines shown on the dashboard (context only, never used by the model) ----
HEADLINES_MAX = 12
HEADLINES_MAX_AGE_DAYS = 4

# ---- data checks ------------------------------------------------------------
MAX_STALE_DAYS = 7          # refuse to run if the S&P 500 data is older than this

# ---- optional AI commentary (needs an Anthropic API key as a GitHub secret) ---
OUTLOOK_MODEL = "claude-sonnet-5-5"

# ---- universe ---------------------------------------------------------------
US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "BAC", "XOM", "CVX",
      "JNJ", "PFE", "PG", "KO", "PEP", "CAT", "HD", "UNH", "WMT"]
UK = ["AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L",
      "BARC.L", "LLOY.L", "BATS.L", "REL.L"]
EU = ["SAP.DE", "SIE.DE", "MC.PA", "ASML.AS"]
JP = ["7203.T", "6758.T"]
STOCKS = US + UK + EU + JP

MARKETS = {"spx": "^GSPC", "ndx": "^NDX", "ftse": "^FTSE", "sx5e": "^STOXX50E", "n225": "^N225"}
NAMES = {"spx": "S&P 500", "ndx": "Nasdaq 100", "ftse": "FTSE 100", "sx5e": "Euro Stoxx 50", "n225": "Nikkei 225"}
MACRO = {"vix": "^VIX", "us10y": "^TNX", "us3m": "^IRX", "dxy": "DX-Y.NYB", "gold": "GC=F",
         "oil": "CL=F", "copper": "HG=F", "hyg": "HYG", "lqd": "LQD"}


def region_of(ticker):
    """Which market's risk scaler applies to a stock."""
    if ticker.endswith(".L"):
        return "ftse"
    if ticker.endswith((".DE", ".PA", ".AS")):
        return "sx5e"
    if ticker.endswith(".T"):
        return "n225"
    return "spx"
