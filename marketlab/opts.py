"""Settings for the news and company-data features, with safe defaults.

Change them in marketlab/config.py (add the same name there). If your config.py is older than these features,
the defaults below apply, so an out-of-date settings file can never break a weekly run.
"""
from . import config as C

DEFAULTS = {
    # ---- news
    "NEWS_ENABLED": True,
    "NEWS_PICK_CHECK": True,              # tone and headlines for the week's top picks
    "NEWS_MIN_DAYS_EXPERIMENT": 1100,     # trading days of news needed before the model experiment can run
    "NEWS_PLACEBOS": 9,                   # shifted-news stand-ins each real result must beat (at least 4)
    "NEWS_EXPERIMENT_BUDGET": 900,        # seconds; the experiment stops starting new markets after this
    "NEWS_BACKFILL_SPANS": ["2900d", "1450d", "720d", "360d"],   # tried in this order, once a month at most
    "NEWS_LIVE_SPANS": ["90d", "3months", "12weeks"],             # GDELT's normal window, written in different units
    "NEWS_BACKFILL_RETRY_DAYS": 28,
    "NEWS_TIME_BUDGET": 420,              # seconds; the news step stops politely after this
    "GDELT_URL": "https://api.gdeltproject.org/api/v2/doc/doc",
    "NEWS_QUERIES": {
        "global": '("stock market" OR "financial markets" OR "Wall Street" OR "global economy")',
        "spx": '("S&P 500" OR "Wall Street" OR "Dow Jones")',
        "ndx": '(Nasdaq OR "tech stocks")',
        "ftse": '("FTSE 100" OR "London stocks" OR "UK stocks")',
        "sx5e": '("Euro Stoxx" OR "European stocks" OR DAX OR CAC)',
        "n225": '(Nikkei OR "Tokyo stocks")',
    },
    "NEWS_STOCK_QUERIES": {
        "AAPL": '("Apple Inc" OR iPhone)', "MSFT": '(Microsoft OR Azure)', "NVDA": '(Nvidia)',
        "AMZN": '("Amazon.com" OR "Amazon Web Services")', "GOOGL": '(Alphabet OR Google)',
        "META": '("Meta Platforms" OR Facebook OR Instagram)', "JPM": '("JPMorgan" OR "JP Morgan")',
        "BAC": '("Bank of America")', "XOM": '("Exxon Mobil" OR ExxonMobil)', "CVX": '(Chevron)',
        "JNJ": '("Johnson & Johnson")', "PFE": '(Pfizer)', "PG": '("Procter & Gamble")', "KO": '("Coca-Cola")',
        "PEP": '(PepsiCo)', "CAT": '(Caterpillar)', "HD": '("Home Depot")', "UNH": '(UnitedHealth)',
        "WMT": '(Walmart)',
        "AZN.L": '(AstraZeneca)', "SHEL.L": '("Shell plc" OR "Royal Dutch Shell")', "HSBA.L": '(HSBC)',
        "ULVR.L": '(Unilever)', "BP.L": '("BP plc")', "GSK.L": '(GSK OR GlaxoSmithKline)', "RIO.L": '("Rio Tinto")',
        "DGE.L": '(Diageo)', "BARC.L": '(Barclays)', "LLOY.L": '("Lloyds Banking Group" OR "Lloyds Bank")',
        "BATS.L": '("British American Tobacco")', "REL.L": '(RELX)',
        "SAP.DE": '("SAP SE" OR "SAP software")', "SIE.DE": '(Siemens)', "MC.PA": '(LVMH OR "Louis Vuitton")',
        "ASML.AS": '(ASML)', "7203.T": '(Toyota)', "6758.T": '("Sony Group" OR "Sony Corp")',
    },
    # ---- company data
    "FUND_ENABLED": True,
    "FUND_HORIZON": 21,                   # trading days over which the company-data score is checked afterwards
    "FUND_MIN_WEEKS": 24,                 # weekly snapshots scored before any verdict is shown
    "FUND_TIME_BUDGET": 420,              # seconds; the company-data step stops politely after this
    "FUND_PAUSE": 0.8,                    # seconds between company requests
    "EARNINGS_WARN_DAYS": 7,
    # ---- paper trading (pretend money: nothing here ever places a real order)
    "PAPER_ENABLED": True,
    "PAPER_MIN_WEEKS": 26,                # weeks of paper results before any verdict is shown
    "PAPER_MIN_GAP_DAYS": 5,              # runs closer together than this count as the same week
    "PAPER_TRADE_MIN": 0.005,             # a weight change smaller than this (0.5% of the account) is "hold"
    # ---- growth watch: the weekly top stocks by upside potential, and the paper portfolio built from them
    "GROWTH_ENABLED": True,
    "GROWTH_TOP_N": 20,
    "GROWTH_TILT": 0.5,                   # share of the portfolio weighted by score (the rest is split equally)
    "GROWTH_MAX_W": 0.10,                 # no single stock above this share of the growth portfolio
    # ---- global drivers: extra market-wide series (rates, currencies, commodities, credit, regions, sectors)
    "MACRO_ENABLED": True,
    "MACRO_TEST_BUDGET": 2400,            # seconds the monthly "do the extra drivers help?" test may use
    "MACRO_MIN_YEARS_WON": 0.6,           # share of test years the stock ranker must improve in
    "MACRO_MIN_AUC_GAIN": 0.002,          # and the average yearly AUC gain must be at least this
    "MACRO_EXTRA": {
        "us5y": "^FVX", "us30y": "^TYX", "vix3m": "^VIX3M",
        "brent": "BZ=F", "natgas": "NG=F", "silver": "SI=F",
        "eurusd": "EURUSD=X", "gbpusd": "GBPUSD=X", "usdjpy": "JPY=X",
        "tlt": "TLT", "eem": "EEM", "china": "FXI", "semis": "SOXX", "smallcap": "IWM", "equalw": "RSP",
    },
}


def get(name):
    return getattr(C, name, DEFAULTS[name])
