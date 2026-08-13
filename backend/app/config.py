"""Central configuration: paths, API access, instrument sets and analytic constants."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BIGGIE_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DB = DATA_DIR / "market.db"
SNAPSHOT_PATH = DATA_DIR / "snapshot.json"
FRONTEND_DIR = ROOT / "frontend"

# --------------------------------------------------------------------------------------
# FMP API
# --------------------------------------------------------------------------------------
FMP_BASE = "https://financialmodelingprep.com/stable"


def api_key() -> str:
    """Resolve the FMP key. ``FMP_API_KEY`` wins so the generic ``API_KEY`` can be reused."""
    key = os.environ.get("FMP_API_KEY") or os.environ.get("API_KEY") or ""
    if not key:
        raise RuntimeError(
            "No FMP API key found. Set FMP_API_KEY (or API_KEY) in the environment."
        )
    return key


# Concurrency / politeness. FMP throttles per minute; these defaults stay well inside
# typical paid limits while still pulling ~1000 symbols in a couple of minutes.
MAX_CONCURRENCY = int(os.environ.get("BIGGIE_CONCURRENCY", "12"))
REQUEST_TIMEOUT = float(os.environ.get("BIGGIE_TIMEOUT", "30"))
MAX_RETRIES = 4

# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------
UNIVERSE_SIZE = int(os.environ.get("BIGGIE_UNIVERSE_SIZE", "1000"))

#: Some freshly-listed names pass the screener but lack enough history to rank, so the
#: candidate pool is over-selected and trimmed back to UNIVERSE_SIZE once history is
#: known. Without this the universe quietly lands short of its target.
UNIVERSE_BUFFER = 1.2
HISTORY_YEARS = 2
# Calendar days pulled; ~504 trading days live inside 2 years, plus slack for holidays.
HISTORY_DAYS = 365 * HISTORY_YEARS + 20

# Liquidity / quality gates. Deliberately broad: anything a retail user could realistically
# trade should survive, so these only cut genuine micro-caps and untradeable names.
MIN_MARKET_CAP = 3.0e8       # $300M
MIN_PRICE = 3.0              # avoids sub-$3 tape where spreads dominate
MIN_AVG_DOLLAR_VOLUME = 1.0e6  # $1M/day median notional
MIN_HISTORY_DAYS = 260       # need at least ~1y of bars to rank a 12-1 window

# Major US listing venues only. This is also what removes OTC / pink sheets, whose data
# quality is too poor to rank on.
ELIGIBLE_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSEAMERICAN", "NYSE AMERICAN"}

# No domicile gate: US-listed foreign issuers and ADRs trade on the same venues with the
# same data quality, and "any stock a user could realistically trade" includes them.

# --------------------------------------------------------------------------------------
# Reference instruments
# --------------------------------------------------------------------------------------
MARKET_ETF = "VTI"
BOND_ETF = "TLT"

# The spec lists ten sector ETFs but omits XLY (Consumer Discretionary). Without it the
# factor model has no sector proxy for ~10% of the universe and the macro sector grid
# reads as broken, so the full GICS set of eleven is used.
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY"]
CORE_INDEX_ETFS = ["SPY", "QQQ", "IWM", "VTI"]
RATES_ETFS = ["TLT", "IEF", "SHY"]
COMMODITY_ETFS = ["GLD", "SLV", "USO"]

#: Everything the macro page ranks.
MACRO_INSTRUMENTS = CORE_INDEX_ETFS + SECTOR_ETFS + RATES_ETFS + COMMODITY_ETFS

#: Everything the factor model and macro page need downloaded.
REFERENCE_TICKERS = sorted(set(MACRO_INSTRUMENTS) | {MARKET_ETF, BOND_ETF})

MACRO_ASSET_CLASS = {
    **{t: "Index" for t in CORE_INDEX_ETFS},
    **{t: "Sector" for t in SECTOR_ETFS},
    **{t: "Rates" for t in RATES_ETFS},
    **{t: "Commodity" for t in COMMODITY_ETFS},
}

MACRO_LABELS = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "VTI": "US Total Market",
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLI": "Industrials", "XLP": "Cons. Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Comm. Services", "XLY": "Cons. Discretionary",
    "TLT": "20Y+ Treasuries", "IEF": "7-10Y Treasuries", "SHY": "1-3Y Treasuries",
    "GLD": "Gold", "SLV": "Silver", "USO": "Crude Oil",
}

# --------------------------------------------------------------------------------------
# Sector classification
# --------------------------------------------------------------------------------------
#: FMP publishes its own sector strings; map them onto GICS sector names, which is the
#: closest available classification and what the sector ETFs are actually built on.
GICS_SECTORS = [
    "Information Technology", "Financials", "Energy", "Health Care", "Industrials",
    "Consumer Staples", "Utilities", "Materials", "Real Estate",
    "Communication Services", "Consumer Discretionary",
]

FMP_SECTOR_TO_GICS = {
    "Technology": "Information Technology",
    "Information Technology": "Information Technology",
    "Financial Services": "Financials",
    "Financial": "Financials",
    "Financials": "Financials",
    "Energy": "Energy",
    "Healthcare": "Health Care",
    "Health Care": "Health Care",
    "Industrials": "Industrials",
    "Industrial Goods": "Industrials",
    "Consumer Defensive": "Consumer Staples",
    "Consumer Staples": "Consumer Staples",
    "Utilities": "Utilities",
    "Basic Materials": "Materials",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Discretionary": "Consumer Discretionary",
}

#: Which sector ETF proxies each GICS sector in the factor model.
SECTOR_TO_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
}

UNKNOWN_SECTOR = "Unclassified"

# --------------------------------------------------------------------------------------
# Ranking windows
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A momentum window: ``lookback`` trading days ending ``skip`` days before today."""

    key: str
    label: str
    lookback: int
    skip: int

    @property
    def total(self) -> int:
        return self.lookback + self.skip


#: Skip logic scales with window length (~8% of the lookback), so *every* window excludes
#: a proportional slice of short-term reversal - not just the 12-month one. Labels are
#: plain durations for that reason: "12-1M" alongside a bare "6M" would imply that only
#: the first window skips anything.
WINDOWS: dict[str, Window] = {
    "mom_12_1": Window("mom_12_1", "12M", 250, 20),
    "mom_6_1": Window("mom_6_1", "6M", 125, 10),
    "mom_3_1": Window("mom_3_1", "3M", 60, 5),
}
DEFAULT_WINDOW = "mom_12_1"

#: Windows used for the per-ticker return sparklines (no skip, plain trailing returns).
RETURN_WINDOWS = [
    ("r_5d", "5D", 5),
    ("r_10d", "10D", 10),
    ("r_1m", "1M", 21),
    ("r_3m", "3M", 63),
    ("r_6m", "6M", 126),
    ("r_12m", "12M", 252),
]

TRADING_DAYS = 252

# Covariance estimation
COV_MIN_OBS = 60            # below this a covariance column is untrustworthy
COV_BLEND_FLOOR = 0.25      # marginal-risk floor as a fraction of standalone vol

# Clustering
DEFAULT_CLUSTER_K = 8
MAX_CLUSTER_K = 25
