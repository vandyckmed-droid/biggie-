"""Price alignment and daily return construction.

Everything downstream (ranking, covariance, HRP, macro) reads from a single aligned
return matrix, so the trading calendar and missing-data policy are decided exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config

#: A stock may be stale for at most this many sessions before its bar is treated as
#: missing rather than carried forward. Longer gaps are real halts, not data noise.
MAX_FORWARD_FILL = 5


@dataclass
class ReturnPanel:
    """Aligned daily returns for a set of symbols on one trading calendar.

    Attributes:
        dates:   ``T`` trading dates (ISO strings), oldest first. Returns are for the
                 transition *into* each date, so ``dates[0]`` is dropped from returns.
        symbols: ``N`` column labels.
        prices:  ``(T, N)`` aligned closes, ``NaN`` where a symbol has no data.
        returns: ``(T-1, N)`` simple daily returns, ``NaN`` where unavailable.
    """

    dates: list[str]
    symbols: list[str]
    prices: np.ndarray
    returns: np.ndarray

    def __post_init__(self) -> None:
        self._index = {s: i for i, s in enumerate(self.symbols)}

    # ------------------------------------------------------------------ accessors
    @property
    def return_dates(self) -> list[str]:
        return self.dates[1:]

    def has(self, symbol: str) -> bool:
        return symbol in self._index

    def col(self, symbol: str) -> int:
        return self._index[symbol]

    def series(self, symbol: str) -> np.ndarray:
        """Daily returns for one symbol."""
        return self.returns[:, self._index[symbol]]

    def price_series(self, symbol: str) -> np.ndarray:
        return self.prices[:, self._index[symbol]]

    def subset(self, symbols: Sequence[str]) -> "ReturnPanel":
        """A panel restricted to ``symbols`` (order preserved, unknown names skipped)."""
        keep = [s for s in symbols if s in self._index]
        idx = [self._index[s] for s in keep]
        return ReturnPanel(
            dates=list(self.dates),
            symbols=keep,
            prices=self.prices[:, idx],
            returns=self.returns[:, idx],
        )

    def window(self, lookback: int, skip: int = 0) -> np.ndarray:
        """``lookback`` rows of returns ending ``skip`` sessions before the last one.

        This is the momentum slice: the newest ``skip`` sessions are dropped to strip
        short-term reversal out of the signal.
        """
        n = self.returns.shape[0]
        end = n - skip
        start = max(0, end - lookback)
        if end <= start:
            return self.returns[:0]
        return self.returns[start:end]

    def coverage(self) -> np.ndarray:
        """Fraction of finite returns per symbol."""
        if self.returns.size == 0:
            return np.zeros(len(self.symbols))
        return np.isfinite(self.returns).mean(axis=0)


def build_panel(
    price_map: dict[str, list[tuple[str, float]]],
    calendar_symbol: str = config.MARKET_ETF,
    min_history: int = config.MIN_HISTORY_DAYS,
) -> ReturnPanel:
    """Align raw price series onto one trading calendar and difference into returns.

    The calendar comes from a reference ETF that trades every session, so individual
    stocks cannot introduce phantom trading days through bad ticks.
    """
    if not price_map:
        raise ValueError("no price data supplied")

    if calendar_symbol in price_map:
        calendar = [d for d, _ in price_map[calendar_symbol]]
    else:
        # Fall back to the union of all observed dates.
        calendar = sorted({d for series in price_map.values() for d, _ in series})

    date_index = {d: i for i, d in enumerate(calendar)}
    t = len(calendar)

    symbols: list[str] = []
    columns: list[np.ndarray] = []

    for symbol, series in price_map.items():
        col = np.full(t, np.nan)
        for d, close in series:
            i = date_index.get(d)
            if i is not None:
                col[i] = close
        col = _forward_fill(col, MAX_FORWARD_FILL)
        if np.isfinite(col).sum() < min_history:
            continue
        symbols.append(symbol)
        columns.append(col)

    if not symbols:
        raise ValueError("no symbol met the minimum history requirement")

    prices = np.column_stack(columns)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = prices[1:] / prices[:-1] - 1.0
    # A single bad tick can produce an absurd jump; clip to a plausible daily range so
    # one dirty bar cannot dominate a covariance estimate.
    returns = np.where(np.isfinite(returns), returns, np.nan)
    returns = np.clip(returns, -0.9, 4.0)

    order = np.argsort(symbols)
    symbols = [symbols[i] for i in order]
    return ReturnPanel(
        dates=calendar,
        symbols=symbols,
        prices=prices[:, order],
        returns=returns[:, order],
    )


def _forward_fill(col: np.ndarray, limit: int) -> np.ndarray:
    """Carry the last observation forward, but only across short gaps."""
    out = col.copy()
    last = np.nan
    run = 0
    for i, v in enumerate(out):
        if np.isfinite(v):
            last, run = v, 0
        elif np.isfinite(last) and run < limit:
            out[i] = last
            run += 1
        else:
            out[i] = np.nan
    return out


# --------------------------------------------------------------------------------------
# Window statistics
# --------------------------------------------------------------------------------------
def annualized_return(window: np.ndarray) -> np.ndarray:
    """Geometric annualised return per column, ignoring missing observations.

    Compounding the actually-observed bars and then scaling by the observation count
    keeps names with short gaps comparable to fully-populated ones.
    """
    if window.size == 0:
        return np.full(window.shape[1] if window.ndim > 1 else 0, np.nan)
    filled = np.where(np.isfinite(window), window, 0.0)
    counts = np.isfinite(window).sum(axis=0)
    growth = np.prod(1.0 + filled, axis=0)
    growth = np.where(growth <= 0, np.nan, growth)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = growth ** (config.TRADING_DAYS / np.maximum(counts, 1)) - 1.0
    return np.where(counts >= 2, out, np.nan)


def mean_daily_return(window: np.ndarray) -> np.ndarray:
    """Window-adjusted mean daily return (missing bars excluded, not zero-filled)."""
    if window.size == 0:
        return np.full(window.shape[1] if window.ndim > 1 else 0, np.nan)
    return np.nanmean(window, axis=0)


def annualized_volatility(window: np.ndarray) -> np.ndarray:
    """Annualised standard deviation per column."""
    if window.size == 0:
        return np.full(window.shape[1] if window.ndim > 1 else 0, np.nan)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(window, axis=0, ddof=1)
    counts = np.isfinite(window).sum(axis=0)
    sd = np.where(counts >= 20, sd, np.nan)
    return sd * np.sqrt(config.TRADING_DAYS)


def trailing_return(prices: np.ndarray, days: int) -> np.ndarray:
    """Simple total return over the last ``days`` sessions, per column."""
    if prices.shape[0] <= days:
        return np.full(prices.shape[1], np.nan)
    start = prices[-(days + 1)]
    end = prices[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = end / start - 1.0
    return np.where(np.isfinite(out), out, np.nan)
