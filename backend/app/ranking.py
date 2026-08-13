"""Momentum ranking across windows, return modes and risk denominators.

Every combination the UI can select is precomputed once per daily refresh, so switching
window / raw-vs-residual / risk denominator on the phone is an instant re-sort of data
already on the device.

**On the risk denominator.** For a single asset, ``sqrt(eᵢᵀ Σ eᵢ)`` is just that asset's
own volatility - shrinkage nudges it, but it is not a different measurement, and an
earlier version of this system wrongly presented it as one. The genuinely different
denominator is *idiosyncratic* volatility: the volatility of what is left after a factor
model removes market and sector exposure. That is what the second risk mode now uses.
The covariance matrix still does the work it is actually good for - HRP, portfolio
volatility, risk contribution, correlation and clustering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from . import config, risk
from .returns import (
    ReturnPanel,
    annualized_return,
    annualized_volatility,
    mean_daily_return,
)

RETURN_MODES = ("raw", "residual")

#: ``total`` divides by the volatility of the return series being ranked.
#: ``idiosyncratic`` divides by factor-residual volatility, isolating stock-specific risk.
RISK_MODES = ("total", "idiosyncratic")
DEFAULT_RISK_MODE = "total"


@dataclass
class WindowResult:
    """Per-symbol statistics for one (window, return mode) pair."""

    window: str
    return_mode: str
    symbols: list[str]
    mean_daily: np.ndarray
    ann_return: np.ndarray
    ann_vol: np.ndarray             # volatility of the ranked return series
    ann_vol_idio: np.ndarray        # factor-residual (idiosyncratic) volatility
    marginal_risk: np.ndarray       # contribution to the equal-weight composite's risk
    risk_contribution: np.ndarray
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    ranks: dict[str, np.ndarray] = field(default_factory=dict)
    shrinkage: float = 0.0
    observations: int = 0


def _score(ann_return: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Return ÷ volatility, guarded against degenerate volatility.

    The floor is a small absolute annualised vol: without it a name whose window happens
    to be nearly flat produces an unbounded score and hijacks the top of the ranking.
    """
    safe = np.where(np.isfinite(vol) & (vol > 0.01), vol, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ann_return / safe


def rank_desc(values: np.ndarray) -> np.ndarray:
    """Dense 1-based ranking, best first. Missing values rank last.

    Rank 1 is the strongest name; with a full universe the worst is rank 1000.
    """
    out = np.full(len(values), len(values), dtype=int)
    finite = np.where(np.isfinite(values))[0]
    if finite.size:
        order = finite[np.argsort(-values[finite], kind="stable")]
        out[order] = np.arange(1, len(order) + 1)
    return out


def percentile_of(ranks: np.ndarray, n: int) -> np.ndarray:
    """Convert 1-based ranks into a 0-100 score where 100 is best."""
    if n <= 1:
        return np.full(len(ranks), 100.0)
    return (1.0 - (ranks - 1) / (n - 1)) * 100.0


def _window_slice(matrix: np.ndarray, window: config.Window) -> np.ndarray:
    """The same lookback/skip slice ``ReturnPanel.window`` takes, for a bare matrix."""
    n = matrix.shape[0]
    end = n - window.skip
    start = max(0, end - window.lookback)
    return matrix[start:end] if end > start else matrix[:0]


def compute_window(
    panel: ReturnPanel,
    window: config.Window,
    *,
    return_mode: str,
    returns_override: np.ndarray | None = None,
    factor_residuals: np.ndarray | None = None,
) -> WindowResult:
    """Compute all statistics for one window under one return mode.

    ``factor_residuals`` is the market+sector residual matrix aligned to ``panel.symbols``;
    its volatility over the same slice becomes the idiosyncratic denominator.
    """
    source = panel if returns_override is None else ReturnPanel(
        dates=panel.dates,
        symbols=panel.symbols,
        prices=panel.prices,
        returns=returns_override,
    )
    slice_ = source.window(window.lookback, window.skip)

    ann_ret = annualized_return(slice_)
    vol_total = annualized_volatility(slice_)

    if factor_residuals is not None:
        vol_idio = annualized_volatility(_window_slice(factor_residuals, window))
    else:
        vol_idio = np.full_like(vol_total, np.nan)

    # The covariance matrix is still estimated here, but for portfolio-level questions
    # rather than to restate each stock's own volatility.
    cov = risk.estimate_covariance(slice_, source.symbols)
    n = len(source.symbols)
    weights = np.full(n, 1.0 / n) if n else np.zeros(0)

    result = WindowResult(
        window=window.key,
        return_mode=return_mode,
        symbols=list(source.symbols),
        mean_daily=mean_daily_return(slice_),
        ann_return=ann_ret,
        ann_vol=vol_total,
        ann_vol_idio=vol_idio,
        marginal_risk=cov.marginal_risk(weights),
        risk_contribution=cov.risk_contribution(weights),
        shrinkage=cov.shrinkage,
        observations=cov.observations,
    )

    for risk_mode, vol in (("total", vol_total), ("idiosyncratic", vol_idio)):
        scores = _score(ann_ret, vol)
        result.scores[risk_mode] = scores
        result.ranks[risk_mode] = rank_desc(scores)

    return result


def compute_all(
    panel: ReturnPanel,
    sectors: dict[str, str],
    *,
    factor_panel: ReturnPanel | None = None,
    windows: Iterable[config.Window] | None = None,
) -> tuple[dict[tuple[str, str], WindowResult], risk.FactorModel]:
    """Every (window, return mode) combination, plus the factor model behind them.

    ``panel`` holds the symbols being ranked; ``factor_panel`` must additionally contain
    the market and sector ETFs, which normally sit outside the ranked universe. Passing
    a stock-only panel for both would leave every beta undefined.

    Returns a dict keyed by ``(window_key, return_mode)``.
    """
    windows = list(windows or config.WINDOWS.values())
    factors = factor_panel if factor_panel is not None else panel

    factor_model = risk.estimate_factor_model(factors, sectors)
    beta_by_symbol = dict(zip(factor_model.symbols, factor_model.beta))
    residual_returns = risk.market_residual_returns(panel, beta_by_symbol, factors)

    # Align the factor-model residual matrix to the ranked universe's column order.
    factor_index = {s: i for i, s in enumerate(factor_model.symbols)}
    cols = [factor_index.get(s) for s in panel.symbols]
    idio = np.full(panel.returns.shape, np.nan)
    for j, src in enumerate(cols):
        if src is not None:
            idio[:, j] = factor_model.residuals[:, src]

    results: dict[tuple[str, str], WindowResult] = {}
    for window in windows:
        results[(window.key, "raw")] = compute_window(
            panel, window, return_mode="raw", factor_residuals=idio
        )
        results[(window.key, "residual")] = compute_window(
            panel, window, return_mode="residual",
            returns_override=residual_returns, factor_residuals=idio,
        )
    return results, factor_model


def market_cap_ranking(symbols: list[str], caps: dict[str, float]) -> np.ndarray:
    """Rank by market capitalisation, largest first."""
    values = np.array([caps.get(s, np.nan) for s in symbols], dtype=float)
    return rank_desc(values)
