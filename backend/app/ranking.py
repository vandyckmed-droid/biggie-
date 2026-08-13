"""Momentum ranking across windows, return modes and risk models.

Every combination the UI can select is precomputed once per refresh, so switching
window / raw-vs-residual / simple-vs-covariance on the phone is an instant re-sort of
data already on the device rather than a round trip.
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
RISK_MODES = ("simple", "covariance")


@dataclass
class WindowResult:
    """Per-symbol statistics for one (window, return mode) pair."""

    window: str
    return_mode: str
    symbols: list[str]
    mean_daily: np.ndarray
    ann_return: np.ndarray
    ann_vol_simple: np.ndarray
    ann_vol_cov: np.ndarray
    marginal_risk: np.ndarray
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


def compute_window(
    panel: ReturnPanel,
    window: config.Window,
    *,
    return_mode: str,
    returns_override: np.ndarray | None = None,
) -> WindowResult:
    """Compute all statistics for one window under one return mode."""
    source = panel if returns_override is None else ReturnPanel(
        dates=panel.dates,
        symbols=panel.symbols,
        prices=panel.prices,
        returns=returns_override,
    )
    slice_ = source.window(window.lookback, window.skip)

    ann_ret = annualized_return(slice_)
    vol_simple = annualized_volatility(slice_)

    cov = risk.estimate_covariance(slice_, source.symbols)
    vol_cov = cov.annual_vol

    # Portfolio-aware view, taken against the equal-weight universe composite.
    n = len(source.symbols)
    weights = np.full(n, 1.0 / n) if n else np.zeros(0)
    marginal = cov.marginal_risk(weights)
    contribution = cov.risk_contribution(weights)

    result = WindowResult(
        window=window.key,
        return_mode=return_mode,
        symbols=list(source.symbols),
        mean_daily=mean_daily_return(slice_),
        ann_return=ann_ret,
        ann_vol_simple=vol_simple,
        ann_vol_cov=vol_cov,
        marginal_risk=marginal,
        risk_contribution=contribution,
        shrinkage=cov.shrinkage,
        observations=cov.observations,
    )

    for risk_mode, vol in (("simple", vol_simple), ("covariance", vol_cov)):
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
    """Every (window, return mode) combination, plus the factor model behind residuals.

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

    results: dict[tuple[str, str], WindowResult] = {}
    for window in windows:
        results[(window.key, "raw")] = compute_window(
            panel, window, return_mode="raw"
        )
        results[(window.key, "residual")] = compute_window(
            panel, window, return_mode="residual", returns_override=residual_returns
        )
    return results, factor_model


def market_cap_ranking(symbols: list[str], caps: dict[str, float]) -> np.ndarray:
    """Rank by market capitalisation, largest first."""
    values = np.array([caps.get(s, np.nan) for s in symbols], dtype=float)
    return rank_desc(values)
