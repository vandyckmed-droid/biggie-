"""Live equal-weight portfolios: one per sector, plus the 1,000-name composite.

These are analytical portfolios, not backtests. Each is a standing equal-weight claim on
its members that is re-derived from the current universe on every refresh, and its
statistics describe the trailing behaviour of *today's* membership.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .returns import ReturnPanel, annualized_return, annualized_volatility
from .risk import CovarianceModel


@dataclass
class PortfolioStats:
    """Trailing statistics for one equal-weight portfolio."""

    key: str
    label: str
    members: list[str]
    weight_each: float
    ann_return: float
    ann_vol: float
    score: float
    beta: float
    max_drawdown: float
    cumulative: list[float]        # normalised equity curve for the sparkline
    dates: list[str]
    diversification_ratio: float   # weighted avg vol / portfolio vol


def _equal_weight_series(panel: ReturnPanel, members: list[str]) -> np.ndarray:
    """Daily return of an equal-weight basket, rebalanced each session.

    Names with a missing bar are simply excluded from that day's average rather than
    treated as zero return, which would silently drag the basket toward flat.
    """
    cols = [panel.col(s) for s in members if panel.has(s)]
    if not cols:
        return np.zeros(0)
    block = panel.returns[:, cols]
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(block).sum(axis=1)
        summed = np.nansum(block, axis=1)
    out = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
    return out


def _max_drawdown(curve: np.ndarray) -> float:
    if curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = curve / peak - 1.0
    return float(np.nanmin(dd)) if np.isfinite(dd).any() else 0.0


def _beta_vs_market(series: np.ndarray, market: np.ndarray) -> float:
    mask = np.isfinite(series) & np.isfinite(market)
    if mask.sum() < 30:
        return float("nan")
    var = float(np.var(market[mask], ddof=1))
    if var <= 0:
        return float("nan")
    return float(np.cov(series[mask], market[mask], ddof=1)[0, 1] / var)


def build_portfolio(
    panel: ReturnPanel,
    key: str,
    label: str,
    members: list[str],
    *,
    cov: CovarianceModel | None = None,
    market_panel: ReturnPanel | None = None,
    curve_points: int = 260,
) -> PortfolioStats | None:
    """Compute trailing statistics for one equal-weight basket.

    ``market_panel`` supplies the market series for the beta calculation; ``panel``
    normally holds only stocks, so it has no VTI column of its own.
    """
    members = [m for m in members if panel.has(m)]
    if not members:
        return None

    series = _equal_weight_series(panel, members)
    if series.size == 0:
        return None

    filled = np.where(np.isfinite(series), series, 0.0)
    curve = np.cumprod(1.0 + filled)

    ann_ret = float(annualized_return(series[:, None])[0])
    ann_vol = float(annualized_volatility(series[:, None])[0])
    score = ann_ret / ann_vol if ann_vol and ann_vol > 0.01 else float("nan")

    market_source = market_panel if market_panel is not None else panel
    market = (
        market_source.returns[:, market_source.col(config.MARKET_ETF)]
        if market_source.has(config.MARKET_ETF)
        else np.full_like(series, np.nan)
    )

    # Diversification ratio: how much correlation is actually buying us.
    div_ratio = float("nan")
    if cov is not None:
        idx = [cov.symbols.index(m) for m in members if m in cov.symbols]
        if idx:
            w = np.zeros(len(cov.symbols))
            w[idx] = 1.0 / len(idx)
            port_vol = cov.portfolio_vol(w)
            weighted_avg = float(np.sum(w * cov.annual_vol))
            if port_vol > 0:
                div_ratio = weighted_avg / port_vol

    tail = slice(max(0, len(curve) - curve_points), None)
    curve_tail = curve[tail]
    normalised = curve_tail / curve_tail[0] if curve_tail.size and curve_tail[0] else curve_tail

    return PortfolioStats(
        key=key,
        label=label,
        members=members,
        weight_each=1.0 / len(members),
        ann_return=ann_ret,
        ann_vol=ann_vol,
        score=score,
        beta=_beta_vs_market(series, market),
        max_drawdown=_max_drawdown(curve),
        cumulative=[round(float(x), 5) for x in normalised],
        dates=panel.return_dates[tail],
        diversification_ratio=div_ratio,
    )


def build_sector_portfolios(
    panel: ReturnPanel,
    sectors: dict[str, str],
    universe: list[str],
    *,
    cov: CovarianceModel | None = None,
    market_panel: ReturnPanel | None = None,
) -> list[PortfolioStats]:
    """One equal-weight portfolio per sector, plus the full-universe composite."""
    by_sector: dict[str, list[str]] = {}
    for symbol in universe:
        by_sector.setdefault(sectors.get(symbol, config.UNKNOWN_SECTOR), []).append(symbol)

    out: list[PortfolioStats] = []
    composite = build_portfolio(
        panel, "UNIVERSE", f"Equal-Weight {len(universe)}", universe,
        cov=cov, market_panel=market_panel,
    )
    if composite:
        out.append(composite)

    for sector in sorted(by_sector):
        stats = build_portfolio(
            panel, f"SECTOR::{sector}", sector, sorted(by_sector[sector]),
            cov=cov, market_panel=market_panel,
        )
        if stats:
            out.append(stats)
    return out


def sector_exposure_table(
    sectors: dict[str, str],
    universe: list[str],
    caps: dict[str, float],
) -> list[dict[str, object]]:
    """Live sector exposure: name count, equal weight and cap weight side by side."""
    counts: dict[str, int] = {}
    cap_totals: dict[str, float] = {}
    for symbol in universe:
        sector = sectors.get(symbol, config.UNKNOWN_SECTOR)
        counts[sector] = counts.get(sector, 0) + 1
        cap_totals[sector] = cap_totals.get(sector, 0.0) + float(caps.get(symbol, 0.0))

    n = max(len(universe), 1)
    total_cap = max(sum(cap_totals.values()), 1.0)
    return [
        {
            "sector": sector,
            "count": counts[sector],
            "equal_weight": counts[sector] / n,
            "cap_weight": cap_totals.get(sector, 0.0) / total_cap,
            "market_cap": cap_totals.get(sector, 0.0),
            "etf": config.SECTOR_TO_ETF.get(sector),
        }
        for sector in sorted(counts, key=lambda s: -counts[s])
    ]
