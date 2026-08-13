"""Macro market intelligence: cross-asset ranking, heatmap and regime detection.

The macro page runs the same risk-adjusted momentum maths as the stock page, but over a
small cross-asset set where the covariance matrix is well conditioned enough to say
something real about regime: how equities and bonds are co-moving, whether leadership is
cyclical or defensive, and how dispersed sector performance has become.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, ranking, risk
from .hrp import correlation_distance, k_medoids
from .returns import (
    ReturnPanel,
    annualized_return,
    annualized_volatility,
    trailing_return,
)

#: Sectors that lead when risk appetite is expanding, and those that lead when it is not.
CYCLICAL = ["XLK", "XLY", "XLI", "XLF", "XLE", "XLB"]
DEFENSIVE = ["XLP", "XLU", "XLV", "XLRE"]

#: Rolling window for the equity/bond correlation read.
CORR_FAST = 60
CORR_SLOW = 250


@dataclass
class MacroAsset:
    symbol: str
    label: str
    asset_class: str
    ann_return: float
    ann_vol: float
    score: float
    rank: int
    percentile: float
    beta: float
    trailing: dict[str, float]
    cluster: int = 0


@dataclass
class MacroRegime:
    state: str                       # Risk-On | Risk-Off | Transition
    score: float                     # -1 (risk-off) .. +1 (risk-on)
    confidence: float                # 0..1, how far from the transition band
    components: list[dict[str, object]] = field(default_factory=list)
    spy_tlt_corr_fast: float = float("nan")
    spy_tlt_corr_slow: float = float("nan")
    sector_dispersion: float = float("nan")
    narrative: str = ""


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < max(20, window // 4):
        return float("nan")
    a, b = a[-window:], b[-window:]
    if a.size < 20 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_macro(
    panel: ReturnPanel,
    *,
    window_key: str = config.DEFAULT_WINDOW,
    cluster_k: int = 5,
) -> tuple[list[MacroAsset], MacroRegime, dict[str, object]]:
    """Rank every macro instrument and derive the regime read."""
    present = [s for s in config.MACRO_INSTRUMENTS if panel.has(s)]
    macro_panel = panel.subset(present)
    window = config.WINDOWS[window_key]

    slice_ = macro_panel.window(window.lookback, window.skip)
    ann_ret = annualized_return(slice_)
    ann_vol = annualized_volatility(slice_)

    # The joint covariance is kept for correlation, clustering and cross-asset risk
    # coupling - not to restate each asset's own volatility.
    cov = risk.estimate_covariance(slice_, macro_panel.symbols)

    chosen = ranking._score(ann_ret, ann_vol)
    ranks = ranking.rank_desc(chosen)
    pcts = ranking.percentile_of(ranks, len(present))

    # Cross-asset betas vs the total market, over the full two years.
    market = (
        panel.returns[:, panel.col(config.MARKET_ETF)]
        if panel.has(config.MARKET_ETF)
        else None
    )

    # Correlation clustering groups instruments that are moving as one bloc.
    corr = cov.correlation()
    clusters = k_medoids(
        correlation_distance(corr), list(macro_panel.symbols), min(cluster_k, len(present))
    )
    cluster_by_symbol = dict(zip(clusters.symbols, clusters.labels))

    trailing_by_key = {
        key: trailing_return(macro_panel.prices, days)
        for key, _label, days in config.RETURN_WINDOWS
    }

    assets: list[MacroAsset] = []
    for i, symbol in enumerate(macro_panel.symbols):
        beta = float("nan")
        if market is not None:
            series = macro_panel.returns[:, i]
            mask = np.isfinite(series) & np.isfinite(market)
            if mask.sum() > 30 and np.var(market[mask], ddof=1) > 0:
                beta = float(
                    np.cov(series[mask], market[mask], ddof=1)[0, 1]
                    / np.var(market[mask], ddof=1)
                )

        assets.append(
            MacroAsset(
                symbol=symbol,
                label=config.MACRO_LABELS.get(symbol, symbol),
                asset_class=config.MACRO_ASSET_CLASS.get(symbol, "Other"),
                ann_return=_safe(ann_ret[i]),
                ann_vol=_safe(ann_vol[i]),
                score=_safe(chosen[i]),
                rank=int(ranks[i]),
                percentile=float(pcts[i]),
                beta=beta,
                trailing={
                    key: _safe(trailing_by_key[key][i])
                    for key, _label, _days in config.RETURN_WINDOWS
                },
                cluster=int(cluster_by_symbol.get(symbol, 0)),
            )
        )

    assets.sort(key=lambda a: a.rank)
    regime = detect_regime(panel, macro_panel, dict(zip(macro_panel.symbols, chosen)))

    diagnostics = {
        "shrinkage": round(float(cov.shrinkage), 4),
        "observations": int(cov.observations),
        "clusters": {
            medoid: members for medoid, members in clusters.groups().items()
        },
        "correlation": {
            "symbols": list(macro_panel.symbols),
            "matrix": [[round(float(v), 4) for v in row] for row in corr],
        },
    }
    return assets, regime, diagnostics


def detect_regime(
    panel: ReturnPanel,
    macro_panel: ReturnPanel,
    scores: dict[str, float],
) -> MacroRegime:
    """Classify the market into Risk-On / Risk-Off / Transition.

    No single indicator is trusted. Four largely independent reads are squashed into a
    common scale and averaged; the middle band is deliberately wide, because "the signals
    disagree" is genuine information rather than something to force into a direction.
    """
    components: list[dict[str, object]] = []

    def add(name: str, raw: float, value: float, detail: str) -> None:
        components.append(
            {
                "name": name,
                "value": _safe(value),
                "signal": _safe(raw),
                "detail": detail,
            }
        )

    # 1. Equities versus long duration. Equities winning is the cleanest risk-on tell.
    spy = scores.get("SPY", float("nan"))
    tlt = scores.get("TLT", float("nan"))
    equity_vs_bond = float("nan")
    if np.isfinite(spy) and np.isfinite(tlt):
        equity_vs_bond = float(np.tanh((spy - tlt) / 1.5))
        add(
            "Equity vs Duration",
            equity_vs_bond,
            spy - tlt,
            f"SPY {spy:.2f} vs TLT {tlt:.2f} risk-adjusted",
        )

    # 2. Cyclical versus defensive sector leadership.
    cyc = [scores[s] for s in CYCLICAL if np.isfinite(scores.get(s, np.nan))]
    dfn = [scores[s] for s in DEFENSIVE if np.isfinite(scores.get(s, np.nan))]
    breadth = float("nan")
    if cyc and dfn:
        gap = float(np.mean(cyc) - np.mean(dfn))
        breadth = float(np.tanh(gap / 1.2))
        add("Cyclical vs Defensive", breadth, gap, f"{len(cyc)} cyclical vs {len(dfn)} defensive")

    # 3. Small caps versus large caps - risk appetite at the speculative end.
    iwm = scores.get("IWM", float("nan"))
    breadth_cap = float("nan")
    if np.isfinite(iwm) and np.isfinite(spy):
        gap = iwm - spy
        breadth_cap = float(np.tanh(gap / 1.2))
        add("Small vs Large Cap", breadth_cap, gap, f"IWM {iwm:.2f} vs SPY {spy:.2f}")

    # 4. Flight to safety: gold and short duration outperforming is defensive.
    gld = scores.get("GLD", float("nan"))
    shy = scores.get("SHY", float("nan"))
    haven = float("nan")
    havens = [v for v in (gld, shy) if np.isfinite(v)]
    if havens and np.isfinite(spy):
        gap = float(spy - np.mean(havens))
        haven = float(np.tanh(gap / 1.5))
        add("Risk vs Havens", haven, gap, "SPY against gold and short duration")

    signals = [v for v in (equity_vs_bond, breadth, breadth_cap, haven) if np.isfinite(v)]
    score = float(np.mean(signals)) if signals else 0.0

    # Correlation and dispersion context.
    corr_fast = corr_slow = float("nan")
    if macro_panel.has("SPY") and macro_panel.has("TLT"):
        a = macro_panel.series("SPY")
        b = macro_panel.series("TLT")
        corr_fast = _rolling_corr(a, b, CORR_FAST)
        corr_slow = _rolling_corr(a, b, CORR_SLOW)

    sector_scores = [
        scores[s] for s in config.SECTOR_ETFS if np.isfinite(scores.get(s, np.nan))
    ]
    dispersion = float(np.std(sector_scores)) if len(sector_scores) > 2 else float("nan")

    # A wide middle band: a weak reading is a transition, not a weak trend.
    if score >= 0.25:
        state = "Risk-On"
    elif score <= -0.25:
        state = "Risk-Off"
    else:
        state = "Transition"

    confidence = float(min(1.0, abs(score) / 0.6))

    narrative = _narrative(state, score, corr_fast, dispersion, components)
    return MacroRegime(
        state=state,
        score=score,
        confidence=confidence,
        components=components,
        spy_tlt_corr_fast=corr_fast,
        spy_tlt_corr_slow=corr_slow,
        sector_dispersion=dispersion,
        narrative=narrative,
    )


def _narrative(
    state: str,
    score: float,
    corr_fast: float,
    dispersion: float,
    components: list[dict[str, object]],
) -> str:
    """One plain sentence explaining the call - the phone has no room for more."""
    if components:
        strongest = max(
            components,
            key=lambda c: abs(c["value"]) if np.isfinite(c["value"]) else 0.0,  # type: ignore[arg-type]
        )
        lead = f"{strongest['name']} is the dominant signal"
    else:
        lead = "Signals are inconclusive"

    bits = [f"{state} at {score:+.2f}", lead]
    if np.isfinite(corr_fast):
        if corr_fast > 0.2:
            bits.append("stocks and bonds are moving together, so duration is not hedging")
        elif corr_fast < -0.2:
            bits.append("the classic negative stock/bond correlation is intact")
        else:
            bits.append("stock/bond correlation is near zero")
    if np.isfinite(dispersion):
        if dispersion > 1.0:
            bits.append("sector dispersion is wide, favouring selection over beta")
        elif dispersion < 0.4:
            bits.append("sectors are moving as one bloc")
    return "; ".join(bits) + "."


def heatmap(assets: list[MacroAsset]) -> dict[str, object]:
    """Grid payload for the macro heatmap, grouped by asset class."""
    values = [a.score for a in assets if np.isfinite(a.score)]
    lo = float(np.percentile(values, 5)) if values else -1.0
    hi = float(np.percentile(values, 95)) if values else 1.0
    bound = max(abs(lo), abs(hi), 0.5)

    groups: dict[str, list[dict[str, object]]] = {}
    for asset in assets:
        groups.setdefault(asset.asset_class, []).append(
            {
                "symbol": asset.symbol,
                "label": asset.label,
                "score": asset.score,
                "rank": asset.rank,
                "ann_return": asset.ann_return,
                "cluster": asset.cluster,
            }
        )

    order = ["Index", "Sector", "Rates", "Commodity"]
    return {
        "scale": {"min": -bound, "max": bound},
        "groups": [
            {"name": name, "cells": groups[name]}
            for name in order
            if name in groups
        ],
    }
