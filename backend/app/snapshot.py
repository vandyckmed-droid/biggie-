"""Snapshot builder: turns raw market data into the single object the app serves.

One refresh does the whole job - pull the universe, top up cached prices, compute every
ranking / risk / portfolio view, and write a self-contained snapshot. The API then only
reads that object, which is what keeps interaction on the phone instant.

The build is deterministic: the same cached prices always produce the same snapshot.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Sequence

import numpy as np

from . import config, hrp, macro, portfolios, ranking, risk, store, universe, validate
from .fmp import FMPClient
from .returns import ReturnPanel, build_panel, trailing_return

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def _clean(value: Any) -> Any:
    """JSON cannot carry NaN/Inf; convert them to null so the client sees a real gap."""
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not math.isfinite(f) else round(f, 6)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


# --------------------------------------------------------------------------------------
# Data acquisition
# --------------------------------------------------------------------------------------
async def refresh_market_data(
    *,
    size: int = config.UNIVERSE_SIZE,
    progress: ProgressFn | None = None,
    force_full: bool = False,
) -> tuple[list[universe.Candidate], universe.UniverseReport]:
    """Refresh the universe and top up the price cache incrementally."""
    def emit(stage: str, pct: float) -> None:
        if progress:
            progress(stage, pct)

    emit("Screening universe", 0.02)
    async with FMPClient() as client:
        rows = await client.screener(
            market_cap_more_than=config.MIN_MARKET_CAP,
            exchanges=("NASDAQ", "NYSE", "AMEX"),
        )
        # Over-select: recent listings clear the screener but fail the history gate, and
        # the pool is trimmed back to `size` once history is actually known.
        report = universe.build_universe(
            rows, size=int(size * config.UNIVERSE_BUFFER)
        )
        selected = report.selected
        log.info(
            "universe: screened %d -> %d selected (%d deduped)",
            report.screened, len(selected), report.deduped,
        )

        store.upsert_securities(
            [
                {
                    "symbol": c.symbol, "name": c.name, "sector": c.sector,
                    "industry": c.industry, "exchange": c.exchange, "country": c.country,
                    "market_cap": c.market_cap, "price": c.price, "volume": c.volume,
                    "kind": "stock",
                }
                for c in selected
            ]
            + [
                {
                    "symbol": t, "name": config.MACRO_LABELS.get(t, t),
                    "sector": None, "industry": None, "exchange": "ETF",
                    "country": "US", "market_cap": None, "price": None,
                    "volume": None, "kind": "etf",
                }
                for t in config.REFERENCE_TICKERS
            ]
        )

        wanted = [c.symbol for c in selected] + list(config.REFERENCE_TICKERS)
        start, end = date.today() - timedelta(days=config.HISTORY_DAYS), date.today()

        cached = {} if force_full else store.latest_dates(wanted)
        to_fetch: list[tuple[str, date]] = []
        for symbol in wanted:
            last = cached.get(symbol)
            if last is None:
                to_fetch.append((symbol, start))
            else:
                last_date = date.fromisoformat(last)
                if last_date < end - timedelta(days=1):
                    # Overlap by a few sessions so late adjustments get corrected.
                    to_fetch.append((symbol, last_date - timedelta(days=5)))

        emit(f"Fetching prices for {len(to_fetch)} symbols", 0.08)
        if to_fetch:
            reference = set(config.REFERENCE_TICKERS)
            # Reference ETFs first. Every beta, every residual and the whole macro page
            # depend on them, so if the vendor throttles partway through a cold pull it
            # must not be these that are missing.
            ordered = sorted(to_fetch, key=lambda pair: pair[0] not in reference)

            by_start: dict[date, list[str]] = {}
            for symbol, s in ordered:
                by_start.setdefault(s, []).append(symbol)
            # Preserve the reference-first ordering inside each start-date group.
            for s in by_start:
                by_start[s].sort(key=lambda sym: sym not in reference)

            done = 0
            total = len(to_fetch)
            for s, syms in by_start.items():
                def on_progress(n: int, t: int, base: int = done, tot: int = total) -> None:
                    emit(f"Fetching prices {base + n}/{tot}", 0.08 + 0.55 * (base + n) / max(tot, 1))

                bars = await client.histories(syms, start=s, end=end, on_progress=on_progress)
                store.upsert_prices(bars)
                done += len(syms)

        if client.throttled:
            log.warning("vendor throttled %d requests during this refresh", client.throttled)

    # A missing market ETF silently produces null betas and an empty macro page, which is
    # far worse than a loud failure - the validation gate would reject it anyway.
    cached_now = store.latest_dates(config.REFERENCE_TICKERS)
    missing = [t for t in config.REFERENCE_TICKERS if t not in cached_now]
    if missing:
        raise RuntimeError(
            "reference instruments have no price history: "
            + ", ".join(missing)
            + ". The vendor probably rate-limited the run; retry, or lower "
            "BIGGIE_RATE_LIMIT."
        )

    # Keep the cache bounded to the configured history length.
    store.prune_prices(date.today() - timedelta(days=config.HISTORY_DAYS + 45))
    emit("Prices cached", 0.65)
    return selected, report


# --------------------------------------------------------------------------------------
# Snapshot assembly
# --------------------------------------------------------------------------------------
def build_snapshot(
    selected: Sequence[universe.Candidate],
    report: universe.UniverseReport | None = None,
    *,
    size: int = config.UNIVERSE_SIZE,
    cluster_k: int = config.DEFAULT_CLUSTER_K,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Compute every view the client needs from cached prices."""
    def emit(stage: str, pct: float) -> None:
        if progress:
            progress(stage, pct)

    t0 = time.time()
    symbols = [c.symbol for c in selected]
    sectors = {c.symbol: c.sector for c in selected}
    caps = {c.symbol: c.market_cap for c in selected}
    names = {c.symbol: c.name for c in selected}

    emit("Loading price history", 0.67)
    all_symbols = symbols + [t for t in config.REFERENCE_TICKERS if t not in symbols]
    price_map = store.load_prices(all_symbols)
    if not price_map:
        raise RuntimeError("price cache is empty - run a refresh first")

    panel = build_panel(price_map)

    # Liquidity, measured from actual traded history rather than a live quote field.
    # This is the real gate: the screener's volume column is zero outside the session.
    dollar_volume = store.median_dollar_volume(
        [s for s in symbols if panel.has(s)], lookback=60
    )
    # Unknown liquidity passes. A symbol with no volume rows cached is a gap in our own
    # data, not evidence of thin trading - and defaulting the unknown case to "drop" is
    # exactly the mistake that emptied the universe in the first place.
    liquid = [
        s for s in symbols
        if panel.has(s)
        and dollar_volume.get(s, float("inf")) >= config.MIN_AVG_DOLLAR_VOLUME
    ]
    dropped = sum(1 for s in symbols if panel.has(s)) - len(liquid)
    if dropped:
        log.info("liquidity filter dropped %d thinly traded names", dropped)

    # `symbols` is ordered by market cap, so trimming here yields exactly the largest
    # `size` names that actually carry enough history to be ranked.
    stock_symbols = liquid[:size]
    stock_panel = panel.subset(stock_symbols)
    log.info("panel: %d dates x %d symbols", len(panel.dates), len(panel.symbols))

    # ---------------------------------------------------------------- rankings
    emit("Computing rankings", 0.72)
    # The factor model reads from the full panel: VTI and the sector ETFs are the
    # regressors, and they deliberately sit outside the ranked universe.
    results, factor_model = ranking.compute_all(
        stock_panel, sectors, factor_panel=panel
    )
    beta_by_symbol = dict(zip(factor_model.symbols, factor_model.beta))
    sector_beta = dict(zip(factor_model.symbols, factor_model.sector_beta))
    r2_by_symbol = dict(zip(factor_model.symbols, factor_model.r_squared))
    idio_by_symbol = dict(zip(factor_model.symbols, factor_model.idio_vol))

    cap_ranks = ranking.market_cap_ranking(stock_symbols, caps)

    trailing = {
        key: trailing_return(stock_panel.prices, days)
        for key, _label, days in config.RETURN_WINDOWS
    }

    # Per-symbol record carrying every precomputed view.
    idx_of = {s: i for i, s in enumerate(stock_symbols)}
    rows: list[dict[str, Any]] = []
    for symbol in stock_symbols:
        i = idx_of[symbol]
        metrics: dict[str, Any] = {}
        for (window_key, return_mode), res in results.items():
            j = res.symbols.index(symbol)
            metrics[f"{window_key}|{return_mode}"] = {
                "ann_return": _clean(res.ann_return[j]),
                "mean_daily": _clean(res.mean_daily[j]),
                "vol": _clean(res.ann_vol[j]),
                "vol_idio": _clean(res.ann_vol_idio[j]),
                "score_total": _clean(res.scores["total"][j]),
                "score_idio": _clean(res.scores["idiosyncratic"][j]),
                "rank_total": int(res.ranks["total"][j]),
                "rank_idio": int(res.ranks["idiosyncratic"][j]),
                "marginal_risk": _clean(res.marginal_risk[j]),
                "risk_contribution": _clean(res.risk_contribution[j]),
            }

        rows.append(
            {
                "symbol": symbol,
                "name": names.get(symbol, symbol),
                "sector": sectors.get(symbol, config.UNKNOWN_SECTOR),
                "market_cap": _clean(caps.get(symbol)),
                "cap_rank": int(cap_ranks[i]),
                "beta": _clean(beta_by_symbol.get(symbol)),
                "sector_beta": _clean(sector_beta.get(symbol)),
                "r_squared": _clean(r2_by_symbol.get(symbol)),
                "idio_vol": _clean(idio_by_symbol.get(symbol)),
                "trailing": {
                    key: _clean(trailing[key][i]) for key, _l, _d in config.RETURN_WINDOWS
                },
                "metrics": metrics,
            }
        )

    # ---------------------------------------------------------------- portfolios
    emit("Building portfolios", 0.85)
    base_window = config.WINDOWS[config.DEFAULT_WINDOW]
    base_cov = risk.estimate_covariance(
        stock_panel.window(base_window.lookback, base_window.skip), stock_panel.symbols
    )
    sector_books = portfolios.build_sector_portfolios(
        stock_panel, sectors, stock_symbols, cov=base_cov, market_panel=panel
    )
    exposure = portfolios.sector_exposure_table(sectors, stock_symbols, caps)

    # ---------------------------------------------------------------- clustering
    emit("Clustering universe", 0.90)
    medoids = hrp.cluster_universe(base_cov, cluster_k)
    cluster_label = {}
    for medoid, members in medoids.groups().items():
        for m in members:
            cluster_label[m] = medoid
    for row in rows:
        row["cluster"] = cluster_label.get(row["symbol"])

    # ---------------------------------------------------------------- macro
    emit("Analysing macro regime", 0.93)
    macro_assets, regime, macro_diag = macro.build_macro(panel, cluster_k=5)
    macro_grid = macro.heatmap(macro_assets)

    # ---------------------------------------------------------------- HRP baseline
    emit("Solving HRP", 0.96)
    # The full 1000-name HRP is the universe-level baseline; watchlist HRP is computed
    # on demand because it changes whenever the user taps a star.
    hrp_universe = hrp.hierarchical_risk_parity(base_cov, k=cluster_k)
    top_hrp = sorted(
        zip(hrp_universe.symbols, hrp_universe.weights), key=lambda kv: -kv[1]
    )[:50]

    built_at = datetime.now(timezone.utc)
    snapshot = {
        "version": 1,
        "built_at": built_at.isoformat(timespec="seconds"),
        "as_of": panel.dates[-1] if panel.dates else None,
        "build_seconds": round(time.time() - t0, 1),
        "universe_size": len(stock_symbols),
        # Recorded so a later re-validation (build_site) judges the snapshot against the
        # size it was actually asked for, not the configured default.
        "requested_size": size,
        "trading_days": len(panel.dates),
        "windows": [
            {
                "key": w.key, "label": w.label,
                "lookback": w.lookback, "skip": w.skip,
            }
            for w in config.WINDOWS.values()
        ],
        "return_windows": [
            {"key": k, "label": lbl, "days": d} for k, lbl, d in config.RETURN_WINDOWS
        ],
        "sectors": sorted({r["sector"] for r in rows}),
        "stocks": rows,
        "sector_exposure": _clean(exposure),
        "portfolios": [
            {
                "key": p.key, "label": p.label, "members": len(p.members),
                "weight_each": _clean(p.weight_each), "ann_return": _clean(p.ann_return),
                "ann_vol": _clean(p.ann_vol), "score": _clean(p.score),
                "beta": _clean(p.beta), "max_drawdown": _clean(p.max_drawdown),
                "diversification_ratio": _clean(p.diversification_ratio),
                "curve": _clean(p.cumulative), "curve_dates": p.dates,
            }
            for p in sector_books
        ],
        "macro": {
            "assets": [
                {
                    "symbol": a.symbol, "label": a.label, "asset_class": a.asset_class,
                    "ann_return": _clean(a.ann_return),
                    "vol": _clean(a.ann_vol),
                    "score": _clean(a.score),
                    "rank": a.rank, "percentile": _clean(a.percentile),
                    "beta": _clean(a.beta), "trailing": _clean(a.trailing),
                    "cluster": a.cluster,
                }
                for a in macro_assets
            ],
            "regime": {
                "state": regime.state,
                "score": _clean(regime.score),
                "confidence": _clean(regime.confidence),
                "components": _clean(regime.components),
                "spy_tlt_corr_fast": _clean(regime.spy_tlt_corr_fast),
                "spy_tlt_corr_slow": _clean(regime.spy_tlt_corr_slow),
                "sector_dispersion": _clean(regime.sector_dispersion),
                "narrative": regime.narrative,
            },
            "heatmap": _clean(macro_grid),
            "diagnostics": _clean(macro_diag),
        },
        "clusters": {
            "k": cluster_k,
            "medoids": medoids.medoids,
            "groups": {m: sorted(v) for m, v in medoids.groups().items()},
            "inertia": _clean(medoids.inertia),
        },
        "hrp_universe": {
            "portfolio_vol": _clean(hrp_universe.portfolio_vol),
            "effective_n": _clean(hrp_universe.effective_n),
            "top_weights": [
                {"symbol": s, "weight": _clean(w)} for s, w in top_hrp
            ],
        },
        "risk_model": {
            "shrinkage": _clean(base_cov.shrinkage),
            "observations": base_cov.observations,
            "factors": factor_model.factor_names,
        },
        "universe_report": {
            "screened": report.screened if report else None,
            "selected": len(stock_symbols),
            "deduped": report.deduped if report else None,
            "excluded": report.excluded if report else {},
        },
    }

    emit("Snapshot ready", 1.0)
    log.info("snapshot built in %.1fs", time.time() - t0)
    return snapshot


def save_snapshot(snapshot: dict[str, Any]) -> None:
    tmp = config.SNAPSHOT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, allow_nan=False))
    tmp.replace(config.SNAPSHOT_PATH)
    store.set_meta("last_build", snapshot["built_at"])


def load_snapshot() -> dict[str, Any] | None:
    if not config.SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(config.SNAPSHOT_PATH.read_text())
    except (ValueError, OSError):
        return None


class RefreshRejected(RuntimeError):
    """A build completed but failed validation, so it was not published."""

    def __init__(self, result: "validate.ValidationResult") -> None:
        super().__init__("; ".join(result.errors) or "validation failed")
        self.result = result


async def full_refresh(
    *,
    size: int = config.UNIVERSE_SIZE,
    cluster_k: int = config.DEFAULT_CLUSTER_K,
    progress: ProgressFn | None = None,
    force_full: bool = False,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Fetch, compute, validate and persist a complete snapshot.

    A snapshot that fails validation is never written, so the previously published
    dataset stays live rather than being replaced by something broken.
    """
    selected, report = await refresh_market_data(
        size=size, progress=progress, force_full=force_full
    )
    snapshot = await asyncio.to_thread(
        build_snapshot, selected, report,
        size=size, cluster_k=cluster_k, progress=progress,
    )

    if not skip_validation:
        result = validate.validate_snapshot(snapshot, expected_size=size)
        snapshot["validation"] = {
            "ok": result.ok,
            "errors": result.errors,
            "warnings": result.warnings,
            "stats": result.stats,
        }
        for message in result.warnings:
            log.warning("validation: %s", message)
        if not result.ok:
            for message in result.errors:
                log.error("validation: %s", message)
            raise RefreshRejected(result)

        previous = load_snapshot()
        if not validate.is_newer(snapshot, previous):
            log.warning(
                "built data (%s) is older than what is published (%s); keeping the "
                "existing dataset",
                snapshot.get("as_of"), (previous or {}).get("as_of"),
            )
            return previous or snapshot

    save_snapshot(snapshot)
    return snapshot
