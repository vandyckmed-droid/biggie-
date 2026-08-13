"""FastAPI application.

Endpoints are deliberately narrow: the list views return only the six fields a row
renders, and the heavy per-ticker detail is fetched on tap. That keeps the payload small
enough for a phone on cellular while still allowing instant client-side re-sorting.
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import date, timedelta
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, hrp, ranking, risk, snapshot as snap, store
from .returns import build_panel, trailing_return

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

WindowKey = Literal["mom_12_1", "mom_6_1", "mom_3_1", "market_cap"]
ReturnMode = Literal["raw", "residual"]
RiskMode = Literal["total", "idiosyncratic"]

DEFAULT_SETTINGS: dict[str, Any] = {
    "window": config.DEFAULT_WINDOW,
    "return_mode": "raw",
    "risk_mode": ranking.DEFAULT_RISK_MODE,
    "cluster_k": config.DEFAULT_CLUSTER_K,
    "sector": None,
}


# --------------------------------------------------------------------------------------
# Snapshot access
# --------------------------------------------------------------------------------------
class _State:
    """In-process cache of the snapshot plus refresh bookkeeping."""

    def __init__(self) -> None:
        self.snapshot: dict[str, Any] | None = None
        self.refreshing = False
        self.stage = "idle"
        self.progress = 0.0
        self.error: str | None = None
        self.lock = asyncio.Lock()

    def load(self) -> dict[str, Any] | None:
        if self.snapshot is None:
            self.snapshot = snap.load_snapshot()
        return self.snapshot


state = _State()


def require_snapshot() -> dict[str, Any]:
    data = state.load()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="No snapshot yet. POST /api/refresh (or run scripts/refresh.py).",
        )
    return data


def _metric(stock: dict[str, Any], window: str, return_mode: str) -> dict[str, Any]:
    return stock.get("metrics", {}).get(f"{window}|{return_mode}", {})


def _row(stock: dict[str, Any], window: str, return_mode: str, risk_mode: str) -> dict[str, Any]:
    """The compact payload a list row renders."""
    m = _metric(stock, window, return_mode)
    idio = risk_mode == "idiosyncratic"
    score_key = "score_idio" if idio else "score_total"
    rank_key = "rank_idio" if idio else "rank_total"
    return {
        "symbol": stock["symbol"],
        "name": stock["name"],
        "sector": stock["sector"],
        "beta": stock.get("beta"),
        "market_cap": stock.get("market_cap"),
        "score": m.get(score_key),
        "score_alt": m.get("score_total" if idio else "score_idio"),
        "rank": m.get(rank_key),
        "ann_return": m.get("ann_return"),
        "vol": m.get("vol_idio" if idio else "vol"),
        "risk_contribution": m.get("risk_contribution"),
        "cluster": stock.get("cluster"),
    }


def _sort_key(row: dict[str, Any]) -> tuple[int, float]:
    """Best first; rows with no score sink to the bottom rather than erroring."""
    score = row.get("score")
    if score is None or not math.isfinite(score):
        return (1, 0.0)
    return (0, -score)


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@router.get("/meta")
def get_meta() -> dict[str, Any]:
    """Everything the client needs to render its chrome before any list loads."""
    data = state.load()
    settings = {**DEFAULT_SETTINGS, **store.get_settings()}
    if data is None:
        return {
            "ready": False,
            "settings": settings,
            "refresh": refresh_status(),
            "windows": [
                {"key": w.key, "label": w.label, "lookback": w.lookback, "skip": w.skip}
                for w in config.WINDOWS.values()
            ],
        }
    return {
        "ready": True,
        "built_at": data["built_at"],
        "as_of": data.get("as_of"),
        "universe_size": data["universe_size"],
        "trading_days": data.get("trading_days"),
        "build_seconds": data.get("build_seconds"),
        "windows": data["windows"],
        "return_windows": data["return_windows"],
        "sectors": data["sectors"],
        "risk_model": data.get("risk_model", {}),
        "universe_report": data.get("universe_report", {}),
        "regime": data.get("macro", {}).get("regime", {}),
        "settings": settings,
        "refresh": refresh_status(),
    }


@router.get("/rankings")
def get_rankings(
    window: WindowKey = Query(config.DEFAULT_WINDOW),
    return_mode: ReturnMode = Query("raw"),
    risk_mode: RiskMode = Query(ranking.DEFAULT_RISK_MODE),
    sector: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Ranked stock rows for the list view."""
    data = require_snapshot()
    stocks = data["stocks"]

    if window == "market_cap":
        rows = [
            {**_row(s, config.DEFAULT_WINDOW, return_mode, risk_mode),
             "rank": s.get("cap_rank"),
             "score": s.get("market_cap")}
            for s in stocks
        ]
        rows.sort(key=lambda r: -(r.get("score") or 0))
    else:
        rows = [_row(s, window, return_mode, risk_mode) for s in stocks]
        rows.sort(key=_sort_key)

    if sector and sector != "All":
        rows = [r for r in rows if r["sector"] == sector]
    if search:
        needle = search.strip().upper()
        rows = [
            r for r in rows
            if needle in r["symbol"] or needle in (r["name"] or "").upper()
        ]

    # Re-number after filtering so the visible list reads 1..N.
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        row["display_rank"] = i

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "window": window,
        "return_mode": return_mode,
        "risk_mode": risk_mode,
        "rows": rows[offset : offset + limit],
    }


@router.get("/stock/{symbol}")
def get_stock(symbol: str) -> dict[str, Any]:
    """Full analytics for one ticker, including the price path for the mini charts."""
    data = require_snapshot()
    symbol = symbol.upper()
    stock = next((s for s in data["stocks"] if s["symbol"] == symbol), None)

    macro_asset = None
    if stock is None:
        macro_asset = next(
            (a for a in data["macro"]["assets"] if a["symbol"] == symbol), None
        )
        if macro_asset is None:
            raise HTTPException(status_code=404, detail=f"{symbol} is not in the universe")

    series = store.load_prices([symbol]).get(symbol, [])
    closes = [c for _d, c in series]
    dates = [d for d, _c in series]

    # Normalised return paths for each mini chart window.
    charts = []
    for key, label, days in config.RETURN_WINDOWS:
        if len(closes) > days:
            seg = closes[-(days + 1) :]
            seg_dates = dates[-(days + 1) :]
        else:
            seg, seg_dates = closes, dates
        if not seg:
            continue
        base = seg[0]
        charts.append(
            {
                "key": key,
                "label": label,
                "days": days,
                "points": [round(c / base - 1.0, 5) for c in seg] if base else [],
                "dates": seg_dates,
                "change": round(seg[-1] / base - 1.0, 5) if base else None,
            }
        )

    payload: dict[str, Any] = {
        "symbol": symbol,
        "charts": charts,
        "last_price": closes[-1] if closes else None,
        "history_days": len(closes),
        "in_watchlist": symbol in store.get_watchlist(),
    }

    if stock is not None:
        payload.update(
            {
                "kind": "stock",
                "name": stock["name"],
                "sector": stock["sector"],
                "market_cap": stock.get("market_cap"),
                "cap_rank": stock.get("cap_rank"),
                "beta": stock.get("beta"),
                "sector_beta": stock.get("sector_beta"),
                "r_squared": stock.get("r_squared"),
                "idio_vol": stock.get("idio_vol"),
                "cluster": stock.get("cluster"),
                "trailing": stock.get("trailing", {}),
                "metrics": stock.get("metrics", {}),
            }
        )
    else:
        payload.update({"kind": "macro", **(macro_asset or {})})
    return payload


@router.get("/macro")
def get_macro() -> dict[str, Any]:
    """Macro dashboard: ranked cross-asset table, heatmap and regime read."""
    data = require_snapshot()
    macro = data["macro"]

    assets = sorted(
        macro["assets"],
        key=lambda a: (a.get("score") is None, -(a.get("score") or 0)),
    )
    rows = [
        {
            "symbol": a["symbol"], "label": a["label"], "asset_class": a["asset_class"],
            "score": a.get("score"), "vol": a.get("vol"),
            "ann_return": a.get("ann_return"), "beta": a.get("beta"),
            "trailing": a.get("trailing", {}), "cluster": a.get("cluster"),
            "rank": i + 1,
        }
        for i, a in enumerate(assets)
    ]
    grid = _rebuild_heatmap(rows)
    return {
        "regime": macro["regime"],
        "assets": rows,
        "heatmap": grid,
        "diagnostics": macro.get("diagnostics", {}),
    }


def _rebuild_heatmap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the heatmap scale for whichever risk mode was requested."""
    values = [r["score"] for r in rows if r["score"] is not None]
    bound = max([abs(v) for v in values] + [0.5]) if values else 1.0
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["asset_class"], []).append(
            {
                "symbol": row["symbol"], "label": row["label"], "score": row["score"],
                "rank": row["rank"], "ann_return": row["ann_return"],
                "cluster": row["cluster"],
            }
        )
    order = ["Index", "Sector", "Rates", "Commodity"]
    return {
        "scale": {"min": -bound, "max": bound},
        "groups": [{"name": n, "cells": groups[n]} for n in order if n in groups],
    }


@router.get("/portfolios")
def get_portfolios() -> dict[str, Any]:
    data = require_snapshot()
    return {
        "sector_exposure": data.get("sector_exposure", []),
        "portfolios": data.get("portfolios", []),
        "hrp_universe": data.get("hrp_universe", {}),
        "clusters": data.get("clusters", {}),
    }


# ------------------------------------------------------------------------- watchlist
class WatchlistPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)


@router.get("/watchlist")
def get_watchlist(
    window: WindowKey = Query(config.DEFAULT_WINDOW),
    return_mode: ReturnMode = Query("raw"),
    risk_mode: RiskMode = Query(ranking.DEFAULT_RISK_MODE),
) -> dict[str, Any]:
    symbols = store.get_watchlist()
    data = state.load()
    rows: list[dict[str, Any]] = []
    if data:
        by_symbol = {s["symbol"]: s for s in data["stocks"]}
        macro_by_symbol = {a["symbol"]: a for a in data["macro"]["assets"]}
        w = config.DEFAULT_WINDOW if window == "market_cap" else window
        for symbol in symbols:
            if symbol in by_symbol:
                rows.append(_row(by_symbol[symbol], w, return_mode, risk_mode))
            elif symbol in macro_by_symbol:
                a = macro_by_symbol[symbol]
                rows.append(
                    {
                        "symbol": a["symbol"], "name": a["label"],
                        "sector": a["asset_class"], "beta": a.get("beta"),
                        "score": a.get("score"), "rank": a.get("rank"),
                        "ann_return": a.get("ann_return"), "vol": a.get("vol"),
                        "market_cap": None, "cluster": a.get("cluster"),
                    }
                )
        rows.sort(key=_sort_key)
    return {"symbols": symbols, "rows": rows}


@router.post("/watchlist")
def add_watchlist(payload: WatchlistPayload) -> dict[str, Any]:
    return {"symbols": store.add_to_watchlist(payload.symbol.upper())}


@router.delete("/watchlist/{symbol}")
def remove_watchlist(symbol: str) -> dict[str, Any]:
    return {"symbols": store.remove_from_watchlist(symbol.upper())}


@router.get("/watchlist/hrp")
def watchlist_hrp(
    k: int = Query(config.DEFAULT_CLUSTER_K, ge=1, le=config.MAX_CLUSTER_K),
    return_mode: ReturnMode = Query("raw"),
    window: WindowKey = Query(config.DEFAULT_WINDOW),
) -> dict[str, Any]:
    """Hierarchical Risk Parity over the current watchlist, computed live.

    The watchlist changes with a single tap, so these weights are always derived on
    demand rather than baked into the snapshot.
    """
    symbols = store.get_watchlist()
    if len(symbols) < 2:
        return {
            "symbols": symbols,
            "weights": [],
            "message": "Add at least two names to build an HRP allocation.",
        }

    needed = sorted(set(symbols) | {config.MARKET_ETF})
    price_map = store.load_prices(needed)
    missing = [s for s in symbols if s not in price_map]
    usable = [s for s in symbols if s in price_map]
    if len(usable) < 2:
        return {"symbols": symbols, "weights": [], "message": "Not enough price history."}

    panel = build_panel(price_map, min_history=60)
    wl_panel = panel.subset([s for s in usable if panel.has(s)])
    if len(wl_panel.symbols) < 2:
        return {"symbols": symbols, "weights": [], "message": "Not enough price history."}

    win = config.WINDOWS[config.DEFAULT_WINDOW if window == "market_cap" else window]
    returns = wl_panel.returns
    if return_mode == "residual":
        factor = risk.estimate_factor_model(panel, {}, use_sector_factors=False)
        betas = dict(zip(factor.symbols, factor.beta))
        returns = risk.market_residual_returns(wl_panel, betas, panel)

    sliced = returns[max(0, returns.shape[0] - win.total) : returns.shape[0] - win.skip]
    cov = risk.estimate_covariance(sliced, wl_panel.symbols, min_obs=20)
    result = hrp.hierarchical_risk_parity(cov, k=min(k, len(wl_panel.symbols)))
    medoids = hrp.cluster_universe(cov, min(k, len(wl_panel.symbols)))

    equal = np.full(len(result.symbols), 1.0 / len(result.symbols))
    corr = cov.correlation()
    off_diag = corr[~np.eye(len(corr), dtype=bool)]

    return {
        "symbols": symbols,
        "missing": missing,
        "weights": [
            {
                "symbol": s,
                "weight": round(float(w), 5),
                "equal_weight": round(float(1.0 / len(result.symbols)), 5),
                "vol": round(float(v), 5),
                "cluster": c,
            }
            for s, w, v, c in zip(
                result.symbols, result.weights, cov.annual_vol, result.clusters
            )
        ],
        "portfolio_vol": round(result.portfolio_vol, 5),
        "equal_weight_vol": round(cov.portfolio_vol(equal), 5),
        "effective_n": round(result.effective_n, 2),
        "avg_correlation": round(float(np.mean(off_diag)), 4) if off_diag.size else None,
        "clusters": medoids.groups(),
        "shrinkage": round(float(cov.shrinkage), 4),
    }


# -------------------------------------------------------------------------- settings
@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **store.get_settings()}


@router.put("/settings")
def put_settings(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {k: v for k, v in payload.items() if k in DEFAULT_SETTINGS}
    if not allowed:
        raise HTTPException(status_code=400, detail="No recognised settings supplied")
    store.save_settings(allowed)
    return {**DEFAULT_SETTINGS, **store.get_settings()}


# --------------------------------------------------------------------------- refresh
def refresh_status() -> dict[str, Any]:
    return {
        "running": state.refreshing,
        "stage": state.stage,
        "progress": round(state.progress, 3),
        "error": state.error,
        "last_build": store.get_meta("last_build"),
    }


@router.get("/refresh")
def get_refresh() -> dict[str, Any]:
    return refresh_status()


@router.post("/refresh")
async def post_refresh(
    force_full: bool = Query(False),
    size: int = Query(config.UNIVERSE_SIZE, ge=10, le=2000),
) -> dict[str, Any]:
    """Kick off a rebuild in the background and return immediately."""
    if state.refreshing:
        return {"started": False, **refresh_status()}

    settings = {**DEFAULT_SETTINGS, **store.get_settings()}

    async def run() -> None:
        state.refreshing = True
        state.error = None
        state.progress = 0.0
        try:
            def progress(stage: str, pct: float) -> None:
                state.stage, state.progress = stage, pct

            data = await snap.full_refresh(
                size=size,
                cluster_k=int(settings.get("cluster_k") or config.DEFAULT_CLUSTER_K),
                progress=progress,
                force_full=force_full,
            )
            state.snapshot = data
            state.stage = "complete"
        except Exception as exc:  # surfaced to the client rather than swallowed
            log.exception("refresh failed")
            state.error = f"{type(exc).__name__}: {exc}"
            state.stage = "failed"
        finally:
            state.refreshing = False

    asyncio.create_task(run())
    return {"started": True, **refresh_status()}


@router.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
    """Ticker/name search across stocks and macro instruments."""
    data = require_snapshot()
    needle = q.strip().upper()
    hits: list[dict[str, Any]] = []
    for s in data["stocks"]:
        if needle in s["symbol"] or needle in (s["name"] or "").upper():
            hits.append({"symbol": s["symbol"], "name": s["name"],
                         "sector": s["sector"], "kind": "stock"})
    for a in data["macro"]["assets"]:
        if needle in a["symbol"] or needle in a["label"].upper():
            hits.append({"symbol": a["symbol"], "name": a["label"],
                         "sector": a["asset_class"], "kind": "macro"})
    hits.sort(key=lambda h: (not h["symbol"].startswith(needle), len(h["symbol"])))
    return {"results": hits[:limit]}


# --------------------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="Biggie - Stock Ranking & Macro Intelligence",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.include_router(router)

    if config.FRONTEND_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=config.FRONTEND_DIR),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(config.FRONTEND_DIR / "index.html")

        @app.get("/manifest.webmanifest", include_in_schema=False)
        def manifest() -> FileResponse:
            return FileResponse(config.FRONTEND_DIR / "manifest.webmanifest")

    @app.get("/health", include_in_schema=False)
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "snapshot": state.load() is not None})

    return app


app = create_app()
