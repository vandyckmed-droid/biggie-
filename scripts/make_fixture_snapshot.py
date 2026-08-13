#!/usr/bin/env python3
"""Write a synthetic snapshot so the site can be built and rendered without market data.

CI has no API key, so without this the render path - the one place the macro field
rename was visible - could never be exercised. The data is fake; the *shape* is real,
because it comes from the same `build_snapshot` the daily pipeline runs.

    BIGGIE_DATA_DIR=/tmp/fixture python scripts/make_fixture_snapshot.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config, snapshot as snap, store, universe  # noqa: E402

STOCK_COUNT = 120
DAYS = 560


def synth(symbols: list[str], seed: int = 7) -> dict[str, list[tuple[str, float]]]:
    rng = np.random.default_rng(seed)
    calendar: list[str] = []
    cursor = date.today()
    while len(calendar) < DAYS:
        if cursor.weekday() < 5:
            calendar.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    calendar.reverse()

    market = rng.normal(0.0005, 0.009, DAYS)
    out: dict[str, list[tuple[str, float]]] = {}
    for i, symbol in enumerate(symbols):
        if symbol == config.MARKET_ETF:
            rets = market.copy()
        else:
            beta = 0.6 + (i % 7) * 0.2
            rets = beta * market + rng.normal(0.0003, 0.009, DAYS)
        closes = 50.0 * np.cumprod(1.0 + rets)
        out[symbol] = list(zip(calendar, [float(c) for c in closes]))
    return out


def main() -> int:
    stocks = [f"SYN{i:03d}" for i in range(STOCK_COUNT)]
    symbols = stocks + list(config.REFERENCE_TICKERS)
    prices = synth(symbols)

    # Feed the real builder through a stubbed store rather than reimplementing it, so
    # the fixture cannot drift from the shape production emits.
    store.load_prices = lambda syms, since=None: {  # type: ignore[assignment]
        s: prices[s] for s in syms if s in prices
    }

    sectors = config.GICS_SECTORS
    candidates = [
        universe.Candidate(
            symbol=s, name=f"Synthetic {s} Inc.", sector=sectors[i % len(sectors)],
            industry="Test", exchange="NASDAQ", country="US",
            market_cap=float(2e12 - i * 5e9), price=100.0, volume=4e6,
        )
        for i, s in enumerate(stocks)
    ]

    data = snap.build_snapshot(candidates, None, size=len(stocks))
    snap.save_snapshot(data)
    print(
        f"Fixture snapshot: {data['universe_size']} stocks, "
        f"{len(data['macro']['assets'])} macro assets, as of {data['as_of']} "
        f"-> {config.SNAPSHOT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
