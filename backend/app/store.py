"""SQLite-backed cache for price history, security metadata and user state.

Prices are the expensive thing to fetch, so they are stored once and refreshed
incrementally: a rebuild only asks FMP for bars newer than what is already on disk.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    close  REAL NOT NULL,
    volume REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS securities (
    symbol      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    industry    TEXT,
    exchange    TEXT,
    country     TEXT,
    market_cap  REAL,
    price       REAL,
    volume      REAL,
    kind        TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS watchlist (
    symbol     TEXT PRIMARY KEY,
    added_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_local = threading.local()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.CACHE_DB, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


def connection() -> sqlite3.Connection:
    """One connection per thread; SQLite objects are not shareable across threads."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------ prices
def upsert_prices(bars: dict[str, Sequence[dict[str, Any]]]) -> int:
    """Insert or replace daily bars. Returns the number of rows written."""
    rows = [
        (sym, b["date"], float(b["close"]), float(b.get("volume") or 0.0))
        for sym, series in bars.items()
        for b in series
    ]
    if not rows:
        return 0
    with transaction() as conn:
        conn.executemany(
            "INSERT INTO prices(symbol, date, close, volume) VALUES (?,?,?,?) "
            "ON CONFLICT(symbol, date) DO UPDATE SET close=excluded.close, "
            "volume=excluded.volume",
            rows,
        )
    return len(rows)


def latest_dates(symbols: Iterable[str]) -> dict[str, str]:
    """Newest cached bar date per symbol, for incremental refresh."""
    syms = list(symbols)
    if not syms:
        return {}
    out: dict[str, str] = {}
    conn = connection()
    for chunk in _chunks(syms, 500):
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT symbol, MAX(date) AS d FROM prices WHERE symbol IN ({marks}) "
            "GROUP BY symbol",
            chunk,
        ).fetchall()
        out.update({r["symbol"]: r["d"] for r in rows if r["d"]})
    return out


def load_prices(symbols: Sequence[str], since: date | None = None) -> dict[str, list[tuple[str, float]]]:
    """Load (date, close) series per symbol, oldest first."""
    if not symbols:
        return {}
    out: dict[str, list[tuple[str, float]]] = {s: [] for s in symbols}
    conn = connection()
    since_s = since.isoformat() if since else "0000-00-00"
    for chunk in _chunks(list(symbols), 500):
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT symbol, date, close FROM prices "
            f"WHERE symbol IN ({marks}) AND date >= ? ORDER BY date",
            [*chunk, since_s],
        ).fetchall()
        for r in rows:
            out[r["symbol"]].append((r["date"], r["close"]))
    return {k: v for k, v in out.items() if v}


def median_dollar_volume(symbols: Sequence[str], lookback: int = 60) -> dict[str, float]:
    """Median close*volume over the most recent ``lookback`` bars, per symbol."""
    out: dict[str, float] = {}
    conn = connection()
    for sym in symbols:
        rows = conn.execute(
            "SELECT close * volume AS dv FROM prices WHERE symbol = ? "
            "ORDER BY date DESC LIMIT ?",
            (sym, lookback),
        ).fetchall()
        vals = sorted(r["dv"] for r in rows if r["dv"] is not None)
        if vals:
            mid = len(vals) // 2
            out[sym] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
    return out


def prune_prices(before: date) -> int:
    """Drop bars older than ``before`` to keep the cache bounded."""
    with transaction() as conn:
        cur = conn.execute("DELETE FROM prices WHERE date < ?", (before.isoformat(),))
    return cur.rowcount


# -------------------------------------------------------------------------- securities
def upsert_securities(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    payload = [
        (
            r["symbol"], r.get("name"), r.get("sector"), r.get("industry"),
            r.get("exchange"), r.get("country"), r.get("market_cap"),
            r.get("price"), r.get("volume"), r.get("kind", "stock"), _now(),
        )
        for r in rows
    ]
    with transaction() as conn:
        conn.executemany(
            "INSERT INTO securities(symbol,name,sector,industry,exchange,country,"
            "market_cap,price,volume,kind,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol) DO UPDATE SET name=excluded.name, sector=excluded.sector,"
            "industry=excluded.industry, exchange=excluded.exchange,"
            "country=excluded.country, market_cap=excluded.market_cap,"
            "price=excluded.price, volume=excluded.volume, kind=excluded.kind,"
            "updated_at=excluded.updated_at",
            payload,
        )


def load_securities(symbols: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    conn = connection()
    if symbols is None:
        rows = conn.execute("SELECT * FROM securities").fetchall()
    else:
        rows = []
        for chunk in _chunks(list(symbols), 500):
            marks = ",".join("?" * len(chunk))
            rows.extend(
                conn.execute(
                    f"SELECT * FROM securities WHERE symbol IN ({marks})", chunk
                ).fetchall()
            )
    return {r["symbol"]: dict(r) for r in rows}


# --------------------------------------------------------------------------- watchlist
def get_watchlist() -> list[str]:
    conn = connection()
    rows = conn.execute("SELECT symbol FROM watchlist ORDER BY added_at").fetchall()
    return [r["symbol"] for r in rows]


def add_to_watchlist(symbol: str) -> list[str]:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO watchlist(symbol, added_at) VALUES (?,?) "
            "ON CONFLICT(symbol) DO NOTHING",
            (symbol.upper(), _now()),
        )
    return get_watchlist()


def remove_from_watchlist(symbol: str) -> list[str]:
    with transaction() as conn:
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
    return get_watchlist()


# ---------------------------------------------------------------------------- settings
def get_settings() -> dict[str, Any]:
    conn = connection()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def save_settings(values: dict[str, Any]) -> dict[str, Any]:
    with transaction() as conn:
        conn.executemany(
            "INSERT INTO settings(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [(k, json.dumps(v)) for k, v in values.items()],
        )
    return get_settings()


# -------------------------------------------------------------------------------- meta
def set_meta(key: str, value: Any) -> None:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )


def get_meta(key: str, default: Any = None) -> Any:
    row = connection().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
