"""Async Financial Modeling Prep client.

Only the handful of endpoints this system needs, wrapped with bounded concurrency,
exponential backoff and 429-aware throttling so a 1,000-symbol pull stays polite.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Iterable, Sequence

import httpx

from . import config

log = logging.getLogger(__name__)

# httpx logs every request URL at INFO, and our URLs carry the API key as a query
# parameter. Never let the key reach a log file or a console.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class FMPError(RuntimeError):
    """Raised when FMP returns an error payload we cannot recover from."""


class _RateLimiter:
    """Paces requests to a sustained rate, regardless of how many workers are running.

    A semaphore alone bounds *concurrency*, not *rate*: six workers finishing quickly
    still burst far past a per-minute quota. This hands out evenly spaced slots so the
    sustained rate is what the plan allows, which is what stops 429s happening at all
    rather than reacting once they do.
    """

    def __init__(self, per_minute: int) -> None:
        self._interval = 60.0 / max(per_minute, 1)
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            start = max(now, self._next)
            self._next = start + self._interval
            delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)


class FMPClient:
    """Thin async wrapper around the FMP `stable` API."""

    def __init__(self, key: str | None = None, concurrency: int | None = None) -> None:
        self._key = key or config.api_key()
        self._sem = asyncio.Semaphore(concurrency or config.MAX_CONCURRENCY)
        self._limiter = _RateLimiter(config.RATE_LIMIT_PER_MIN)
        self._client: httpx.AsyncClient | None = None
        # Set when a 429 is seen; every worker waits this out before its next request.
        self._cooldown_until = 0.0
        self.throttled = 0

    async def __aenter__(self) -> "FMPClient":
        self._client = httpx.AsyncClient(
            base_url=config.FMP_BASE,
            timeout=config.REQUEST_TIMEOUT,
            headers={"User-Agent": "biggie-ranking/1.0"},
            limits=httpx.Limits(max_connections=config.MAX_CONCURRENCY * 2),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---------------------------------------------------------------- internals
    def _redact(self, text: str, limit: int = 200) -> str:
        """Strip the API key out of anything that might be logged or raised."""
        return text.replace(self._key, "***")[:limit]

    async def _respect_cooldown(self) -> None:
        loop = asyncio.get_running_loop()
        delay = self._cooldown_until - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

    async def _get(self, path: str, **params: Any) -> Any:
        """GET a JSON endpoint, retrying transient failures with exponential backoff."""
        if self._client is None:
            raise RuntimeError("FMPClient must be used as an async context manager")

        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = self._key
        last_error: Exception | None = None

        for attempt in range(config.MAX_RETRIES):
            await self._respect_cooldown()
            await self._limiter.acquire()
            try:
                async with self._sem:
                    resp = await self._client.get(path, params=params)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
            else:
                if resp.status_code == 429:
                    # Pause every worker, not just this one: the quota is per account, so
                    # letting the others keep firing just deepens the hole.
                    self.throttled += 1
                    loop = asyncio.get_running_loop()
                    backoff = min(5.0 * (2**attempt), config.MAX_BACKOFF)
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            backoff = max(backoff, float(retry_after))
                        except ValueError:
                            pass
                    self._cooldown_until = max(self._cooldown_until, loop.time() + backoff)
                    last_error = FMPError("rate limited")
                    continue
                if resp.status_code in (500, 502, 503, 504):
                    last_error = FMPError(f"server error {resp.status_code}")
                elif resp.status_code >= 400:
                    # 401/403 etc. are terminal - retrying will not help.
                    raise FMPError(
                        f"{path} -> HTTP {resp.status_code}: {self._redact(resp.text)}"
                    )
                else:
                    try:
                        payload = resp.json()
                    except ValueError as exc:
                        raise FMPError(f"{path} -> non-JSON response") from exc
                    if isinstance(payload, dict) and "Error Message" in payload:
                        raise FMPError(
                            f"{path} -> {self._redact(str(payload['Error Message']))}"
                        )
                    return payload

            await asyncio.sleep(min(1.5 * (2**attempt), config.MAX_BACKOFF))

        raise FMPError(f"{path} failed after {config.MAX_RETRIES} attempts: {last_error}")

    # ---------------------------------------------------------------- endpoints
    async def screener(
        self,
        *,
        market_cap_more_than: float,
        exchanges: Iterable[str] = ("NASDAQ", "NYSE", "AMEX"),
        limit: int = 5000,
        actively_trading: bool = True,
    ) -> list[dict[str, Any]]:
        """Company screener - the source of the tradeable universe and its sectors."""
        rows = await self._get(
            "/company-screener",
            marketCapMoreThan=int(market_cap_more_than),
            exchange=",".join(exchanges),
            isEtf="false",
            isFund="false",
            isActivelyTrading="true" if actively_trading else None,
            limit=limit,
        )
        return rows if isinstance(rows, list) else []

    async def history(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> list[dict[str, Any]]:
        """Daily bars for one symbol, oldest first.

        Uses the dividend-adjusted series so total-return momentum is not distorted by
        distributions - which matters a great deal for TLT, XLU and other high yielders.
        """
        path = (
            "/historical-price-eod/dividend-adjusted"
            if adjusted
            else "/historical-price-eod/full"
        )
        rows = await self._get(
            path, symbol=symbol, **{"from": start.isoformat(), "to": end.isoformat()}
        )
        if not isinstance(rows, list):
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            close = row.get("adjClose", row.get("close"))
            if close is None or row.get("date") is None:
                continue
            try:
                close_f = float(close)
            except (TypeError, ValueError):
                continue
            if close_f <= 0:
                continue
            out.append(
                {
                    "date": str(row["date"])[:10],
                    "close": close_f,
                    "volume": float(row.get("volume") or 0.0),
                }
            )
        out.sort(key=lambda r: r["date"])
        return out

    async def histories(
        self,
        symbols: Sequence[str],
        *,
        start: date,
        end: date,
        on_progress: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch many symbols concurrently. Symbols that fail are simply omitted."""
        results: dict[str, list[dict[str, Any]]] = {}
        done = 0
        total = len(symbols)
        lock = asyncio.Lock()

        async def one(sym: str) -> None:
            nonlocal done
            try:
                bars = await self.history(sym, start=start, end=end)
            except FMPError as exc:
                log.warning("history failed for %s: %s", sym, exc)
                bars = []
            async with lock:
                if bars:
                    results[sym] = bars
                done += 1
                if on_progress and (done % 25 == 0 or done == total):
                    on_progress(done, total)

        await asyncio.gather(*(one(s) for s in symbols))
        return results


def default_window() -> tuple[date, date]:
    """The (start, end) calendar range covering the configured history length."""
    end = date.today()
    return end - timedelta(days=config.HISTORY_DAYS), end
