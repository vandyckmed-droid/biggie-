"""End-to-end shape tests: synthetic prices -> snapshot -> published payload.

The analytics tests check that the maths is right. These check the *contract* - that what
the pipeline emits is what the browser reads. That gap is how a macro rename shipped a
heatmap of em dashes while every Python test passed, `node --check` passed, and the site
built without complaint.

No network: prices are synthesised and the store is stubbed.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app import config, contract, snapshot as snap, store, universe, validate  # noqa: E402


# --------------------------------------------------------------------------- fixtures
def synth_prices(symbols, days=560, seed=11):
    """Correlated random-walk closes on a weekday calendar ending today.

    The calendar must end today or the freshness check rejects the build, and VTI must
    *be* the market factor rather than another noisy asset - regressing on a noisy proxy
    attenuates every beta and the plausibility check then fails for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    calendar: list[str] = []
    cursor = date.today()
    while len(calendar) < days:
        if cursor.weekday() < 5:
            calendar.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    calendar.reverse()

    market = rng.normal(0.0005, 0.009, days)

    def to_closes(rets):
        return list(zip(calendar, [float(c) for c in 50.0 * np.cumprod(1.0 + rets)]))

    out = {}
    for i, symbol in enumerate(symbols):
        if symbol == config.MARKET_ETF:
            rets = market.copy()
        else:
            beta = 0.6 + (i % 7) * 0.2
            rets = beta * market + rng.normal(0.0002, 0.008, days)
        out[symbol] = to_closes(rets)
    return out


def candidates(symbols):
    sectors = config.GICS_SECTORS
    return [
        universe.Candidate(
            symbol=s, name=f"{s} Corporation", sector=sectors[i % len(sectors)],
            industry="Test", exchange="NASDAQ", country="US",
            market_cap=float(1e12 - i * 1e9), price=100.0, volume=5e6,
        )
        for i, s in enumerate(symbols)
    ]


@pytest.fixture
def built(monkeypatch):
    """A full snapshot built from synthetic data, plus its published core payload."""
    stocks = [f"AA{i:03d}" for i in range(60)]
    symbols = stocks + list(config.REFERENCE_TICKERS)
    prices = synth_prices(symbols)

    monkeypatch.setattr(store, "load_prices", lambda syms, since=None: {
        s: prices[s] for s in syms if s in prices
    })

    snapshot = snap.build_snapshot(candidates(stocks), None, size=len(stocks))

    import build_site  # noqa: PLC0415 - imported here so the stub is already installed

    core = build_site.build_core(snapshot)
    core["meta"]["schema_version"] = contract.SCHEMA_VERSION
    return snapshot, core


# ------------------------------------------------------------------------------ tests
class TestPublishedContract:
    def test_core_payload_satisfies_the_client_contract(self, built):
        _snapshot, core = built
        result = contract.check_core(core)
        assert result.ok, result.errors

    def test_every_macro_asset_carries_a_score(self, built):
        """The exact regression that shipped: macro assets renamed out from under the
        client, so every heatmap cell rendered as an em dash and the ranking sorted on
        `undefined`."""
        _snapshot, core = built
        assets = core["macro"]["assets"]
        assert assets
        for asset in assets:
            assert asset.get("score") is not None, f"{asset['symbol']} has no score"
            assert asset.get("vol") is not None

    def test_stock_metrics_expose_both_risk_denominators(self, built):
        _snapshot, core = built
        metrics = core["stocks"][0]["metrics"]
        for key, block in metrics.items():
            assert set(block) >= contract.STOCK_METRIC_FIELDS, key
            assert block["score_total"] is not None
            assert block["vol"] is not None

    def test_contract_rejects_a_renamed_macro_field(self, built):
        _snapshot, core = built
        for asset in core["macro"]["assets"]:
            asset["score_cov"] = asset.pop("score")
        result = contract.check_core(core)
        assert not result.ok
        assert any("macro" in e for e in result.errors)

    def test_contract_rejects_a_present_but_empty_field(self, built):
        """A field that exists and is null renders exactly like a missing one."""
        _snapshot, core = built
        for asset in core["macro"]["assets"]:
            asset["score"] = None
        result = contract.check_core(core)
        assert not result.ok

    def test_contract_rejects_renamed_stock_volatility(self, built):
        _snapshot, core = built
        for stock in core["stocks"]:
            for block in stock["metrics"].values():
                block["vol_simple"] = block.pop("vol")
        assert not contract.check_core(core).ok


class TestSnapshotValidation:
    def test_a_synthetic_build_passes_validation(self, built):
        snapshot, _core = built
        result = validate.validate_snapshot(snapshot, expected_size=60)
        assert result.ok, result.errors

    def test_betas_are_populated_and_plausible(self, built):
        snapshot, _core = built
        betas = [s["beta"] for s in snapshot["stocks"] if s["beta"] is not None]
        assert len(betas) == len(snapshot["stocks"])
        median = sorted(betas)[len(betas) // 2]
        assert 0.4 <= median <= 1.8, median

    def test_macro_regime_is_one_of_the_three_states(self, built):
        snapshot, _core = built
        assert snapshot["macro"]["regime"]["state"] in {"Risk-On", "Risk-Off", "Transition"}


class TestPublishedSeries:
    def test_series_share_one_dated_calendar(self, built, monkeypatch):
        """Aligning by array position would pair a new listing's first week against an
        established name's last week, and truncating to the shortest series would let one
        IPO shorten the history behind every watchlist."""
        import build_site  # noqa: PLC0415

        snapshot, _core = built
        symbols = [s["symbol"] for s in snapshot["stocks"]]
        series = build_site.build_returns(symbols + list(config.REFERENCE_TICKERS))

        assert series["dates"], "series must publish its trading calendar"
        assert series["dates"] == sorted(series["dates"])
        width = len(series["dates"])
        for symbol, row in zip(series["symbols"], series["data"]):
            assert len(row) == width, f"{symbol} is not on the shared calendar"

    def test_missing_observations_are_null_not_zero(self, monkeypatch):
        """A gap filled with 0.0 reads as a flat trading day and deflates volatility."""
        import build_site  # noqa: PLC0415

        symbols = ["GAPPY"] + list(config.REFERENCE_TICKERS)
        prices = synth_prices(symbols, days=400)
        # Punch a real hole: three consecutive sessions with no bar at all.
        prices["GAPPY"] = [p for i, p in enumerate(prices["GAPPY"]) if i not in (200, 201, 202)]

        monkeypatch.setattr(store, "load_prices", lambda syms, since=None: {
            s: prices[s] for s in syms if s in prices
        })
        series = build_site.build_returns(symbols)
        row = series["data"][series["symbols"].index("GAPPY")]
        assert any(v is None for v in row), "the gap should surface as null"
        assert all(v != 0.0 for v in row if v is not None)
