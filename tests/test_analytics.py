"""Tests for the analytic core.

These use synthetic data with known answers rather than live market data, so a failure
points at the maths rather than at whatever the market did overnight.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config, hrp, ranking, risk, universe, validate  # noqa: E402
from app.returns import (  # noqa: E402
    ReturnPanel,
    annualized_return,
    annualized_volatility,
    build_panel,
    trailing_return,
)


# ---------------------------------------------------------------------- universe
class TestExclusions:
    """The universe must contain plain common equity and nothing else."""

    @pytest.mark.parametrize(
        "symbol,name,expected",
        [
            # Preferreds: FMP encodes them as a -P<letter> suffix.
            ("FITB-PA", "Fifth Third Bancorp", "preferred"),
            ("SEAL-PB", "Seapeak LLC", "preferred"),
            ("MER-PK", "Merrill Lynch Capital Trust I GTD CAP 6.45%", "preferred"),
            # Derivatives and units.
            ("ACME-WT", "Acme Corp Warrants", "warrant"),
            ("ACME-RT", "Acme Corp Rights", "right"),
            ("SPAC-UN", "SPAC Holdings Units", "unit"),
            ("SPAC-U", "SPAC Holdings Unit", "unit"),
            # Fixed income masquerading as equity.
            ("XYZ", "Acme Capital 6.25% Senior Notes due 2032", "baby_bond"),
            ("ABC", "Acme Trust Preferred Securities", "baby_bond"),
            # Fund wrappers.
            ("DEF", "Barclays ETN linked to something", "fund_or_etn"),
            ("GHI", "Abrdn Income Credit Strategies Fund", "fund_or_etn"),
        ],
    )
    def test_structurally_unsuitable_is_excluded(self, symbol, name, expected):
        assert universe.exclusion_reason(symbol, name) == expected

    @pytest.mark.parametrize(
        "symbol,name",
        [
            # Bare single-letter suffixes are real share classes, not preferreds.
            ("BRK-A", "Berkshire Hathaway Inc."),
            ("BRK-B", "Berkshire Hathaway Inc."),
            ("BF-B", "Brown-Forman Corporation"),
            ("MOG-A", "Moog Inc."),
            ("MKC-V", "McCormick & Company, Incorporated"),
            ("AAPL", "Apple Inc."),
            ("GOOGL", "Alphabet Inc."),
        ],
    )
    def test_common_equity_survives(self, symbol, name):
        assert universe.exclusion_reason(symbol, name) is None


class TestDeduplication:
    def _row(self, symbol, name, cap, price=100.0, volume=1e6):
        return {
            "symbol": symbol, "companyName": name, "marketCap": cap,
            "price": price, "volume": volume, "sector": "Technology",
            "industry": "Software", "exchangeShortName": "NASDAQ",
            "country": "US", "isEtf": False, "isFund": False,
            "isActivelyTrading": True,
        }

    def test_dual_class_collapses_to_most_liquid(self):
        rows = [
            self._row("GOOG", "Alphabet Inc.", 2e12, volume=1e6),
            self._row("GOOGL", "Alphabet Inc.", 2e12, volume=5e6),
        ]
        report = universe.build_universe(rows)
        symbols = [c.symbol for c in report.selected]
        assert symbols == ["GOOGL"], "the more liquid class should win"
        assert report.deduped == 1

    def test_share_classes_with_differing_roots_still_collapse(self):
        # GOOG/GOOGL differ by a trailing letter; naive fixed-length stripping breaks here.
        for pair in [("FOX", "FOXA"), ("DISCA", "DISCK"), ("BRK-A", "BRK-B")]:
            rows = [self._row(pair[0], "Same Co Inc.", 1e11),
                    self._row(pair[1], "Same Co Inc.", 1e11, volume=2e6)]
            report = universe.build_universe(rows)
            assert len(report.selected) == 1, f"{pair} should collapse"

    def test_different_issuers_are_not_merged(self):
        rows = [
            self._row("AAPL", "Apple Inc.", 3e12),
            self._row("MSFT", "Microsoft Corporation", 3e12),
        ]
        report = universe.build_universe(rows)
        assert len({c.symbol for c in report.selected}) == 2

    def test_takes_largest_by_market_cap(self):
        rows = [self._row(f"S{i}", f"Company {i} Inc.", cap=i * 1e9) for i in range(1, 21)]
        report = universe.build_universe(rows, size=5)
        caps = [c.market_cap for c in report.selected]
        assert caps == sorted(caps, reverse=True)
        assert len(report.selected) == 5
        assert caps[0] == 20e9

    def test_otc_and_funds_are_dropped(self):
        rows = [
            {**self._row("PINK", "Pink Sheet Co", 1e10), "exchangeShortName": "OTC"},
            {**self._row("FUND", "Some Fund", 1e10), "isFund": True},
            self._row("GOOD", "Good Company Inc.", 1e10),
        ]
        report = universe.build_universe(rows)
        assert [c.symbol for c in report.selected] == ["GOOD"]


# ----------------------------------------------------------------------- returns
def make_panel(n_days=520, n_syms=12, seed=7, drift=None):
    """A synthetic panel with a shared market factor plus idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, n_days)
    symbols = [f"S{i:02d}" for i in range(n_syms)] + [config.MARKET_ETF]
    betas = np.linspace(0.5, 2.0, n_syms)

    cols = []
    for i in range(n_syms):
        mu = 0.0 if drift is None else drift[i]
        cols.append(betas[i] * market + rng.normal(mu, 0.008, n_days))
    cols.append(market)

    returns = np.column_stack(cols)
    prices = np.vstack([np.ones(len(symbols)), np.cumprod(1 + returns, axis=0) ])
    dates = [f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_days + 1)]
    return ReturnPanel(dates=dates, symbols=symbols, prices=prices, returns=returns), betas


class TestReturns:
    def test_window_applies_skip(self):
        panel, _ = make_panel(n_days=300)
        w = panel.window(lookback=100, skip=20)
        assert w.shape[0] == 100
        # The window must end 20 rows before the last observation.
        np.testing.assert_allclose(w[-1], panel.returns[-21])

    def test_window_shorter_than_history_is_truncated_not_wrapped(self):
        panel, _ = make_panel(n_days=50)
        w = panel.window(lookback=250, skip=20)
        assert w.shape[0] == 30

    def test_annualized_return_matches_compounding(self):
        # A constant 0.1%/day for exactly one trading year.
        window = np.full((config.TRADING_DAYS, 1), 0.001)
        expected = 1.001**config.TRADING_DAYS - 1
        np.testing.assert_allclose(annualized_return(window)[0], expected, rtol=1e-9)

    def test_annualized_volatility_scales_by_sqrt_time(self):
        rng = np.random.default_rng(0)
        window = rng.normal(0, 0.02, (2000, 1))
        vol = annualized_volatility(window)[0]
        assert vol == pytest.approx(0.02 * np.sqrt(config.TRADING_DAYS), rel=0.06)

    def test_missing_bars_do_not_zero_fill(self):
        window = np.array([[0.01], [np.nan], [0.01]])
        # Two observed +1% days, not three days one of which was flat.
        assert annualized_return(window)[0] > 0

    def test_trailing_return(self):
        prices = np.array([[100.0], [110.0], [121.0]])
        np.testing.assert_allclose(trailing_return(prices, 2)[0], 0.21)

    def test_build_panel_uses_market_calendar(self):
        price_map = {
            config.MARKET_ETF: [(f"2024-01-{d:02d}", 100.0 + d) for d in range(1, 29)],
            # A phantom date that is not a real session must be ignored.
            "AAA": [(f"2024-01-{d:02d}", 50.0 + d) for d in range(1, 29)] + [("2024-01-31", 99.0)],
        }
        panel = build_panel(price_map, min_history=10)
        assert "2024-01-31" not in panel.dates
        assert len(panel.dates) == 28


# -------------------------------------------------------------------------- risk
class TestCovariance:
    def test_shrinkage_is_positive_definite_when_names_exceed_observations(self):
        rng = np.random.default_rng(1)
        # 40 observations, 80 names: the sample covariance is singular by construction.
        window = rng.normal(0, 0.01, (40, 80))
        model = risk.estimate_covariance(window, [f"S{i}" for i in range(80)])
        eigenvalues = np.linalg.eigvalsh(model.cov)
        assert eigenvalues.min() > 0, "shrunk covariance must be invertible"
        assert 0 < model.shrinkage <= 1

    def test_diagonal_matches_standalone_variance(self):
        rng = np.random.default_rng(2)
        window = rng.normal(0, 0.02, (500, 5))
        model = risk.estimate_covariance(window, list("ABCDE"))
        sample_vol = np.std(window, axis=0, ddof=1) * np.sqrt(config.TRADING_DAYS)
        # Shrinkage nudges toward the average variance; it should not distort.
        np.testing.assert_allclose(model.annual_vol, sample_vol, rtol=0.25)

    def test_risk_contributions_sum_to_one(self):
        rng = np.random.default_rng(3)
        window = rng.normal(0, 0.015, (300, 20))
        model = risk.estimate_covariance(window, [f"S{i}" for i in range(20)])
        w = np.full(20, 1 / 20)
        assert model.risk_contribution(w).sum() == pytest.approx(1.0)

    def test_portfolio_vol_below_weighted_average_when_uncorrelated(self):
        rng = np.random.default_rng(4)
        window = rng.normal(0, 0.02, (600, 10))  # independent columns
        model = risk.estimate_covariance(window, [f"S{i}" for i in range(10)])
        w = np.full(10, 0.1)
        assert model.portfolio_vol(w) < float(np.sum(w * model.annual_vol))

    def test_correlation_is_bounded_with_unit_diagonal(self):
        rng = np.random.default_rng(5)
        model = risk.estimate_covariance(rng.normal(0, 0.01, (200, 8)), list("ABCDEFGH"))
        corr = model.correlation()
        np.testing.assert_allclose(np.diag(corr), 1.0)
        assert corr.min() >= -1.0 and corr.max() <= 1.0


class TestFactorModel:
    def test_recovers_known_betas(self):
        panel, true_betas = make_panel(n_days=520, n_syms=10)
        model = risk.estimate_factor_model(panel, {}, use_sector_factors=False)
        estimated = np.array([
            model.beta[model.symbols.index(f"S{i:02d}")] for i in range(10)
        ])
        np.testing.assert_allclose(estimated, true_betas, atol=0.12)

    def test_beta_is_univariate_and_positive_for_long_equity(self):
        # A joint market+sector regression produces partial betas that can go negative;
        # the reported beta must be the plain univariate one.
        panel, _ = make_panel(n_syms=8)
        model = risk.estimate_factor_model(panel, {}, use_sector_factors=False)
        stock_betas = [model.beta[model.symbols.index(f"S{i:02d}")] for i in range(8)]
        assert all(b > 0 for b in stock_betas)

    def test_residuals_strip_market_exposure(self):
        panel, _ = make_panel(n_days=520, n_syms=6)
        model = risk.estimate_factor_model(panel, {}, use_sector_factors=False)
        betas = dict(zip(model.symbols, model.beta))
        resid = risk.market_residual_returns(panel, betas)

        market = panel.series(config.MARKET_ETF)
        for i in range(6):
            j = panel.col(f"S{i:02d}")
            before = abs(np.corrcoef(panel.returns[:, j], market)[0, 1])
            after = abs(np.corrcoef(resid[:, j], market)[0, 1])
            assert after < before
            assert after < 0.15, "residuals should be near-orthogonal to the market"

    def test_residuals_work_when_market_lives_in_another_panel(self):
        # The ranked universe holds only stocks; VTI sits in the reference panel.
        panel, _ = make_panel(n_syms=6)
        stocks = panel.subset([f"S{i:02d}" for i in range(6)])
        assert not stocks.has(config.MARKET_ETF)

        model = risk.estimate_factor_model(panel, {}, use_sector_factors=False)
        betas = dict(zip(model.symbols, model.beta))
        resid = risk.market_residual_returns(stocks, betas, panel)
        assert resid.shape == stocks.returns.shape
        assert not np.allclose(resid, stocks.returns)


# ----------------------------------------------------------------------- ranking
class TestRanking:
    def test_rank_is_one_based_and_best_first(self):
        ranks = ranking.rank_desc(np.array([0.1, 0.9, 0.5]))
        assert list(ranks) == [3, 1, 2]

    def test_missing_scores_rank_last(self):
        ranks = ranking.rank_desc(np.array([0.5, np.nan, 0.9]))
        assert ranks[2] == 1 and ranks[0] == 2
        assert ranks[1] == 3

    def test_higher_return_at_equal_risk_ranks_higher(self):
        drift = np.array([0.002, 0.001, 0.0])
        panel, _ = make_panel(n_days=400, n_syms=3, drift=drift)
        stocks = panel.subset(["S00", "S01", "S02"])
        result = ranking.compute_window(
            stocks, config.WINDOWS["mom_12_1"], return_mode="raw"
        )
        order = np.argsort(result.ranks["total"])
        assert [result.symbols[i] for i in order] == ["S00", "S01", "S02"]

    def test_score_guards_against_near_zero_volatility(self):
        # A flat series would otherwise divide by ~0 and dominate the ranking.
        scores = ranking._score(np.array([0.5]), np.array([1e-9]))
        assert not np.isfinite(scores[0])

    def test_idiosyncratic_vol_is_below_total_vol(self):
        """The factor-residual denominator must be a genuinely different measure.

        sqrt(eiT * Sigma * ei) is just the stock's own volatility, so the second risk
        mode uses what is left after market and sector exposure are removed. Removing
        explanatory factors can only reduce variance, so the ratio must sit below 1.
        """
        panel, _ = make_panel(n_days=520, n_syms=8)
        stocks = panel.subset([f"S{i:02d}" for i in range(8)])
        results, _ = ranking.compute_all(stocks, {}, factor_panel=panel)
        res = results[("mom_12_1", "raw")]

        ratios = res.ann_vol_idio / res.ann_vol
        finite = ratios[np.isfinite(ratios)]
        assert finite.size == 8
        assert (finite < 1.0).all(), "residual vol should be below total vol"
        assert (finite > 0.05).all(), "residual vol should not collapse to zero"
        # The two denominators must actually reorder some names, or the toggle is a lie.
        assert not np.array_equal(res.ranks["total"], res.ranks["idiosyncratic"])

    def test_all_window_and_mode_combinations_present(self):
        panel, _ = make_panel(n_syms=6)
        stocks = panel.subset([f"S{i:02d}" for i in range(6)])
        results, factors = ranking.compute_all(stocks, {}, factor_panel=panel)
        for key in config.WINDOWS:
            for mode in ("raw", "residual"):
                assert (key, mode) in results
        assert np.isfinite(factors.beta).any()

    def test_skip_windows_scale_with_lookback(self):
        for window in config.WINDOWS.values():
            ratio = window.skip / window.lookback
            assert 0.05 <= ratio <= 0.12, f"{window.key} skip is out of proportion"


# --------------------------------------------------------------------------- HRP
class TestHRP:
    def _model(self, seed=0, n=12):
        rng = np.random.default_rng(seed)
        # Two correlated blocs plus noise - the structure HRP is supposed to find.
        f1, f2 = rng.normal(0, 0.01, 400), rng.normal(0, 0.01, 400)
        cols = [
            (f1 if i < n // 2 else f2) + rng.normal(0, 0.004 * (1 + i), 400)
            for i in range(n)
        ]
        window = np.column_stack(cols)
        return risk.estimate_covariance(window, [f"S{i:02d}" for i in range(n)])

    def test_weights_are_a_valid_allocation(self):
        result = hrp.hierarchical_risk_parity(self._model())
        assert result.weights.sum() == pytest.approx(1.0)
        assert (result.weights > 0).all()

    def test_lower_volatility_names_get_more_weight(self):
        model = self._model()
        result = hrp.hierarchical_risk_parity(model)
        vols = model.annual_vol
        assert np.corrcoef(result.weights, vols)[0, 1] < 0, (
            "HRP should tilt away from the riskier names"
        )

    def test_beats_equal_weight_on_volatility(self):
        model = self._model()
        result = hrp.hierarchical_risk_parity(model)
        equal = np.full(len(model.symbols), 1 / len(model.symbols))
        assert result.portfolio_vol < model.portfolio_vol(equal)

    def test_is_deterministic(self):
        model = self._model()
        a = hrp.hierarchical_risk_parity(model)
        b = hrp.hierarchical_risk_parity(model)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_correlation_distance_is_a_metric(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        d = hrp.correlation_distance(corr)
        assert d[0, 0] == 0
        assert d[0, 1] == pytest.approx(np.sqrt(0.5 * 0.5))
        # Perfectly correlated -> distance 0; perfectly anti-correlated -> 1.
        assert hrp.correlation_distance(np.array([[1.0, 1.0], [1.0, 1.0]]))[0, 1] == 0
        assert hrp.correlation_distance(np.array([[1.0, -1.0], [-1.0, 1.0]]))[0, 1] == 1

    def test_single_name_allocation(self):
        model = risk.estimate_covariance(
            np.random.default_rng(0).normal(0, 0.01, (100, 1)), ["ONLY"]
        )
        result = hrp.hierarchical_risk_parity(model)
        assert result.weights.tolist() == [1.0]


class TestMedoids:
    def test_finds_planted_clusters(self):
        # Three tight groups; medoids must land one per group.
        d = np.full((9, 9), 1.0)
        for start in (0, 3, 6):
            for i in range(start, start + 3):
                for j in range(start, start + 3):
                    d[i, j] = 0.05
        np.fill_diagonal(d, 0.0)
        result = hrp.k_medoids(d, [f"S{i}" for i in range(9)], k=3)
        groups = result.groups()
        assert len(groups) == 3
        assert sorted(len(v) for v in groups.values()) == [3, 3, 3]

    def test_medoids_are_real_symbols(self):
        d = np.random.default_rng(0).random((10, 10))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        symbols = [f"S{i}" for i in range(10)]
        result = hrp.k_medoids(d, symbols, k=3)
        assert all(m in symbols for m in result.medoids)

    def test_is_deterministic(self):
        d = np.random.default_rng(1).random((15, 15))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        symbols = [f"S{i}" for i in range(15)]
        a = hrp.k_medoids(d, symbols, k=4)
        b = hrp.k_medoids(d, symbols, k=4)
        assert a.labels == b.labels and a.medoids == b.medoids

    def test_k_larger_than_population_is_clamped(self):
        d = np.zeros((3, 3))
        result = hrp.k_medoids(d, ["A", "B", "C"], k=10)
        assert len(result.medoids) == 3


# ---------------------------------------------------------------------- validation
class TestValidation:
    """The daily pipeline must refuse to publish a broken dataset."""

    def _snapshot(self, **overrides):
        stocks = [
            {
                "symbol": f"S{i:03d}",
                "sector": ["Energy", "Financials", "Health Care", "Industrials",
                           "Information Technology"][i % 5],
                "beta": 1.0,
                "metrics": {"mom_12_1|raw": {"score_total": 1.0}},
            }
            for i in range(1000)
        ]
        base = {
            "as_of": date.today().isoformat(),
            "trading_days": 500,
            "stocks": stocks,
            "macro": {
                "assets": [{"symbol": f"E{i}"} for i in range(21)],
                "regime": {"state": "Risk-On"},
            },
            "portfolios": [{"key": "UNIVERSE"}],
        }
        base.update(overrides)
        return base

    def test_healthy_snapshot_passes(self):
        result = validate.validate_snapshot(self._snapshot())
        assert result.ok, result.errors
        assert result.stats["universe_size"] == 1000

    def test_empty_universe_fails(self):
        assert not validate.validate_snapshot(self._snapshot(stocks=[])).ok

    def test_truncated_universe_fails(self):
        snap = self._snapshot()
        snap["stocks"] = snap["stocks"][:100]
        result = validate.validate_snapshot(snap)
        assert not result.ok
        assert any("universe has" in e for e in result.errors)

    def test_stale_data_fails(self):
        stale = (date.today() - timedelta(days=30)).isoformat()
        result = validate.validate_snapshot(self._snapshot(as_of=stale))
        assert not result.ok
        assert any("days old" in e for e in result.errors)

    def test_missing_scores_fail(self):
        snap = self._snapshot()
        for s in snap["stocks"][:500]:
            s["metrics"]["mom_12_1|raw"]["score_total"] = None
        result = validate.validate_snapshot(snap)
        assert not result.ok
        assert any("score" in e for e in result.errors)

    def test_implausible_median_beta_fails(self):
        # This is the exact regression that shipped once: a joint market+sector
        # regression produced a median beta near zero.
        snap = self._snapshot()
        for s in snap["stocks"]:
            s["beta"] = -0.1
        result = validate.validate_snapshot(snap)
        assert not result.ok
        assert any("beta" in e for e in result.errors)

    def test_invalid_regime_fails(self):
        snap = self._snapshot()
        snap["macro"]["regime"]["state"] = "Bullish"
        assert not validate.validate_snapshot(snap).ok

    def test_missing_composite_portfolio_fails(self):
        assert not validate.validate_snapshot(self._snapshot(portfolios=[])).ok

    def test_older_data_is_not_newer(self):
        old = {"as_of": "2026-01-01"}
        new = {"as_of": "2026-02-01"}
        assert validate.is_newer(new, old)
        assert not validate.is_newer(old, new)
        assert validate.is_newer(new, None)


class TestWindowLabelling:
    """Every window skips proportionally, and the labels must not imply otherwise."""

    def test_all_windows_have_a_skip(self):
        for window in config.WINDOWS.values():
            assert window.skip > 0, f"{window.key} must skip recent data"

    def test_skip_matches_the_specified_ratios(self):
        expected = {"mom_12_1": (250, 20), "mom_6_1": (125, 10), "mom_3_1": (60, 5)}
        actual = {k: (w.lookback, w.skip) for k, w in config.WINDOWS.items()}
        assert actual == expected

    def test_labels_do_not_single_out_the_twelve_month_window(self):
        """`12-1M` beside a bare `6M` reads as "only this one skips", which is false."""
        labels = [w.label for w in config.WINDOWS.values()]
        assert labels == ["12M", "6M", "3M"]
        for label in labels:
            assert "-1" not in label and "–1" not in label

    def test_each_window_excludes_its_own_recent_days(self):
        panel, _ = make_panel(n_days=520, n_syms=4)
        n = panel.returns.shape[0]
        for window in config.WINDOWS.values():
            sliced = panel.window(window.lookback, window.skip)
            assert sliced.shape[0] == window.lookback
            # The final row of the slice must be `skip` rows back from the newest bar.
            np.testing.assert_array_equal(sliced[-1], panel.returns[n - window.skip - 1])

    def test_skipping_changes_the_result(self):
        """A skip that made no difference would be a silent no-op."""
        panel, _ = make_panel(n_days=400, n_syms=4)
        for window in config.WINDOWS.values():
            with_skip = panel.window(window.lookback, window.skip)
            without = panel.window(window.lookback, 0)
            assert not np.array_equal(with_skip, without), f"{window.key} skip is inert"


class TestLiquidityGate:
    """Liquidity must come from traded history, not from a live quote field.

    The vendor's screener reports `volume` for the *current session* and resets it to
    zero once the session settles. The daily pipeline runs after the close, so gating on
    it discarded ~89% of the universe and failed the build every single night -
    deterministically, at exactly the hour the job runs.
    """

    def _row(self, symbol, volume, cap=1e10, price=100.0):
        return {
            "symbol": symbol, "companyName": f"{symbol} Inc.", "marketCap": cap,
            "price": price, "volume": volume, "sector": "Technology",
            "industry": "Software", "exchangeShortName": "NASDAQ", "country": "US",
            "isEtf": False, "isFund": False, "isActivelyTrading": True,
        }

    def test_zero_live_volume_does_not_empty_the_universe(self):
        rows = [self._row(f"Z{i:03d}", volume=0) for i in range(200)]
        report = universe.build_universe(rows)
        assert len(report.selected) == 200, (
            "a post-close screener reports zero volume; that means 'not trading now', "
            "not 'illiquid'"
        )
        assert "illiquid" not in report.excluded

    def test_missing_volume_is_also_tolerated(self):
        rows = [self._row("AAA", volume=None), self._row("BBB", volume=0.0)]
        report = universe.build_universe(rows)
        assert len(report.selected) == 2

    def test_a_genuinely_thin_name_is_still_dropped_when_volume_is_known(self):
        rows = [
            self._row("THIN", volume=100, price=1000.0),   # $100k/day - too thin
            self._row("DEEP", volume=1_000_000, price=100.0),
        ]
        report = universe.build_universe(rows)
        assert [c.symbol for c in report.selected] == ["DEEP"]
        assert report.excluded.get("illiquid") == 1
