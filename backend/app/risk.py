"""Risk model: shrinkage covariance, market beta, factor regression and residual risk.

Two ideas do the work here.

1. **Shrinkage.** With ~250 observations and ~1000 names the sample covariance matrix is
   singular and its extreme eigenvalues are noise. Ledoit-Wolf pulls it toward a scaled
   identity, which is what makes per-name risk estimates and HRP stable.

2. **Factor decomposition.** Regressing each stock on the market and its sector ETF
   splits total risk into a factor part and an idiosyncratic part, so ranking can be run
   on alpha rather than on beta exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.covariance import LedoitWolf

from . import config
from .returns import ReturnPanel


# --------------------------------------------------------------------------------------
# Covariance
# --------------------------------------------------------------------------------------
@dataclass
class CovarianceModel:
    """A covariance estimate over one window, plus the quantities derived from it."""

    symbols: list[str]
    cov: np.ndarray            # (N, N) daily covariance
    shrinkage: float           # Ledoit-Wolf intensity actually applied
    observations: int

    @property
    def variance(self) -> np.ndarray:
        return np.clip(np.diag(self.cov), 1e-12, None)

    @property
    def daily_vol(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def annual_vol(self) -> np.ndarray:
        """sqrt(eᵢᵀ Σ eᵢ) annualised - the covariance-adjusted standalone volatility."""
        return self.daily_vol * np.sqrt(config.TRADING_DAYS)

    def correlation(self) -> np.ndarray:
        d = self.daily_vol
        corr = self.cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def portfolio_vol(self, weights: np.ndarray) -> float:
        """sqrt(wᵀ Σ w), annualised."""
        w = np.asarray(weights, dtype=float)
        return float(np.sqrt(max(w @ self.cov @ w, 0.0)) * np.sqrt(config.TRADING_DAYS))

    def marginal_risk(self, weights: np.ndarray) -> np.ndarray:
        """Each name's marginal contribution to portfolio risk, ∂σ_p/∂wᵢ (annualised).

        This is the portfolio-aware view: a name that is volatile but uncorrelated with
        the book contributes far less risk than its standalone volatility suggests.
        """
        w = np.asarray(weights, dtype=float)
        port_var = float(w @ self.cov @ w)
        if port_var <= 0:
            return np.zeros(len(w))
        return (self.cov @ w) / np.sqrt(port_var) * np.sqrt(config.TRADING_DAYS)

    def risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """Fraction of total portfolio variance attributable to each name (sums to 1)."""
        w = np.asarray(weights, dtype=float)
        port_var = float(w @ self.cov @ w)
        if port_var <= 0:
            return np.zeros(len(w))
        return w * (self.cov @ w) / port_var


def estimate_covariance(
    window: np.ndarray,
    symbols: Sequence[str],
    min_obs: int = config.COV_MIN_OBS,
) -> CovarianceModel:
    """Ledoit-Wolf shrunk covariance over a return window.

    Missing observations are mean-imputed per column before estimation - dropping whole
    rows would discard most of the panel when even a few names have gaps.
    """
    symbols = list(symbols)
    if window.size == 0 or window.shape[0] < 2:
        n = len(symbols)
        return CovarianceModel(symbols, np.eye(n) * 1e-6, 0.0, 0)

    x = np.asarray(window, dtype=float)
    col_mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    filled = np.where(np.isfinite(x), x, col_mean)

    # A column with almost no real data carries no information; force it to its own
    # variance only, so it cannot invent correlations.
    sparse = np.isfinite(x).sum(axis=0) < min_obs

    try:
        lw = LedoitWolf(assume_centered=False).fit(filled)
        cov = np.asarray(lw.covariance_, dtype=float)
        shrinkage = float(lw.shrinkage_)
    except (ValueError, np.linalg.LinAlgError):
        cov = np.cov(filled, rowvar=False)
        cov = np.atleast_2d(cov)
        shrinkage = 0.0

    if sparse.any():
        idx = np.where(sparse)[0]
        diag = np.diag(cov).copy()
        cov[idx, :] = 0.0
        cov[:, idx] = 0.0
        cov[idx, idx] = np.maximum(diag[idx], 1e-10)

    cov = _make_psd(cov)
    return CovarianceModel(symbols, cov, shrinkage, int(x.shape[0]))


def _make_psd(cov: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Symmetrise and clip negative eigenvalues so the matrix is usable everywhere."""
    cov = (cov + cov.T) / 2.0
    diag = np.diag(cov)
    if np.all(np.isfinite(cov)) and np.all(diag > 0):
        try:
            np.linalg.cholesky(cov + np.eye(len(cov)) * floor)
            return cov
        except np.linalg.LinAlgError:
            pass
    vals, vecs = np.linalg.eigh(np.nan_to_num(cov))
    vals = np.clip(vals, floor, None)
    return (vecs * vals) @ vecs.T


# --------------------------------------------------------------------------------------
# Beta and residual returns
# --------------------------------------------------------------------------------------
@dataclass
class FactorModel:
    """Per-stock exposures to the market and (optionally) sector factors.

    ``beta`` is the plain univariate regression on the market, which is what the residual
    toggle and the UI mean by "beta". The sector loading comes from a separate joint
    regression; the two are reported side by side rather than mixed, because the market
    and its sector ETFs are ~0.9 correlated and a joint market coefficient is a *partial*
    beta that routinely goes negative and reads as nonsense.
    """

    symbols: list[str]
    beta: np.ndarray                 # univariate market beta vs VTI
    alpha: np.ndarray                # daily intercept of the univariate fit
    r_squared: np.ndarray            # R² of the joint (market + sector) fit
    sector_beta: np.ndarray          # exposure to the stock's own sector ETF
    residuals: np.ndarray            # (T, N) joint-model residual series
    idio_vol: np.ndarray             # annualised idiosyncratic volatility
    factor_names: list[str]


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares of ``y`` on ``x`` with an intercept. Returns (coefs, resid, R²)."""
    design = np.column_stack([np.ones(len(x)), x])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    resid = y - fitted
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0
    return coefs, resid, r2


def estimate_factor_model(
    panel: ReturnPanel,
    sectors: dict[str, str],
    *,
    use_sector_factors: bool = True,
) -> FactorModel:
    """Regress every stock on the market factor and its own sector ETF.

    Using only the stock's own sector - rather than all eleven - keeps the regression
    well conditioned on a two-year window and avoids the collinearity that makes
    multi-sector loadings uninterpretable.
    """
    market_col = panel.col(config.MARKET_ETF) if panel.has(config.MARKET_ETF) else None
    n_obs, n_sym = panel.returns.shape

    beta = np.full(n_sym, np.nan)
    alpha = np.full(n_sym, np.nan)
    r2 = np.full(n_sym, np.nan)
    sector_beta = np.full(n_sym, np.nan)
    residuals = np.full((n_obs, n_sym), np.nan)
    idio_vol = np.full(n_sym, np.nan)

    if market_col is None:
        return FactorModel(
            list(panel.symbols), beta, alpha, r2, sector_beta, residuals, idio_vol, []
        )

    market = panel.returns[:, market_col]

    for j, symbol in enumerate(panel.symbols):
        y_raw = panel.returns[:, j]

        # 1. Univariate market beta - the number the UI shows and the residual toggle uses.
        uni_mask = np.isfinite(y_raw) & np.isfinite(market)
        if uni_mask.sum() < config.COV_MIN_OBS:
            continue
        uni_coefs, _uni_resid, _ = _ols(y_raw[uni_mask], market[uni_mask, None])
        alpha[j] = uni_coefs[0]
        beta[j] = uni_coefs[1]

        # 2. Joint market + own-sector regression, for idiosyncratic risk.
        factors = [market]
        sector_etf = config.SECTOR_TO_ETF.get(sectors.get(symbol, ""), "")
        if use_sector_factors and sector_etf and panel.has(sector_etf) and sector_etf != symbol:
            factors.append(panel.returns[:, panel.col(sector_etf)])

        stacked = np.column_stack(factors)
        mask = np.isfinite(y_raw) & np.all(np.isfinite(stacked), axis=1)
        if mask.sum() < config.COV_MIN_OBS:
            continue

        coefs, resid, r_sq = _ols(y_raw[mask], stacked[mask])
        if len(coefs) > 2:
            sector_beta[j] = coefs[2]
        r2[j] = r_sq
        residuals[mask, j] = resid
        idio_vol[j] = float(np.std(resid, ddof=1)) * np.sqrt(config.TRADING_DAYS)

    return FactorModel(
        list(panel.symbols), beta, alpha, r2, sector_beta, residuals, idio_vol,
        [config.MARKET_ETF] + (["sector"] if use_sector_factors else []),
    )


def market_residual_returns(
    panel: ReturnPanel,
    betas: dict[str, float],
    market_panel: ReturnPanel | None = None,
) -> np.ndarray:
    """Residual return = stock return − β × market return.

    This is the spec's plain residual definition, kept separate from the full factor
    regression so the UI toggle stays simple and explainable.

    ``market_panel`` supplies the market series when ``panel`` itself holds only stocks
    (the usual case - the reference ETFs live outside the ranked universe). Both panels
    must share a trading calendar, which ``ReturnPanel.subset`` guarantees.
    """
    source = market_panel if market_panel is not None else panel
    if not source.has(config.MARKET_ETF):
        return panel.returns.copy()

    market = source.returns[:, source.col(config.MARKET_ETF)]
    if market.shape[0] != panel.returns.shape[0]:
        raise ValueError("market panel and target panel are on different calendars")

    b = np.array([betas.get(s, 1.0) for s in panel.symbols], dtype=float)
    b = np.where(np.isfinite(b), b, 1.0)
    return panel.returns - np.outer(market, b)
