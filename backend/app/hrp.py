"""Hierarchical Risk Parity and correlation-distance clustering.

HRP (Lopez de Prado) allocates without inverting the covariance matrix, which is exactly
the property needed here: with ~250 observations and up to 1000 names, mean-variance
weights would be dominated by estimation error. The tree is built on the correlation
distance d(i,j) = sqrt(0.5 * (1 - ρ)), risk is split recursively down that tree, and the
result is a set of weights that respects correlation structure without ever solving an
ill-conditioned system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from . import config
from .risk import CovarianceModel


# --------------------------------------------------------------------------------------
def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """d(i,j) = sqrt(0.5 * (1 - ρᵢⱼ)) - a proper metric on correlations."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


def _linkage(distance: np.ndarray, method: str = "single") -> np.ndarray:
    condensed = squareform(distance, checks=False)
    return linkage(condensed, method=method)


def _inverse_variance_weights(cov: np.ndarray, idx: np.ndarray) -> np.ndarray:
    var = np.clip(np.diag(cov)[idx], 1e-12, None)
    inv = 1.0 / var
    return inv / inv.sum()


def _cluster_variance(cov: np.ndarray, idx: np.ndarray) -> float:
    """Variance of the inverse-variance-weighted portfolio of one cluster."""
    w = _inverse_variance_weights(cov, idx)
    sub = cov[np.ix_(idx, idx)]
    return float(w @ sub @ w)


def _bisect(order: list[int]) -> list[tuple[list[int], list[int]]]:
    """Recursively split an ordered list in half, yielding (left, right) pairs."""
    out: list[tuple[list[int], list[int]]] = []
    stack = [order]
    while stack:
        group = stack.pop()
        if len(group) <= 1:
            continue
        mid = len(group) // 2
        left, right = group[:mid], group[mid:]
        out.append((left, right))
        stack.extend([left, right])
    return out


@dataclass
class HRPResult:
    symbols: list[str]
    weights: np.ndarray
    order: list[int]              # quasi-diagonalised ordering
    linkage_matrix: np.ndarray
    portfolio_vol: float
    effective_n: float            # 1 / sum(w²): how many names the book really holds
    clusters: list[int]           # cluster id per symbol at the configured k

    def as_dict(self) -> dict[str, float]:
        return {s: float(w) for s, w in zip(self.symbols, self.weights)}


def hierarchical_risk_parity(
    model: CovarianceModel,
    *,
    k: int = config.DEFAULT_CLUSTER_K,
    linkage_method: str = "single",
) -> HRPResult:
    """Allocate weights by recursive bisection of the correlation tree."""
    symbols = list(model.symbols)
    n = len(symbols)
    if n == 0:
        return HRPResult([], np.zeros(0), [], np.zeros((0, 4)), 0.0, 0.0, [])
    if n == 1:
        return HRPResult(symbols, np.ones(1), [0], np.zeros((0, 4)),
                         model.portfolio_vol(np.ones(1)), 1.0, [1])

    cov = model.cov
    corr = model.correlation()
    dist = correlation_distance(corr)
    link = _linkage(dist, linkage_method)

    # Quasi-diagonalisation: reorder so similar assets sit adjacent.
    order = [int(i) for i in leaves_list(link)]

    weights = np.ones(n)
    for left, right in _bisect(order):
        l_idx, r_idx = np.array(left), np.array(right)
        var_l = _cluster_variance(cov, l_idx)
        var_r = _cluster_variance(cov, r_idx)
        total = var_l + var_r
        # Allocate inversely to cluster variance - the riskier side gets less.
        alpha = 1.0 - var_l / total if total > 0 else 0.5
        weights[l_idx] *= alpha
        weights[r_idx] *= 1.0 - alpha

    total = weights.sum()
    weights = weights / total if total > 0 else np.full(n, 1.0 / n)

    k_eff = int(np.clip(k, 1, max(1, n)))
    clusters = fcluster(link, t=k_eff, criterion="maxclust").astype(int).tolist()

    return HRPResult(
        symbols=symbols,
        weights=weights,
        order=order,
        linkage_matrix=link,
        portfolio_vol=model.portfolio_vol(weights),
        effective_n=float(1.0 / np.sum(weights**2)) if np.any(weights) else 0.0,
        clusters=clusters,
    )


# --------------------------------------------------------------------------------------
# Medoid clustering
# --------------------------------------------------------------------------------------
@dataclass
class MedoidResult:
    labels: list[int]
    medoids: list[str]
    symbols: list[str]
    inertia: float

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {m: [] for m in self.medoids}
        for sym, label in zip(self.symbols, self.labels):
            if 0 <= label < len(self.medoids):
                out[self.medoids[label]].append(sym)
        return out


def k_medoids(
    distance: np.ndarray,
    symbols: list[str],
    k: int,
    *,
    max_iter: int = 100,
    seed: int = 0,
) -> MedoidResult:
    """Partition around medoids on a precomputed distance matrix.

    Medoids are used rather than centroids because a medoid is an actual ticker - "this
    cluster is the AAPL cluster" is interpretable in a way a synthetic centroid is not.
    Initialisation is deterministic (k-means++ style with a fixed seed) so a refresh
    reproduces the same clusters given the same data.
    """
    n = len(symbols)
    k = int(np.clip(k, 1, max(1, n)))
    if n == 0:
        return MedoidResult([], [], [], 0.0)
    if k >= n:
        return MedoidResult(list(range(n)), list(symbols), list(symbols), 0.0)

    rng = np.random.default_rng(seed)

    # Deterministic k-means++ seeding: start from the most central point, then greedily
    # take the point furthest from everything already chosen.
    medoids = [int(np.argmin(distance.sum(axis=1)))]
    while len(medoids) < k:
        nearest = distance[:, medoids].min(axis=1)
        nearest[medoids] = -1.0
        best = float(nearest.max())
        ties = np.where(nearest >= best - 1e-12)[0]
        medoids.append(int(ties[0] if len(ties) == 1 else rng.choice(ties)))

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(distance[:, medoids], axis=1)
        moved = False
        for c in range(k):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            sub = distance[np.ix_(members, members)]
            candidate = int(members[int(np.argmin(sub.sum(axis=1)))])
            if candidate != medoids[c]:
                medoids[c] = candidate
                moved = True
        if not moved:
            break

    labels = np.argmin(distance[:, medoids], axis=1)
    inertia = float(distance[np.arange(n), np.array(medoids)[labels]].sum())
    return MedoidResult(
        labels=[int(x) for x in labels],
        medoids=[symbols[i] for i in medoids],
        symbols=list(symbols),
        inertia=inertia,
    )


def cluster_universe(model: CovarianceModel, k: int) -> MedoidResult:
    """Medoid clustering of a covariance model using correlation distance."""
    corr = model.correlation()
    dist = correlation_distance(corr)
    return k_medoids(dist, list(model.symbols), k)
