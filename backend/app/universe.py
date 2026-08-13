"""Universe construction: eligibility, exclusions, deduplication and top-N selection.

The goal is a clean set of plain common equities. Everything with embedded optionality,
a fixed-income claim, or a fund wrapper is removed, because those instruments do not
behave like equity and would corrupt both the ranking and the covariance model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import config

# --------------------------------------------------------------------------------------
# Symbol-shape rules
# --------------------------------------------------------------------------------------
# FMP encodes non-common share classes as a hyphen suffix. Empirically:
#   preferred  -> BAC-PB, FITB-PA, OAK-PA   (suffix starts with P + a letter)
#   warrant    -> XYZ-WT, XYZ-WS, XYZ.WS
#   right      -> XYZ-RT, XYZ-R
#   unit       -> XYZ-UN, XYZ-U, XYZ.U
#   class      -> BRK-A, BF-B, MOG-A, CRD-A  (bare single letter -> a real common share)
# The bare-letter case must be preserved; collapsing it would drop Berkshire.
_PREFERRED_SUFFIX = re.compile(r"^P[A-Z]?$")
_WARRANT_SUFFIX = re.compile(r"^W[STI]?$")
_RIGHT_SUFFIX = re.compile(r"^R(T|TS)?$")
_UNIT_SUFFIX = re.compile(r"^U(N|NIT)?$")

_SUFFIX_SPLIT = re.compile(r"[-.]")

# --------------------------------------------------------------------------------------
# Name-shape rules
# --------------------------------------------------------------------------------------
#: Fixed-income claims dressed up as exchange-listed equity ("baby bonds"), plus the
#: preferred/trust-preferred family. Matched case-insensitively against the issuer name.
_DEBT_PATTERNS = [
    r"\bnotes?\b", r"\bdebenture", r"\bsubordinated\b", r"\bbaby bond",
    r"\bsenior\b.*\bdue\b", r"\bdue \d{4}", r"\d(\.\d+)?\s*%",
    r"\bcapital trust\b", r"\btrust preferred\b", r"\bfixed[- ]rate\b",
    r"\bfixed[/ ]floating\b", r"\bcumulative\b", r"\bperpetual\b",
]
_PREFERRED_PATTERNS = [
    r"\bpreferred\b", r"\bpfd\b", r"\bdepositary (share|receipt)s? .*preferred",
    r"\bpref\b", r"\bseries [a-z]\b.*\bpreferred",
]
_DERIVATIVE_PATTERNS = [
    r"\bwarrants?\b", r"\brights?\b(?! ?management)", r"\bunits?\b",
    r"\bsubscription rights\b",
]
_FUND_PATTERNS = [
    r"\betn\b", r"\bexchange[- ]traded note", r"\bstructured note",
    r"\bclosed[- ]end\b", r"\bclosed end fund", r"\bincome fund\b",
    r"\bmunicipal .*fund\b", r"\bterm trust\b", r"\bstrategies fund\b",
    r"\bopportunities fund\b", r"\bindex trust\b", r"\bunit investment trust\b",
]

_COMPILED = {
    "baby_bond": [re.compile(p, re.I) for p in _DEBT_PATTERNS],
    "preferred": [re.compile(p, re.I) for p in _PREFERRED_PATTERNS],
    "derivative": [re.compile(p, re.I) for p in _DERIVATIVE_PATTERNS],
    "fund": [re.compile(p, re.I) for p in _FUND_PATTERNS],
}

#: Suffixes stripped when collapsing share classes of the same issuer.
_NAME_NOISE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|plc|ltd|limited|lp|llc|llp|"
    r"holdings?|group|sa|s\.a|nv|n\.v|ag|ab|se|the|class [a-z]|cl [a-z]|"
    r"common stock|ordinary shares?|adr|ads|american depositary shares?)\b",
    re.I,
)


def _normalize_name(name: str) -> str:
    """Collapse an issuer name to a comparison key so share classes group together."""
    cleaned = _NAME_NOISE.sub(" ", (name or "").lower())
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# --------------------------------------------------------------------------------------
@dataclass
class Candidate:
    symbol: str
    name: str
    sector: str
    industry: str
    exchange: str
    country: str
    market_cap: float
    price: float
    volume: float

    @property
    def dollar_volume(self) -> float:
        return self.price * self.volume


@dataclass
class UniverseReport:
    """Audit trail for how the universe was narrowed - surfaced in the UI."""

    screened: int = 0
    selected: list[Candidate] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)
    deduped: int = 0

    def drop(self, reason: str, n: int = 1) -> None:
        self.excluded[reason] = self.excluded.get(reason, 0) + n


# --------------------------------------------------------------------------------------
def exclusion_reason(symbol: str, name: str, industry: str = "") -> str | None:
    """Return why this instrument is structurally unsuitable, or ``None`` if it is fine."""
    sym = (symbol or "").upper()
    if not sym:
        return "bad_symbol"

    parts = _SUFFIX_SPLIT.split(sym)
    if len(parts) > 1:
        suffix = parts[-1]
        if _PREFERRED_SUFFIX.match(suffix):
            return "preferred"
        if _WARRANT_SUFFIX.match(suffix):
            return "warrant"
        if _RIGHT_SUFFIX.match(suffix):
            return "right"
        if _UNIT_SUFFIX.match(suffix):
            return "unit"
        if len(suffix) > 2:
            # Anything longer than a share-class letter pair is not a common share.
            return "non_common"

    haystack = f"{name or ''} {industry or ''}"
    for reason, patterns in _COMPILED.items():
        if any(p.search(haystack) for p in patterns):
            # A trailing "% ..." in a name is the single strongest baby-bond tell, but
            # map each family to the label the spec uses.
            return {
                "baby_bond": "baby_bond",
                "preferred": "preferred",
                "derivative": "warrant_right_unit",
                "fund": "fund_or_etn",
            }[reason]
    return None


def _eligible(row: dict[str, Any]) -> str | None:
    """Liquidity / listing gates. Returns a drop reason or ``None``."""
    exchange = (row.get("exchangeShortName") or row.get("exchange") or "").upper()
    if exchange not in config.ELIGIBLE_EXCHANGES:
        return "otc_or_foreign_venue"
    if row.get("isEtf") or row.get("isFund"):
        return "fund_or_etn"
    if row.get("isActivelyTrading") is False:
        return "not_trading"

    cap = row.get("marketCap") or 0.0
    if cap < config.MIN_MARKET_CAP:
        return "below_min_market_cap"
    price = row.get("price") or 0.0
    if price < config.MIN_PRICE:
        return "below_min_price"

    # Liquidity is NOT judged here. The screener's `volume` is a *live session* figure
    # that the vendor resets to zero once the session settles - after the close it reads
    # 0 for ~89% of the universe. Gating on it made the nightly pipeline discard almost
    # everything and fail, deterministically, at exactly the hour it runs.
    #
    # A zero here means "not trading right now", not "illiquid". Real liquidity is
    # measured from cached daily history in `filter_by_liquidity` once prices exist.
    volume = row.get("volume") or 0.0
    if volume > 0 and price * volume < config.MIN_AVG_DOLLAR_VOLUME:
        return "illiquid"
    return None


def build_universe(
    screener_rows: Sequence[dict[str, Any]],
    size: int = config.UNIVERSE_SIZE,
) -> UniverseReport:
    """Filter, deduplicate and take the largest ``size`` names by market capitalisation."""
    report = UniverseReport(screened=len(screener_rows))
    kept: list[Candidate] = []

    for row in screener_rows:
        symbol = (row.get("symbol") or "").upper()
        name = row.get("companyName") or ""

        reason = _eligible(row)
        if reason:
            report.drop(reason)
            continue

        reason = exclusion_reason(symbol, name, row.get("industry") or "")
        if reason:
            report.drop(reason)
            continue

        raw_sector = (row.get("sector") or "").strip()
        kept.append(
            Candidate(
                symbol=symbol,
                name=name,
                sector=config.FMP_SECTOR_TO_GICS.get(raw_sector, config.UNKNOWN_SECTOR),
                industry=row.get("industry") or "",
                exchange=(row.get("exchangeShortName") or "").upper(),
                country=row.get("country") or "",
                market_cap=float(row.get("marketCap") or 0.0),
                price=float(row.get("price") or 0.0),
                volume=float(row.get("volume") or 0.0),
            )
        )

    deduped = _dedupe(kept, report)
    deduped.sort(key=lambda c: c.market_cap, reverse=True)
    report.selected = deduped[:size]
    return report


#: Ticker roots must overlap by at least this many characters to be the same issuer.
_ROOT_PREFIX_MATCH = 3


def _ticker_root(symbol: str) -> str:
    """The part of a ticker before any share-class suffix (``BRK-A`` -> ``BRK``)."""
    return _SUFFIX_SPLIT.split(symbol.upper())[0]


def _same_issuer_root(a: str, b: str) -> bool:
    """Do two ticker roots plausibly belong to one issuer?

    ``GOOG``/``GOOGL`` and ``DISCA``/``DISCK`` differ only in a trailing class letter, so
    a shared prefix is the right test - fixed-length stripping breaks on ``GOOG``.
    """
    shortest = min(len(a), len(b))
    if shortest == 0:
        return False
    shared = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        shared += 1
    # Allow at most one trailing class character to differ on each side.
    return shared >= min(_ROOT_PREFIX_MATCH, shortest) and (
        len(a) - shared <= 1 and len(b) - shared <= 1
    )


def _better(a: Candidate, b: Candidate) -> bool:
    """Is ``a`` the more tradeable listing of the two?

    Dollar volume first - that is the line an actual order would go to. Market cap breaks
    ties, since FMP frequently reports the same total cap for every class.
    """
    return (a.dollar_volume, a.market_cap, -len(a.symbol)) > (
        b.dollar_volume, b.market_cap, -len(b.symbol)
    )


def _dedupe(candidates: Iterable[Candidate], report: UniverseReport) -> list[Candidate]:
    """Collapse duplicate listings of one issuer, keeping the most liquid share class.

    Dual-class issuers (GOOG/GOOGL, BRK-A/BRK-B) would otherwise occupy two slots and
    double-count one company's risk in the covariance matrix. Both the normalised issuer
    name *and* the ticker root must agree before two rows are merged, so unrelated
    companies with similar names are never silently dropped.
    """
    buckets: dict[str, list[Candidate]] = {}
    seen_symbols: set[str] = set()

    for cand in candidates:
        if cand.symbol in seen_symbols:
            report.drop("duplicate_symbol")
            continue
        seen_symbols.add(cand.symbol)
        buckets.setdefault(_normalize_name(cand.name) or cand.symbol, []).append(cand)

    survivors: list[Candidate] = []
    for group in buckets.values():
        # Within one issuer name, cluster listings whose ticker roots agree; each cluster
        # contributes exactly one line.
        clusters: list[list[Candidate]] = []
        for cand in group:
            root = _ticker_root(cand.symbol)
            for cluster in clusters:
                if _same_issuer_root(root, _ticker_root(cluster[0].symbol)):
                    cluster.append(cand)
                    break
            else:
                clusters.append([cand])

        for cluster in clusters:
            best = cluster[0]
            for other in cluster[1:]:
                if _better(other, best):
                    best = other
            report.deduped += len(cluster) - 1
            survivors.append(best)

    return survivors
