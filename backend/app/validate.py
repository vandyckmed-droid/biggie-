"""Dataset validation.

A daily pipeline that publishes whatever it produced is worse than one that publishes
nothing: a silently truncated universe or a wall of nulls looks like a working app
showing wrong numbers. Every build is checked here before it is allowed to replace the
live dataset, and a failed check leaves yesterday's good data in place.

Checks are deliberately coarse. They catch the failures that actually happen - a vendor
outage returning empty history, a screener change gutting the universe, a stale date -
without being so tight that an unusual but real market breaks the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from . import config


@dataclass
class ValidationResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "All checks passed."
        parts = [f"{len(self.errors)} error(s)", f"{len(self.warnings)} warning(s)"]
        return ", ".join(parts)


#: A dataset older than this is stale enough to be worth failing on. Generous because
#: of weekends and market holidays - four calendar days covers a long weekend.
MAX_STALE_DAYS = 5

#: Fractions of the universe that must carry usable values.
MIN_SCORE_COVERAGE = 0.90
MIN_BETA_COVERAGE = 0.90

#: A universe this far below target means the screener or the history gate broke.
MIN_UNIVERSE_FRACTION = 0.80


def validate_snapshot(
    snapshot: dict[str, Any],
    *,
    expected_size: int = config.UNIVERSE_SIZE,
    today: date | None = None,
) -> ValidationResult:
    """Check a freshly built snapshot is fit to publish."""
    result = ValidationResult()
    today = today or date.today()

    stocks = snapshot.get("stocks") or []
    result.stats["universe_size"] = len(stocks)

    # ---- structure
    if not stocks:
        result.fail("snapshot contains no stocks")
        return result

    floor = int(expected_size * MIN_UNIVERSE_FRACTION)
    if len(stocks) < floor:
        result.fail(f"universe has {len(stocks)} names, expected at least {floor}")
    elif len(stocks) < expected_size:
        result.warn(f"universe has {len(stocks)} names, short of {expected_size}")

    # ---- freshness
    as_of = snapshot.get("as_of")
    result.stats["as_of"] = as_of
    if not as_of:
        result.fail("snapshot has no as_of date")
    else:
        try:
            as_of_date = date.fromisoformat(as_of)
        except ValueError:
            result.fail(f"as_of is not a date: {as_of!r}")
        else:
            age = (today - as_of_date).days
            result.stats["age_days"] = age
            if age > MAX_STALE_DAYS:
                result.fail(f"data is {age} days old (limit {MAX_STALE_DAYS})")
            elif age < 0:
                result.fail(f"as_of {as_of} is in the future")

    trading_days = snapshot.get("trading_days") or 0
    result.stats["trading_days"] = trading_days
    if trading_days < config.MIN_HISTORY_DAYS:
        result.fail(
            f"only {trading_days} trading days of history, need {config.MIN_HISTORY_DAYS}"
        )

    # ---- value coverage
    window = config.DEFAULT_WINDOW
    key = f"{window}|raw"
    scored = sum(
        1 for s in stocks
        if (s.get("metrics", {}).get(key, {}) or {}).get("score_total") is not None
    )
    coverage = scored / len(stocks)
    result.stats["score_coverage"] = round(coverage, 4)
    if coverage < MIN_SCORE_COVERAGE:
        result.fail(
            f"only {coverage:.1%} of names have a {window} score "
            f"(need {MIN_SCORE_COVERAGE:.0%})"
        )

    betas = [s.get("beta") for s in stocks if s.get("beta") is not None]
    beta_coverage = len(betas) / len(stocks)
    result.stats["beta_coverage"] = round(beta_coverage, 4)
    if beta_coverage < MIN_BETA_COVERAGE:
        result.fail(f"only {beta_coverage:.1%} of names have a beta")

    if betas:
        median_beta = sorted(betas)[len(betas) // 2]
        result.stats["median_beta"] = round(median_beta, 3)
        # Long equity against a total-market index should sit near 1. A median outside
        # this band means the regression is wired wrong, which has happened before.
        if not 0.4 <= median_beta <= 1.8:
            result.fail(f"median beta {median_beta:.2f} is implausible for long equity")

    # ---- sectors
    sectors = {s.get("sector") for s in stocks}
    result.stats["sectors"] = len(sectors)
    if len(sectors) < 5:
        result.fail(f"only {len(sectors)} distinct sectors present")
    unclassified = sum(1 for s in stocks if s.get("sector") == config.UNKNOWN_SECTOR)
    if unclassified > len(stocks) * 0.15:
        result.warn(f"{unclassified} names are unclassified")

    # ---- macro
    macro = snapshot.get("macro") or {}
    assets = macro.get("assets") or []
    result.stats["macro_assets"] = len(assets)
    if len(assets) < len(config.MACRO_INSTRUMENTS) * 0.7:
        result.fail(f"only {len(assets)} macro instruments present")

    regime = (macro.get("regime") or {}).get("state")
    result.stats["regime"] = regime
    if regime not in {"Risk-On", "Risk-Off", "Transition"}:
        result.fail(f"regime state is invalid: {regime!r}")

    # ---- portfolios
    books = snapshot.get("portfolios") or []
    result.stats["portfolios"] = len(books)
    if not any(b.get("key") == "UNIVERSE" for b in books):
        result.fail("universe composite portfolio is missing")

    return result


def is_newer(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    """Is ``candidate`` at least as fresh as the dataset already published?

    Guards against a vendor replaying older history and silently rolling the app back.
    """
    if incumbent is None:
        return True
    new, old = candidate.get("as_of"), incumbent.get("as_of")
    if not old:
        return True
    if not new:
        return False
    return new >= old
