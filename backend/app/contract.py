"""The published data contract between the pipeline and the frontend.

A renamed field is invisible to every check the project had: Python tests pass because
the engine is self-consistent, `node --check` passes because the JavaScript is still
valid, and the site builds because JSON has no schema. The result was a deployable
artifact whose macro scores were all `undefined` - rendered as an em dash, sorted as if
every asset scored the same.

So the field names the client reads are declared here, checked when the site is built,
and asserted in the test suite. If the engine renames a field, the build fails loudly
instead of publishing a page full of blanks.

`frontend/offline.js` and `frontend/app.js` are the consumers. When a name changes here,
they change too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Bumped whenever the published payload changes shape incompatibly. Stamped into
#: core.json so a stale cached client can tell it is looking at an unfamiliar payload.
SCHEMA_VERSION = 3

#: Per-window, per-return-mode statistics the ranking list and detail sheet read.
STOCK_METRIC_FIELDS = frozenset({
    "ann_return", "mean_daily", "vol", "vol_idio",
    "score_total", "score_idio", "rank_total", "rank_idio",
    "marginal_risk", "risk_contribution",
})

#: Top-level per-stock fields.
STOCK_FIELDS = frozenset({
    "symbol", "name", "sector", "market_cap", "cap_rank",
    "beta", "sector_beta", "r_squared", "idio_vol", "trailing", "metrics",
})

#: Macro assets carry a single score - see PRODUCT_SPEC.md §13.
MACRO_ASSET_FIELDS = frozenset({
    "symbol", "label", "asset_class", "ann_return", "vol", "score",
    "rank", "beta", "trailing", "cluster",
})

MACRO_REGIME_FIELDS = frozenset({"state", "score", "narrative", "components"})

CORE_SECTIONS = frozenset({
    "meta", "stocks", "macro", "portfolios", "sector_exposure",
    "clusters", "hrp_universe",
})

META_FIELDS = frozenset({
    "built_at", "as_of", "universe_size", "windows", "return_windows",
    "sectors", "market_etf", "schema_version",
})


@dataclass
class ContractResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)


def _missing(required: frozenset[str], present: Any, label: str, result: ContractResult) -> None:
    if not isinstance(present, dict):
        result.fail(f"{label} is not an object")
        return
    gap = required - set(present)
    if gap:
        result.fail(f"{label} is missing {sorted(gap)}")


def check_core(core: dict[str, Any]) -> ContractResult:
    """Verify a built core payload carries every field the client reads.

    Checks presence *and* that the headline numbers are actually populated - a field
    that exists but is null everywhere renders exactly as a missing field does.
    """
    result = ContractResult()

    gap = CORE_SECTIONS - set(core)
    if gap:
        result.fail(f"core payload is missing sections {sorted(gap)}")
        return result

    _missing(META_FIELDS, core.get("meta"), "meta", result)

    stocks = core.get("stocks") or []
    if not stocks:
        result.fail("core payload has no stocks")
    else:
        _missing(STOCK_FIELDS, stocks[0], "stocks[0]", result)
        metrics = stocks[0].get("metrics") or {}
        if not metrics:
            result.fail("stocks[0].metrics is empty")
        else:
            key = sorted(metrics)[0]
            _missing(STOCK_METRIC_FIELDS, metrics[key], f"stocks[0].metrics[{key}]", result)

        # A populated-but-null field is indistinguishable from a missing one on screen.
        scored = sum(
            1 for s in stocks
            if (s.get("metrics", {}).get("mom_12_1|raw") or {}).get("score_total") is not None
        )
        if scored < len(stocks) * 0.9:
            result.fail(
                f"only {scored}/{len(stocks)} stocks have a populated score_total"
            )

    macro = core.get("macro") or {}
    assets = macro.get("assets") or []
    if not assets:
        result.fail("core payload has no macro assets")
    else:
        _missing(MACRO_ASSET_FIELDS, assets[0], "macro.assets[0]", result)
        scored = sum(1 for a in assets if a.get("score") is not None)
        if scored < len(assets):
            result.fail(
                f"only {scored}/{len(assets)} macro assets have a populated score"
            )

    _missing(MACRO_REGIME_FIELDS, macro.get("regime"), "macro.regime", result)
    return result
