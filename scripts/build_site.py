#!/usr/bin/env python3
"""Build the deployable static site.

Emits a directory the phone can load directly from any static host. The split matters:

  core.json    everything the four tabs need to render - rankings, macro, sectors.
  series.json  daily return series, needed only for per-ticker charts and watchlist
               HRP, so it is fetched in the background after first paint.

Nothing here talks to a market-data vendor; it reads the validated snapshot the daily
pipeline produced. The API key never reaches this stage, so it cannot leak into the
published output.

    python scripts/build_site.py -o site
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config, contract, snapshot as snap, store, validate  # noqa: E402
from app.returns import build_panel  # noqa: E402

#: Longest chart window plus the longest ranking skip.
RETURN_HISTORY = 320
RETURN_PRECISION = 5


def build_returns(symbols: list[str]) -> dict[str, object]:
    """Daily returns per symbol on one shared, dated trading calendar.

    Alignment is by *date*, not by array position. Slicing each symbol's own list to a
    common length silently pairs a recent listing's first week against an established
    name's last week, and truncating everyone to the shortest series lets one new IPO
    shorten the history behind every watchlist HRP.

    Missing observations are emitted as `null` rather than dropped or zero-filled, so the
    client can exclude them the same way the Python does.
    """
    price_map = store.load_prices(symbols)
    if not price_map:
        return {"dates": [], "symbols": [], "data": []}

    panel = build_panel(price_map, min_history=2)
    dates = panel.return_dates[-RETURN_HISTORY:]
    block = panel.returns[-RETURN_HISTORY:]

    rows: list[list[float | None]] = []
    for j, _symbol in enumerate(panel.symbols):
        column = block[:, j]
        rows.append([
            None if not math.isfinite(v) else round(float(v), RETURN_PRECISION)
            for v in column
        ])

    return {"dates": dates, "symbols": list(panel.symbols), "data": rows}


def build_core(data: dict) -> dict:
    """The payload every tab needs on first paint."""
    return {
        "meta": {
            "built_at": data["built_at"],
            "as_of": data.get("as_of"),
            "universe_size": data["universe_size"],
            "trading_days": data.get("trading_days"),
            "build_seconds": data.get("build_seconds"),
            "windows": data["windows"],
            "return_windows": data["return_windows"],
            "sectors": data["sectors"],
            "risk_model": data.get("risk_model", {}),
            "universe_report": data.get("universe_report", {}),
            "validation": data.get("validation", {}),
            "market_etf": config.MARKET_ETF,
            "schema_version": contract.SCHEMA_VERSION,
        },
        "stocks": data["stocks"],
        "macro": data["macro"],
        "portfolios": data.get("portfolios", []),
        "sector_exposure": data.get("sector_exposure", []),
        "clusters": data.get("clusters", {}),
        "hrp_universe": data.get("hrp_universe", {}),
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Biggie</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=5">
<meta name="theme-color" content="#101010">
<meta name="description" content="Risk-adjusted stock ranking and macro market intelligence">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Biggie">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23101010'/%3E%3Cg fill='%233987e5'%3E%3Crect x='12' y='34' width='8' height='18' rx='2'/%3E%3Crect x='24' y='25' width='8' height='27' rx='2'/%3E%3Crect x='36' y='17' width='8' height='35' rx='2'/%3E%3Crect x='48' y='11' width='8' height='41' rx='2'/%3E%3C/g%3E%3C/svg%3E">
<link rel="apple-touch-icon" href="icon.png">
<link rel="stylesheet" href="styles.css">
</head>
<body>
<div id="app">
  <section class="page" id="page-stocks" role="tabpanel" aria-label="Stocks">
    <header class="topbar">
      <h1>Stocks <small id="stocks-count"></small></h1>
      <div class="chips" id="window-chips" role="group" aria-label="Ranking window"></div>
      <p class="window-note" id="window-note"></p>
      <div class="chips" id="sector-chips" role="group" aria-label="Sector filter"></div>
      <input class="search" id="stock-search" type="search" inputmode="search"
             placeholder="Search ticker or company" autocomplete="off"
             autocorrect="off" spellcheck="false" aria-label="Search stocks">
    </header>
    <div class="list" id="stock-list"></div>
    <div class="sentinel" id="stock-sentinel"></div>
    <div class="loading" id="stock-loading" hidden>Loading&hellip;</div>
  </section>

  <section class="page" id="page-watchlist" role="tabpanel" aria-label="Watchlist" hidden>
    <header class="topbar"><h1>Watchlist <small id="watchlist-count"></small></h1></header>
    <div id="watchlist-body"></div>
  </section>

  <section class="page" id="page-macro" role="tabpanel" aria-label="Macro market" hidden>
    <header class="topbar">
      <h1>Macro Market <small id="macro-asof"></small></h1>
      <div class="chips" id="macro-view-chips" role="group" aria-label="Macro view"></div>
    </header>
    <div id="macro-body"></div>
  </section>

  <section class="page" id="page-settings" role="tabpanel" aria-label="Settings" hidden>
    <header class="topbar"><h1>Settings</h1></header>
    <div id="settings-body"></div>
  </section>

  <nav class="tabbar" role="tablist" aria-label="Sections">
    <button class="tab" role="tab" data-tab="stocks" aria-selected="true">
      <span class="glyph" aria-hidden="true">&#9638;</span><span>Stocks</span></button>
    <button class="tab" role="tab" data-tab="watchlist" aria-selected="false">
      <span class="glyph" aria-hidden="true">&#9733;</span><span>Watchlist</span></button>
    <button class="tab" role="tab" data-tab="macro" aria-selected="false">
      <span class="glyph" aria-hidden="true">&#9680;</span><span>Macro</span></button>
    <button class="tab" role="tab" data-tab="settings" aria-selected="false">
      <span class="glyph" aria-hidden="true">&#9881;</span><span>Settings</span></button>
  </nav>
</div>

<div class="sheet-backdrop" id="sheet-backdrop"></div>
<div class="sheet" id="sheet" role="dialog" aria-modal="true" aria-labelledby="sheet-sym">
  <div class="sheet-handle"></div>
  <div class="sheet-head">
    <div class="sheet-title">
      <h2 id="sheet-sym">&mdash;</h2>
      <p id="sheet-name"></p>
    </div>
    <button class="star" id="sheet-star" aria-label="Toggle watchlist">&#9734;</button>
    <button class="star" id="sheet-close" aria-label="Close">&#10005;</button>
  </div>
  <div class="sheet-body" id="sheet-body"></div>
</div>

<div class="toast" id="toast" role="status" aria-live="polite"></div>

<script src="storage.js"></script>
<script src="data-adapter.js"></script>
<script src="app.js"></script>
</body>
</html>
"""

MANIFEST = {
    "name": "Biggie — Stock Ranking & Macro",
    "short_name": "Biggie",
    "description": "Risk-adjusted stock ranking and macro market intelligence",
    "start_url": "./",
    "scope": "./",
    "display": "standalone",
    "orientation": "portrait",
    "background_color": "#101010",
    "theme_color": "#101010",
    "icons": [
        {
            "src": (
                "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
                "viewBox='0 0 192 192'%3E%3Crect width='192' height='192' rx='42' "
                "fill='%23101010'/%3E%3Cg fill='%233987e5'%3E%3Crect x='38' y='104' "
                "width='22' height='50' rx='4'/%3E%3Crect x='72' y='76' width='22' "
                "height='78' rx='4'/%3E%3Crect x='106' y='52' width='22' height='102' "
                "rx='4'/%3E%3Crect x='140' y='34' width='22' height='120' rx='4'/%3E"
                "%3C/g%3E%3C/svg%3E"
            ),
            "sizes": "192x192",
            "type": "image/svg+xml",
            "purpose": "any",
        }
    ],
}


def _escape_non_ascii(source: str) -> str:
    r"""Rewrite non-ASCII characters as JavaScript ``\uXXXX`` escapes.

    The single-file bundle may be served by a host whose skeleton owns the ``<head>``,
    so a ``<meta charset>`` written here would land too late to affect decoding. Keeping
    the output pure ASCII removes the dependency on the document charset entirely.
    """
    return "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in source)


def render_single_file(site: Path, core: dict, series: dict) -> str:
    """Inline the whole app into one HTML file, for hosts that serve a single page."""
    body = (site / "index.html").read_text(encoding="utf-8")
    # Strip the document scaffolding; the host supplies <head> and <body>.
    start = body.index("<div id=\"app\">")
    end = body.index("<script src=")
    markup = body[start:end]

    css = (site / "styles.css").read_text(encoding="utf-8")
    storage = _escape_non_ascii((site / "storage.js").read_text(encoding="utf-8"))
    adapter = _escape_non_ascii((site / "data-adapter.js").read_text(encoding="utf-8"))
    app = _escape_non_ascii((site / "app.js").read_text(encoding="utf-8"))

    def blob(payload: dict) -> str:
        # `</script>` inside JSON would close the tag early; escaping the slash is the
        # standard fix and stays valid JSON. json.dumps already escapes non-ASCII.
        return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    return f"""<meta charset="utf-8">
<title>Biggie Market Terminal</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=5">
<meta name="theme-color" content="#101010">
<style>
{css}
html, body {{ height: 100%; margin: 0; }}
</style>

{markup}
<script id="core-data" type="application/json">{blob(core)}</script>
<script id="series-data" type="application/json">{blob(series)}</script>
<script>
window.__BIGGIE_CORE__ = JSON.parse(document.getElementById('core-data').textContent);
window.__BIGGIE_SERIES__ = JSON.parse(document.getElementById('series-data').textContent);
</script>
<script>
{storage}
</script>
<script>
{adapter}
</script>
<script>
{app}
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deployable static site")
    parser.add_argument("-o", "--output", default="site")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--single-file",
        metavar="PATH",
        help="also emit a self-contained HTML bundle with the data inlined",
    )
    args = parser.parse_args()

    data = snap.load_snapshot()
    if data is None:
        print("No snapshot found. Run scripts/refresh.py first.", file=sys.stderr)
        return 1

    # Never publish a dataset that would not have passed the pipeline's own gate.
    if not args.skip_validation:
        result = validate.validate_snapshot(
            data, expected_size=data.get("requested_size") or config.UNIVERSE_SIZE
        )
        for message in result.warnings:
            print(f"  warning: {message}")
        if not result.ok:
            print("Refusing to build: snapshot failed validation", file=sys.stderr)
            for message in result.errors:
                print(f"  error: {message}", file=sys.stderr)
            return 1

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    frontend = config.FRONTEND_DIR
    (out / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    shutil.copy(frontend / "styles.css", out / "styles.css")
    shutil.copy(frontend / "app.js", out / "app.js")
    shutil.copy(frontend / "storage.js", out / "storage.js")
    # iOS ignores SVG for home-screen icons; without this PNG the shortcut is blank.
    shutil.copy(frontend / "icon.png", out / "icon.png")
    shutil.copy(frontend / "offline.js", out / "data-adapter.js")
    (out / "manifest.webmanifest").write_text(json.dumps(MANIFEST, indent=2))
    # Tell GitHub Pages not to run the output through Jekyll.
    (out / ".nojekyll").write_text("")

    data_dir = out / "data"
    data_dir.mkdir()

    core = build_core(data)

    # The contract check is the gate that a renamed field cannot slip past: Python tests,
    # `node --check` and JSON serialisation all stay happy while the client reads nothing.
    contract_result = contract.check_core(core)
    if not contract_result.ok:
        print("Refusing to build: core payload breaks the client contract", file=sys.stderr)
        for message in contract_result.errors:
            print(f"  error: {message}", file=sys.stderr)
        return 1

    core_path = data_dir / "core.json"
    core_path.write_text(json.dumps(core, separators=(",", ":")))

    stock_symbols = [s["symbol"] for s in data["stocks"]]
    macro_symbols = [a["symbol"] for a in data["macro"]["assets"]]
    wanted = list(dict.fromkeys(stock_symbols + macro_symbols + [config.MARKET_ETF]))
    series = build_returns(wanted)
    series_path = data_dir / "series.json"
    series_path.write_text(json.dumps(series, separators=(",", ":")))

    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "as_of": data.get("as_of"),
                "built_at": data["built_at"],
                "universe_size": data["universe_size"],
            },
            indent=2,
        )
    )

    if args.single_file:
        bundle = Path(args.single_file)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text(
            render_single_file(out, core, series), encoding="utf-8"
        )
        print(f"  bundle       {bundle} ({bundle.stat().st_size / 1e6:.2f} MB)")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Built {out}/")
    print(f"  as of        {data.get('as_of')}  ({data['universe_size']} names)")
    print(f"  core.json    {core_path.stat().st_size / 1e6:.2f} MB  (first paint)")
    print(f"  series.json  {series_path.stat().st_size / 1e6:.2f} MB  (lazy)")
    print(f"  total        {total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
