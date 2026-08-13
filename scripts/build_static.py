#!/usr/bin/env python3
"""Pack the app into one self-contained HTML file.

The result has no backend: the snapshot plus a daily-return series per symbol is
embedded in the page, and `frontend/offline.js` answers the same routes the live API
does. That makes the app shareable as a single URL, at the cost of a fixed snapshot -
refreshing market data still requires running the real thing.

    python scripts/build_static.py -o dist/biggie.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config, snapshot as snap, store  # noqa: E402

#: Longest chart window plus the longest ranking window's skip, so every view the UI
#: can render has the observations it needs without shipping all two years.
RETURN_HISTORY = 320
RETURN_PRECISION = 5


def build_returns(symbols: list[str]) -> dict[str, object]:
    """Daily simple returns per symbol, trimmed and rounded to keep the page small."""
    price_map = store.load_prices(symbols)
    kept: list[str] = []
    rows: list[list[float]] = []

    for symbol in symbols:
        series = price_map.get(symbol)
        if not series or len(series) < 2:
            continue
        closes = [c for _d, c in series][-(RETURN_HISTORY + 1) :]
        returns = [
            round(closes[i] / closes[i - 1] - 1.0, RETURN_PRECISION)
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if not returns:
            continue
        kept.append(symbol)
        rows.append(returns)

    # Align every series to a common length so the browser can treat it as a matrix.
    width = min((len(r) for r in rows), default=0)
    return {"symbols": kept, "data": [r[-width:] for r in rows] if width else []}


def build_payload() -> dict[str, object]:
    data = snap.load_snapshot()
    if data is None:
        raise SystemExit("No snapshot found. Run scripts/refresh.py first.")

    stock_symbols = [s["symbol"] for s in data["stocks"]]
    macro_symbols = [a["symbol"] for a in data["macro"]["assets"]]
    wanted = list(dict.fromkeys(stock_symbols + macro_symbols + [config.MARKET_ETF]))

    meta = {
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
        "market_etf": config.MARKET_ETF,
    }

    return {
        "meta": meta,
        "stocks": data["stocks"],
        "macro": data["macro"],
        "portfolios": data.get("portfolios", []),
        "sector_exposure": data.get("sector_exposure", []),
        "clusters": data.get("clusters", {}),
        "hrp_universe": data.get("hrp_universe", {}),
        "returns": build_returns(wanted),
    }


def _escape_non_ascii(source: str) -> str:
    r"""Rewrite every non-ASCII character as a JavaScript ``\uXXXX`` escape.

    Valid inside string literals (where it decodes back to the original glyph) and
    harmless inside comments, so it can be applied to a whole script safely.
    """
    return "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in source)


def render(payload: dict[str, object]) -> str:
    frontend = config.FRONTEND_DIR
    css = (frontend / "styles.css").read_text(encoding="utf-8")
    app_js = (frontend / "app.js").read_text(encoding="utf-8")
    offline_js = (frontend / "offline.js").read_text(encoding="utf-8")

    # `</script>` inside JSON would close the tag early; escaping the slash is the
    # standard fix and stays valid JSON. json.dumps already escapes non-ASCII to \uXXXX.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    # Emit a pure-ASCII page. A `<meta charset>` written here lands inside the host's
    # <body>, far too late to influence decoding, so the file must not depend on the
    # document charset at all. Escaping the JS keeps every beta, sigma and arrow glyph
    # correct whatever encoding the host declares.
    app_js = _escape_non_ascii(app_js)
    offline_js = _escape_non_ascii(offline_js)

    # The charset declaration is not optional: without it a browser opening this file
    # falls back to Latin-1 and every beta, sigma and em-dash renders as mojibake.
    return f"""<meta charset="utf-8">
<title>Biggie Market Terminal</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=5">
<meta name="theme-color" content="#101010">
<style>
{css}
/* The host page supplies its own frame; keep the app inside the viewport. */
html, body {{ height: 100%; margin: 0; }}
</style>

<div id="app">
  <section class="page" id="page-stocks" role="tabpanel" aria-label="Stocks">
    <header class="topbar">
      <h1>Stocks <small id="stocks-count"></small></h1>
      <div class="chips" id="window-chips" role="group" aria-label="Ranking window"></div>
      <div class="chips" id="sector-chips" role="group" aria-label="Sector filter"></div>
      <input class="search" id="stock-search" type="search" inputmode="search"
             placeholder="Search ticker or company" autocomplete="off"
             autocorrect="off" spellcheck="false" aria-label="Search stocks">
    </header>
    <div class="list" id="stock-list"></div>
    <div class="sentinel" id="stock-sentinel"></div>
    <div class="loading" id="stock-loading" hidden>Loading...</div>
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

<script id="biggie-data" type="application/json">{blob}</script>
<script>
window.__BIGGIE__ = JSON.parse(document.getElementById('biggie-data').textContent);
</script>
<script>
{offline_js}
</script>
<script>
{app_js}
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the standalone HTML app")
    parser.add_argument("-o", "--output", default="dist/biggie.html")
    args = parser.parse_args()

    payload = build_payload()
    html = render(payload)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    size = out.stat().st_size
    returns = payload["returns"]
    print(f"Wrote {out}  ({size / 1e6:.2f} MB)")
    print(f"  stocks        {len(payload['stocks'])}")
    print(f"  macro assets  {len(payload['macro']['assets'])}")
    print(f"  return series {len(returns['symbols'])} x "
          f"{len(returns['data'][0]) if returns['data'] else 0} days")
    if size > 16_000_000:
        print("  WARNING: over the 16MB hosting limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
