# Biggie — Product Specification (Version 3)

This document is the authoritative description of product intent, architecture,
analytics and UX. It supersedes Versions 1 and 2.

It describes **intent, not an inflexible implementation contract**. Prioritise
correctness, usability, robustness, simplicity and performance over literal readings.
Make reasonable engineering tradeoffs where requirements are ambiguous or impractical;
simplify, approximate or substitute methods where the core intent is preserved. Do not
get stuck on edge cases, wording, or precision that does not materially affect the user.
Use best-practice financial, statistical and data-engineering judgment where details are
underspecified.

Update this file when a product decision materially changes the specification.

---

## 1. What this is

A phone-first stock ranking and macro market intelligence application. It maintains a
live 1,000-name US equity universe, ranks it on risk-adjusted momentum, reads the
cross-asset tape for a market regime, and builds risk-parity allocations from a
user-selected watchlist.

**It is not a backtester.** All portfolios are current analytical representations
derived from the latest available data.

## 2. Development autonomy

Routine implementation work — repo structure, branches, commits, merges, dependencies,
architecture choices, deployment mechanics — is owned by whoever is building. Keep the
repository clean and `main` functional. Continue from the existing application rather
than replacing working systems.

After meaningful changes: test, fix obvious issues, commit and merge, update the live
deployment, provide the phone-testable URL, and summarise what changed.

## 3. Production architecture

The app needs fresh market information **once per trading day**.

```
scheduled daily job → market-data API → analytical pipeline → static artifacts
                    → deployed static site → iPhone
```

An always-running application server is not required. Heavy calculations belong in the
daily pipeline; the phone consumes precomputed results. Lightweight user-specific
calculations (sorting, filtering, watchlist HRP) may run in the browser.

## 4. API key

The key must remain private. It must never appear in browser JavaScript, static JSON,
public source files, client-side environment variables, browser-visible logs, or
repository history. The scheduled pipeline may use it to produce public analytical
outputs.

## 5. Daily update pipeline

Runs once per trading day after market close, and must:

1. Determine whether new market data is available.
2. Download only what is needed; reuse cached history.
3. Update the universe, price/return histories, rankings, beta and residual metrics,
   regularized covariance/factor models, sector composites, macro analytics and
   clustering.
4. **Validate** the generated outputs.
5. Publish the new snapshot — and **preserve the prior valid snapshot if the update
   fails**.

Expose the latest successful market-data date clearly but unobtrusively in the UI.
Manual refresh may remain for development but must not be required.

## 6. Universe

Screen for reasonably liquid, realistically tradeable stocks. Keep eligibility broad: a
stock only needs to be something a typical retail investor could reasonably trade.
Deduplicate, then select the largest **1,000 by market capitalisation**.

Explicitly exclude structurally unsuitable instruments: baby bonds, preferred shares,
warrants, rights, SPAC units and other units with embedded derivatives, ETNs, structured
notes, closed-end funds (unless intentionally supported), and poor-quality OTC / pink
sheet securities.

Goal: a clean universe of ordinary common equities suitable for ranking and portfolio
construction.

## 7. Sector classification

Map every stock to its GICS sector, or the closest reliable equivalent. Maintain current
sector membership and exposure information.

## 8. Equal-weight analytical portfolios

- An equal-weight composite for **each sector**.
- A single equal-weight composite across **all 1,000 stocks**, used as the internal
  broad-universe benchmark.

## 9. Market and factor history

Maintain ~2 years of daily history for:

| Group | Instruments |
|---|---|
| Broad equity | VTI, SPY, QQQ, IWM |
| Sectors | XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLB, XLRE, XLC (and XLY — see note) |
| Bonds | TLT, IEF, SHY |
| Alternatives | GLD, SLV, USO |

> **Note.** XLY (Consumer Discretionary) is included beyond the listed ten. Without it
> the factor model has no sector proxy for ~10% of the universe and the macro sector grid
> is visibly incomplete.

## 10. Primary ranking

12-month risk-adjusted momentum with a proportional recent-data skip: ~250 trading days,
skipping the most recent ~20.

Calculate mean daily return, annualised return, annualised volatility, and:

```
score = annualised return ÷ annualised volatility
```

Rank the full universe strongest to weakest.

## 11. Additional ranking windows

Support market capitalisation plus 12-month, 6-month and 3-month risk-adjusted returns.

**The skip must scale with the window:**

| Window | Lookback | Skip |
|---|---|---|
| 12 month | 250 days | 20 |
| 6 month | 125 days | 10 |
| 3 month | 60 days | 5 |

Do **not** implement 12-month with a skip and the shorter windows without one. The
proportional skip is part of the intended methodology. Precompute all outputs daily so
switching windows is instantaneous.

## 12. Ranking UI labels

Labels must accurately reflect the methodology. A row of `12–1M | 6M | 3M` is misleading
when all three windows skip proportionally. Use plain labels (`12M | 6M | 3M`) with a
concise nearby explanation of the skip. Keep the UI minimal.

## 13. Covariance and risk model

Maintain a covariance model from synchronised daily returns. With ~1,000 stocks and only
~2 years of observations, **do not rely on the raw sample covariance matrix** — prefer
Ledoit-Wolf shrinkage, a factor covariance, or another defensible regularized method.

Use covariance for: HRP, portfolio construction, portfolio volatility, risk
contribution, correlation analysis, diversification analysis, clustering, factor
decomposition and residual covariance.

**For an individual stock, `sqrt(eᵢᵀ Σ eᵢ)` is simply that stock's volatility.** Do not
present it as a novel "covariance-adjusted" per-stock score. A genuinely different
per-stock risk measure may instead use residual or factor-adjusted volatility.

## 14. Beta

Estimate each stock's beta versus VTI over ~2 years of daily returns. Beta may appear in
per-ticker analytics and wherever it provides useful context, but must not clutter the
main ranking list.

## 15. Raw and residual momentum

Support a raw mode and a market-adjusted (residual) mode:

```
residual return = stock return − beta × VTI return
```

Where useful, improve this with a factor model using a broad market factor and the
relevant sector factor. Calculate residual returns, residual volatility and residual
momentum. Maintain regularized residual covariance where useful.

## 16. Mobile-first UI

Designed primarily for iPhone. Priorities: fast, minimal, high contrast, adequate
typography, low clutter, smooth scrolling, instant-feeling interactions, and both light
and dark mode.

Bottom navigation: **Stocks · Watchlist · Macro · Settings**.

## 17. Stocks page

Displays the ranked universe. Each row prioritises: rank, ticker, company, sector,
risk-adjusted score, and a watchlist selection control. Beta may be shown if it stays
visually clean. Do not overload rows with poorly interpretable statistics.

## 18. No Σ COV on stock rows

The ambiguous per-row covariance percentage is removed — it was unclear and consumed
valuable screen space. **This does not remove covariance from the analytical system**,
where it remains essential for HRP, portfolio risk, correlation structure, clustering,
diversification and risk contribution.

## 19. Direct watchlist control

The space previously used by Σ COV carries a direct watchlist control:

- Unselected: a restrained `+`.
- Selected: an unmistakable state (checkmark or filled marker).
- One tap adds; one tap removes. No detail page required.
- State changes immediately and stays synchronised with the Watchlist page.
- Visually secondary to ticker and score.

## 20. Watchlist persistence

The watchlist must survive page reload, closing the tab, closing the browser, returning
later, and normal deployment updates where practical. Persistent local browser storage
is acceptable for the static architecture. Do not add authentication or cloud
infrastructure solely for this.

## 21. Settings persistence

User settings must persist across sessions — ranking window, raw vs residual mode,
theme, cluster k, risk mode and any other user-adjustable setting. **Restore saved
settings before or during initial render** so the UI does not visibly reset and then
jump. The app should reopen substantially as the user left it.

## 22. Per-ticker view

Tapping a stock opens detailed analytics with return charts for 5-day, 10-day, 1-month,
3-month, 6-month and 12-month windows. Keep charts clean, mobile-optimised, restrained
and easy to compare. Useful analytics include return, annualised volatility,
risk-adjusted score, beta, raw/residual results, and relevant correlation or portfolio
risk information. Do not expose every statistic simply because it exists.

## 23. Watchlist / portfolio page

Shows explicitly selected stocks with persistent state, plus portfolio analytics: HRP
weights, portfolio volatility, correlation information, diversification diagnostics and
risk contribution where useful. Small watchlist-specific calculations may run in the
browser.

## 24. HRP

Hierarchical Risk Parity for selected-portfolio construction, on correlation distance:

```
d(i,j) = sqrt(0.5 × (1 − ρᵢⱼ))
```

Use regularized covariance. Perform hierarchical clustering and recursive allocation
based on cluster risk. Recompute when the watchlist changes. The purpose is to reduce
naive concentration in highly correlated securities.

## 25. Medoid clustering

Support correlation/covariance-based clustering, preferring medoids where
interpretability matters (the representative stays an actual stock). Use
correlation-derived distance with adjustable k. Perform universe-scale clustering in the
daily pipeline rather than repeatedly on the phone. Clustering is secondary to the
ranking and watchlist workflow.

## 26. Macro page

Covers equity (SPY, QQQ, IWM, VTI), the sector ETFs, bonds (TLT, IEF, SHY) and
alternatives (GLD, SLV, USO). Uses the same risk-adjusted momentum framework as the
stock universe, with joint macro covariance/correlation analytics.

## 27. Macro heatmap

A compact grid whose primary visual signal is risk-adjusted performance. Correlation
clustering or covariance grouping may be incorporated where it improves interpretation.
Do not sacrifice clarity to expose complex analytics.

## 28. Macro regime

A simple deterministic indicator from equity/bond behaviour, SPY/TLT correlation, sector
breadth, sector dispersion and cross-asset behaviour. Output only **Risk-On**,
**Transition** or **Risk-Off**. Keep it interpretable; do not turn it into a black-box
forecasting model.

## 29. Settings page

Minimal surface: ranking method and window, raw vs residual mode, cluster k, risk mode,
theme. Do not expose technical implementation controls unless they materially improve
the experience. All user-facing settings persist.

## 30. Static deployment

Deploy permanently on a simple, inexpensive platform supporting static hosting,
automated deployment, scheduled workflows and secure secrets. Provide a stable URL that
works well on iPhone. The temporary coding machine must not be required for ordinary use.

## 31. Performance

Precompute expensive universe-wide calculations once daily. Cache reusable intermediate
data. Avoid repeatedly downloading two years of history for 1,000 securities. Generate
compact, mobile-friendly artifacts and lazy-load per-ticker history where useful.

Sorting, filtering, switching ranking windows, changing raw/residual, adding or removing
watchlist stocks, and switching tabs should all feel immediate.

## 32. Determinism

Given identical market data, universe and configuration, the pipeline must produce
identical results.

Normal production behaviour is *daily refresh → validated snapshot → static deployment →
fast phone interaction*, **not** *phone repeatedly downloads raw data and rebuilds the
analytical engine*.

---

## Appendix: implementation notes

These record decisions made while building, so they are not relitigated.

**Dual-class deduplication.** Both the normalised issuer name *and* the ticker root must
agree before two listings merge (GOOG/GOOGL, BRK-A/BRK-B), keeping the more liquid line.
Bare single-letter suffixes are real share classes and are kept — a naive "strip after
the hyphen" rule drops Berkshire. Preferred shares are `-P<letter>`.

**Beta is univariate.** Reported beta comes from a plain regression on VTI, kept separate
from the joint market+sector regression that supplies sector loading and idiosyncratic
volatility. VTI and its sector ETFs are ~0.9 correlated, so a joint market coefficient is
a *partial* beta that routinely goes negative; an early build shipped a median beta of
−0.12 this way. The validation gate now fails a build whose median beta is implausible.

**The second risk denominator is idiosyncratic volatility**, from the market + own-sector
residual — median ≈0.87× total volatility, moving names ~20 rank positions. This exists
because `sqrt(eᵢᵀ Σ eᵢ)` was not a second measurement (see §13).

**Payload split.** `core.json` carries everything the tabs render and is the only thing
first paint waits on; `series.json` holds daily returns for charts and watchlist HRP and
is prefetched in the background.

**Storage degrades explicitly.** `frontend/storage.js` probes localStorage by writing to
it, falls back to sessionStorage then memory, and reports which it got — silently
swallowing storage errors is what makes a watchlist quietly fail to persist. Settings
written by an older release are normalised on read; both former risk modes map to
`total`, never to `idiosyncratic`, because that is a different measurement and changing
it silently would change a user's rankings without them asking.

**The published payload is a contract, and it is enforced.** `backend/app/contract.py`
declares the field names the client reads; `build_site.py` refuses to publish a payload
that breaks them, and `scripts/smoke_test.js` renders the built site at 390×844 and
asserts on *rendered text*. This exists because a macro field rename once shipped a
heatmap where every cell read "—": the Python suite passed, `node --check` passed, JSON
has no schema, and a count-based UI check saw 21 cells and called it fine. Only reading
the pixels catches that class of bug.

**Returns are never computed from carried-forward prices.** Forward-filling a halted
stock manufactures a run of 0% days followed by one catch-up day, which understates
volatility and feeds a distorted covariance into beta, clustering and HRP. The price
path is still filled for charts; returns come from observed closes only, with genuine
gaps left as missing.

**Published series share one dated calendar.** Aligning by array position pairs a recent
listing's first week against an established name's last week, and truncating everyone to
the shortest series lets a single IPO shorten the history behind every watchlist. Gaps
are published as `null` so the browser excludes them exactly as the Python does.

**List rows always show the global universe rank.** Renumbering after a sector or search
filter makes a sector leader read as universe #1.
