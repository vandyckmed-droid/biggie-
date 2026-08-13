# Biggie

A phone-first stock ranking and macro market intelligence app. It maintains a live
1,000-name equity universe, ranks it on risk-adjusted momentum with a shrunk covariance
model, and reads the cross-asset tape for a risk-on / risk-off regime call.

This is **not a backtester**. Every portfolio is a standing analytical claim that is
re-derived from current data on each refresh.

<img src="docs/stocks.png" width="260" alt="Stocks tab"> <img src="docs/macro.png" width="260" alt="Macro tab"> <img src="docs/watchlist.png" width="260" alt="Watchlist with HRP">

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

export FMP_API_KEY=your_key          # or API_KEY
.venv/bin/python scripts/refresh.py  # build the first snapshot (~40s cold, ~15s warm)

.venv/bin/python -m uvicorn app.api:app --app-dir backend --host 0.0.0.0 --port 8000
```

Open `http://<your-machine>:8000` on your phone. It installs as a PWA from the
browser's "Add to Home Screen".

```bash
.venv/bin/python -m pytest tests/ -q   # 55 tests, no network required
```

## How it works

### Universe (`universe.py`)

Screens all actively-traded NASDAQ / NYSE / AMEX common equity above $300M market cap,
then takes the largest 1,000 by market capitalisation.

Structurally unsuitable instruments are removed, because they do not behave like equity
and would corrupt both the ranking and the covariance matrix:

| Removed | How it is detected |
|---|---|
| Preferred shares | `-P<letter>` ticker suffix (`FITB-PA`), or "preferred"/"pfd" in the name |
| Baby bonds, trust preferreds | "notes", "debenture", "due 2032", a `%` coupon in the name |
| Warrants / rights / units | `-WT`, `-RT`, `-UN`, `-U` suffixes, or the words in the name |
| ETNs, structured notes, CEFs | `isEtf`/`isFund` flags plus fund-name patterns |
| OTC / pink sheets | Not on a major listing venue |

Bare single-letter suffixes are deliberately **kept** — `BRK-A`, `BF-B` and `MOG-A` are
real common shares, and a naive "strip anything after a hyphen" rule would drop
Berkshire.

Dual-class listings are then collapsed to one line per issuer (GOOG/GOOGL, FOX/FOXA,
BRK-A/BRK-B), keeping the most liquid class. Both the normalised issuer name *and* the
ticker root must agree before two rows merge, so unrelated companies are never silently
dropped. Without this, one company would occupy two slots and double-count its risk.

US-listed foreign issuers and ADRs are kept — they trade on the same venues with the
same data quality.

### Risk model (`risk.py`)

With ~250 observations and 1,000 names the sample covariance matrix is singular and its
extreme eigenvalues are noise, so **Ledoit-Wolf shrinkage** is used throughout (measured
intensity ≈ 0.26 at full universe size). This is what makes per-name volatility, HRP and
clustering stable rather than artefacts of estimation error.

Two regressions are run per stock, and reported separately:

- **Market beta** — univariate, against VTI, over two years. This is the number the UI
  shows and the residual toggle uses.
- **Joint market + own-sector regression** — supplies the sector loading and the
  idiosyncratic volatility.

They are kept apart on purpose. VTI and its sector ETFs are ~0.9 correlated, so a *joint*
market coefficient is a partial beta that routinely goes negative and reads as nonsense.
An early build reported exactly that (median beta −0.12); the univariate figure gives a
median of 1.02, with NVDA at 1.93 and consumer staples near 0.2.

### Ranking (`ranking.py`)

`Annualised return ÷ annualised volatility` over a momentum window, where volatility is
either the plain standard deviation or `sqrt(eᵢᵀ Σ eᵢ)` from the shrunk covariance.

Skip length scales with the lookback so each window strips a proportional slice of
short-term reversal:

| Window | Lookback | Skip |
|---|---|---|
| 12–1M | 250d | 20d |
| 6M | 125d | 10d |
| 3M | 60d | 5d |

Market cap is available as a fourth ranking. Every combination of window × (raw ·
market-adjusted) × (simple σ · covariance σ) is precomputed on refresh, so switching any
of them on the phone is an instant re-sort rather than a round trip.

In market-adjusted mode, `residual = stock return − β × VTI return` is applied to every
daily observation before the window statistics and the covariance are computed.

### Portfolios, HRP and clustering

- Live equal-weight portfolios per GICS sector plus a 1,000-name composite, each with
  trailing return, volatility, beta, max drawdown and diversification ratio.
- **Hierarchical Risk Parity** on the correlation distance `d(i,j) = sqrt(0.5(1−ρ))`:
  single-linkage tree, quasi-diagonalisation, recursive bisection allocating inversely to
  cluster variance. No matrix inversion, which is the point — mean-variance weights on a
  matrix this noisy would be dominated by estimation error.
- **k-medoids** on the same distance. Medoids rather than centroids because "the LRCX
  cluster" is interpretable in a way a synthetic centroid is not. Deterministic seeding,
  so a refresh reproduces the same clusters from the same data.

Watchlist HRP is solved on demand, since the watchlist changes with a single tap.

### Macro (`macro.py`)

21 instruments — SPY/QQQ/IWM/VTI, all eleven sector ETFs, TLT/IEF/SHY, GLD/SLV/USO —
ranked with the same maths and a joint covariance matrix.

The regime call blends four largely independent reads, each squashed through `tanh`:
equity vs duration, cyclical vs defensive leadership, small vs large cap, and risk assets
vs havens. The middle band is deliberately wide: "the signals disagree" is real
information, so a weak reading is reported as **Transition** rather than forced into a
direction. SPY/TLT rolling correlation and sector dispersion are reported alongside as
context.

> The spec lists ten sector ETFs and omits XLY. All eleven are used — without it the
> factor model has no sector proxy for ~10% of the universe and the macro sector grid
> reads as broken.

## Architecture

```
backend/app/
  config.py      paths, instrument sets, windows, thresholds
  fmp.py         async FMP client - retry, 429 backoff, key redaction
  store.py       SQLite cache: prices, securities, watchlist, settings
  universe.py    eligibility, exclusions, dedup, top-N
  returns.py     calendar alignment, return matrix, window statistics
  risk.py        Ledoit-Wolf covariance, beta, factor model, residuals
  ranking.py     momentum windows across every mode combination
  portfolios.py  equal-weight sector and composite books
  hrp.py         hierarchical risk parity, k-medoids
  macro.py       cross-asset ranking, heatmap, regime detection
  snapshot.py    orchestration -> one self-contained snapshot
  api.py         FastAPI routes
frontend/        three static files, no build step
```

Prices live in SQLite and refresh incrementally — a rebuild only asks for bars newer than
what is cached, with a few days of overlap so late adjustments get corrected. A full
cold build of 1,000 names takes ~40s; a warm rebuild ~14s. Snapshot builds are
deterministic: the same cached prices always produce the same output.

The API serves compact rows (six fields) rather than the whole snapshot, so the phone
never downloads 1.9MB to render a list. Endpoint latency is 3–9ms at full universe size.

## Design notes

**The heatmap is blue↔red, not green↔red.** Red-green is the one pair red-green colour
blindness cannot separate, and it is the single most common form. Every cell also prints
its numeric score, so colour is a second channel rather than the only one. Return figures
elsewhere use green/red *with an explicit +/− sign*, which is the redundant channel there.

**Both themes are selected, not flipped.** The dark diverging ramp runs dark→mid so a
single white ink colour clears 4.5:1 on every step; the light ramp runs light→dark and
flips ink on the two darkest steps.

The categorical pair used for HRP weight vs equal weight passes CVD separation, the
normal-vision floor and contrast in both modes.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FMP_API_KEY` / `API_KEY` | — | Financial Modeling Prep key (required) |
| `BIGGIE_DATA_DIR` | `./data` | Cache and snapshot location |
| `BIGGIE_UNIVERSE_SIZE` | `1000` | Names in the universe |
| `BIGGIE_CONCURRENCY` | `12` | Parallel FMP requests |

```bash
python scripts/refresh.py --size 250   # smaller universe, for a quick check
python scripts/refresh.py --full       # ignore the cache, refetch everything
```

## Limitations

- Sector classification comes from FMP and is mapped onto GICS sector names. It is the
  closest available classification, not licensed GICS data.
- The universe is rebuilt from a current screen each refresh, so it reflects today's
  membership. Trailing statistics describe how *today's* members behaved, which is not
  the same as a survivorship-free historical study — that would need point-in-time
  constituent data, and this system is explicitly not a backtester.
- Ranking uses dividend-adjusted closes; intraday and corporate-action edge cases inherit
  whatever the vendor reports.
