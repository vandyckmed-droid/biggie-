# Working on this repository

**Read [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) before making any meaningful product or
architectural change.** It is the authoritative description of what this system is meant
to do, how it is meant to be built, and why. This file only tells you how to work; the
spec tells you what to build.

## How to treat the specification

- `PRODUCT_SPEC.md` describes **intent, not an inflexible implementation contract**.
  Prioritise correctness, usability, robustness, simplicity and performance over literal
  wording. Make reasonable engineering tradeoffs where the spec is ambiguous or
  impractical, and use best-practice financial and data-engineering judgment where it is
  underspecified.
- **Update `PRODUCT_SPEC.md`** when a product decision materially changes the
  specification. A spec that has drifted from the system is worse than no spec.

## How to work

- **Preserve what works.** Continue from the existing implementation; do not restart or
  rebuild working functionality without cause.
- **Work incrementally.** Small, verified changes beat sweeping rewrites.
- **Keep `main` functional**, and handle repo mechanics — branches, commits, merges,
  dependencies, deployment — autonomously.
- **Test before calling anything done.** `pytest tests/ -q` must pass, and UI changes
  should be exercised at a phone-sized viewport (390×844), not just reasoned about.
- **Never let the API key reach the browser**, published JSON, repository contents, or
  logs. `scripts/check_no_secrets.py` enforces this in CI.

## Commands

```bash
pytest tests/ -q                      # analytics + validation tests, no network needed
python scripts/refresh.py             # rebuild the snapshot (needs FMP_API_KEY)
python scripts/build_site.py -o site  # build the deployable static site
python scripts/check_no_secrets.py    # credential tripwire
node --check frontend/app.js          # frontend has no build step; just syntax-check
```

## Shape of the system

A scheduled daily job builds a validated analytical snapshot and publishes a static
site; the phone consumes precomputed results. There is no server in production.

```
.github/workflows/daily.yml  after-close refresh -> validate -> deploy to Pages
backend/app/                 the analytical engine (see PRODUCT_SPEC.md §13-15)
  validate.py                the publish gate - a failed build is never deployed
frontend/                    static shell; offline.js serves the API surface from JSON
scripts/                     refresh, site build, secret scan
```

Two invariants worth knowing before you change anything:

1. **The covariance matrix is for portfolio questions, not single-stock volatility.**
   For one asset, `sqrt(eᵢᵀ Σ eᵢ)` is just that asset's own volatility. Do not
   reintroduce it as a distinct "covariance-adjusted" per-stock metric.
2. **Every ranking window applies a proportional skip** (250/20, 125/10, 60/5). Labels
   and copy must not imply that only the 12-month window skips recent data.
