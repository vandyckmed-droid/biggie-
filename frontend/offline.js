/* Static-site data adapter.
 *
 * The deployed site has no backend: the daily pipeline publishes JSON and this answers
 * the same routes the live API does. Two payloads, loaded separately, because the phone
 * should not wait on data it is not looking at yet:
 *
 *   data/core.json    every tab's first paint - rankings, macro, sectors
 *   data/series.json  daily returns, needed only for per-ticker charts and watchlist
 *                     HRP, so it is fetched lazily and prefetched in the background
 *
 * The numerics that matter - Ledoit-Wolf shrinkage, correlation distance, HRP recursive
 * bisection, k-medoids - are ported from the Python so a watchlist allocation computed
 * here matches what the pipeline would produce.
 */
'use strict';

(function () {
  // Resolve data URLs against this script, so the site works from a subpath such as
  // a GitHub Pages project URL (/<repo>/) as well as from a domain root. In the
  // single-file bundle this script is inline, so `src` is empty and there is nothing
  // to resolve against - fall back to a relative base rather than throwing.
  const BASE = (() => {
    const src = document.currentScript && document.currentScript.src;
    if (!src) return './';
    try {
      return new URL('.', src).href;
    } catch {
      return './';
    }
  })();

  let DATA = null;
  let SERIES = null;
  let corePromise = null;
  let seriesPromise = null;
  let STOCK_BY_SYMBOL = new Map();
  let MACRO_BY_SYMBOL = new Map();
  let RETURN_INDEX = new Map();

  /* Two delivery shapes share this adapter: the deployed site fetches the JSON files,
     and the single-file build inlines them on `window`. Reading the inline copy first
     keeps one code path instead of two adapters that can drift apart. */
  async function fetchJSON(name, inlineKey) {
    const inline = window[inlineKey];
    if (inline) return inline;
    const res = await fetch(BASE + 'data/' + name, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Could not load market data (${res.status})`);
    return res.json();
  }

  function ensureCore() {
    if (!corePromise) {
      corePromise = fetchJSON('core.json', '__BIGGIE_CORE__').then((json) => {
        DATA = json;
        STOCK_BY_SYMBOL = new Map(json.stocks.map((s) => [s.symbol, s]));
        MACRO_BY_SYMBOL = new Map(json.macro.assets.map((a) => [a.symbol, a]));
        return json;
      }).catch((err) => { corePromise = null; throw err; });
    }
    return corePromise;
  }

  function ensureSeries() {
    if (!seriesPromise) {
      seriesPromise = fetchJSON('series.json', '__BIGGIE_SERIES__').then((json) => {
        SERIES = json;
        RETURN_INDEX = new Map(json.symbols.map((s, i) => [s, i]));
        return json;
      }).catch((err) => { seriesPromise = null; throw err; });
    }
    return seriesPromise;
  }

  // Let the shell warm the series file once the list is on screen.
  window.__PREFETCH_SERIES__ = () => ensureSeries().catch(() => {});

  const TRADING_DAYS = 252;
  const WATCHLIST_KEY = 'watchlist';
  const SETTINGS_KEY = 'settings';

  // ------------------------------------------------------------------ storage
  const store = window.BiggieStore;
  const readJSON = (key, fallback) => store.get(key, fallback);
  const writeJSON = (key, value) => store.set(key, value);

  const getWatchlist = () => {
    const list = store.get(WATCHLIST_KEY, []);
    return Array.isArray(list) ? list : [];
  };
  const setWatchlist = (list) => { store.set(WATCHLIST_KEY, list); return list; };

  // ------------------------------------------------------------- linear algebra
  /** Demean each column of a T x N matrix. */
  function demean(x) {
    const t = x.length, n = x[0].length;
    const means = new Array(n).fill(0);
    for (let i = 0; i < t; i++) for (let j = 0; j < n; j++) means[j] += x[i][j];
    for (let j = 0; j < n; j++) means[j] /= t;
    return x.map((row) => row.map((v, j) => v - means[j]));
  }

  /**
   * Ledoit-Wolf shrinkage toward a scaled identity.
   *
   * Ported from the same estimator sklearn uses, so the browser and the server agree:
   * the sample covariance is pulled toward mu*I by an intensity derived from how noisy
   * the sample estimate is relative to its dispersion.
   */
  function ledoitWolf(returns) {
    const x = demean(returns);
    const t = x.length, n = x[0].length;

    // Sample covariance S = X'X / T
    const s = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let k = 0; k < t; k++) {
      const row = x[k];
      for (let i = 0; i < n; i++) {
        const xi = row[i];
        if (!xi) continue;
        for (let j = i; j < n; j++) s[i][j] += xi * row[j];
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) { s[i][j] /= t; s[j][i] = s[i][j]; }
    }

    let mu = 0;
    for (let i = 0; i < n; i++) mu += s[i][i];
    mu /= n;

    // d2 = ||S - mu I||^2 / n
    let d2 = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const d = s[i][j] - (i === j ? mu : 0);
        d2 += d * d;
      }
    }
    d2 /= n;

    // b2 = average squared deviation of the per-observation outer products from S
    let b2 = 0;
    for (let k = 0; k < t; k++) {
      const row = x[k];
      let acc = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const d = row[i] * row[j] - s[i][j];
          acc += d * d;
        }
      }
      b2 += acc / n;
    }
    b2 /= t * t;
    b2 = Math.min(b2, d2);

    const shrinkage = d2 > 0 ? Math.max(0, Math.min(1, b2 / d2)) : 0;
    const cov = s.map((row, i) =>
      row.map((v, j) => (1 - shrinkage) * v + (i === j ? shrinkage * mu : 0)),
    );
    return { cov, shrinkage };
  }

  const variances = (cov) => cov.map((row, i) => Math.max(row[i], 1e-14));
  const annualVol = (cov) => variances(cov).map((v) => Math.sqrt(v) * Math.sqrt(TRADING_DAYS));

  function correlation(cov) {
    const sd = variances(cov).map(Math.sqrt);
    return cov.map((row, i) =>
      row.map((v, j) => (i === j ? 1 : Math.max(-1, Math.min(1, v / (sd[i] * sd[j]))))),
    );
  }

  function portfolioVol(cov, w) {
    let acc = 0;
    for (let i = 0; i < w.length; i++) {
      for (let j = 0; j < w.length; j++) acc += w[i] * cov[i][j] * w[j];
    }
    return Math.sqrt(Math.max(acc, 0)) * Math.sqrt(TRADING_DAYS);
  }

  /** d(i,j) = sqrt(0.5 * (1 - rho)) */
  const correlationDistance = (corr) =>
    corr.map((row, i) => row.map((v, j) => (i === j ? 0 : Math.sqrt(Math.max(0, 0.5 * (1 - v))))));

  // ------------------------------------------------------------------------ HRP
  /**
   * Single-linkage agglomeration that yields the quasi-diagonal leaf order directly.
   * Merging leaf lists in place is equivalent to scipy's leaves_list on the same tree,
   * without needing scipy's linkage-matrix bookkeeping.
   */
  function quasiDiagonalOrder(dist) {
    const n = dist.length;
    if (n <= 1) return [0].slice(0, n);
    let clusters = Array.from({ length: n }, (_, i) => [i]);

    const linkDistance = (a, b) => {
      let best = Infinity;
      for (const i of a) for (const j of b) best = Math.min(best, dist[i][j]);
      return best;
    };

    while (clusters.length > 1) {
      let bi = 0, bj = 1, best = Infinity;
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const d = linkDistance(clusters[i], clusters[j]);
          if (d < best) { best = d; bi = i; bj = j; }
        }
      }
      const merged = clusters[bi].concat(clusters[bj]);
      clusters = clusters.filter((_, i) => i !== bi && i !== bj);
      clusters.push(merged);
    }
    return clusters[0];
  }

  function clusterVariance(cov, idx) {
    const inv = idx.map((i) => 1 / Math.max(cov[i][i], 1e-14));
    const total = inv.reduce((a, b) => a + b, 0);
    const w = inv.map((v) => v / total);
    let acc = 0;
    for (let a = 0; a < idx.length; a++) {
      for (let b = 0; b < idx.length; b++) acc += w[a] * cov[idx[a]][idx[b]] * w[b];
    }
    return acc;
  }

  /** Recursive bisection: split the ordered leaves, allocate inversely to cluster risk. */
  function hierarchicalRiskParity(cov) {
    const n = cov.length;
    if (n === 0) return [];
    if (n === 1) return [1];

    const order = quasiDiagonalOrder(correlationDistance(correlation(cov)));
    const weights = new Array(n).fill(1);
    const stack = [order];

    while (stack.length) {
      const group = stack.pop();
      if (group.length <= 1) continue;
      const mid = Math.floor(group.length / 2);
      const left = group.slice(0, mid);
      const right = group.slice(mid);

      const vl = clusterVariance(cov, left);
      const vr = clusterVariance(cov, right);
      const total = vl + vr;
      const alpha = total > 0 ? 1 - vl / total : 0.5;

      left.forEach((i) => { weights[i] *= alpha; });
      right.forEach((i) => { weights[i] *= 1 - alpha; });
      stack.push(left, right);
    }

    const sum = weights.reduce((a, b) => a + b, 0);
    return sum > 0 ? weights.map((w) => w / sum) : new Array(n).fill(1 / n);
  }

  /** Partition around medoids; deterministic seeding so refreshes reproduce clusters. */
  function kMedoids(dist, symbols, k) {
    const n = symbols.length;
    k = Math.max(1, Math.min(k, n));
    if (n === 0) return { labels: [], medoids: [] };
    if (k >= n) return { labels: symbols.map((_, i) => i), medoids: symbols.slice() };

    const rowSums = dist.map((row) => row.reduce((a, b) => a + b, 0));
    const medoids = [rowSums.indexOf(Math.min(...rowSums))];
    while (medoids.length < k) {
      let best = -1, bestIdx = 0;
      for (let i = 0; i < n; i++) {
        if (medoids.includes(i)) continue;
        const nearest = Math.min(...medoids.map((m) => dist[i][m]));
        if (nearest > best) { best = nearest; bestIdx = i; }
      }
      medoids.push(bestIdx);
    }

    let labels = new Array(n).fill(0);
    for (let iter = 0; iter < 100; iter++) {
      labels = dist.map((row) => {
        let best = 0;
        for (let c = 1; c < medoids.length; c++) {
          if (row[medoids[c]] < row[medoids[best]]) best = c;
        }
        return best;
      });
      let moved = false;
      for (let c = 0; c < medoids.length; c++) {
        const members = [];
        labels.forEach((l, i) => { if (l === c) members.push(i); });
        if (!members.length) continue;
        let best = members[0], bestCost = Infinity;
        for (const cand of members) {
          const cost = members.reduce((acc, m) => acc + dist[cand][m], 0);
          if (cost < bestCost) { bestCost = cost; best = cand; }
        }
        if (best !== medoids[c]) { medoids[c] = best; moved = true; }
      }
      if (!moved) break;
    }
    return { labels, medoids: medoids.map((i) => symbols[i]) };
  }

  // ------------------------------------------------------------------ data access
  const seriesFor = (symbol) => {
    const i = RETURN_INDEX.get(symbol);
    return i === undefined || !SERIES ? null : SERIES.data[i];
  };

  /** Cumulative path over the last `days` observations, as return-from-start. */
  function chartPoints(symbol, days) {
    const series = seriesFor(symbol);
    if (!series) return null;
    const slice = series.slice(Math.max(0, series.length - days));
    const points = [0];
    let level = 1;
    for (const r of slice) { level *= 1 + r; points.push(level - 1); }
    return points;
  }

  const metricOf = (stock, window, mode) =>
    (stock.metrics || {})[`${window}|${mode}`] || {};

  function rowFor(stock, window, mode, riskMode) {
    const m = metricOf(stock, window, mode);
    const scoreKey = riskMode === 'idiosyncratic' ? 'score_idio' : 'score_total';
    const rankKey = riskMode === 'idiosyncratic' ? 'rank_idio' : 'rank_total';
    return {
      symbol: stock.symbol,
      name: stock.name,
      sector: stock.sector,
      beta: stock.beta,
      market_cap: stock.market_cap,
      score: m[scoreKey],
      rank: m[rankKey],
      ann_return: m.ann_return,
      vol: riskMode === 'idiosyncratic' ? m.vol_cov : m.vol_simple,
      risk_contribution: m.risk_contribution,
      cluster: stock.cluster,
    };
  }

  const sortRows = (rows) =>
    rows.sort((a, b) => {
      const av = a.score, bv = b.score;
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return bv - av;
    });

  // --------------------------------------------------------------------- routes
  function rankings(params) {
    const window = params.get('window') || 'mom_12_1';
    const mode = params.get('return_mode') || 'raw';
    const riskMode = params.get('risk_mode') || 'idiosyncratic';
    const sector = params.get('sector');
    const search = (params.get('search') || '').trim().toUpperCase();
    const limit = Number(params.get('limit') || 60);
    const offset = Number(params.get('offset') || 0);

    let rows;
    if (window === 'market_cap') {
      rows = DATA.stocks.map((s) => ({
        ...rowFor(s, 'mom_12_1', mode, riskMode),
        rank: s.cap_rank,
        score: s.market_cap,
      }));
      rows.sort((a, b) => (b.score || 0) - (a.score || 0));
    } else {
      rows = sortRows(DATA.stocks.map((s) => rowFor(s, window, mode, riskMode)));
    }

    if (sector && sector !== 'All') rows = rows.filter((r) => r.sector === sector);
    if (search) {
      rows = rows.filter(
        (r) => r.symbol.includes(search) || (r.name || '').toUpperCase().includes(search),
      );
    }
    rows.forEach((r, i) => { r.display_rank = i + 1; });

    return {
      total: rows.length,
      offset,
      limit,
      window,
      return_mode: mode,
      risk_mode: riskMode,
      rows: rows.slice(offset, offset + limit),
    };
  }

  function stockDetail(symbol) {
    symbol = symbol.toUpperCase();
    const stock = STOCK_BY_SYMBOL.get(symbol);
    const macroAsset = stock ? null : MACRO_BY_SYMBOL.get(symbol);
    if (!stock && !macroAsset) throw new Error(`${symbol} is not in the universe`);

    const charts = [];
    for (const w of DATA.meta.return_windows) {
      const points = chartPoints(symbol, w.days);
      if (!points) continue;
      charts.push({
        key: w.key,
        label: w.label,
        days: w.days,
        points,
        change: points[points.length - 1],
      });
    }

    const base = {
      symbol,
      charts,
      history_days: (seriesFor(symbol) || []).length,
      in_watchlist: getWatchlist().includes(symbol),
    };
    return stock
      ? { ...base, kind: 'stock', ...stock }
      : { ...base, kind: 'macro', ...macroAsset, name: macroAsset.label };
  }

  function macro(params) {
    const riskMode = params.get('risk_mode') || 'idiosyncratic';
    const scoreKey = riskMode === 'idiosyncratic' ? 'score_idio' : 'score_total';
    const volKey = riskMode === 'idiosyncratic' ? 'vol_idio' : 'vol';

    const assets = DATA.macro.assets
      .slice()
      .sort((a, b) => (b[scoreKey] ?? -Infinity) - (a[scoreKey] ?? -Infinity))
      .map((a, i) => ({
        symbol: a.symbol, label: a.label, asset_class: a.asset_class,
        score: a[scoreKey], vol: a[volKey], ann_return: a.ann_return,
        beta: a.beta, trailing: a.trailing, cluster: a.cluster, rank: i + 1,
      }));

    const values = assets.map((a) => a.score).filter((v) => v != null);
    const bound = values.length ? Math.max(...values.map(Math.abs), 0.5) : 1;
    const groups = {};
    assets.forEach((a) => {
      (groups[a.asset_class] = groups[a.asset_class] || []).push({
        symbol: a.symbol, label: a.label, score: a.score,
        rank: a.rank, ann_return: a.ann_return, cluster: a.cluster,
      });
    });

    return {
      regime: DATA.macro.regime,
      assets,
      heatmap: {
        scale: { min: -bound, max: bound },
        groups: ['Index', 'Sector', 'Rates', 'Commodity']
          .filter((n) => groups[n])
          .map((n) => ({ name: n, cells: groups[n] })),
      },
      diagnostics: DATA.macro.diagnostics || {},
      risk_mode: riskMode,
    };
  }

  function watchlistRows(params) {
    const symbols = getWatchlist();
    const rawWindow = params.get('window') || 'mom_12_1';
    const window = rawWindow === 'market_cap' ? 'mom_12_1' : rawWindow;
    const mode = params.get('return_mode') || 'raw';
    const riskMode = params.get('risk_mode') || 'idiosyncratic';

    const rows = [];
    symbols.forEach((symbol) => {
      const stock = STOCK_BY_SYMBOL.get(symbol);
      if (stock) { rows.push(rowFor(stock, window, mode, riskMode)); return; }
      const asset = MACRO_BY_SYMBOL.get(symbol);
      if (asset) {
        rows.push({
          symbol: asset.symbol, name: asset.label, sector: asset.asset_class,
          beta: asset.beta, market_cap: null, cluster: asset.cluster,
          score: riskMode === 'idiosyncratic' ? asset.score_cov : asset.score_simple,
          vol: riskMode === 'idiosyncratic' ? asset.vol_cov : asset.vol_simple,
          ann_return: asset.ann_return, rank: asset.rank,
        });
      }
    });
    return { symbols, rows: sortRows(rows) };
  }

  function watchlistHrp(params) {
    const symbols = getWatchlist();
    const k = Number(params.get('k') || 8);
    const mode = params.get('return_mode') || 'raw';
    const rawWindow = params.get('window') || 'mom_12_1';
    const windowKey = rawWindow === 'market_cap' ? 'mom_12_1' : rawWindow;
    const win = DATA.meta.windows.find((w) => w.key === windowKey) || DATA.meta.windows[0];

    if (symbols.length < 2) {
      return { symbols, weights: [], message: 'Add at least two names to build an HRP allocation.' };
    }

    const usable = symbols.filter((s) => seriesFor(s));
    if (usable.length < 2) {
      return { symbols, weights: [], message: 'Not enough price history.' };
    }

    // Same slice the server takes: `lookback` observations ending `skip` days back.
    const market = seriesFor(DATA.meta.market_etf) || null;
    const cut = (arr) => {
      const end = arr.length - win.skip;
      return arr.slice(Math.max(0, end - win.lookback), end);
    };

    const columns = usable.map((symbol) => {
      let series = seriesFor(symbol);
      if (mode === 'residual' && market) {
        const beta = (STOCK_BY_SYMBOL.get(symbol) || MACRO_BY_SYMBOL.get(symbol) || {}).beta;
        const b = Number.isFinite(beta) ? beta : 1;
        series = series.map((r, i) => r - b * (market[i] ?? 0));
      }
      return cut(series);
    });

    const t = Math.min(...columns.map((c) => c.length));
    const matrix = [];
    for (let i = 0; i < t; i++) matrix.push(columns.map((c) => c[c.length - t + i]));

    const { cov, shrinkage } = ledoitWolf(matrix);
    const weights = hierarchicalRiskParity(cov);
    const vols = annualVol(cov);
    const equal = new Array(usable.length).fill(1 / usable.length);
    const corr = correlation(cov);
    const medoids = kMedoids(correlationDistance(corr), usable, Math.min(k, usable.length));

    let offSum = 0, offCount = 0;
    for (let i = 0; i < corr.length; i++) {
      for (let j = 0; j < corr.length; j++) {
        if (i !== j) { offSum += corr[i][j]; offCount++; }
      }
    }

    const groups = {};
    medoids.medoids.forEach((m) => { groups[m] = []; });
    medoids.labels.forEach((label, i) => {
      const medoid = medoids.medoids[label];
      if (medoid) groups[medoid].push(usable[i]);
    });

    return {
      symbols,
      missing: symbols.filter((s) => !seriesFor(s)),
      weights: usable.map((symbol, i) => ({
        symbol,
        weight: weights[i],
        equal_weight: 1 / usable.length,
        vol: vols[i],
        cluster: medoids.labels[i],
      })),
      portfolio_vol: portfolioVol(cov, weights),
      equal_weight_vol: portfolioVol(cov, equal),
      effective_n: 1 / weights.reduce((a, w) => a + w * w, 0),
      avg_correlation: offCount ? offSum / offCount : null,
      clusters: groups,
      shrinkage,
    };
  }

  function search(params) {
    const needle = (params.get('q') || '').trim().toUpperCase();
    const hits = [];
    DATA.stocks.forEach((s) => {
      if (s.symbol.includes(needle) || (s.name || '').toUpperCase().includes(needle)) {
        hits.push({ symbol: s.symbol, name: s.name, sector: s.sector, kind: 'stock' });
      }
    });
    DATA.macro.assets.forEach((a) => {
      if (a.symbol.includes(needle) || a.label.toUpperCase().includes(needle)) {
        hits.push({ symbol: a.symbol, name: a.label, sector: a.asset_class, kind: 'macro' });
      }
    });
    hits.sort((a, b) =>
      (a.symbol.startsWith(needle) ? 0 : 1) - (b.symbol.startsWith(needle) ? 0 : 1)
      || a.symbol.length - b.symbol.length);
    return { results: hits.slice(0, Number(params.get('limit') || 20)) };
  }

  const DEFAULT_SETTINGS = {
    window: 'mom_12_1', return_mode: 'raw', risk_mode: 'total',
    cluster_k: 8, sector: null,
  };

  // ------------------------------------------------------------------- dispatch
  //: Routes that read daily return series rather than precomputed aggregates.
  const NEEDS_SERIES = (route) =>
    route.startsWith('/stock/') || route === '/watchlist/hrp';

  window.__OFFLINE_API__ = async function offlineApi(path, options = {}) {
    const [route, query] = path.split('?');
    const params = new URLSearchParams(query || '');
    const method = (options.method || 'GET').toUpperCase();

    await ensureCore();
    if (NEEDS_SERIES(route)) await ensureSeries();

    if (route === '/meta') {
      return {
        ...DATA.meta,
        ready: true,
        settings: { ...DEFAULT_SETTINGS, ...readJSON(SETTINGS_KEY, {}) },
        refresh: { running: false, stage: 'daily pipeline', progress: 0, error: null },
        storage: { kind: store.kind, durable: store.durable },
        regime: DATA.macro.regime,
      };
    }
    if (route === '/rankings') return rankings(params);
    if (route.startsWith('/stock/')) return stockDetail(decodeURIComponent(route.slice(7)));
    if (route === '/macro') return macro(params);
    if (route === '/portfolios') {
      return {
        sector_exposure: DATA.sector_exposure,
        portfolios: DATA.portfolios,
        hrp_universe: DATA.hrp_universe,
        clusters: DATA.clusters,
      };
    }
    if (route === '/watchlist/hrp') return watchlistHrp(params);
    if (route === '/watchlist') {
      if (method === 'POST') {
        const symbol = JSON.parse(options.body || '{}').symbol.toUpperCase();
        const list = getWatchlist();
        if (!list.includes(symbol)) list.push(symbol);
        return { symbols: setWatchlist(list) };
      }
      return watchlistRows(params);
    }
    if (route.startsWith('/watchlist/') && method === 'DELETE') {
      const symbol = decodeURIComponent(route.slice(11)).toUpperCase();
      return { symbols: setWatchlist(getWatchlist().filter((s) => s !== symbol)) };
    }
    if (route === '/settings') {
      if (method === 'PUT') {
        const patch = JSON.parse(options.body || '{}');
        const merged = { ...readJSON(SETTINGS_KEY, {}), ...patch };
        writeJSON(SETTINGS_KEY, merged);
        return { ...DEFAULT_SETTINGS, ...merged };
      }
      return { ...DEFAULT_SETTINGS, ...readJSON(SETTINGS_KEY, {}) };
    }
    if (route === '/refresh') {
      // The hosted build ships a fixed snapshot; there is nothing to refresh against.
      return {
        running: false,
        stage: 'daily pipeline',
        progress: 0,
        error: 'Data refreshes automatically after each market close.',
        last_build: DATA.meta.built_at,
      };
    }
    if (route === '/search') return search(params);
    throw new Error(`Unknown route ${route}`);
  };
})();
