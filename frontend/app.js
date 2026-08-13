/* Biggie - client.
 *
 * Deliberately dependency-free and build-step-free: the whole app is three static
 * files, so a phone gets it in one round trip and every interaction after that is
 * either local or a small JSON fetch.
 */
'use strict';

// ---------------------------------------------------------------------- helpers
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

const num = (v) => (v == null || Number.isNaN(v) ? null : v);
const pct = (v, digits = 1) =>
  num(v) == null ? '—' : `${(v * 100).toFixed(digits)}%`;
const signedPct = (v, digits = 1) =>
  num(v) == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;
const fixed = (v, digits = 2) => (num(v) == null ? '—' : v.toFixed(digits));
const signed = (v, digits = 2) =>
  num(v) == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`;

function money(v) {
  if (num(v) == null) return '—';
  const abs = Math.abs(v);
  if (abs >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
  return `$${v.toFixed(0)}`;
}

/** Sign class for return-style numbers. The +/- sign is always printed too, so the
 *  colour is reinforcing rather than load-bearing. */
const signClass = (v) => (num(v) == null ? 'flat' : v > 0 ? 'pos' : v < 0 ? 'neg' : 'flat');

async function api(path, options) {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

let toastTimer;
function toast(message) {
  const node = $('#toast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove('show'), 2200);
}

// ------------------------------------------------------------------ app state
const state = {
  tab: 'stocks',
  meta: null,
  settings: {
    window: 'mom_12_1',
    return_mode: 'raw',
    risk_mode: 'covariance',
    cluster_k: 8,
    sector: null,
  },
  stocks: { rows: [], total: 0, offset: 0, loading: false, done: false, search: '' },
  watchlist: new Set(),
  macroView: 'heatmap',
  sheetSymbol: null,
};

const PAGE_SIZE = 60;

// --------------------------------------------------------------------- charts
/** A sparkline. One series, so no legend - the card header names it.
 *  Baseline at the window's starting value makes "up or down since then" readable
 *  without axes, which is the only question a 42px chart can honestly answer. */
function sparkline(points, width = 150, height = 42) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');

  if (!points || points.length < 2) return svg;

  const lo = Math.min(...points, 0);
  const hi = Math.max(...points, 0);
  const span = hi - lo || 1;
  const pad = 3;
  const x = (i) => (i / (points.length - 1)) * width;
  const y = (v) => height - pad - ((v - lo) / span) * (height - pad * 2);

  const last = points[points.length - 1];
  const stroke = last > 0 ? 'var(--good)' : last < 0 ? 'var(--critical)' : 'var(--text-secondary)';

  // Zero line: the "flat since the start of the window" reference.
  const zero = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  zero.setAttribute('x1', 0); zero.setAttribute('x2', width);
  zero.setAttribute('y1', y(0)); zero.setAttribute('y2', y(0));
  zero.setAttribute('stroke', 'var(--border)');
  zero.setAttribute('stroke-width', '1');
  zero.setAttribute('stroke-dasharray', '2 3');
  svg.appendChild(zero);

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', points.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' '));
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', stroke);
  path.setAttribute('stroke-width', '2');
  path.setAttribute('stroke-linejoin', 'round');
  path.setAttribute('stroke-linecap', 'round');
  path.setAttribute('vector-effect', 'non-scaling-stroke');
  svg.appendChild(path);

  // End marker, ringed in the surface colour so it stays legible over the line.
  const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  dot.setAttribute('cx', width); dot.setAttribute('cy', y(last));
  dot.setAttribute('r', '2.5');
  dot.setAttribute('fill', stroke);
  dot.setAttribute('stroke', 'var(--surface-2)');
  dot.setAttribute('stroke-width', '1.5');
  svg.appendChild(dot);

  return svg;
}

/** Diverging colour for the heatmap: red (negative) - neutral - blue (positive). */
function divergingColor(value, bound) {
  if (num(value) == null) return 'var(--surface-2)';
  const t = Math.max(-1, Math.min(1, value / (bound || 1)));
  const mag = Math.abs(t);
  const side = t >= 0 ? 'pos' : 'neg';
  if (mag < 0.12) return 'var(--div-neutral)';
  const step = mag < 0.38 ? 1 : mag < 0.62 ? 2 : mag < 0.85 ? 3 : 4;
  return `var(--div-${side}-${step})`;
}

/** Cell ink. The dark ramp runs dark->mid so white clears 4.5:1 everywhere; the light
 *  ramp runs light->dark, so ink flips to white on the two darkest steps. */
function divergingInk(value, bound) {
  if (num(value) == null) return 'var(--text-muted)';
  const mag = Math.abs(Math.max(-1, Math.min(1, value / (bound || 1))));
  if (document.documentElement.dataset.theme === 'light') {
    return mag > 0.62 ? '#fff' : 'var(--text-primary)';
  }
  return mag < 0.12 ? 'var(--text-secondary)' : '#fff';
}

// ------------------------------------------------------------------- rendering
function stockRow(row, opts = {}) {
  const node = el('button', 'row');
  node.type = 'button';
  node.dataset.symbol = row.symbol;

  node.appendChild(el('div', 'row-rank', opts.rank ?? row.display_rank ?? row.rank ?? '—'));

  const main = el('div', 'row-main');
  const sym = el('div', 'row-sym');
  sym.appendChild(el('span', null, row.symbol));
  if (state.watchlist.has(row.symbol)) {
    const star = el('span', null, '★');
    star.style.color = 'var(--warning)';
    star.style.fontSize = '11px';
    sym.appendChild(star);
  }
  main.appendChild(sym);
  main.appendChild(el('div', 'row-sub', `${row.sector || '—'} · β ${fixed(row.beta)}`));
  node.appendChild(main);

  const metrics = el('div', 'row-metrics');

  const scoreBlock = el('div', 'metric-block');
  const isCap = state.settings.window === 'market_cap';
  const scoreVal = el(
    'div',
    `metric-val ${isCap ? 'flat' : signClass(row.score)}`,
    isCap ? money(row.score) : signed(row.score),
  );
  scoreBlock.appendChild(scoreVal);
  scoreBlock.appendChild(el('div', 'metric-cap', isCap ? 'mkt cap' : 'score'));
  metrics.appendChild(scoreBlock);

  if (!isCap) {
    const volBlock = el('div', 'metric-block');
    volBlock.appendChild(el('div', 'metric-val flat', pct(row.vol, 0)));
    volBlock.appendChild(
      el('div', 'metric-cap', state.settings.risk_mode === 'covariance' ? 'σ cov' : 'σ'),
    );
    metrics.appendChild(volBlock);
  }

  node.appendChild(metrics);
  node.addEventListener('click', () => openSheet(row.symbol));
  return node;
}

// --------------------------------------------------------------------- Stocks
async function loadStocks(reset = false) {
  if (state.stocks.loading) return;
  if (reset) {
    state.stocks = { ...state.stocks, rows: [], offset: 0, done: false };
    $('#stock-list').textContent = '';
  }
  if (state.stocks.done) return;

  state.stocks.loading = true;
  $('#stock-loading').hidden = false;

  const params = new URLSearchParams({
    window: state.settings.window,
    return_mode: state.settings.return_mode,
    risk_mode: state.settings.risk_mode,
    limit: PAGE_SIZE,
    offset: state.stocks.offset,
  });
  if (state.settings.sector) params.set('sector', state.settings.sector);
  if (state.stocks.search) params.set('search', state.stocks.search);

  try {
    const data = await api(`/rankings?${params}`);
    const list = $('#stock-list');
    data.rows.forEach((row) => list.appendChild(stockRow(row)));
    state.stocks.total = data.total;
    state.stocks.offset += data.rows.length;
    state.stocks.done = state.stocks.offset >= data.total || data.rows.length === 0;
    $('#stocks-count').textContent = `${data.total} names`;
    if (data.total === 0) {
      list.appendChild(el('div', 'empty', 'Nothing matches that filter.'));
    }
  } catch (err) {
    $('#stock-list').appendChild(el('div', 'empty', err.message));
    state.stocks.done = true;
  } finally {
    state.stocks.loading = false;
    $('#stock-loading').hidden = true;
  }
}

function renderWindowChips() {
  const box = $('#window-chips');
  box.textContent = '';
  const windows = [
    ...(state.meta?.windows || []).map((w) => ({ key: w.key, label: w.label })),
    { key: 'market_cap', label: 'Mkt Cap' },
  ];
  windows.forEach((w) => {
    const chip = el('button', 'chip', w.label);
    chip.type = 'button';
    chip.setAttribute('aria-pressed', String(state.settings.window === w.key));
    chip.addEventListener('click', () => {
      state.settings.window = w.key;
      persistSettings({ window: w.key });
      renderWindowChips();
      loadStocks(true);
    });
    box.appendChild(chip);
  });
}

/** GICS sector names are far too long for a phone chip row; these abbreviations keep
 *  the row scannable instead of forcing a scroll past three chips. */
const SECTOR_SHORT = {
  'Information Technology': 'Info Tech',
  'Communication Services': 'Comm Svcs',
  'Consumer Discretionary': 'Cons Disc',
  'Consumer Staples': 'Staples',
  'Health Care': 'Health',
  'Real Estate': 'Real Est',
  Industrials: 'Indust',
  Financials: 'Fins',
  Materials: 'Mats',
  Utilities: 'Utils',
  Energy: 'Energy',
};

function renderSectorChips() {
  const box = $('#sector-chips');
  box.textContent = '';
  const sectors = ['All', ...(state.meta?.sectors || [])];
  sectors.forEach((sector) => {
    const active = (sector === 'All' && !state.settings.sector) || state.settings.sector === sector;
    const chip = el('button', 'chip', sector === 'All' ? 'All' : SECTOR_SHORT[sector] || sector);
    chip.title = sector;
    chip.type = 'button';
    chip.setAttribute('aria-pressed', String(active));
    chip.addEventListener('click', () => {
      state.settings.sector = sector === 'All' ? null : sector;
      persistSettings({ sector: state.settings.sector });
      renderSectorChips();
      loadStocks(true);
    });
    box.appendChild(chip);
  });
}

// ------------------------------------------------------------------ Watchlist
async function loadWatchlist() {
  const body = $('#watchlist-body');
  body.textContent = '';

  const params = new URLSearchParams({
    window: state.settings.window,
    return_mode: state.settings.return_mode,
    risk_mode: state.settings.risk_mode,
  });

  let data;
  try {
    data = await api(`/watchlist?${params}`);
  } catch (err) {
    body.appendChild(el('div', 'empty', err.message));
    return;
  }

  state.watchlist = new Set(data.symbols);
  $('#watchlist-count').textContent = `${data.symbols.length} held`;

  if (!data.symbols.length) {
    body.appendChild(
      el('div', 'empty', 'No names yet. Tap any stock or macro asset, then the ☆ to add it here.'),
    );
    return;
  }

  const listCard = el('div', 'card');
  listCard.appendChild(el('h2', null, 'Holdings'));
  const list = el('div');
  data.rows.forEach((row, i) => list.appendChild(stockRow(row, { rank: i + 1 })));
  listCard.appendChild(list);
  body.appendChild(listCard);

  // HRP allocation is computed live, because the watchlist changes on a single tap.
  const hrpCard = el('div', 'card');
  hrpCard.appendChild(el('h2', null, 'HRP Allocation'));
  hrpCard.appendChild(el('div', 'loading', 'Solving…'));
  body.appendChild(hrpCard);

  try {
    const hrp = await api(
      `/watchlist/hrp?k=${state.settings.cluster_k}&return_mode=${state.settings.return_mode}&window=${state.settings.window}`,
    );
    renderHrp(hrpCard, hrp);
  } catch (err) {
    hrpCard.textContent = '';
    hrpCard.appendChild(el('h2', null, 'HRP Allocation'));
    hrpCard.appendChild(el('div', 'empty', err.message));
  }
}

function renderHrp(card, hrp) {
  card.textContent = '';
  card.appendChild(el('h2', null, 'HRP Allocation'));

  if (!hrp.weights.length) {
    card.appendChild(el('div', 'empty', hrp.message || 'Not enough data.'));
    return;
  }

  // Fixed 2x2 rather than auto-fit: four stats reflow to 3+1 at phone widths.
  const stats = el('div', 'stat-grid two-up');
  [
    // Labels are uppercased in CSS, which mangles Greek (rho -> capital Rho reads as P).
    ['HRP vol', pct(hrp.portfolio_vol)],
    ['Equal-wt vol', pct(hrp.equal_weight_vol)],
    ['Effective N', fixed(hrp.effective_n, 1)],
    ['Avg corr', fixed(hrp.avg_correlation)],
  ].forEach(([label, value]) => {
    const box = el('div');
    box.appendChild(el('div', 'stat-label', label));
    box.appendChild(el('div', 'stat-value', value));
    stats.appendChild(box);
  });
  card.appendChild(stats);

  // Two series on screen -> legend is mandatory.
  const legend = el('div', 'legend-keys');
  legend.style.marginTop = '14px';
  [
    ['HRP weight', 'var(--series-1)', 'legend-swatch'],
    ['Equal weight', 'var(--series-2)', 'legend-swatch line'],
  ].forEach(([label, color, cls]) => {
    const key = el('div', 'legend-key');
    const sw = el('span', cls);
    sw.style.background = color;
    key.appendChild(sw);
    key.appendChild(el('span', null, label));
    legend.appendChild(key);
  });
  card.appendChild(legend);

  const max = Math.max(...hrp.weights.map((w) => w.weight), 0.0001);
  const sorted = [...hrp.weights].sort((a, b) => b.weight - a.weight);
  sorted.forEach((w) => {
    const row = el('div', 'weight-row');
    row.appendChild(el('div', 'weight-sym', w.symbol));

    const track = el('div', 'weight-track');
    const fill = el('div', 'weight-fill');
    fill.style.width = `${(w.weight / max) * 100}%`;
    track.appendChild(fill);
    const ref = el('div', 'weight-ref');
    ref.style.left = `${(w.equal_weight / max) * 100}%`;
    ref.title = 'Equal weight';
    track.appendChild(ref);
    row.appendChild(track);

    row.appendChild(el('div', 'weight-val', pct(w.weight, 1)));
    card.appendChild(row);
  });

  const note = el('div', 'setting-help');
  note.style.marginTop = '10px';
  const shrunk = hrp.equal_weight_vol > 0
    ? (1 - hrp.portfolio_vol / hrp.equal_weight_vol) * 100
    : 0;
  note.textContent =
    `Weights come from recursive bisection of the correlation tree (Ledoit-Wolf shrinkage ` +
    `${fixed(hrp.shrinkage)}). HRP volatility is ${shrunk >= 0 ? shrunk.toFixed(1) + '% below' : (-shrunk).toFixed(1) + '% above'} ` +
    `the equal-weight book.`;
  card.appendChild(note);

  // Only worth showing once clustering actually groups something - with k >= n every
  // cluster is a singleton and the line just repeats the holdings list.
  const grouped = Object.entries(hrp.clusters || {}).filter(([, m]) => m.length > 1);
  if (grouped.length) {
    const cl = el('div', 'setting-help');
    cl.style.marginTop = '6px';
    cl.textContent = 'Clusters — ' + grouped
      .map(([medoid, members]) => `${medoid}: ${members.join(', ')}`)
      .join(' · ');
    card.appendChild(cl);
  }
}

// ---------------------------------------------------------------------- Macro
async function loadMacro() {
  const body = $('#macro-body');
  body.textContent = '';
  body.appendChild(el('div', 'loading', 'Loading…'));

  let data;
  try {
    data = await api(`/macro?risk_mode=${state.settings.risk_mode}`);
  } catch (err) {
    body.textContent = '';
    body.appendChild(el('div', 'empty', err.message));
    return;
  }

  body.textContent = '';
  body.appendChild(renderRegime(data.regime));
  if (state.macroView === 'heatmap') {
    body.appendChild(renderHeatmap(data.heatmap));
  } else {
    body.appendChild(renderMacroList(data.assets));
  }
  $('#macro-asof').textContent = state.meta?.as_of || '';
}

function renderRegime(regime) {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Market Regime'));

  const cls = regime.state === 'Risk-On' ? 'regime-on'
    : regime.state === 'Risk-Off' ? 'regime-off' : 'regime-mid';
  const icon = regime.state === 'Risk-On' ? '▲'
    : regime.state === 'Risk-Off' ? '▼' : '◆';

  const head = el('div', 'regime');
  // Icon + label, never colour alone.
  const badge = el('div', `regime-badge ${cls}`);
  badge.appendChild(el('span', 'icon', icon));
  badge.appendChild(el('span', null, regime.state));
  head.appendChild(badge);

  const score = el('div', `regime-score ${cls}`, signed(regime.score));
  head.appendChild(score);
  card.appendChild(head);

  card.appendChild(el('div', 'narrative', regime.narrative || ''));

  (regime.components || []).forEach((c) => {
    const row = el('div', 'component');
    const left = el('div');
    left.appendChild(el('div', 'component-name', c.name));
    left.appendChild(el('div', 'component-detail', c.detail || ''));
    row.appendChild(left);

    const bar = el('div', 'signal-bar');
    const fill = el('div', 'signal-fill');
    const v = Math.max(-1, Math.min(1, c.value || 0));
    const width = Math.abs(v) * 50;
    fill.style.width = `${width}%`;
    if (v >= 0) {
      fill.style.left = '50%';
      fill.style.background = 'var(--good)';
    } else {
      fill.style.right = '50%';
      fill.style.background = 'var(--critical)';
    }
    bar.appendChild(fill);
    bar.title = signed(c.value);
    row.appendChild(bar);
    card.appendChild(row);
  });

  const dl = el('dl');
  dl.style.marginTop = '12px';
  [
    ['SPY / TLT ρ (60d)', fixed(regime.spy_tlt_corr_fast)],
    ['SPY / TLT ρ (250d)', fixed(regime.spy_tlt_corr_slow)],
    ['Sector dispersion', fixed(regime.sector_dispersion)],
    ['Confidence', pct(regime.confidence, 0)],
  ].forEach(([k, v]) => {
    const row = el('div', 'kv');
    row.appendChild(el('dt', null, k));
    row.appendChild(el('dd', null, v));
    dl.appendChild(row);
  });
  card.appendChild(dl);
  return card;
}

function renderHeatmap(grid) {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Risk-Adjusted Return'));

  const bound = grid.scale?.max || 1;
  (grid.groups || []).forEach((group) => {
    const box = el('div', 'heat-group');
    box.appendChild(el('h3', null, group.name));
    const g = el('div', 'heat-grid');
    group.cells.forEach((cell) => {
      const node = el('button', 'heat-cell');
      node.type = 'button';
      node.style.background = divergingColor(cell.score, bound);
      node.style.color = divergingInk(cell.score, bound);
      node.appendChild(el('div', 'heat-sym', cell.symbol));
      node.appendChild(el('div', 'heat-label', cell.label));
      // The number is always printed, so the colour is a second channel, not the only one.
      node.appendChild(el('div', 'heat-score', signed(cell.score)));
      node.setAttribute(
        'aria-label',
        `${cell.symbol} ${cell.label}, score ${signed(cell.score)}, rank ${cell.rank}`,
      );
      node.addEventListener('click', () => openSheet(cell.symbol));
      g.appendChild(node);
    });
    box.appendChild(g);
    card.appendChild(box);
  });

  const legend = el('div', 'legend');
  legend.appendChild(el('span', null, signed(-bound)));
  legend.appendChild(el('div', 'legend-ramp'));
  legend.appendChild(el('span', null, signed(bound)));
  card.appendChild(legend);
  card.appendChild(
    el('div', 'setting-help', 'Annualised return ÷ volatility over the ranking window. Tap any cell for detail.'),
  );
  return card;
}

function renderMacroList(assets) {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Ranked Cross-Asset'));
  assets.forEach((a, i) => {
    const row = el('button', 'row');
    row.type = 'button';
    row.appendChild(el('div', 'row-rank', i + 1));

    const main = el('div', 'row-main');
    const sym = el('div', 'row-sym');
    sym.appendChild(el('span', null, a.symbol));
    if (state.watchlist.has(a.symbol)) {
      const star = el('span', null, '★');
      star.style.color = 'var(--warning)';
      star.style.fontSize = '11px';
      sym.appendChild(star);
    }
    main.appendChild(sym);
    main.appendChild(el('div', 'row-sub', `${a.label} · ${a.asset_class} · β ${fixed(a.beta)}`));
    row.appendChild(main);

    const metrics = el('div', 'row-metrics');
    const s = el('div', 'metric-block');
    s.appendChild(el('div', `metric-val ${signClass(a.score)}`, signed(a.score)));
    s.appendChild(el('div', 'metric-cap', 'score'));
    metrics.appendChild(s);
    const v = el('div', 'metric-block');
    v.appendChild(el('div', 'metric-val flat', pct(a.vol, 0)));
    v.appendChild(el('div', 'metric-cap', 'σ'));
    metrics.appendChild(v);
    row.appendChild(metrics);

    row.addEventListener('click', () => openSheet(a.symbol));
    card.appendChild(row);
  });
  return card;
}

// -------------------------------------------------------------- detail sheet
async function openSheet(symbol) {
  state.sheetSymbol = symbol;
  $('#sheet-sym').textContent = symbol;
  $('#sheet-name').textContent = '';
  $('#sheet-body').textContent = '';
  $('#sheet-body').appendChild(el('div', 'loading', 'Loading…'));
  $('#sheet').classList.add('open');
  $('#sheet-backdrop').classList.add('open');

  let data;
  try {
    data = await api(`/stock/${encodeURIComponent(symbol)}`);
  } catch (err) {
    $('#sheet-body').textContent = '';
    $('#sheet-body').appendChild(el('div', 'empty', err.message));
    return;
  }

  $('#sheet-name').textContent = data.name || data.label || '';
  updateStar(data.in_watchlist);
  renderSheet(data);
}

function updateStar(on) {
  const star = $('#sheet-star');
  star.textContent = on ? '★' : '☆';
  star.classList.toggle('on', !!on);
}

function renderSheet(data) {
  const body = $('#sheet-body');
  body.textContent = '';

  // -------- headline stats
  const head = el('div', 'card');
  const stats = el('div', 'stat-grid');
  const isStock = data.kind === 'stock';
  const metricKey = `${state.settings.window === 'market_cap' ? 'mom_12_1' : state.settings.window}|${state.settings.return_mode}`;
  const m = (data.metrics || {})[metricKey] || {};
  const scoreKey = state.settings.risk_mode === 'covariance' ? 'score_cov' : 'score_simple';
  const volKey = state.settings.risk_mode === 'covariance' ? 'vol_cov' : 'vol_simple';

  const entries = isStock
    ? [
        ['Score', signed(m[scoreKey]), signClass(m[scoreKey])],
        ['Rank', m[state.settings.risk_mode === 'covariance' ? 'rank_cov' : 'rank_simple'] ?? '—', ''],
        ['Ann. return', signedPct(m.ann_return), signClass(m.ann_return)],
        ['Volatility', pct(m[volKey]), ''],
        ['Beta', fixed(data.beta), ''],
        ['Market cap', money(data.market_cap), ''],
      ]
    : [
        ['Score', signed(data[scoreKey]), signClass(data[scoreKey])],
        ['Rank', data.rank ?? '—', ''],
        ['Ann. return', signedPct(data.ann_return), signClass(data.ann_return)],
        ['Volatility', pct(data[volKey]), ''],
        ['Beta', fixed(data.beta), ''],
        ['Class', data.asset_class || '—', ''],
      ];

  entries.forEach(([label, value, cls]) => {
    const box = el('div');
    box.appendChild(el('div', 'stat-label', label));
    box.appendChild(el('div', `stat-value ${cls}`, value));
    stats.appendChild(box);
  });
  head.appendChild(stats);
  body.appendChild(head);

  // -------- return windows
  const charts = el('div', 'card');
  charts.appendChild(el('h2', null, 'Return by Window'));
  const grid = el('div', 'chart-grid');
  (data.charts || []).forEach((c) => {
    const mini = el('div', 'mini');
    const head2 = el('div', 'mini-head');
    head2.appendChild(el('span', 'mini-win', c.label));
    head2.appendChild(el('span', `mini-chg ${signClass(c.change)}`, signedPct(c.change)));
    mini.appendChild(head2);
    const svg = sparkline(c.points);
    svg.setAttribute('aria-label', `${c.label} return ${signedPct(c.change)}`);
    mini.appendChild(svg);
    grid.appendChild(mini);
  });
  charts.appendChild(grid);
  body.appendChild(charts);

  // -------- risk analytics
  const riskCard = el('div', 'card');
  riskCard.appendChild(el('h2', null, 'Risk Analytics'));
  const dl = el('dl');
  const rows = isStock
    ? [
        ['Market beta (VTI)', fixed(data.beta)],
        ['Sector beta', fixed(data.sector_beta)],
        ['Factor R²', fixed(data.r_squared)],
        ['Idiosyncratic vol', pct(data.idio_vol)],
        ['Covariance σ', pct(m.vol_cov)],
        ['Standalone σ', pct(m.vol_simple)],
        ['Marginal risk', pct(m.marginal_risk)],
        ['Risk contribution', m.risk_contribution == null ? '—' : `${(m.risk_contribution * 100).toFixed(3)}%`],
        ['Cluster', data.cluster || '—'],
        ['Cap rank', data.cap_rank ?? '—'],
      ]
    : [
        ['Market beta (VTI)', fixed(data.beta)],
        ['Covariance σ', pct(data.vol_cov)],
        ['Standalone σ', pct(data.vol_simple)],
        ['Asset class', data.asset_class || '—'],
      ];
  rows.forEach(([k, v]) => {
    const row = el('div', 'kv');
    row.appendChild(el('dt', null, k));
    row.appendChild(el('dd', null, v));
    dl.appendChild(row);
  });
  riskCard.appendChild(dl);

  if (isStock) {
    // Raw vs residual, right where the numbers it changes are shown.
    const seg = el('div', 'seg');
    ['raw', 'residual'].forEach((mode) => {
      const b = el('button', null, mode === 'raw' ? 'Raw returns' : 'Market-adjusted');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(state.settings.return_mode === mode));
      b.addEventListener('click', () => {
        state.settings.return_mode = mode;
        persistSettings({ return_mode: mode });
        renderSheet(data);
        refreshActiveTab();
      });
      seg.appendChild(b);
    });
    riskCard.appendChild(seg);
    riskCard.appendChild(
      el('div', 'setting-help',
        'Market-adjusted strips β × VTI out of every daily return, so the score reflects ' +
        'alpha rather than market exposure.'),
    );
  }
  body.appendChild(riskCard);

  // -------- window comparison
  if (isStock && data.metrics) {
    const cmp = el('div', 'card');
    cmp.appendChild(el('h2', null, 'Across Windows'));
    const dl2 = el('dl');
    (state.meta?.windows || []).forEach((w) => {
      const wm = data.metrics[`${w.key}|${state.settings.return_mode}`] || {};
      const row = el('div', 'kv');
      row.appendChild(el('dt', null, `${w.label} (skip ${w.skip}d)`));
      const dd = el('dd', null,
        `${signed(wm[scoreKey])}  ·  #${wm[state.settings.risk_mode === 'covariance' ? 'rank_cov' : 'rank_simple'] ?? '—'}`);
      dd.className = signClass(wm[scoreKey]);
      row.appendChild(dd);
      dl2.appendChild(row);
    });
    cmp.appendChild(dl2);
    body.appendChild(cmp);
  }
}

function closeSheet() {
  $('#sheet').classList.remove('open');
  $('#sheet-backdrop').classList.remove('open');
  state.sheetSymbol = null;
}

async function toggleStar() {
  const symbol = state.sheetSymbol;
  if (!symbol) return;
  try {
    if (state.watchlist.has(symbol)) {
      const r = await api(`/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' });
      state.watchlist = new Set(r.symbols);
      toast(`${symbol} removed`);
    } else {
      const r = await api('/watchlist', {
        method: 'POST',
        body: JSON.stringify({ symbol }),
      });
      state.watchlist = new Set(r.symbols);
      toast(`${symbol} added`);
    }
    updateStar(state.watchlist.has(symbol));
  } catch (err) {
    toast(err.message);
  }
}

// ------------------------------------------------------------------- Settings
function renderSettings() {
  const body = $('#settings-body');
  body.textContent = '';

  const card = el('div', 'card');

  const segment = (label, help, key, options) => {
    const box = el('div', 'setting');
    box.appendChild(el('div', 'setting-label', label));
    box.appendChild(el('div', 'setting-help', help));
    const seg = el('div', 'seg');
    options.forEach(([value, text]) => {
      const b = el('button', null, text);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(state.settings[key] === value));
      b.addEventListener('click', () => {
        state.settings[key] = value;
        persistSettings({ [key]: value });
        renderSettings();
        refreshActiveTab();
      });
      seg.appendChild(b);
    });
    box.appendChild(seg);
    return box;
  };

  card.appendChild(segment(
    'Ranking window',
    'How much history the momentum signal reads, and how many recent days it skips to strip short-term reversal.',
    'window',
    [
      ...(state.meta?.windows || []).map((w) => [w.key, w.label]),
      ['market_cap', 'Mkt Cap'],
    ],
  ));

  card.appendChild(segment(
    'Return mode',
    'Market-adjusted subtracts β × VTI from every daily return, isolating alpha from market exposure.',
    'return_mode',
    [['raw', 'Raw'], ['residual', 'Market-adjusted']],
  ));

  card.appendChild(segment(
    'Covariance mode',
    'Full covariance uses a Ledoit-Wolf shrunk matrix for volatility — more stable than a raw standard deviation when names outnumber observations.',
    'risk_mode',
    [['simple', 'Simple σ'], ['covariance', 'Full covariance']],
  ));

  // Cluster k
  const kBox = el('div', 'setting');
  kBox.appendChild(el('div', 'setting-label', 'Cluster granularity'));
  const kHelp = el('div', 'setting-help', `k = ${state.settings.cluster_k} medoid clusters`);
  kBox.appendChild(kHelp);
  const slider = el('input', 'slider');
  slider.type = 'range';
  slider.min = '2';
  slider.max = '25';
  slider.value = String(state.settings.cluster_k);
  slider.addEventListener('input', () => {
    kHelp.textContent = `k = ${slider.value} medoid clusters`;
  });
  slider.addEventListener('change', () => {
    state.settings.cluster_k = Number(slider.value);
    persistSettings({ cluster_k: state.settings.cluster_k });
    if (state.tab === 'watchlist') loadWatchlist();
  });
  kBox.appendChild(slider);
  card.appendChild(kBox);

  // Theme
  const themeBox = el('div', 'setting');
  themeBox.appendChild(el('div', 'setting-label', 'Appearance'));
  const seg = el('div', 'seg');
  [['dark', 'Dark'], ['light', 'Light']].forEach(([value, text]) => {
    const b = el('button', null, text);
    b.type = 'button';
    b.setAttribute('aria-pressed', String(document.documentElement.dataset.theme === value));
    b.addEventListener('click', () => {
      document.documentElement.dataset.theme = value;
      try { localStorage.setItem('biggie-theme', value); } catch { /* private mode */ }
      renderSettings();
      if (state.tab === 'macro') loadMacro();
    });
    seg.appendChild(b);
  });
  themeBox.appendChild(seg);
  card.appendChild(themeBox);
  body.appendChild(card);

  // ---- data card
  const dataCard = el('div', 'card');
  dataCard.appendChild(el('h2', null, 'Data'));
  const dl = el('dl');
  const meta = state.meta || {};
  const report = meta.universe_report || {};
  [
    ['Universe', meta.universe_size ?? '—'],
    ['As of', meta.as_of || '—'],
    ['Trading days', meta.trading_days ?? '—'],
    ['Screened', report.screened ?? '—'],
    ['Duplicates collapsed', report.deduped ?? '—'],
    ['LW shrinkage', fixed(meta.risk_model?.shrinkage)],
    ['Built', meta.built_at ? meta.built_at.replace('T', ' ').replace('+00:00', ' UTC') : '—'],
    ['Build time', meta.build_seconds ? `${meta.build_seconds}s` : '—'],
  ].forEach(([k, v]) => {
    const row = el('div', 'kv');
    row.appendChild(el('dt', null, k));
    row.appendChild(el('dd', null, String(v)));
    dl.appendChild(row);
  });
  dataCard.appendChild(dl);

  if (report.excluded && Object.keys(report.excluded).length) {
    const ex = el('div', 'setting-help');
    ex.style.marginTop = '8px';
    ex.textContent = 'Excluded — ' + Object.entries(report.excluded)
      .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
      .join(' · ');
    dataCard.appendChild(ex);
  }

  const refreshBtn = el('button', 'btn', 'Refresh market data');
  refreshBtn.style.marginTop = '14px';
  const progress = el('div', 'progress');
  progress.hidden = true;
  const fill = el('div', 'progress-fill');
  fill.style.width = '0%';
  progress.appendChild(fill);
  const stage = el('div', 'setting-help');

  refreshBtn.addEventListener('click', async () => {
    refreshBtn.disabled = true;
    progress.hidden = false;
    try {
      await api('/refresh', { method: 'POST' });
      pollRefresh(fill, stage, refreshBtn);
    } catch (err) {
      toast(err.message);
      refreshBtn.disabled = false;
    }
  });
  dataCard.appendChild(refreshBtn);
  dataCard.appendChild(progress);
  dataCard.appendChild(stage);
  body.appendChild(dataCard);
}

async function pollRefresh(fill, stage, btn) {
  try {
    const status = await api('/refresh');
    fill.style.width = `${(status.progress || 0) * 100}%`;
    stage.textContent = status.stage || '';
    if (status.running) {
      setTimeout(() => pollRefresh(fill, stage, btn), 1200);
      return;
    }
    btn.disabled = false;
    if (status.error) {
      stage.textContent = status.error;
      toast('Refresh failed');
    } else {
      toast('Market data updated');
      await bootstrap(true);
    }
  } catch (err) {
    btn.disabled = false;
    stage.textContent = err.message;
  }
}

async function persistSettings(patch) {
  try {
    await api('/settings', { method: 'PUT', body: JSON.stringify(patch) });
  } catch {
    /* Settings are a convenience; a failed save must never block interaction. */
  }
}

// ------------------------------------------------------------------ nav / init
function refreshActiveTab() {
  if (state.tab === 'stocks') loadStocks(true);
  else if (state.tab === 'watchlist') loadWatchlist();
  else if (state.tab === 'macro') loadMacro();
}

function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tab').forEach((b) => {
    b.setAttribute('aria-selected', String(b.dataset.tab === tab));
  });
  ['stocks', 'watchlist', 'macro', 'settings'].forEach((name) => {
    $(`#page-${name}`).hidden = name !== tab;
  });
  if (tab === 'watchlist') loadWatchlist();
  else if (tab === 'macro') loadMacro();
  else if (tab === 'settings') renderSettings();
  else if (!state.stocks.rows.length) loadStocks(true);
}

function renderMacroChips() {
  const box = $('#macro-view-chips');
  box.textContent = '';
  [['heatmap', 'Heatmap'], ['list', 'Ranked list']].forEach(([key, label]) => {
    const chip = el('button', 'chip', label);
    chip.type = 'button';
    chip.setAttribute('aria-pressed', String(state.macroView === key));
    chip.addEventListener('click', () => {
      state.macroView = key;
      renderMacroChips();
      loadMacro();
    });
    box.appendChild(chip);
  });
}

async function bootstrap(silent = false) {
  try {
    const meta = await api('/meta');
    state.meta = meta;
    state.settings = { ...state.settings, ...(meta.settings || {}) };
    renderWindowChips();
    renderSectorChips();
    renderMacroChips();

    try {
      const wl = await api('/watchlist');
      state.watchlist = new Set(wl.symbols);
    } catch { /* watchlist is optional at boot */ }

    if (!meta.ready) {
      $('#stock-list').textContent = '';
      $('#stock-list').appendChild(
        el('div', 'empty',
          'No snapshot yet.\nOpen Settings and tap “Refresh market data”, or run scripts/refresh.py.'),
      );
      return;
    }
    if (!silent) switchTab(state.tab);
    else refreshActiveTab();
  } catch (err) {
    $('#stock-list').appendChild(el('div', 'empty', err.message));
  }
}

function init() {
  try {
    const saved = localStorage.getItem('biggie-theme');
    if (saved) document.documentElement.dataset.theme = saved;
  } catch { /* private mode */ }

  document.querySelectorAll('.tab').forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
  $('#sheet-close').addEventListener('click', closeSheet);
  $('#sheet-backdrop').addEventListener('click', closeSheet);
  $('#sheet-star').addEventListener('click', toggleStar);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSheet();
  });

  let searchTimer;
  $('#stock-search').addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    const value = e.target.value;
    searchTimer = setTimeout(() => {
      state.stocks.search = value;
      loadStocks(true);
    }, 220);
  });

  // Infinite scroll rather than pagination controls - fewer taps on a phone.
  const observer = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && state.tab === 'stocks') loadStocks(false);
  }, { rootMargin: '400px' });
  observer.observe($('#stock-sentinel'));

  bootstrap();
}

document.addEventListener('DOMContentLoaded', init);
