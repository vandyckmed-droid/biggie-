/* Browser smoke test for the built static site.
 *
 * The macro regression that shipped was valid JavaScript reading field names the
 * pipeline no longer emitted. Nothing in the Python suite, `node --check`, or the site
 * build could see it - only rendering the page and reading the pixels can.
 *
 * So this drives the real site at iPhone size and asserts on *rendered text*, not on
 * element counts: a heatmap with 21 cells that all read "—" passes a count check and
 * fails a user.
 *
 *   node scripts/smoke_test.js http://127.0.0.1:8000/
 */
'use strict';

const { chromium } = require('playwright');

const URL = process.argv[2] || 'http://127.0.0.1:8000/';
const VIEWPORT = { width: 390, height: 844 };

const failures = [];
const check = (ok, message) => {
  if (!ok) failures.push(message);
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${message}`);
};

/** Does this cell text contain a real signed number, rather than an em dash? */
const hasNumber = (text) => /[-+]?\d+\.\d+/.test(text || '');

async function run(page, theme) {
  console.log(`\n--- ${theme} ---`);
  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#stock-list .row', { timeout: 30000 });
  await page.waitForTimeout(800);

  // ---- Stocks
  const rows = await page.locator('#stock-list .row').count();
  check(rows > 0, `stocks list renders (${rows} rows)`);

  const firstScore = await page.locator('#stock-list .metric-val').first().innerText();
  check(hasNumber(firstScore), `stock score is a number ("${firstScore}")`);

  const rank = await page.locator('#stock-list .row-rank').first().innerText();
  check(/^\d+$/.test(rank.trim()), `stock rank is populated ("${rank.trim()}")`);

  const note = await page.locator('#window-note').innerText();
  check(/skipping the most recent \d+ days/.test(note), `window note states the skip ("${note}")`);

  // ---- Direct watchlist selection
  const pick = page.locator('#stock-list .row').first().locator('.row-pick');
  const box = await pick.boundingBox();
  check(box && box.width >= 40 && box.height >= 40,
    `watchlist control is a full touch target (${Math.round(box?.width)}x${Math.round(box?.height)})`);
  await pick.click();
  await page.waitForTimeout(300);
  check((await pick.getAttribute('aria-pressed')) === 'true', 'tapping + selects the stock');
  const stored = await page.evaluate(() => window.BiggieStore.get('watchlist'));
  check(Array.isArray(stored) && stored.length === 1, `selection persisted (${JSON.stringify(stored)})`);

  // ---- Detail sheet
  await page.locator('#stock-list .row-open').first().click();
  await page.waitForTimeout(900);
  const charts = await page.locator('.mini').count();
  check(charts === 6, `detail sheet renders 6 return charts (got ${charts})`);
  const sheetScore = await page.locator('#sheet-body .stat-value').first().innerText();
  check(hasNumber(sheetScore), `detail sheet score is a number ("${sheetScore}")`);
  await page.locator('#sheet-close').click();
  await page.waitForTimeout(300);
  check(await page.locator('#sheet').evaluate((n) => n.hasAttribute('inert')),
    'closed sheet leaves the accessibility tree');

  // ---- Macro: the exact regression this test exists for
  await page.locator('.tab[data-tab=macro]').click();
  await page.waitForTimeout(1200);
  const cells = await page.locator('.heat-cell').allInnerTexts();
  check(cells.length > 0, `macro heatmap renders (${cells.length} cells)`);
  const blank = cells.filter((t) => !hasNumber(t));
  check(blank.length === 0, `every heatmap cell shows a score (${blank.length} blank)`);

  const regime = await page.locator('.regime-badge').innerText();
  check(/Risk-On|Risk-Off|Transition/.test(regime), `regime is one of the three states ("${regime}")`);

  await page.locator('#macro-view-chips .chip', { hasText: 'Ranked list' }).click();
  await page.waitForTimeout(700);
  const macroScores = await page.locator('#macro-body .metric-val').allInnerTexts();
  check(macroScores.length > 0 && macroScores.every(hasNumber),
    `macro ranked list scores are numbers (${macroScores.length} rows)`);

  await page.locator('#macro-view-chips .chip', { hasText: 'Sectors' }).click();
  await page.waitForTimeout(700);
  check(await page.locator('.sector-row').count() > 0, 'sector exposure renders');

  // ---- Watchlist + HRP
  await page.locator('.tab[data-tab=watchlist]').click();
  await page.waitForTimeout(2500);
  check(await page.locator('#watchlist-body .row').count() > 0, 'watchlist shows the selection');

  // ---- Settings
  await page.locator('.tab[data-tab=settings]').click();
  await page.waitForTimeout(600);
  const pressed = await page.locator('.seg button[aria-pressed=true]').count();
  check(pressed >= 3, `every settings group has a selected option (${pressed})`);
  const settingsText = await page.locator('#settings-body').innerText();
  check(!/Refresh market data/.test(settingsText),
    'static build does not offer a refresh button that cannot work');
}

(async () => {
  // Honour an explicit browser path when one is provided (this repo's dev container
  // ships Chromium at a fixed location rather than in Playwright's own cache).
  const launchOptions = process.env.CHROMIUM_PATH
    ? { executablePath: process.env.CHROMIUM_PATH }
    : {};
  const browser = await chromium.launch(launchOptions);
  const errors = [];

  for (const theme of ['dark', 'light']) {
    const context = await browser.newContext({
      viewport: VIEWPORT, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
      colorScheme: theme,
    });
    const page = await context.newPage();
    page.on('pageerror', (e) => errors.push(`${theme}: ${e.message}`));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(`${theme} console: ${m.text().slice(0, 200)}`);
    });
    await run(page, theme);
    await context.close();
  }

  await browser.close();

  console.log('');
  if (errors.length) {
    console.log('JavaScript errors:');
    errors.forEach((e) => console.log(`  ${e}`));
    failures.push(`${errors.length} JavaScript error(s)`);
  }

  if (failures.length) {
    console.error(`\n${failures.length} check(s) failed:`);
    failures.forEach((f) => console.error(`  - ${f}`));
    process.exit(1);
  }
  console.log('All smoke checks passed.');
})();
