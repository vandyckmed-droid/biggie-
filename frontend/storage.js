/* Persistent local storage with graceful degradation.
 *
 * The watchlist and settings must survive closing the tab, closing the browser, and
 * coming back days later. `localStorage` does that - but it is not always available:
 * a sandboxed iframe without `allow-same-origin`, Safari private browsing, or a blocked
 * third-party context all make it throw on *access*, not just on write.
 *
 * The previous version wrapped every call in try/catch and returned a default on
 * failure, which silently degraded to "nothing ever persists" with no way to tell.
 * This picks the best backing store available once, exposes which one it got, and only
 * falls back to memory when the browser genuinely offers nothing durable.
 */
'use strict';

window.BiggieStore = (function () {
  const PREFIX = 'biggie-';

  /** Probe a Storage object by actually writing to it; merely reading can succeed
   *  where writing later throws (Safari private mode historically did exactly that). */
  function usable(factory) {
    try {
      const store = factory();
      if (!store) return null;
      const probe = `${PREFIX}__probe__`;
      store.setItem(probe, '1');
      const ok = store.getItem(probe) === '1';
      store.removeItem(probe);
      return ok ? store : null;
    } catch {
      return null;
    }
  }

  const memory = (() => {
    const map = new Map();
    return {
      getItem: (k) => (map.has(k) ? map.get(k) : null),
      setItem: (k, v) => map.set(k, String(v)),
      removeItem: (k) => map.delete(k),
    };
  })();

  const backing =
    usable(() => window.localStorage) ||
    usable(() => window.sessionStorage) ||
    memory;

  const kind =
    backing === memory
      ? 'memory'
      : backing === window.sessionStorage
        ? 'session'
        : 'local';

  /* Settings persisted by an older release can name modes that no longer exist. The
     risk denominator changed from simple/covariance to total/idiosyncratic, and a stored
     "covariance" would leave no button selected (and, against the dev API, return a 422).

     Both former modes normalise to `total`, deliberately. The old "covariance" was the
     stock's own volatility with shrinkage applied - so it *was* the total-risk
     denominator. `idiosyncratic` is a different measurement, and silently opting an
     existing user into it would change their rankings without them asking. */
  const SETTINGS_ALLOWED = {
    window: ['mom_12_1', 'mom_6_1', 'mom_3_1', 'market_cap'],
    return_mode: ['raw', 'residual'],
    risk_mode: ['total', 'idiosyncratic'],
  };
  const LEGACY_RISK_MODE = { simple: 'total', covariance: 'total' };

  function migrateSettings(saved) {
    if (!saved || typeof saved !== 'object') return null;
    const out = {};
    for (const [key, value] of Object.entries(saved)) {
      if (key === 'risk_mode' && LEGACY_RISK_MODE[value]) {
        out[key] = LEGACY_RISK_MODE[value];
        continue;
      }
      const allowed = SETTINGS_ALLOWED[key];
      // Drop unrecognised values rather than passing them on; a stale enum reaching the
      // UI shows as "nothing selected" and reaching the API as a 422.
      if (allowed && !allowed.includes(value)) continue;
      if (key === 'cluster_k') {
        const k = Number(value);
        if (Number.isFinite(k) && k >= 2 && k <= 25) out[key] = k;
        continue;
      }
      out[key] = value;
    }
    return out;
  }

  return {
    /** Which store won: 'local' persists across sessions, the others do not. */
    kind,
    durable: kind === 'local',

    get(key, fallback = null) {
      try {
        const raw = backing.getItem(PREFIX + key);
        if (raw === null) return fallback;
        const value = JSON.parse(raw);
        return key === 'settings' ? (migrateSettings(value) ?? fallback) : value;
      } catch {
        return fallback;
      }
    },

    /** Exposed for tests; the migration is otherwise applied transparently on read. */
    migrateSettings,

    set(key, value) {
      try {
        backing.setItem(PREFIX + key, JSON.stringify(value));
        return true;
      } catch {
        // Quota exhausted or the store vanished mid-session; keep the app usable.
        return false;
      }
    },

    remove(key) {
      try {
        backing.removeItem(PREFIX + key);
      } catch { /* nothing durable to clean up */ }
    },
  };
})();
