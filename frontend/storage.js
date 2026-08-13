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

  return {
    /** Which store won: 'local' persists across sessions, the others do not. */
    kind,
    durable: kind === 'local',

    get(key, fallback = null) {
      try {
        const raw = backing.getItem(PREFIX + key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch {
        return fallback;
      }
    },

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
