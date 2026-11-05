/**
 * UI strings. The UI language is independent of the data languages; the
 * default comes from the browser and the choice is kept in localStorage.
 */
import cs from './cs.json';
import en from './en.json';

export type Key = keyof typeof en;
const DICTS: Record<string, Record<Key, string>> = { cs, en };
export const UI_LANGUAGES = Object.keys(DICTS);
const STORE_KEY = 'wombatlm.uiLang';

function initial(): string {
  try {
    const saved = localStorage.getItem(STORE_KEY);
    if (saved && saved in DICTS) return saved;
  } catch { /* storage may be blocked */ }
  const nav = (globalThis.navigator?.language ?? 'en').slice(0, 2);
  return nav in DICTS ? nav : 'en';
}

let current = initial();

export function uiLang(): string {
  return current;
}

export function setUiLang(lang: string): void {
  if (!(lang in DICTS)) return;
  current = lang;
  try { localStorage.setItem(STORE_KEY, lang); } catch { /* ignore */ }
}

/** Translate a key, filling {placeholders}. Unknown keys fall back to English, then to the key. */
export function t(key: Key | string, vars: Record<string, string | number> = {}): string {
  const s = DICTS[current][key as Key] ?? DICTS.en[key as Key] ?? key;
  return s.replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? `{${k}}`));
}
