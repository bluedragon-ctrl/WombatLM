/** Tiny DOM helper: h('div', { class: 'x' }, child, 'text'). */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K, attrs: Record<string, string | number | boolean> = {}, ...children: (Node | string | null | undefined)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false) continue;
    if (k === 'class') el.className = String(v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined) el.append(c);
  return el;
}

/** One shared floating tooltip that follows the mouse. */
let tip: HTMLDivElement | null = null;
export function showTip(e: MouseEvent, text: string | null): void {
  if (!tip) {
    tip = h('div', { class: 'tooltip', 'data-test': 'tooltip' });
    document.body.append(tip);
  }
  if (!text) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.textContent = text;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${e.clientY + 16}px`;
}
