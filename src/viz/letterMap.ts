/**
 * Letter map: every letter (and task token) is a row of the letter table
 * (32 numbers). PCA squeezes those 32 numbers into the two directions with
 * the most variation, so letters the model treats alike end up close.
 *
 * Each new frame is rotated / mirrored to match the previous one
 * (Procrustes) and the points glide there, so the map never jumps.
 *
 * Clicking a letter selects it: lines go to its closest letters, measured on
 * all 32 numbers (the flat map keeps only part of the differences).
 */
import { alignTo } from '../nn/pca';
import { prepare } from './canvas';
import { GRID, INK, LETTER_KINDS, LetterKind } from './colors';

export interface LetterPoint {
  label: string;
  kind: LetterKind;
  /** Full token, shown on hover. */
  token: string;
  /** How often it appears in the learning data (-1: not counted). */
  count?: number;
}

/** Category of a token for colour and marker. */
export function letterKind(token: string): LetterKind {
  if (token.startsWith('<')) return token === '<eos>' ? 'other' : 'task';
  const nfd = token.normalize('NFD');
  if (nfd.length > 1) return 'accented';
  if ('aeiouy'.includes(token)) return 'vowel';
  if (/^[a-z]$/.test(token)) return 'consonant';
  return 'other';
}

export function letterLabel(token: string): string {
  if (token === '<eos>') return '⏎';
  if (token === ' ') return '␣';
  if (token.startsWith('<')) return token.slice(1, -1);
  return token;
}

const PAD = 22;

export class LetterMapView {
  private points: LetterPoint[] = [];
  private target: Float32Array | null = null;
  private shown: Float32Array | null = null;
  private view = { x0: -1, x1: 1, y0: -1, y1: 1 };
  private viewTarget = { x0: -1, x1: 1, y0: -1, y1: 1 };
  private raf = 0;
  private hover: number | null = null;
  private near: number[][] = [];
  /** Token of the clicked letter, kept across frames. */
  private selected: string | null = null;
  /** Called when the selection or its closest letters change. */
  onSelect: (sel: { point: LetterPoint; near: LetterPoint[] } | null) => void = () => {};

  constructor(private canvas: HTMLCanvasElement, private size: () => number) {
    canvas.addEventListener('mousemove', (e) => {
      const i = this.pointAt(e.offsetX, e.offsetY);
      if (i !== this.hover) { this.hover = i; this.draw(); }
      canvas.title = i === null ? '' : this.points[i].token;
      canvas.style.cursor = i !== null && this.points[i].kind !== 'task' ? 'pointer' : '';
    });
    canvas.addEventListener('mouseleave', () => { this.hover = null; this.draw(); });
    canvas.addEventListener('click', (e) => {
      const i = this.pointAt(e.offsetX, e.offsetY);
      const token = i === null || this.points[i].kind === 'task' ? null : this.points[i].token;
      this.select(token === this.selected ? null : token);
    });
    // Keyboard: focus the map and type a letter to select it; Escape clears.
    canvas.tabIndex = 0;
    canvas.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { this.select(null); return; }
      const p = this.points.find((q) => q.kind !== 'task' && q.token === e.key.toLowerCase());
      if (p) { this.select(p.token); e.preventDefault(); }
    });
  }

  /** Select a letter by token (null clears). Task tokens cannot be selected. */
  select(token: string | null): void {
    this.selected = token;
    this.draw();
    this.emit();
  }

  private selectedIndex(): number | null {
    if (this.selected === null) return null;
    const i = this.points.findIndex((p) => p.token === this.selected);
    return i < 0 || this.points[i].kind === 'task' ? null : i;
  }

  private emit(): void {
    const i = this.selectedIndex();
    this.onSelect(i === null ? null : { point: this.points[i], near: (this.near[i] ?? []).map((j) => this.points[j]) });
  }

  /** Forget the previous frame (new model): the next update appears without gliding. */
  reset(): void {
    this.target = this.shown = null;
  }

  update(points: LetterPoint[], coords: Float32Array, near: number[][] = []): void {
    const same = this.target && this.target.length === coords.length && points.every((p, i) => p.token === this.points[i]?.token);
    this.points = points;
    this.near = near;
    if (this.selected !== null && this.selectedIndex() === null) this.selected = null;
    this.emit();
    this.target = same ? alignTo(this.target!, coords) : coords;
    if (!same || !this.shown) {
      this.shown = this.target.slice();
      this.viewTarget = bounds(this.target);
      this.view = { ...this.viewTarget };
    } else {
      this.viewTarget = bounds(this.target);
    }
    this.animate();
  }

  private animate(): void {
    if (this.raf) return;
    const tick = () => {
      this.raf = 0;
      if (!this.shown || !this.target) return;
      let moving = false;
      const range = Math.max(this.view.x1 - this.view.x0, this.view.y1 - this.view.y0);
      for (let i = 0; i < this.shown.length; i++) {
        const d = this.target[i] - this.shown[i];
        if (Math.abs(d) > range * 1e-3) { this.shown[i] += d * 0.18; moving = true; } else this.shown[i] = this.target[i];
      }
      for (const k of ['x0', 'x1', 'y0', 'y1'] as const) {
        const d = this.viewTarget[k] - this.view[k];
        if (Math.abs(d) > range * 1e-3) { this.view[k] += d * 0.18; moving = true; } else this.view[k] = this.viewTarget[k];
      }
      this.draw();
      if (moving) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private toScreen(i: number, s: number): [number, number] {
    const { x0, x1, y0, y1 } = this.view;
    const span = Math.max(x1 - x0, y1 - y0) || 1;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const k = (s - 2 * PAD) / span;
    return [s / 2 + (this.shown![2 * i] - cx) * k, s / 2 - (this.shown![2 * i + 1] - cy) * k];
  }

  private pointAt(x: number, y: number): number | null {
    if (!this.shown) return null;
    const s = this.size();
    let best: number | null = null, bestD = 14 * 14;
    for (let i = 0; i < this.points.length; i++) {
      const [px, py] = this.toScreen(i, s);
      const d = (px - x) ** 2 + (py - y) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  draw(): void {
    const s = this.size();
    const ctx = prepare(this.canvas, s, s);
    ctx.strokeStyle = GRID;
    ctx.strokeRect(0.5, 0.5, s - 1, s - 1);
    if (!this.shown) return;
    ctx.textAlign = 'center';
    const sel = this.selectedIndex();
    const near = new Set(sel === null ? [] : this.near[sel] ?? []);
    // Lines from the selected letter to its closest letters, thicker for closer ones.
    if (sel !== null) {
      const [sx, sy] = this.toScreen(sel, s);
      (this.near[sel] ?? []).forEach((j, rank) => {
        const [x, y] = this.toScreen(j, s);
        ctx.strokeStyle = SELECT;
        ctx.lineWidth = 3 - rank * 0.5;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(x, y); ctx.stroke();
      });
    }
    // Task tokens first, so the letters sit on top; the selection last.
    const rank = (i: number) => (i === sel ? 3 : near.has(i) ? 2 : this.points[i].kind === 'task' ? 0 : 1);
    const order = this.points.map((_, i) => i).sort((a, b) => rank(a) - rank(b));
    for (const i of order) {
      const p = this.points[i];
      const [x, y] = this.toScreen(i, s);
      const style = LETTER_KINDS[p.kind];
      const hot = i === this.hover || i === sel;
      const r = p.kind === 'task' ? 5 : 10;
      // With a selection, everything else fades so the closest letters stand out.
      ctx.globalAlpha = sel === null || i === sel || near.has(i) ? 1 : 0.3;
      if (i === sel || near.has(i)) {
        ctx.fillStyle = '#fff';
        marker(ctx, 'circle', x, y, r + 4, false);
        ctx.strokeStyle = SELECT;
        ctx.lineWidth = i === sel ? 3 : 1.5;
        ctx.stroke();
      }
      ctx.fillStyle = style.color + (hot ? '66' : '22');
      ctx.strokeStyle = style.color;
      ctx.lineWidth = hot ? 2.5 : 1.5;
      marker(ctx, style.shape, x, y, r);
      if (p.kind === 'task') {
        ctx.fillStyle = style.color;
        ctx.font = `${hot ? 'bold ' : ''}11px system-ui, sans-serif`;
        ctx.textAlign = 'left';
        ctx.fillText(p.label, x + 7, y);
        ctx.textAlign = 'center';
      } else {
        ctx.fillStyle = INK;
        ctx.font = `${hot ? 'bold ' : ''}14px system-ui, sans-serif`;
        ctx.fillText(p.label, x, y + 1);
      }
    }
    ctx.globalAlpha = 1;
  }
}

function bounds(c: Float32Array): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < c.length; i += 2) {
    x0 = Math.min(x0, c[i]); x1 = Math.max(x1, c[i]);
    y0 = Math.min(y0, c[i + 1]); y1 = Math.max(y1, c[i + 1]);
  }
  return Number.isFinite(x0) ? { x0, x1, y0, y1 } : { x0: -1, x1: 1, y0: -1, y1: 1 };
}

/** Ring and lines around the clicked letter and its closest letters. */
const SELECT = '#0f172a';

export function marker(ctx: CanvasRenderingContext2D, shape: string, x: number, y: number, r: number, stroke = true): void {
  ctx.beginPath();
  switch (shape) {
    case 'circle': ctx.arc(x, y, r, 0, Math.PI * 2); break;
    case 'square': ctx.rect(x - r * 0.88, y - r * 0.88, r * 1.76, r * 1.76); break;
    case 'diamond': ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.15, y); ctx.lineTo(x, y + r * 1.15); ctx.lineTo(x - r * 1.15, y); ctx.closePath(); break;
    case 'triangle': ctx.moveTo(x, y - r * 1.1); ctx.lineTo(x + r, y + r * 0.8); ctx.lineTo(x - r, y + r * 0.8); ctx.closePath(); break;
    case 'star':
      for (let k = 0; k < 10; k++) {
        const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r * 0.45 : r;
        if (k === 0) ctx.moveTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); else ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
      }
      ctx.closePath();
      break;
  }
  ctx.fill();
  if (stroke) ctx.stroke();
}
