/** A small line chart used by the loss chart and the scoreboard. */
import { prepare } from './canvas';
import { GRID, MUTED } from './colors';

export interface Series {
  points: [number, number][];
  color: string;
  dash?: readonly number[];
  width?: number;
}

export interface ChartOptions {
  width: number;
  height: number;
  yMin: number;
  yMax: number;
  /** Format y-axis labels. */
  yLabel?: (v: number) => string;
  yTicks?: number[];
  log?: boolean;
}

export function drawLineChart(canvas: HTMLCanvasElement, series: Series[], o: ChartOptions): void {
  const ctx = prepare(canvas, o.width, o.height);
  const left = 40, right = 8, top = 8, bottom = 20;
  const W = o.width - left - right, H = o.height - top - bottom;
  const xMax = Math.max(1, ...series.flatMap((s) => (s.points.length ? [s.points[s.points.length - 1][0]] : [])));
  const fy = (v: number) => (o.log ? Math.log10(Math.max(v, 1e-4)) : v);
  const y0 = fy(o.yMin), y1 = fy(o.yMax);
  const X = (x: number) => left + (x / xMax) * W;
  const Y = (y: number) => top + H - ((Math.min(Math.max(fy(y), y0), y1) - y0) / (y1 - y0 || 1)) * H;

  ctx.strokeStyle = GRID;
  ctx.fillStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  for (const v of o.yTicks ?? [o.yMin, (o.yMin + o.yMax) / 2, o.yMax]) {
    ctx.beginPath();
    ctx.moveTo(left, Y(v));
    ctx.lineTo(left + W, Y(v));
    ctx.stroke();
    ctx.fillText(o.yLabel ? o.yLabel(v) : String(v), left - 4, Y(v));
  }
  ctx.textAlign = 'left';
  ctx.fillText('0', left, top + H + 11);
  ctx.textAlign = 'right';
  ctx.fillText(String(Math.round(xMax)), left + W, top + H + 11);

  for (const s of series) {
    if (s.points.length === 0) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width ?? 2;
    ctx.setLineDash(s.dash ? [...s.dash] : []);
    ctx.beginPath();
    s.points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y))));
    if (s.points.length === 1) ctx.lineTo(X(s.points[0][0]) + 2, Y(s.points[0][1]));
    ctx.stroke();
  }
  ctx.setLineDash([]);
}
