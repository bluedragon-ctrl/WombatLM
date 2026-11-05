/** HiDPI-aware canvas setup: size in CSS pixels, drawing in CSS pixels. */
export function prepare(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '13px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  return ctx;
}

/** Width the canvas should use: its parent's content width. */
export function parentWidth(canvas: HTMLCanvasElement, fallback = 400): number {
  const w = canvas.parentElement?.clientWidth ?? fallback;
  return Math.max(200, w || fallback);
}
