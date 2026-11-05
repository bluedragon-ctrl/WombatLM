/**
 * Saving and loading weights, and growing the vocabulary of a trained model.
 *
 * Binary layout:
 *   "ZVAI" (4 bytes) | header length (uint32 LE) | header JSON (UTF-8) |
 *   padding to a multiple of 4 | each parameter's data, each padded to 4 bytes.
 * Data encodings: f32, f16 (IEEE half) or int8 with one scale per matrix.
 */
import { Rng } from './rng';
import { Tensor } from './tensor';
import { ParamSet } from './transformer';

export type Encoding = 'f32' | 'f16' | 'int8';

export interface ParamHeader {
  name: string;
  shape: number[];
  scale?: number;
}

export interface WeightsHeader {
  format: 1;
  encoding: Encoding;
  /** Free-form metadata (model kind, config, vocabulary tokens, step…). */
  meta: Record<string, unknown>;
  params: ParamHeader[];
}

const MAGIC = 'ZVAI';

// ---- half-precision helpers ------------------------------------------------

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

export function toHalf(value: number): number {
  f32buf[0] = value;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / nan
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign; // too small -> 0
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    if ((mant >>> (shift - 1)) & 1) half++; // round
    return sign | half;
  }
  let half = sign | (e << 10) | (mant >>> 13);
  if (mant & 0x1000) half++; // round to nearest
  return half;
}

export function fromHalf(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (mant / 1024);
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

// ---- encode / decode ---------------------------------------------------------

const pad4 = (n: number) => (n + 3) & ~3;

export function encodeWeights(params: readonly Tensor[], meta: Record<string, unknown>, encoding: Encoding = 'f32'): ArrayBuffer {
  const headers: ParamHeader[] = params.map((p) => {
    const h: ParamHeader = { name: p.name, shape: p.shape.slice() };
    if (encoding === 'int8') {
      let max = 0;
      for (let i = 0; i < p.size; i++) max = Math.max(max, Math.abs(p.data[i]));
      h.scale = max > 0 ? max / 127 : 1;
    }
    return h;
  });
  const header: WeightsHeader = { format: 1, encoding, meta, params: headers };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const bytesPer = encoding === 'f32' ? 4 : encoding === 'f16' ? 2 : 1;
  let total = pad4(8 + headerBytes.length);
  for (const p of params) total += pad4(p.size * bytesPer);
  const buf = new ArrayBuffer(total);
  const u8 = new Uint8Array(buf);
  const view = new DataView(buf);
  for (let i = 0; i < 4; i++) u8[i] = MAGIC.charCodeAt(i);
  view.setUint32(4, headerBytes.length, true);
  u8.set(headerBytes, 8);
  let off = pad4(8 + headerBytes.length);
  params.forEach((p, idx) => {
    for (let i = 0; i < p.size; i++) {
      const v = p.data[i];
      if (encoding === 'f32') view.setFloat32(off + i * 4, v, true);
      else if (encoding === 'f16') view.setUint16(off + i * 2, toHalf(v), true);
      else view.setInt8(off + i, Math.max(-127, Math.min(127, Math.round(v / headers[idx].scale!))));
    }
    off += pad4(p.size * bytesPer);
  });
  return buf;
}

export interface DecodedWeights {
  header: WeightsHeader;
  /** Parameter name -> values (always float32 after decoding). */
  tensors: Map<string, Float32Array>;
}

export function decodeWeights(buf: ArrayBuffer): DecodedWeights {
  const u8 = new Uint8Array(buf);
  const view = new DataView(buf);
  const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (magic !== MAGIC) throw new Error('Not a weights file');
  const hLen = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hLen))) as WeightsHeader;
  const bytesPer = header.encoding === 'f32' ? 4 : header.encoding === 'f16' ? 2 : 1;
  let off = pad4(8 + hLen);
  const tensors = new Map<string, Float32Array>();
  for (const p of header.params) {
    const n = p.shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if (header.encoding === 'f32') data[i] = view.getFloat32(off + i * 4, true);
      else if (header.encoding === 'f16') data[i] = fromHalf(view.getUint16(off + i * 2, true));
      else data[i] = view.getInt8(off + i) * p.scale!;
    }
    tensors.set(p.name, data);
    off += pad4(n * bytesPer);
  }
  return { header, tensors };
}

/** Copy decoded weights into a model. Every parameter must be present with the right size. */
export function loadInto(ps: ParamSet, tensors: Map<string, Float32Array>): void {
  for (const t of ps.list) {
    const src = tensors.get(t.name);
    if (!src) throw new Error(`Missing weights for ${t.name}`);
    if (src.length !== t.size) throw new Error(`Wrong size for ${t.name}: ${src.length} vs ${t.size}`);
    t.data.set(src);
  }
}

// ---- vocabulary growth -----------------------------------------------------

/**
 * Re-index the vocabulary-sized parameters of a model from `oldTokens` to
 * `newTokens`, in place. Rows are matched by token text, so the order may
 * change. Tokens that are new get small random rows (or zero for 1-D
 * parameters such as an output bias). Every other weight is untouched, so the
 * model keeps what it knows.
 *
 * `vocabParams` lists the parameters whose first (or only) dimension is the
 * vocabulary, e.g. the embedding table [V, d] and the output bias [V].
 */
export function growVocab(
  vocabParams: readonly Tensor[], oldTokens: readonly string[], newTokens: readonly string[], rng: Rng, initStd = 0.02,
): void {
  const oldIndex = new Map(oldTokens.map((t, i) => [t, i]));
  const missing = oldTokens.filter((t) => !newTokens.includes(t));
  if (missing.length) throw new Error(`New vocabulary drops tokens: ${missing.join(' ')}`);
  for (const p of vocabParams) {
    if (p.shape[0] !== oldTokens.length) throw new Error(`${p.name} is not vocabulary-sized`);
    const rowSize = p.size / p.shape[0];
    const data = new Float32Array(newTokens.length * rowSize);
    newTokens.forEach((tok, i) => {
      const o = oldIndex.get(tok);
      if (o !== undefined) data.set(p.data.subarray(o * rowSize, (o + 1) * rowSize), i * rowSize);
      else if (p.shape.length > 1) for (let j = 0; j < rowSize; j++) data[i * rowSize + j] = rng.normal() * initStd;
    });
    p.data = data;
    p.shape = [newTokens.length, ...p.shape.slice(1)];
    p.grad = null;
  }
}
