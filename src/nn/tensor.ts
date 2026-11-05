/**
 * A Tensor is a block of numbers with a shape, e.g. [3, 4] is a 3×4 table.
 * Numbers are stored row-major in one flat Float32Array.
 *
 * `grad` holds "how much the loss would change if this number went up a bit".
 * It is filled in by `Tape.backward` (see autograd.ts).
 */
export class Tensor {
  data: Float32Array;
  shape: number[];
  grad: Float32Array | null = null;
  requiresGrad: boolean;
  /** Optional human name, used for parameters ("enc0.attn.Wq"). */
  name: string;

  constructor(data: Float32Array, shape: number[], requiresGrad = false, name = '') {
    if (data.length !== numel(shape)) {
      throw new Error(`Tensor data length ${data.length} does not match shape [${shape}]`);
    }
    this.data = data;
    this.shape = shape;
    this.requiresGrad = requiresGrad;
    this.name = name;
  }

  get size(): number {
    return this.data.length;
  }

  /** Allocate the gradient buffer if needed and return it. */
  ensureGrad(): Float32Array {
    if (!this.grad || this.grad.length !== this.data.length) {
      this.grad = new Float32Array(this.data.length);
    }
    return this.grad;
  }

  zeroGrad(): void {
    if (this.grad) this.grad.fill(0);
  }

  static zeros(shape: number[], requiresGrad = false, name = ''): Tensor {
    return new Tensor(new Float32Array(numel(shape)), shape, requiresGrad, name);
  }

  static from(values: ArrayLike<number>, shape: number[], requiresGrad = false): Tensor {
    return new Tensor(Float32Array.from(values), shape, requiresGrad);
  }

  /** The single value of a 1-element tensor (e.g. a loss). */
  item(): number {
    if (this.data.length !== 1) throw new Error('item() needs a tensor with one element');
    return this.data[0];
  }
}

export function numel(shape: readonly number[]): number {
  let n = 1;
  for (const s of shape) n *= s;
  return n;
}

export function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export function assertShape(t: Tensor, rank: number, what: string): void {
  if (t.shape.length !== rank) {
    throw new Error(`${what}: expected rank ${rank}, got shape [${t.shape}]`);
  }
}
