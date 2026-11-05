/**
 * Reverse-mode automatic differentiation with a "tape".
 *
 * While a tape is active, every operation that touches a trainable tensor
 * writes down a small "backward" function: how to send the gradient from its
 * output back to its inputs. `backward(loss)` then plays the tape in reverse,
 * from the loss back to every weight. This is backpropagation.
 */
import { Tensor } from './tensor';

export type BackwardFn = () => void;

let activeTape: Tape | null = null;

export class Tape {
  private fns: BackwardFn[] = [];

  /** Run `fn` with this tape recording. Operations outside `run` are not recorded. */
  run<T>(fn: () => T): T {
    const previous = activeTape;
    activeTape = this;
    try {
      return fn();
    } finally {
      activeTape = previous;
    }
  }

  push(fn: BackwardFn): void {
    this.fns.push(fn);
  }

  get length(): number {
    return this.fns.length;
  }

  /** Send gradients from `loss` (a 1-element tensor) back through the tape, then clear it. */
  backward(loss: Tensor): void {
    if (loss.size !== 1) throw new Error('backward() needs a scalar loss');
    loss.ensureGrad()[0] = 1;
    for (let i = this.fns.length - 1; i >= 0; i--) this.fns[i]();
    this.fns = [];
  }

  /** Forget everything recorded (e.g. after an evaluation pass). */
  clear(): void {
    this.fns = [];
  }
}

/** True if an op with these inputs must be recorded. */
export function isRecording(inputs: readonly Tensor[]): boolean {
  return activeTape !== null && inputs.some((t) => t.requiresGrad);
}

/**
 * Called by every op after computing its output. If a tape is recording and
 * any input needs a gradient, the output is marked as needing one too and the
 * backward function is stored.
 */
export function recordOp(inputs: readonly Tensor[], out: Tensor, backward: BackwardFn): void {
  if (!isRecording(inputs)) return;
  out.requiresGrad = true;
  activeTape!.push(backward);
}
