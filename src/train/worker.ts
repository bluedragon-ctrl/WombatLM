/// <reference lib="webworker" />
/**
 * Web Worker entry: owns the model and the training loop (see engine.ts).
 */
import { Engine } from './engine';
import type { ToWorker } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

// Zero-delay yields via a MessageChannel: setTimeout(0) is clamped to ≥ 4 ms
// when nested, which would waste a quarter of the time at full speed.
const channel = new MessageChannel();
const queue: (() => void)[] = [];
channel.port1.onmessage = () => queue.shift()?.();

const engine = new Engine({
  post: (msg, transfer = []) => self.postMessage(msg, transfer),
  schedule: (fn, ms) => {
    if (ms <= 0) { queue.push(fn); channel.port2.postMessage(0); } else setTimeout(fn, ms);
  },
  now: () => performance.now(),
});

self.onmessage = (e: MessageEvent<ToWorker>) => engine.handle(e.data);
