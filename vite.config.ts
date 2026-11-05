import { defineConfig } from 'vitest/config';

// `base` is relative so the built site works on GitHub Pages under /<repo>/.
export default defineConfig({
  base: './',
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true },
  test: {
    include: ['tests/**/*.test.ts'],
    testTimeout: 60_000,
  },
});
