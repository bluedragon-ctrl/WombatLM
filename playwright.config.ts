import { defineConfig } from '@playwright/test';

// Uses the pre-installed Chromium (Playwright is pinned to match it).
export default defineConfig({
  testDir: 'e2e',
  timeout: 60_000,
  use: { baseURL: 'http://localhost:4173', viewport: { width: 1400, height: 900 } },
  webServer: { command: 'npm run build && npx vite preview --port 4173 --strictPort', url: 'http://localhost:4173', reuseExistingServer: true, timeout: 120_000 },
});
