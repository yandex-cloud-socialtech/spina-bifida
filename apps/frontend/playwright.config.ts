import {defineConfig, devices} from '@playwright/test';

const externalBaseUrl = process.env.PLAYWRIGHT_BASE_URL;

export default defineConfig({
  testDir: './e2e',
  workers: externalBaseUrl ? 1 : undefined,
  snapshotPathTemplate: '{testDir}/{testFilePath}-snapshots/{arg}-{projectName}{ext}',
  fullyParallel: true,
  forbidOnly: true,
  retries: 0,
  reporter: 'line',
  use: {
    baseURL: externalBaseUrl ?? 'http://127.0.0.1:4173',
    locale: 'ru-RU',
    timezoneId: 'Europe/Moscow',
    colorScheme: 'light',
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
  },
  expect: {
    toHaveScreenshot: {maxDiffPixelRatio: 0.01},
  },
  projects: [
    {
      name: 'chromium-1440',
      use: {...devices['Desktop Chrome'], viewport: {width: 1440, height: 900}},
    },
    {
      name: 'chromium-1024',
      use: {...devices['Desktop Chrome'], viewport: {width: 1024, height: 900}},
    },
  ],
  webServer: externalBaseUrl
    ? undefined
    : {
        command: 'npm run dev -- --host 127.0.0.1 --port 4173',
        url: 'http://127.0.0.1:4173',
        reuseExistingServer: true,
      },
});
