import react from '@vitejs/plugin-react';
import {defineConfig} from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'jsdom',
    environmentOptions: {
      jsdom: {url: 'http://localhost/'},
    },
    setupFiles: ['./src/test/setup.ts'],
    server: {
      deps: {
        inline: [/@gravity-ui\/uikit/],
      },
    },
  },
});
