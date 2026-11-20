import {expect, test, type Page} from '@playwright/test';

const liveModels = process.env.LIVE_MODEL_E2E === '1';
const email = process.env.E2E_ADMIN_EMAIL ?? '';
const password = process.env.E2E_ADMIN_PASSWORD ?? '';

async function login(page: Page) {
  await page.goto('/login');
  await page.getByLabel('Электронная почта').fill(email);
  await page.getByLabel('Пароль').fill(password);
  await page.getByRole('button', {name: 'Войти'}).click();
  await expect(page).toHaveURL(/\/$/);
}

test.describe('live Compose model workflow', () => {
  test.skip(!liveModels, 'Set LIVE_MODEL_E2E=1 for model-backed Compose analysis');
  test.skip(!email || !password, 'Set E2E_ADMIN_EMAIL and E2E_ADMIN_PASSWORD');

  test('runs examples through queue, overlay viewer and feedback', async ({page}) => {
    await login(page);
    await page.getByRole('button', {name: 'Открыть примеры'}).click();
    await expect(page.getByText('4 из 10')).toBeVisible();
    await page.getByRole('button', {name: 'Запустить анализ'}).click();

    const terminal = page.locator(
      'text=/Анализ завершён|Недостаточное качество|Техническая ошибка/',
    );
    await expect(terminal.first()).toBeVisible({timeout: 360_000});

    if (await page.getByRole('heading', {name: 'Плоскость'}).isVisible()) {
      await expect(page.getByRole('img', {name: 'Оригинал'})).toBeVisible();
      await expect(page.getByRole('button', {name: 'ROI'})).toHaveAttribute('aria-pressed', 'true');
      await expect(page.getByRole('button', {name: 'Heatmap'})).toHaveAttribute('aria-pressed', 'true');
      await page.getByRole('radio', {name: 'Результат полезен'}).check();
      await page.getByRole('button', {name: 'Отправить'}).click();
      await expect(page.getByRole('status')).toContainText('Обратная связь отправлена');
    }
  });
});
