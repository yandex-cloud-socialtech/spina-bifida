import {expect, test, type Page} from '@playwright/test';

const live = process.env.LIVE_E2E === '1';
const adminEmail = process.env.E2E_ADMIN_EMAIL ?? '';
const adminPassword = process.env.E2E_ADMIN_PASSWORD ?? '';

async function login(page: Page) {
  await page.goto('/login');
  await page.getByLabel('Электронная почта').fill(adminEmail);
  await page.getByLabel('Пароль').fill(adminPassword);
  await page.getByRole('button', {name: 'Войти'}).click();
  await expect(page).toHaveURL(/\/$/);
}

test.describe('live Compose administration', () => {
  test.skip(!live, 'Set LIVE_E2E=1 and PLAYWRIGHT_BASE_URL for Compose browser tests');
  test.skip(!adminEmail || !adminPassword, 'Set E2E_ADMIN_EMAIL and E2E_ADMIN_PASSWORD');
  test.describe.configure({mode: 'serial'});

  test('logs in, manages a user, switches all locales and logs out', async ({page}, testInfo) => {
    const suffix = `${testInfo.project.name}-${Date.now()}`;
    const userEmail = `e2e-${suffix}@example.test`;
    await login(page);

    await page.goto('/admin');
    await expect(page.getByRole('heading', {name: 'Пользователи', level: 1})).toBeVisible();
    await page.getByLabel('Электронная почта пользователя').fill(userEmail);
    await page.getByLabel('Временный пароль').fill('E2eStrongPass123');
    await page.getByLabel('Роль', {exact: true}).selectOption('user');
    await page.locator('#admin-user-locale').selectOption('kk');
    await page.getByRole('button', {name: 'Добавить пользователя'}).click();
    await expect(page.getByRole('status')).toContainText('Пользователь добавлен');
    await expect(page.getByText(userEmail)).toBeVisible();

    await page.getByRole('button', {name: `Заблокировать ${userEmail}`}).click();
    await expect(page.getByRole('button', {name: `Разблокировать ${userEmail}`})).toBeVisible();
    await page.getByRole('button', {name: `Разблокировать ${userEmail}`}).click();
    await expect(page.getByRole('button', {name: `Заблокировать ${userEmail}`})).toBeVisible();

    await page.getByRole('button', {name: `Удалить ${userEmail}`}).click();
    await page.getByRole('dialog').getByRole('button', {name: 'Удалить пользователя'}).click();
    await expect(page.getByRole('status')).toContainText('Пользователь удалён');

    const locale = page.locator('.locale-control select');
    for (const value of ['en', 'kk', 'uz', 'ru']) {
      await locale.selectOption(value);
      await expect(locale).toHaveValue(value);
    }

    await page.locator('.profile-menu summary').click();
    await page.getByRole('button', {name: 'Выйти'}).click();
    await expect(page).toHaveURL(/\/login$/);
  });
});
