import {ThemeProvider} from '@gravity-ui/uikit';
import {QueryClient, QueryClientProvider} from '@tanstack/react-query';
import {render, screen, waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import {afterAll, afterEach, beforeAll, describe, expect, it} from 'vitest';

import {apiClient} from '../../api/client';
import type {AuthResponse, UserDto} from '../../api/contracts';
import {currentUserQueryKey} from './useCurrentUser';
import {TermsGate} from './TermsGate';

const server = setupServer();
const doctor: UserDto = {
  id: 'u1',
  email: 'doctor@example.test',
  role: 'user',
  status: 'active',
  locale: 'ru',
};

function auth(termsAccepted: boolean): AuthResponse {
  return {
    user: doctor,
    csrf_token: 'csrf-1',
    terms_accepted: termsAccepted,
    terms_version: '2026-09-22',
  };
}

function renderGate(initialAuth: AuthResponse) {
  const queryClient = new QueryClient({defaultOptions: {queries: {retry: false}}});
  queryClient.setQueryData(currentUserQueryKey, initialAuth);
  apiClient.csrfToken = initialAuth.csrf_token;
  const view = render(
    <ThemeProvider theme="light">
      <QueryClientProvider client={queryClient}>
        <TermsGate auth={initialAuth}>
          <div>Рабочая область</div>
        </TermsGate>
      </QueryClientProvider>
    </ThemeProvider>,
  );
  return {queryClient, ...view};
}

beforeAll(() => server.listen({onUnhandledRequest: 'error'}));
afterEach(() => {
  server.resetHandlers();
  apiClient.clearSession();
});
afterAll(() => server.close());

describe('TermsGate', () => {
  it('blocks the application with all terms until the user accepts them', async () => {
    renderGate(auth(false));
    const user = userEvent.setup();

    expect(screen.queryByText('Рабочая область')).toBeNull();
    expect(screen.getByRole('heading', {name: '1. Назначение сервиса'})).toBeVisible();
    expect(screen.getByRole('heading', {name: '2. Целевая аудитория'})).toBeVisible();
    expect(screen.getByRole('heading', {name: '3. Ограничение ответственности'})).toBeVisible();
    expect(screen.getByRole('heading', {name: '4. Обработка данных'})).toBeVisible();
    expect(screen.getByRole('heading', {name: '5. Согласие пользователя'})).toBeVisible();
    expect(screen.getByRole('button', {name: /соглашаюсь/i})).toHaveFocus();

    await user.keyboard('{Escape}');
    expect(screen.getByRole('dialog')).toBeVisible();
    expect(screen.queryByRole('button', {name: 'Закрыть'})).toBeNull();
  });

  it('keeps the gate open after a failed acceptance request', async () => {
    server.use(
      http.post('/api/v1/auth/terms/accept', () =>
        HttpResponse.json(
          {error: {code: 'network_error', params: {}, request_id: 'terms-failure'}},
          {status: 503},
        ),
      ),
    );
    renderGate(auth(false));
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', {name: /соглашаюсь/i}));

    expect(await screen.findByRole('alert')).toBeVisible();
    expect(screen.getByRole('dialog')).toBeVisible();
    expect(screen.queryByText('Рабочая область')).toBeNull();
  });

  it('updates the current-user cache and unlocks the current session after acceptance', async () => {
    server.use(
      http.post('/api/v1/auth/terms/accept', ({request}) => {
        expect(request.headers.get('X-CSRF-Token')).toBe('csrf-1');
        return HttpResponse.json({terms_accepted: true, terms_version: '2026-09-22'});
      }),
    );
    const {queryClient} = renderGate(auth(false));
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', {name: /соглашаюсь/i}));

    expect(await screen.findByText('Рабочая область')).toBeVisible();
    await waitFor(() =>
      expect(queryClient.getQueryData<AuthResponse>(currentUserQueryKey)).toMatchObject({
        terms_accepted: true,
        terms_version: '2026-09-22',
      }),
    );
  });

  it('uses the auth response on reload and reopens for a new unaccepted login', () => {
    const {rerender} = renderGate(auth(true));
    expect(screen.getByText('Рабочая область')).toBeVisible();

    rerender(
      <ThemeProvider theme="light">
        <QueryClientProvider client={new QueryClient()}>
          <TermsGate auth={auth(false)}>
            <div>Рабочая область</div>
          </TermsGate>
        </QueryClientProvider>
      </ThemeProvider>,
    );

    expect(screen.getByRole('dialog')).toBeVisible();
    expect(screen.queryByText('Рабочая область')).toBeNull();
  });
});
