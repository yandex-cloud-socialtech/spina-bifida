import {QueryClient} from '@tanstack/react-query';
import {render, screen, waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import {afterAll, afterEach, beforeAll, describe, expect, it} from 'vitest';

import {apiClient} from '../../api/client';
import {App} from '../../app/App';
import {AppProviders} from '../../app/providers';

const server = setupServer(
  http.get('/api/v1/system/info', () =>
    HttpResponse.json({
      application_version: '0.1.0',
      model_versions: ['v1', 'v2'],
      default_model_version: 'v2',
      schema_version: '0003',
      device: 'cpu',
    }),
  ),
);

const doctor = {
  id: 'u1',
  email: 'doctor@example.test',
  role: 'user',
  status: 'active',
  locale: 'ru',
};

function renderApp(queryClient = new QueryClient({defaultOptions: {queries: {retry: false}}})) {
  return render(
    <AppProviders queryClient={queryClient}>
      <App />
    </AppProviders>,
  );
}

beforeAll(() => server.listen({onUnhandledRequest: 'error'}));
afterEach(() => {
  server.resetHandlers();
  apiClient.clearSession();
  window.history.replaceState({}, '', '/');
  localStorage.clear();
});
afterAll(() => server.close());

describe('authentication routes', () => {
  it('logs in with the bootstrap CSRF token and opens the terms gate', async () => {
    window.history.replaceState({}, '', '/login');
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json(
          {
            error: {code: 'authentication_required', params: {}, request_id: 'r0'},
            csrf_token: 'bootstrap-csrf',
          },
          {status: 401},
        ),
      ),
      http.post('/api/v1/auth/login', async ({request}) => {
        expect(request.headers.get('X-CSRF-Token')).toBe('bootstrap-csrf');
        expect(await request.json()).toEqual({
          email: 'doctor@example.test',
          password: 'StrongPass123',
        });
        return HttpResponse.json({
          user: doctor,
          csrf_token: 'signed-in-csrf',
          terms_accepted: false,
          terms_version: '2026-09-22',
        });
      }),
    );
    renderApp();
    const user = userEvent.setup();
    await waitFor(() => expect(apiClient.csrfToken).toBe('bootstrap-csrf'));

    await user.type(screen.getByLabelText('Электронная почта'), 'doctor@example.test');
    await user.type(screen.getByLabelText('Пароль'), 'StrongPass123');
    await user.click(screen.getByRole('button', {name: 'Войти'}));

    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    expect(screen.queryByRole('main', {name: 'Рабочая область анализа'})).toBeNull();
  });

  it('redirects an anonymous user from the workspace to login', async () => {
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json(
          {
            error: {code: 'authentication_required', params: {}, request_id: 'r1'},
            csrf_token: 'csrf-1',
          },
          {status: 401},
        ),
      ),
    );

    renderApp();

    expect(await screen.findByRole('heading', {name: 'Вход в сервис'})).toBeInTheDocument();
    expect(window.location.pathname).toBe('/login');
  });

  it('shows the workspace to a user and the admin route to an admin', async () => {
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json({user: doctor, csrf_token: 'csrf-1', terms_accepted: true, terms_version: '2026-09-22'}),
      ),
    );
    renderApp();
    expect(await screen.findByRole('main', {name: 'Рабочая область анализа'})).toBeInTheDocument();

    window.history.replaceState({}, '', '/admin');
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json({user: {...doctor, role: 'admin'}, csrf_token: 'csrf-2', terms_accepted: true, terms_version: '2026-09-22'}),
      ),
      http.get('/api/v1/admin/users', () => HttpResponse.json({users: []})),
    );
    renderApp();
    expect(await screen.findByRole('heading', {name: 'Пользователи'})).toBeInTheDocument();
  });

  it('redirects an ordinary user away from /admin', async () => {
    window.history.replaceState({}, '', '/admin');
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json({user: doctor, csrf_token: 'csrf-1', terms_accepted: true, terms_version: '2026-09-22'}),
      ),
    );

    renderApp();

    await waitFor(() => expect(window.location.pathname).toBe('/'));
    expect(await screen.findByRole('main', {name: 'Рабочая область анализа'})).toBeVisible();
  });

  it('clears cached user and analysis queries on logout', async () => {
    const queryClient = new QueryClient({defaultOptions: {queries: {retry: false}}});
    queryClient.setQueryData(['analysis', 'j1'], {id: 'j1'});
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json({user: doctor, csrf_token: 'csrf-1', terms_accepted: true, terms_version: '2026-09-22'}),
      ),
      http.post('/api/v1/auth/logout', () => new HttpResponse(null, {status: 204})),
    );
    renderApp(queryClient);
    const user = userEvent.setup();

    await user.click(await screen.findByLabelText('Профиль'));
    await user.click(screen.getByRole('button', {name: 'Выйти'}));

    await waitFor(() => expect(window.location.pathname).toBe('/login'));
    expect(queryClient.getQueryData(['analysis', 'j1'])).toBeUndefined();
    expect(queryClient.getQueryData(['current-user'])).toBeUndefined();
  });
});
