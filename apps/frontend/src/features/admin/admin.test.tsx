import {QueryClient} from '@tanstack/react-query';
import {render, screen, waitFor, within} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {delay, http, HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import {afterAll, afterEach, beforeAll, describe, expect, it, vi} from 'vitest';

import {apiClient} from '../../api/client';
import type {UserDto} from '../../api/contracts';
import {App} from '../../app/App';
import {AppProviders} from '../../app/providers';
import {adminUsersQueryKey} from './UserManagementPage';

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
const admin: UserDto = {
  id: 'admin-1',
  email: 'admin@example.test',
  role: 'admin',
  status: 'active',
  locale: 'ru',
};
const doctor: UserDto = {
  id: 'doctor-1',
  email: 'doctor@example.test',
  role: 'user',
  status: 'active',
  locale: 'kk',
};

function renderAdmin(queryClient = new QueryClient({defaultOptions: {queries: {retry: false}}})) {
  window.history.replaceState({}, '', '/admin');
  server.use(
    http.get('/api/v1/auth/me', () =>
      HttpResponse.json({user: admin, csrf_token: 'csrf-admin', terms_accepted: true, terms_version: '2026-09-22'}),
    ),
  );
  return {
    queryClient,
    ...render(
      <AppProviders queryClient={queryClient}>
        <App />
      </AppProviders>,
    ),
  };
}

beforeAll(() => server.listen({onUnhandledRequest: 'error'}));
afterEach(() => {
  server.resetHandlers();
  apiClient.clearSession();
  window.history.replaceState({}, '', '/');
});
afterAll(() => server.close());

describe('admin user management', () => {
  it('renders loading, error and empty states', async () => {
    server.use(
      http.get('/api/v1/admin/users', async () => {
        await delay(80);
        return HttpResponse.json({users: []});
      }),
    );
    renderAdmin();
    expect(await screen.findByText('Загрузка пользователей…')).toBeVisible();
    expect(await screen.findByText('Пользователей пока нет.')).toBeVisible();
  });

  it('shows a stable list error with retry', async () => {
    server.use(
      http.get('/api/v1/admin/users', () =>
        HttpResponse.json(
          {error: {code: 'network_error', params: {}, request_id: 'admin-list'}},
          {status: 503},
        ),
      ),
    );
    renderAdmin();
    expect(await screen.findByRole('alert')).toHaveTextContent('Не удалось загрузить пользователей.');
    expect(screen.getByRole('button', {name: 'Повторить'})).toBeVisible();
  });

  it('creates a user without retaining the write-only password', async () => {
    let createdBody: unknown;
    server.use(
      http.get('/api/v1/admin/users', () => HttpResponse.json({users: [admin]})),
      http.post('/api/v1/admin/users', async ({request}) => {
        createdBody = await request.json();
        return HttpResponse.json({user: doctor}, {status: 201});
      }),
    );
    renderAdmin();
    const user = userEvent.setup();
    await screen.findByRole('table', {name: 'Пользователи сервиса'});

    await user.type(screen.getByLabelText('Электронная почта пользователя'), 'doctor@example.test');
    await user.type(screen.getByLabelText('Временный пароль'), 'StrongPass123');
    await user.selectOptions(screen.getByLabelText('Роль'), 'user');
    await user.selectOptions(screen.getByLabelText('Язык пользователя'), 'kk');
    await user.click(screen.getByRole('button', {name: 'Добавить пользователя'}));

    await waitFor(() =>
      expect(createdBody).toEqual({
        email: 'doctor@example.test',
        password: 'StrongPass123',
        role: 'user',
        locale: 'kk',
      }),
    );
    expect(screen.getByLabelText('Временный пароль')).toHaveValue('');
    expect(await screen.findByRole('status')).toHaveTextContent('Пользователь добавлен');
  });

  it('blocks, unblocks and deletes with an email-only confirmation', async () => {
    let users = [admin, doctor];
    const mutationBodies: unknown[] = [];
    server.use(
      http.get('/api/v1/admin/users', () => HttpResponse.json({users})),
      http.patch('/api/v1/admin/users/:id', async ({params, request}) => {
        const body = (await request.json()) as {blocked: boolean};
        mutationBodies.push(body);
        users = users.map((item) =>
          item.id === params.id
            ? {...item, status: body.blocked ? ('blocked' as const) : ('active' as const)}
            : item,
        );
        return HttpResponse.json({user: users.find((item) => item.id === params.id)});
      }),
      http.delete('/api/v1/admin/users/:id', ({params}) => {
        const removed = users.find((item) => item.id === params.id) as UserDto;
        users = users.filter((item) => item.id !== params.id);
        return HttpResponse.json({user: {...removed, status: 'deleted'}});
      }),
    );
    const queryClient = new QueryClient({defaultOptions: {queries: {retry: false}}});
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
    renderAdmin(queryClient);
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', {name: 'Заблокировать doctor@example.test'}));
    await waitFor(() => expect(mutationBodies).toEqual([{blocked: true}]));
    await user.click(await screen.findByRole('button', {name: 'Разблокировать doctor@example.test'}));
    await waitFor(() => expect(mutationBodies).toEqual([{blocked: true}, {blocked: false}]));

    const deleteButton = await screen.findByRole('button', {name: 'Удалить doctor@example.test'});
    await user.click(deleteButton);
    const dialog = screen.getByRole('dialog', {name: 'Удалить пользователя?'});
    expect(dialog).toHaveTextContent('doctor@example.test');
    expect(dialog).not.toHaveTextContent('StrongPass123');
    await user.click(within(dialog).getByRole('button', {name: 'Удалить пользователя'}));
    await waitFor(() => expect(screen.queryByText('doctor@example.test')).toBeNull());
    expect(screen.getByRole('heading', {name: 'Пользователи'})).toHaveFocus();
    expect(invalidate).toHaveBeenCalledWith({queryKey: adminUsersQueryKey});
  });

  it('maps duplicate and last-admin errors while disabling self-blocking', async () => {
    let selfPatchCalls = 0;
    server.use(
      http.get('/api/v1/admin/users', () => HttpResponse.json({users: [admin, {...doctor, role: 'admin'}]})),
      http.post('/api/v1/admin/users', () =>
        HttpResponse.json(
          {error: {code: 'email_exists', params: {}, request_id: 'duplicate'}},
          {status: 409},
        ),
      ),
      http.delete('/api/v1/admin/users/admin-1', () =>
        HttpResponse.json(
          {error: {code: 'last_active_admin', params: {}, request_id: 'last-admin'}},
          {status: 409},
        ),
      ),
      http.patch('/api/v1/admin/users/admin-1', () => {
        selfPatchCalls += 1;
        return HttpResponse.json(
          {error: {code: 'last_active_admin', params: {}, request_id: 'last-admin'}},
          {status: 409},
        );
      }),
    );
    renderAdmin();
    const user = userEvent.setup();
    await screen.findByRole('table', {name: 'Пользователи сервиса'});

    await user.type(screen.getByLabelText('Электронная почта пользователя'), 'admin@example.test');
    await user.type(screen.getByLabelText('Временный пароль'), 'StrongPass123');
    await user.click(screen.getByRole('button', {name: 'Добавить пользователя'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Пользователь с такой почтой уже существует.');

    const selfBlock = screen.getByRole('button', {name: 'Заблокировать admin@example.test'});
    expect(selfBlock).toBeDisabled();
    expect(selfBlock).toHaveAttribute('title', 'Нельзя заблокировать собственную учётную запись.');
    await user.click(selfBlock);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(selfPatchCalls).toBe(0);
    expect(screen.getByRole('button', {name: 'Заблокировать doctor@example.test'})).toBeEnabled();

    await user.click(screen.getByRole('button', {name: 'Удалить admin@example.test'}));
    const dialog = screen.getByRole('dialog', {name: 'Удалить пользователя?'});
    await user.click(within(dialog).getByRole('button', {name: 'Удалить пользователя'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Нельзя изменить последнего активного администратора.');
  });

  it('redirects an ordinary user away from the admin route', async () => {
    window.history.replaceState({}, '', '/admin');
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json({user: doctor, csrf_token: 'csrf-doctor', terms_accepted: true, terms_version: '2026-09-22'}),
      ),
    );
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );
    await waitFor(() => expect(window.location.pathname).toBe('/'));
    expect(await screen.findByRole('main')).toBeVisible();
  });
});
