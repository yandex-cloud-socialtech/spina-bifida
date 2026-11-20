import {render, screen, waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {StrictMode} from 'react';
import {BrowserRouter, useNavigate} from 'react-router-dom';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {ApiError, apiClient} from '../../api/client';
import {App} from '../../app/App';
import {AppRoutes} from '../../app/routes';
import {AppProviders} from '../../app/providers';
import {ProfileMenu} from '../../components/ProfileMenu';
import {UserTable} from '../admin/UserTable';
import {AccountPage} from './AccountPage';

const doctor = {id: 'u1', email: 'doctor@example.test', role: 'user', status: 'active', locale: 'ru'} as const;
let request: ReturnType<typeof vi.spyOn>;
function renderPage(mode: Parameters<typeof AccountPage>[0]['mode']) {
  return render(<AppProviders><BrowserRouter><AccountPage mode={mode} /></BrowserRouter></AppProviders>);
}
function OpenAnotherEmail({mode}: {mode: 'verify-email' | 'reset-password'}) {
  const navigate = useNavigate();
  return <button onClick={() => navigate(`/${mode}#token=new-link-0123456789abcdefghijklmnopqrstuvwxyz`)}>
    Open latest email
  </button>;
}

beforeEach(() => {
  request = vi.spyOn(apiClient, 'request').mockResolvedValue({status: 'accepted'});
});
afterEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState({}, '', '/');
});

describe('registration and passwords', () => {
  it('registers with ordinary user fields and explains moderation', async () => {
    renderPage('register');
    const user = userEvent.setup();
    await user.type(screen.getByLabelText('Электронная почта'), 'doctor@example.test');
    await user.type(screen.getByLabelText('Новый пароль'), 'StrongPass123');
    await user.type(screen.getByLabelText('Повторите пароль'), 'StrongPass123');
    await user.click(screen.getByRole('button', {name: 'Зарегистрироваться'}));
    expect(await screen.findByRole('status')).toHaveTextContent('ссылка подтверждения');
    expect(request).toHaveBeenCalledWith('/api/v1/auth/register', expect.objectContaining({
      method: 'POST', json: expect.objectContaining({email: 'doctor@example.test', locale: 'ru'}),
    }));
    expect(screen.queryByLabelText('Новый пароль')).toBeNull();
  });

  it('checks password confirmation before a reset request', async () => {
    window.history.replaceState({}, '', '/reset-password#token=single-use-token');
    renderPage('reset-password');
    const user = userEvent.setup();
    await user.type(screen.getByLabelText('Новый пароль'), 'StrongPass123');
    await user.type(screen.getByLabelText('Повторите пароль'), 'DifferentPass123');
    await user.click(screen.getByRole('button', {name: 'Сохранить пароль'}));
    expect(screen.getByRole('alert')).toHaveTextContent('Пароли не совпадают');
    expect(request).not.toHaveBeenCalledWith('/api/v1/auth/reset-password', expect.anything());
  });

  it('consumes the fragment only after explicit confirmation and removes it from the address bar', async () => {
    window.history.replaceState({}, '', '/verify-email#token=single-use-token');
    renderPage('verify-email');
    expect(window.location.hash).toBe('');
    expect(request).not.toHaveBeenCalledWith('/api/v1/auth/verify-email', expect.anything());
    await userEvent.setup().click(screen.getByRole('button', {name: 'Подтвердить адрес'}));
    expect(request).toHaveBeenCalledWith('/api/v1/auth/verify-email', expect.objectContaining({
      json: expect.objectContaining({token: 'single-use-token'}),
    }));
    expect(await screen.findByRole('status')).toHaveTextContent('ожидает модерации');
  });

  it.each(['verify-email', 'reset-password'] as const)(
    'preserves the email link through the complete %s application mount', async (mode) => {
      const token = 'test-link-0123456789abcdefghijklmnopqrstuvwxyz';
      window.history.replaceState({}, '', `/${mode}#token=${token}`);
      render(<StrictMode><AppProviders><App /></AppProviders></StrictMode>);
      const user = userEvent.setup();
      if (mode === 'reset-password') {
        await user.type(screen.getByLabelText('Новый пароль'), 'AnotherPass123');
        await user.type(screen.getByLabelText('Повторите пароль'), 'AnotherPass123');
      }
      await user.click(screen.getByRole('button', {name: mode === 'verify-email' ?
        'Подтвердить адрес' : 'Сохранить пароль'}));
      expect(request).toHaveBeenCalledWith(`/api/v1/auth/${mode}`, expect.objectContaining({
        json: expect.objectContaining({token}),
      }));
      expect(window.location.hash).toBe('');
    },
  );

  it.each(['verify-email', 'reset-password'] as const)(
    'uses the latest %s link when its fragment changes in the same tab', async (mode) => {
      window.history.replaceState({}, '', `/${mode}#token=old-link-0123456789abcdefghijklmnopqrstuvwxyz`);
      render(<AppProviders><BrowserRouter><AppRoutes /><OpenAnotherEmail mode={mode} />
        </BrowserRouter></AppProviders>);
      const user = userEvent.setup();
      await user.click(screen.getByRole('button', {name: 'Open latest email'}));
      if (mode === 'reset-password') {
        await user.type(screen.getByLabelText('Новый пароль'), 'AnotherPass123');
        await user.type(screen.getByLabelText('Повторите пароль'), 'AnotherPass123');
      }
      await user.click(screen.getByRole('button', {name: mode === 'verify-email' ?
        'Подтвердить адрес' : 'Сохранить пароль'}));
      expect(request).toHaveBeenCalledWith(`/api/v1/auth/${mode}`, expect.objectContaining({
        json: expect.objectContaining({token: 'new-link-0123456789abcdefghijklmnopqrstuvwxyz'}),
      }));
      expect(window.location.hash).toBe('');
    },
  );

  it('clears the previous link error when a new verification link is opened', async () => {
    window.history.replaceState({}, '', '/verify-email#token=old-link-0123456789abcdefghijklmnopqrstuvwxyz');
    request.mockImplementation((path: string) => path === '/api/v1/auth/verify-email' ?
      Promise.reject(new ApiError('invalid_token', 400)) : Promise.resolve({}));
    render(<AppProviders><BrowserRouter><AppRoutes /><OpenAnotherEmail mode="verify-email" />
      </BrowserRouter></AppProviders>);
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', {name: 'Подтвердить адрес'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Ссылка недействительна');
    await user.click(screen.getByRole('button', {name: 'Open latest email'}));
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByRole('button', {name: 'Подтвердить адрес'})).toBeEnabled();
  });

  it('cannot submit a reset without a link token', () => {
    renderPage('reset-password');
    expect(screen.getByRole('button', {name: 'Сохранить пароль'})).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent('Ссылка недействительна');
  });

  it('asks for a current password when changing a password', async () => {
    renderPage('change-password');
    const user = userEvent.setup();
    await user.type(screen.getByLabelText('Текущий пароль'), 'StrongPass123');
    await user.type(screen.getByLabelText('Новый пароль'), 'AnotherPass123');
    await user.type(screen.getByLabelText('Повторите пароль'), 'AnotherPass123');
    await user.click(screen.getByRole('button', {name: 'Сменить пароль'}));
    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/v1/auth/change-password',
      expect.objectContaining({json: expect.objectContaining({
        current_password: 'StrongPass123', password: 'AnotherPass123',
      })})));
  });

  it('shows the admin panel link only to admins, and password change to everyone', async () => {
    const view = render(<AppProviders><BrowserRouter><ProfileMenu user={doctor} /></BrowserRouter></AppProviders>);
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', {name: 'Профиль'}));
    expect(screen.getByRole('link', {name: 'Сменить пароль'})).toHaveAttribute('href', '/change-password');
    expect(screen.queryByRole('link', {name: 'Админ-панель'})).toBeNull();
    view.unmount();
    render(<AppProviders><BrowserRouter><ProfileMenu user={{...doctor, role: 'admin'}} /></BrowserRouter></AppProviders>);
    await user.click(screen.getByRole('button', {name: 'Профиль'}));
    expect(screen.getByRole('link', {name: 'Админ-панель'})).toHaveAttribute('href', '/admin');
  });

  it('offers moderation only for verified pending users', async () => {
    const moderate = vi.fn();
    render(<UserTable currentUserId="admin-1" users={[{...doctor, status: 'pending'}]} pendingUserId={null}
      onModerate={moderate} onToggleBlock={vi.fn()} onDelete={vi.fn()} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', {name: `Одобрить ${doctor.email}`}));
    expect(moderate).toHaveBeenCalledWith(expect.objectContaining({id: doctor.id}), true);
    await user.click(screen.getByRole('button', {name: `Отклонить ${doctor.email}`}));
    expect(moderate).toHaveBeenCalledWith(expect.objectContaining({id: doctor.id}), false);
    expect(screen.queryByRole('button', {name: `Разблокировать ${doctor.email}`})).toBeNull();
  });
});
