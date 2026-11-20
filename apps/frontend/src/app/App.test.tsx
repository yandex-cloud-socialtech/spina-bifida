import {render, screen, waitFor, within} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {apiClient} from '../api/client';
import type {CreateAnalysisResponse} from '../api/contracts';
import {App} from './App';
import {AppProviders} from './providers';

const LONG_EMAIL = `doctor.${'research.'.repeat(24)}@example.test`;
const SYSTEM_INFO = {
  application_version: '0.1.0',
  model_versions: ['v1', 'v2'] as const,
  default_model_version: 'v2' as const,
  schema_version: 'head',
  device: 'cpu',
};

let request: ReturnType<typeof vi.spyOn>;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return {promise, resolve, reject};
}

function createdAnalysis(): CreateAnalysisResponse {
  return {
    batch_id: 'batch-app',
    jobs: [{
      id: 'job-app',
      state: 'queued',
      frame_id: 'frame-app',
      batch_position: 0,
      polling_url: '/api/v1/analyses/job-app',
    }],
  };
}

vi.mock('../features/auth/useCurrentUser', () => ({
  currentUserQueryKey: ['current-user'],
  useCurrentUser: () => ({
    data: {
      user: {
        id: 'u1',
        email: LONG_EMAIL,
        role: 'user',
        status: 'active',
        locale: 'ru',
      },
      csrf_token: 'csrf-1',
      terms_accepted: true,
      terms_version: '2026-09-22',
    },
    isPending: false,
    isError: false,
    error: null,
  }),
}));

describe('App', () => {
  beforeEach(() => {
    request = vi.spyOn(apiClient, 'request').mockImplementation((path: string) => {
      if (path === '/api/v1/system/info') {
        return Promise.resolve(SYSTEM_INFO);
      }
      return Promise.reject(new Error(`unexpected request: ${path}`));
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders the approved header identity and research-only badge', async () => {
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );

    expect(await screen.findByRole('banner')).toBeInTheDocument();
    expect(screen.getByRole('banner')).toHaveTextContent('Spina Bifida');
    expect(screen.getByRole('banner')).toHaveTextContent('Только в исследовательских целях');
    expect(screen.getByRole('banner')).not.toHaveTextContent('Исследовательский сервис');
  });

  it('shows three official partner links and keeps the model label inline', async () => {
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );

    const banner = await screen.findByRole('banner');
    expect(within(banner).queryByText('SB', {exact: true})).not.toBeInTheDocument();
    expect(within(banner).getByRole('link', {name: 'Yandex Cloud'})).toHaveAttribute(
      'href',
      'https://yandex.cloud/',
    );
    expect(
      within(banner).getByRole('link', {name: 'Благотворительный фонд «Спина бифида»'}),
    ).toHaveAttribute('href', 'https://helpspinabifida.ru/');
    expect(
      within(banner).getByRole('link', {name: 'НМИЦ АГП им. В.И. Кулакова'}),
    ).toHaveAttribute('href', 'https://ncagp.ru/');
    const model = within(banner).getByRole('combobox', {name: 'Модель'});
    const modelControl = within(banner).getByText('Модель').closest('label');
    expect(modelControl).toContainElement(model);
    expect(modelControl).toHaveClass('model-control_inline');
  });

  it('waits for system info and starts with its default model', async () => {
    const info = deferred<typeof SYSTEM_INFO>();
    request.mockImplementation((path: string) => {
      if (path === '/api/v1/system/info') {
        return info.promise;
      }
      return Promise.reject(new Error(`unexpected request: ${path}`));
    });
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );

    expect(screen.queryByRole('combobox', {name: 'Модель'})).not.toBeInTheDocument();

    info.resolve(SYSTEM_INFO);

    expect(await screen.findByRole('combobox', {name: 'Модель'})).toHaveValue('v2');
  });

  it('switches the shared analysis model from the header selector', async () => {
    const user = userEvent.setup();
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );

    const model = await screen.findByRole('combobox', {name: 'Модель'});
    await user.selectOptions(model, 'v1');

    expect(model).toHaveValue('v1');
  });

  it('opens a one-column profile menu from a compact avatar and closes it accessibly', async () => {
    const user = userEvent.setup();
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );

    const profile = await screen.findByTestId('profile-menu');
    const trigger = within(profile).getByRole('button', {name: 'Профиль'});
    expect(within(trigger).getByText('DO')).toHaveClass('profile-avatar');
    expect(trigger).not.toHaveTextContent(LONG_EMAIL);

    await user.click(trigger);

    const menu = screen.getByRole('dialog', {name: 'Профиль'});
    const rows = Array.from(menu.children);
    expect(rows).toHaveLength(6);
    expect(rows[0]).toHaveTextContent(LONG_EMAIL);
    expect(rows[1]).toContainElement(screen.getByLabelText('Тема'));
    expect(rows[2]).toContainElement(screen.getByLabelText('Язык'));
    expect(rows[3]).toContainElement(screen.getByRole('link', {name: 'Сменить пароль'}));
    expect(rows[4]?.tagName).toBe('HR');
    expect(rows[5]).toContainElement(screen.getByRole('button', {name: 'Выйти'}));

    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog', {name: 'Профиль'})).not.toBeInTheDocument();

    await user.click(trigger);
    await user.click(screen.getByText('Spina Bifida'));
    expect(screen.queryByRole('dialog', {name: 'Профиль'})).not.toBeInTheDocument();
  });

  it('disables model selection while an analysis submission is unresolved and re-enables it on success', async () => {
    const submission = deferred<CreateAnalysisResponse>();
    request.mockImplementation((path: string) => {
      if (path === '/api/v1/system/info') {
        return Promise.resolve(SYSTEM_INFO);
      }
      if (path === '/api/v1/analyses') {
        return submission.promise;
      }
      return Promise.reject(new Error(`unexpected request: ${path}`));
    });
    const user = userEvent.setup();
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );

    await user.upload(
      await screen.findByLabelText('Загрузить изображения'),
      new File(['frame'], 'frame.jpg', {type: 'image/jpeg'}),
    );
    await user.click(await screen.findByRole('button', {name: 'Запустить анализ'}));

    const model = await screen.findByRole('combobox', {name: 'Модель'});
    await waitFor(() => expect(model).toBeDisabled());
    await user.selectOptions(model, 'v1');
    expect(model).toHaveValue('v2');

    submission.resolve(createdAnalysis());
    await waitFor(() => expect(model).toBeEnabled());
  });

  it('re-enables model selection after a failed analysis submission', async () => {
    const submission = deferred<CreateAnalysisResponse>();
    request.mockImplementation((path: string) => {
      if (path === '/api/v1/system/info') {
        return Promise.resolve(SYSTEM_INFO);
      }
      if (path === '/api/v1/analyses') {
        return submission.promise;
      }
      return Promise.reject(new Error(`unexpected request: ${path}`));
    });
    const user = userEvent.setup();
    render(
      <AppProviders>
        <App />
      </AppProviders>,
    );

    await user.upload(
      await screen.findByLabelText('Загрузить изображения'),
      new File(['frame'], 'frame.jpg', {type: 'image/jpeg'}),
    );
    await user.click(await screen.findByRole('button', {name: 'Запустить анализ'}));

    const model = await screen.findByRole('combobox', {name: 'Модель'});
    await waitFor(() => expect(model).toBeDisabled());

    submission.reject(new Error('network'));
    await waitFor(() => expect(model).toBeEnabled());
  });
});
