import {http, HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import {afterAll, afterEach, beforeAll, describe, expect, it} from 'vitest';

import {apiClient} from './client';

const server = setupServer();

beforeAll(() => server.listen({onUnhandledRequest: 'error'}));
afterEach(() => {
  server.resetHandlers();
  apiClient.clearSession();
});
afterAll(() => server.close());

describe('apiClient', () => {
  it('uses same-origin credentials and attaches CSRF only to mutations', async () => {
    let getCredentials: RequestCredentials | undefined;
    let postCredentials: RequestCredentials | undefined;
    let getCsrf: string | null = null;
    let csrf: string | null = null;
    server.use(
      http.get('/api/v1/auth/me', ({request}) => {
        getCredentials = request.credentials;
        getCsrf = request.headers.get('X-CSRF-Token');
        return HttpResponse.json({
          user: {id: 'u1', email: 'doctor@example.test', role: 'user', status: 'active', locale: 'ru'},
          csrf_token: 'csrf-1',
          terms_accepted: true,
          terms_version: '2026-09-22',
        });
      }),
      http.post('/api/v1/analyses', ({request}) => {
        postCredentials = request.credentials;
        csrf = request.headers.get('X-CSRF-Token');
        return HttpResponse.json({id: 'j1'}, {status: 202});
      }),
    );

    await apiClient.request('/api/v1/auth/me');
    await apiClient.request('/api/v1/analyses', {method: 'POST', body: new FormData()});

    expect(getCredentials).toBe('include');
    expect(getCsrf).toBeNull();
    expect(postCredentials).toBe('include');
    expect(csrf).toBe('csrf-1');
  });

  it('normalizes the stable error envelope and bootstrap CSRF', async () => {
    server.use(
      http.get('/api/v1/auth/me', () =>
        HttpResponse.json(
          {
            error: {
              code: 'authentication_required',
              params: {},
              request_id: 'request-1',
            },
            csrf_token: 'bootstrap-csrf',
          },
          {status: 401},
        ),
      ),
    );

    await expect(apiClient.request('/api/v1/auth/me')).rejects.toMatchObject({
      code: 'authentication_required',
      status: 401,
      requestId: 'request-1',
    });
    expect(apiClient.csrfToken).toBe('bootstrap-csrf');
  });

  it('refreshes me once after 401 and retries the original request once', async () => {
    let protectedCalls = 0;
    let refreshCalls = 0;
    server.use(
      http.get('/api/v1/protected', () => {
        protectedCalls += 1;
        if (protectedCalls === 1) {
          return HttpResponse.json(
            {error: {code: 'authentication_required', params: {}, request_id: 'r1'}},
            {status: 401},
          );
        }
        return HttpResponse.json({ok: true});
      }),
      http.get('/api/v1/auth/me', () => {
        refreshCalls += 1;
        return HttpResponse.json({
          user: {id: 'u1', email: 'doctor@example.test', role: 'user', status: 'active', locale: 'ru'},
          csrf_token: 'csrf-refreshed',
          terms_accepted: true,
          terms_version: '2026-09-22',
        });
      }),
    );

    await expect(apiClient.request<{ok: boolean}>('/api/v1/protected')).resolves.toEqual({
      ok: true,
    });
    expect(refreshCalls).toBe(1);
    expect(protectedCalls).toBe(2);
  });

  it('does not enter a retry loop after the retried request returns 401', async () => {
    let protectedCalls = 0;
    let refreshCalls = 0;
    server.use(
      http.get('/api/v1/protected', () => {
        protectedCalls += 1;
        return HttpResponse.json(
          {error: {code: 'authentication_required', params: {}, request_id: 'r2'}},
          {status: 401},
        );
      }),
      http.get('/api/v1/auth/me', () => {
        refreshCalls += 1;
        return HttpResponse.json({
          user: {id: 'u1', email: 'doctor@example.test', role: 'user', status: 'active', locale: 'ru'},
          csrf_token: 'csrf-refreshed',
          terms_accepted: true,
          terms_version: '2026-09-22',
        });
      }),
    );

    await expect(apiClient.request('/api/v1/protected')).rejects.toMatchObject({
      code: 'authentication_required',
      status: 401,
    });
    expect(refreshCalls).toBe(1);
    expect(protectedCalls).toBe(2);
  });
});
