import type {ApiErrorEnvelope, AuthResponse} from './contracts';

export interface ApiRequestOptions extends Omit<RequestInit, 'body'> {
  body?: BodyInit | null;
  json?: unknown;
}

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    public readonly status: number,
    public readonly params: Record<string, unknown> = {},
    public readonly requestId: string | null = null,
  ) {
    super(code);
    this.name = 'ApiError';
  }
}

const MUTATING_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);
const ME_PATH = '/api/v1/auth/me';

class ApiClient {
  csrfToken: string | null = null;

  clearSession(): void {
    this.csrfToken = null;
  }

  async request<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
    return this.perform<T>(path, options, true);
  }

  private async perform<T>(
    path: string,
    options: ApiRequestOptions,
    allowRefresh: boolean,
  ): Promise<T> {
    const url = new URL(path, window.location.origin);
    if (url.origin !== window.location.origin) {
      throw new ApiError('cross_origin_request_blocked', 0);
    }
    const {json, ...requestOptions} = options;
    const method = (requestOptions.method ?? 'GET').toUpperCase();
    const headers = new Headers(requestOptions.headers);
    let body = requestOptions.body;
    if (json !== undefined) {
      headers.set('Content-Type', 'application/json');
      body = JSON.stringify(json);
    }
    if (MUTATING_METHODS.has(method) && this.csrfToken) {
      headers.set('X-CSRF-Token', this.csrfToken);
    }

    let response: Response;
    try {
      response = await fetch(url, {
        ...requestOptions,
        method,
        headers,
        body,
        credentials: 'include',
      });
    } catch (error) {
      throw new ApiError('network_error', 0, {
        reason: error instanceof Error ? error.message : 'unknown',
      });
    }

    const payload = await this.parsePayload(response);
    this.captureCsrf(payload);
    if (response.ok) {
      return payload as T;
    }

    const apiError = this.toApiError(response.status, payload);
    if (response.status === 401 && allowRefresh && url.pathname !== ME_PATH) {
      try {
        await this.perform<AuthResponse>(ME_PATH, {}, false);
      } catch {
        throw apiError;
      }
      return this.perform<T>(path, options, false);
    }
    throw apiError;
  }

  private async parsePayload(response: Response): Promise<unknown> {
    if (response.status === 204) {
      return undefined;
    }
    const contentType = response.headers.get('Content-Type') ?? '';
    if (!contentType.includes('application/json')) {
      if (response.ok) {
        return undefined;
      }
      throw new ApiError('invalid_api_response', response.status);
    }
    try {
      return await response.json();
    } catch {
      throw new ApiError('invalid_api_response', response.status);
    }
  }

  private captureCsrf(payload: unknown): void {
    if (
      payload &&
      typeof payload === 'object' &&
      'csrf_token' in payload &&
      typeof payload.csrf_token === 'string'
    ) {
      this.csrfToken = payload.csrf_token;
    }
  }

  private toApiError(status: number, payload: unknown): ApiError {
    if (this.isErrorEnvelope(payload)) {
      return new ApiError(
        payload.error.code,
        status,
        payload.error.params,
        payload.error.request_id,
      );
    }
    return new ApiError('invalid_api_response', status);
  }

  private isErrorEnvelope(payload: unknown): payload is ApiErrorEnvelope {
    if (!payload || typeof payload !== 'object' || !('error' in payload)) {
      return false;
    }
    const error = payload.error;
    return Boolean(
      error &&
        typeof error === 'object' &&
        'code' in error &&
        typeof error.code === 'string',
    );
  }
}

export const apiClient = new ApiClient();
