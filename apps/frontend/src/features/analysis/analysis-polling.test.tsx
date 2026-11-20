import {QueryClient} from '@tanstack/react-query';
import {render, waitFor} from '@testing-library/react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {apiClient} from '../../api/client';
import type {AnalysisJobDto} from '../../api/contracts';
import {AppProviders} from '../../app/providers';
import {AnalysisWorkspace} from './AnalysisWorkspace';
import {AnalysisController} from './model';

function polledJob(id: string, state: AnalysisJobDto['state']): AnalysisJobDto {
  return {
    id,
    state,
    stage: state === 'quality_check' ? 'quality_check' : null,
    progress: state === 'completed' ? 100 : 40,
    queue_position: null,
    attempt: 0,
    frames: [{id: `frame-${id.at(-1)}`, original_asset_url: `/assets/${id}`}],
    result: null,
    error: null,
    expires_at: '2026-09-23T00:00:00Z',
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('analysis polling integration', () => {
  it('polls two active jobs independently and keeps refetching only the unfinished sibling', async () => {
    const controller = new AnalysisController({
      request: vi.fn().mockResolvedValue({
        batch_id: 'batch-1',
        jobs: [
          {
            id: 'job-0',
            state: 'queued',
            frame_id: 'frame-0',
            batch_position: 0,
            polling_url: '/api/v1/analyses/job-0',
          },
          {
            id: 'job-1',
            state: 'queued',
            frame_id: 'frame-1',
            batch_position: 1,
            polling_url: '/api/v1/analyses/job-1',
          },
        ],
      }),
      createObjectURL: (candidate) => `blob:${(candidate as File).name}`,
      revokeObjectURL: vi.fn(),
    });
    await controller.addFiles([
      new File(['one'], 'one.jpg', {type: 'image/jpeg'}),
      new File(['two'], 'two.jpg', {type: 'image/jpeg'}),
    ]);
    await controller.start();
    const calls = new Map<string, number>();
    vi.spyOn(apiClient, 'request').mockImplementation(async (path) => {
      const count = (calls.get(path) ?? 0) + 1;
      calls.set(path, count);
      if (path.endsWith('job-0')) {
        return polledJob('job-0', 'completed');
      }
      return polledJob('job-1', count === 1 ? 'quality_check' : 'completed');
    });
    const queryClient = new QueryClient({
      defaultOptions: {queries: {retry: false}},
    });

    render(
      <AppProviders queryClient={queryClient}>
        <AnalysisWorkspace controller={controller} />
      </AppProviders>,
    );

    await waitFor(() => {
      expect(controller.snapshot.frames.map(({job}) => job?.state)).toEqual([
        'completed',
        'quality_check',
      ]);
    });
    await waitFor(
      () => {
        expect(controller.snapshot.frames.map(({job}) => job?.state)).toEqual([
          'completed',
          'completed',
        ]);
      },
      {timeout: 2_500},
    );
    expect(calls.get('/api/v1/analyses/job-0')).toBe(1);
    expect(calls.get('/api/v1/analyses/job-1')).toBeGreaterThanOrEqual(2);
  });

  it('keeps a retried job queued until a fresh poll replaces the recoverable response', async () => {
    const request = vi.fn(async (path: string) => {
      if (path === '/api/v1/analyses') {
        return {
          batch_id: 'batch-1',
          jobs: [
            {
              id: 'job-0',
              state: 'queued',
              frame_id: 'frame-0',
              batch_position: 0,
              polling_url: '/api/v1/analyses/job-0',
            },
          ],
        };
      }
      if (path === '/api/v1/analyses/job-0/retry') {
        return {id: 'job-0', state: 'queued', attempt: 2};
      }
      throw new Error(`Unexpected request: ${path}`);
    });
    const controller = new AnalysisController({
      request,
      createObjectURL: (candidate) => `blob:${(candidate as File).name}`,
      revokeObjectURL: vi.fn(),
    });
    await controller.addFiles([new File(['one'], 'one.jpg', {type: 'image/jpeg'})]);
    await controller.start();
    let polls = 0;
    vi.spyOn(apiClient, 'request').mockImplementation(async () => {
      polls += 1;
      return polledJob('job-0', polls === 1 ? 'failed_recoverable' : 'completed');
    });
    const queryClient = new QueryClient({
      defaultOptions: {queries: {retry: false}},
    });

    render(
      <AppProviders queryClient={queryClient}>
        <AnalysisWorkspace controller={controller} />
      </AppProviders>,
    );

    await waitFor(() => expect(controller.snapshot.frames[0].job?.state).toBe('failed_recoverable'));
    await controller.retryFrame(controller.snapshot.frames[0].id);
    expect(controller.snapshot.frames[0].job?.state).toBe('queued');
    await waitFor(
      () => expect(controller.snapshot.frames[0].job?.state).toBe('completed'),
      {timeout: 2_500},
    );
    expect(polls).toBe(2);
  });
});
