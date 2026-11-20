import {describe, expect, it, vi} from 'vitest';

import {ApiError} from '../../api/client';
import type {
  AnalysisJobDto,
  CreateAnalysisResponse,
  CreatedAnalysisJobDto,
  JobState,
} from '../../api/contracts';
import {AnalysisController, type AnalysisSnapshot} from './model';

function file(name: string, type: string, content = name): File {
  return new File([content], name, {type, lastModified: 1});
}

function jpeg(name: string, content = name): File {
  return file(name, 'image/jpeg', content);
}

function createdJob(position: number, overrides: Partial<CreatedAnalysisJobDto> = {}): CreatedAnalysisJobDto {
  return {
    id: `job-${position}`,
    state: 'queued',
    frame_id: `backend-frame-${position}`,
    batch_position: position,
    polling_url: `/api/v1/analyses/job-${position}`,
    ...overrides,
  };
}

function createdBatch(
  count: number,
  jobs = Array.from({length: count}, (_, index) => createdJob(index)),
): CreateAnalysisResponse {
  return {batch_id: 'batch-1', jobs};
}

function job(
  id: string,
  state: AnalysisJobDto['state'],
  overrides: Partial<AnalysisJobDto> = {},
): AnalysisJobDto {
  return {
    id,
    state,
    stage: state === 'quality_check' ? 'quality_check' : null,
    progress: state === 'completed' ? 100 : 40,
    attempt: 0,
    queue_position: state === 'queued' ? 2 : null,
    frames: [],
    result: null,
    error: state === 'failed_recoverable' ? {code: 'inference_failed'} : null,
    expires_at: '2026-09-23T00:00:00Z',
    ...overrides,
  };
}

function createController(overrides: ConstructorParameters<typeof AnalysisController>[0] = {}) {
  let sequence = 0;
  return new AnalysisController({
    createObjectURL: () => `blob:preview-${++sequence}`,
    revokeObjectURL: vi.fn(),
    request: vi.fn(),
    initialModelVersion: 'v1',
    ...overrides,
  });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((accept, fail) => {
    resolve = accept;
    reject = fail;
  });
  return {promise, resolve, reject};
}

async function startedController(
  count = 2,
  request = vi.fn().mockResolvedValue(createdBatch(count)),
) {
  const controller = createController({request});
  await controller.addFiles(Array.from({length: count}, (_, index) => jpeg(`frame-${index}.jpg`)));
  await controller.start();
  return {controller, request};
}

describe('AnalysisController', () => {
  it('accepts ten files and rejects an eleventh without changing the ten drafts', async () => {
    const controller = createController();
    await controller.addFiles(Array.from({length: 10}, (_, index) => jpeg(`${index}.jpg`)));
    const originalNames = controller.snapshot.frames.map(({name}) => name);

    await controller.addFiles([jpeg('eleven.jpg')]);

    expect(controller.snapshot.frames).toHaveLength(10);
    expect(controller.snapshot.frames.map(({name}) => name)).toEqual(originalNames);
    expect(controller.snapshot.error).toMatchObject({code: 'too_many_frames', fileName: 'eleven.jpg'});
  });

  it('validates a newly selected group atomically before adding any file', async () => {
    const controller = createController();
    await controller.addFiles(Array.from({length: 9}, (_, index) => jpeg(`${index}.jpg`)));
    const originalFrames = controller.snapshot.frames;

    await controller.addFiles([jpeg('valid-tenth.jpg'), file('corrupt.txt', 'text/plain')]);

    expect(controller.snapshot.frames).toBe(originalFrames);
    expect(controller.snapshot.frames).toHaveLength(9);
    expect(controller.snapshot.error).toMatchObject({
      code: 'unsupported_format',
      fileName: 'corrupt.txt',
    });
  });

  it('skips duplicate files while adding the unique files from the same selection', async () => {
    const controller = createController();
    await controller.addFiles([jpeg('existing.jpg')]);
    const duplicateExisting = jpeg('existing.jpg');
    const duplicateInsideSelection = jpeg('duplicate.jpg', 'same');

    await controller.addFiles([
      duplicateExisting,
      duplicateInsideSelection,
      duplicateInsideSelection,
      jpeg('new.jpg'),
    ]);

    expect(controller.snapshot.frames.map(({name}) => name)).toEqual([
      'existing.jpg',
      'duplicate.jpg',
      'new.jpg',
    ]);
    expect(controller.snapshot.error).toBeNull();
  });

  it('reports a duplicate only when the selection contains no new files', async () => {
    const controller = createController();
    await controller.addFiles([jpeg('existing.jpg')]);
    const originalFrames = controller.snapshot.frames;

    await controller.addFiles([jpeg('existing.jpg')]);

    expect(controller.snapshot.frames).toBe(originalFrames);
    expect(controller.snapshot.error).toMatchObject({
      code: 'duplicate_image',
      fileName: 'existing.jpg',
    });
  });

  it('retries without a server-detected duplicate and starts the remaining queue', async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new ApiError('duplicate_image', 422, {file_index: 1}))
      .mockResolvedValueOnce(createdBatch(2));
    const revokeObjectURL = vi.fn();
    const controller = createController({request, revokeObjectURL});
    await controller.addFiles([jpeg('first.jpg'), jpeg('duplicate-copy.jpg'), jpeg('third.jpg')]);
    const duplicate = controller.snapshot.frames[1];
    controller.selectFrame(duplicate.id);

    await controller.start();

    expect(request).toHaveBeenCalledTimes(2);
    const firstSubmission = request.mock.calls[0][1]?.body as FormData;
    const retriedSubmission = request.mock.calls[1][1]?.body as FormData;
    expect(firstSubmission.getAll('frames').map((part) => (part as File).name)).toEqual([
      'first.jpg',
      'duplicate-copy.jpg',
      'third.jpg',
    ]);
    expect(retriedSubmission.getAll('frames').map((part) => (part as File).name)).toEqual([
      'first.jpg',
      'third.jpg',
    ]);
    expect(controller.snapshot.frames.map(({name}) => name)).toEqual(['first.jpg', 'third.jpg']);
    expect(controller.snapshot.selectedFrameId).toBe(controller.snapshot.frames[0].id);
    expect(controller.snapshot.batchId).toBe('batch-1');
    expect(controller.snapshot.error).toBeNull();
    expect(revokeObjectURL).toHaveBeenCalledWith(duplicate.previewUrl);
  });

  it('loads all four checked-in examples in one atomic action', async () => {
    const loadExamples = vi.fn().mockResolvedValue([
      jpeg('norm-sagittal.jpg'),
      jpeg('norm-axial.jpg'),
      jpeg('patology-sagittal.jpg'),
      jpeg('patology-axial.jpg'),
    ]);
    const controller = createController({loadExamples});

    await controller.useExamples();

    expect(loadExamples).toHaveBeenCalledOnce();
    expect(controller.snapshot.frames.map(({name}) => name)).toEqual([
      'norm-sagittal.jpg',
      'norm-axial.jpg',
      'patology-sagittal.jpg',
      'patology-axial.jpg',
    ]);
    expect(controller.snapshot.frames.every(({source}) => source === 'example')).toBe(true);
  });

  it('ignores examples that finish loading after a manual frame was added', async () => {
    const examples = deferred<File[]>();
    const controller = createController({loadExamples: () => examples.promise});
    const pendingExamples = controller.useExamples();

    await controller.addFiles([jpeg('manual.jpg')]);
    examples.resolve([
      jpeg('norm-sagittal.jpg'),
      jpeg('norm-axial.jpg'),
      jpeg('patology-sagittal.jpg'),
      jpeg('patology-axial.jpg'),
    ]);
    await pendingExamples;

    expect(controller.snapshot.frames.map(({name}) => name)).toEqual(['manual.jpg']);
  });

  it('submits frames in draft order with the explicit model and maps shuffled jobs by batch position', async () => {
    const request = vi.fn().mockResolvedValue(
      createdBatch(3, [createdJob(2), createdJob(0), createdJob(1)]),
    );
    const controller = createController({request});
    await controller.addFiles([jpeg('one.jpg'), jpeg('two.jpg'), jpeg('three.jpg')]);
    await controller.changeModel('v2');

    await controller.start();

    const submitted = request.mock.calls[0][1]?.body as FormData;
    expect(submitted.getAll('frames').map((part) => (part as File).name)).toEqual([
      'one.jpg',
      'two.jpg',
      'three.jpg',
    ]);
    expect(submitted.get('model_version')).toBe('v2');
    expect(controller.snapshot.batchId).toBe('batch-1');
    expect(controller.snapshot.frames.map(({jobId}) => jobId)).toEqual(['job-0', 'job-1', 'job-2']);
    expect(controller.snapshot.frames.map(({backendFrameId}) => backendFrameId)).toEqual([
      'backend-frame-0',
      'backend-frame-1',
      'backend-frame-2',
    ]);
  });

  it('rejects a non-contiguous batch response without assigning jobs to frames', async () => {
    const request = vi.fn().mockResolvedValue(
      createdBatch(2, [createdJob(0), createdJob(2)]),
    );
    const controller = createController({request});
    await controller.addFiles([jpeg('one.jpg'), jpeg('two.jpg')]);

    await controller.start();

    expect(controller.snapshot.batchId).toBeNull();
    expect(controller.snapshot.frames.every(({jobId}) => jobId === null)).toBe(true);
    expect(controller.snapshot.error?.code).toBe('invalid_analysis_response');
  });

  it.each([
    ['empty batch id', {...createdBatch(2), batch_id: ''}, false],
    ['whitespace batch id', {...createdBatch(2), batch_id: '   '}, false],
    [
      'empty job id',
      createdBatch(2, [createdJob(0, {id: ''}), createdJob(1)]),
      true,
    ],
    [
      'whitespace job id',
      createdBatch(2, [createdJob(0, {id: '  '}), createdJob(1)]),
      true,
    ],
    [
      'empty frame id',
      createdBatch(2, [createdJob(0, {frame_id: ''}), createdJob(1)]),
      true,
    ],
    [
      'whitespace frame id',
      createdBatch(2, [createdJob(0, {frame_id: '  '}), createdJob(1)]),
      true,
    ],
    [
      'empty polling url',
      createdBatch(2, [createdJob(0, {polling_url: ''}), createdJob(1)]),
      true,
    ],
    [
      'whitespace polling url',
      createdBatch(2, [createdJob(0, {polling_url: '  '}), createdJob(1)]),
      true,
    ],
    [
      'duplicate job ids',
      createdBatch(2, [createdJob(0, {id: 'same-job'}), createdJob(1, {id: 'same-job'})]),
      true,
    ],
    [
      'duplicate frame ids',
      createdBatch(2, [
        createdJob(0, {frame_id: 'same-frame'}),
        createdJob(1, {frame_id: 'same-frame'}),
      ]),
      true,
    ],
  ])('rejects malformed create response with %s', async (_label, response, cleansBatch) => {
    const request = vi.fn().mockResolvedValueOnce(response).mockResolvedValueOnce(undefined);
    const controller = createController({request});
    await controller.addFiles([jpeg('one.jpg'), jpeg('two.jpg')]);

    await controller.start();

    expect(controller.snapshot.batchId).toBeNull();
    expect(controller.snapshot.frames.every(({jobId}) => jobId === null)).toBe(true);
    expect(controller.snapshot.error?.code).toBe('invalid_analysis_response');
    const deleteCalls = request.mock.calls.filter(([, options]) => options?.method === 'DELETE');
    expect(deleteCalls).toHaveLength(cleansBatch ? 1 : 0);
    if (cleansBatch) {
      expect(deleteCalls[0][0]).toBe('/api/v1/analyses/batches/batch-1');
    }
  });

  it('tracks active polling and terminal state independently for every frame', async () => {
    const {controller} = await startedController(3);

    controller.applyJob(job('job-0', 'completed'));
    controller.applyJob(job('job-1', 'quality_check'));
    controller.applyJob(job('job-2', 'failed_recoverable'));

    expect(controller.pollingJobIds).toEqual(['job-1']);
    expect(controller.snapshot.frames.map(({job}) => job?.state)).toEqual([
      'completed',
      'quality_check',
      'failed_recoverable',
    ]);
    controller.applyJob(job('unrelated', 'completed'));
    expect(controller.snapshot.frames.map(({jobId}) => jobId)).toEqual(['job-0', 'job-1', 'job-2']);
  });

  it('retries only the selected failed job and leaves completed siblings usable', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(2))
      .mockResolvedValueOnce({id: 'job-1', state: 'queued', attempt: 2});
    const {controller} = await startedController(2, request);
    controller.applyJob(job('job-0', 'completed'));
    controller.applyJob(job('job-1', 'failed_recoverable', {attempt: 1}));

    await controller.retryFrame(controller.snapshot.frames[1].id);

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/job-1/retry', {method: 'POST'});
    expect(controller.snapshot.frames[0].job?.state).toBe('completed');
    expect(controller.snapshot.frames[1].job).toMatchObject({state: 'queued', attempt: 2});
  });

  it('posts one retry while the selected frame retry is pending and clears the guard on success', async () => {
    const retry = deferred<{id: string; state: JobState; attempt: number}>();
    const request = vi.fn((path: string) => {
      if (path.endsWith('/retry')) {
        return retry.promise;
      }
      return Promise.resolve(createdBatch(1));
    });
    const {controller} = await startedController(1, request);
    const frame = controller.snapshot.frames[0];
    controller.applyJob(job('job-0', 'failed_recoverable', {attempt: 1}));

    const first = controller.retryFrame(frame.id);
    const second = controller.retryFrame(frame.id);

    expect(request.mock.calls.filter(([path]) => path.endsWith('/retry'))).toHaveLength(1);
    retry.resolve({id: 'job-0', state: 'queued', attempt: 2});
    await Promise.all([first, second]);
    expect(controller.snapshot.frames[0].job).toMatchObject({state: 'queued', attempt: 2});
  });

  it('clears the per-frame retry guard after an error so the retry can be attempted again', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(1))
      .mockRejectedValueOnce(new ApiError('retry_failed', 503))
      .mockResolvedValueOnce({id: 'job-0', state: 'queued', attempt: 2});
    const {controller} = await startedController(1, request);
    const frame = controller.snapshot.frames[0];
    controller.applyJob(job('job-0', 'failed_recoverable', {attempt: 1}));

    await controller.retryFrame(frame.id);
    await controller.retryFrame(frame.id);

    expect(request.mock.calls.filter(([path]) => path.endsWith('/retry'))).toHaveLength(2);
    expect(controller.snapshot.frames[0].job).toMatchObject({state: 'queued', attempt: 2});
  });

  it('deletes the server batch before revoking previews on clear', async () => {
    const events: string[] = [];
    const request = vi.fn(async (path: string, options = {}) => {
      if ((options as {method?: string}).method === 'DELETE') {
        events.push(`delete:${path}`);
        return undefined;
      }
      return createdBatch(2);
    });
    const revokeObjectURL = vi.fn((url: string) => events.push(`revoke:${url}`));
    const controller = createController({request, revokeObjectURL});
    await controller.addFiles([jpeg('one.jpg'), jpeg('two.jpg')]);
    await controller.start();

    await controller.clear();

    expect(events[0]).toBe('delete:/api/v1/analyses/batches/batch-1');
    expect(events.slice(1)).toEqual(['revoke:blob:preview-1', 'revoke:blob:preview-2']);
    expect(controller.snapshot.frames).toEqual([]);
    expect(controller.snapshot.batchId).toBeNull();
  });

  it.each(['adding', 'replacing'] as const)(
    'does not create or retain previews when disposed while %s waits for batch deletion',
    async (operation) => {
      const deletion = deferred<unknown>();
      const request = vi
        .fn()
        .mockResolvedValueOnce(createdBatch(1))
        .mockImplementationOnce(() => deletion.promise);
      const createObjectURL = vi.fn(() => `blob:preview-${createObjectURL.mock.calls.length}`);
      const revokeObjectURL = vi.fn();
      const controller = createController({request, createObjectURL, revokeObjectURL});
      await controller.addFiles([jpeg('one.jpg')]);
      await controller.start();
      let notifications = 0;
      controller.subscribe(() => {
        notifications += 1;
      });
      const pending = operation === 'adding'
        ? controller.addFiles([jpeg('two.jpg')])
        : controller.replaceFrame(controller.snapshot.frames[0].id, jpeg('two.jpg'));
      await Promise.resolve();
      controller.dispose();
      const notificationsAfterDispose = notifications;

      deletion.resolve(undefined);
      await pending;

      expect(controller.snapshot.frames).toEqual([]);
      expect(createObjectURL).toHaveBeenCalledTimes(1);
      expect(revokeObjectURL).toHaveBeenCalledTimes(1);
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:preview-1');
      expect(notifications).toBe(notificationsAfterDispose);
    },
  );

  it('serializes overlapping clear, remove and replace without double revocation or crashes', async () => {
    const deletion = deferred<unknown>();
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(2))
      .mockImplementationOnce(() => deletion.promise);
    const createObjectURL = vi.fn((candidate: Blob) => `blob:${(candidate as File).name}`);
    const revokeObjectURL = vi.fn();
    const controller = createController({request, createObjectURL, revokeObjectURL});
    await controller.addFiles([jpeg('one.jpg'), jpeg('two.jpg')]);
    await controller.start();
    const firstFrame = controller.snapshot.frames[0].id;
    const secondFrame = controller.snapshot.frames[1].id;

    const clearing = controller.clear();
    await Promise.resolve();
    const removing = controller.removeFrame(firstFrame);
    const replacing = controller.replaceFrame(secondFrame, jpeg('replacement.jpg'));
    deletion.resolve(undefined);
    await expect(Promise.all([clearing, removing, replacing])).resolves.toEqual([
      undefined,
      undefined,
      undefined,
    ]);

    expect(request.mock.calls.filter(([, options]) => options?.method === 'DELETE')).toHaveLength(1);
    expect(createObjectURL).toHaveBeenCalledTimes(2);
    expect(controller.snapshot.frames).toEqual([]);
    expect(revokeObjectURL.mock.calls.map(([url]) => url)).toEqual([
      'blob:one.jpg',
      'blob:two.jpg',
    ]);
  });

  it('disposes idempotently and ignores later mutations and notifications', async () => {
    const createObjectURL = vi.fn((candidate: Blob) => `blob:${(candidate as File).name}`);
    const revokeObjectURL = vi.fn();
    const controller = createController({createObjectURL, revokeObjectURL});
    await controller.addFiles([jpeg('one.jpg'), jpeg('two.jpg')]);
    let notifications = 0;
    controller.subscribe(() => {
      notifications += 1;
    });

    controller.dispose();
    controller.dispose();
    await controller.addFiles([jpeg('late.jpg')]);
    controller.selectFrame('draft-1');

    expect(controller.snapshot.frames).toEqual([]);
    expect(createObjectURL).toHaveBeenCalledTimes(2);
    expect(revokeObjectURL.mock.calls.map(([url]) => url)).toEqual([
      'blob:one.jpg',
      'blob:two.jpg',
    ]);
    expect(notifications).toBe(0);
  });

  it('changes model after batch deletion while preserving files, previews and order', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(2))
      .mockResolvedValueOnce(undefined);
    const revokeObjectURL = vi.fn();
    const controller = createController({request, revokeObjectURL});
    await controller.addFiles([jpeg('one.jpg'), jpeg('two.jpg')]);
    await controller.start();
    controller.applyJob(job('job-0', 'completed'));
    controller.applyJob(job('job-1', 'failed_recoverable'));
    const originals = controller.snapshot.frames.map(({file, name, previewUrl, id}) => ({
      file,
      name,
      previewUrl,
      id,
    }));

    await controller.changeModel('v2');

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/batches/batch-1', {method: 'DELETE'});
    expect(controller.snapshot.modelVersion).toBe('v2');
    expect(controller.snapshot.batchId).toBeNull();
    expect(
      controller.snapshot.frames.map(({file, name, previewUrl, id}) => ({
        file,
        name,
        previewUrl,
        id,
      })),
    ).toEqual(originals);
    expect(
      controller.snapshot.frames.every(({jobId, job, feedback}) => !jobId && !job && !feedback),
    ).toBe(true);
    expect(revokeObjectURL).not.toHaveBeenCalled();
  });

  it('keeps the old model and complete frame state when batch deletion fails', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(1))
      .mockRejectedValueOnce(new ApiError('delete_failed', 503));
    const {controller} = await startedController(1, request);
    const completed = job('job-0', 'completed');
    controller.applyJob(completed);
    const oldFrames = controller.snapshot.frames;

    await controller.changeModel('v2');

    expect(controller.snapshot.modelVersion).toBe('v1');
    expect(controller.snapshot.frames).toBe(oldFrames);
    expect(controller.snapshot.frames[0].job).toBe(completed);
    expect(controller.snapshot.error?.code).toBe('delete_failed');
    expect(controller.snapshot.modelPending).toBe(false);
  });

  it('changes model during an in-flight create and discards the stale server batch', async () => {
    const create = deferred<CreateAnalysisResponse>();
    const request = vi.fn(async (_path: string, options = {}) => {
      if ((options as {method?: string}).method === 'POST') {
        return create.promise;
      }
      return undefined;
    });
    const controller = createController({request});
    await controller.addFiles([jpeg('one.jpg')]);
    const pendingStart = controller.start();

    await controller.changeModel('v2');
    expect(controller.snapshot.submitting).toBe(false);
    create.resolve(createdBatch(1));
    await pendingStart;

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/batches/batch-1', {method: 'DELETE'});
    expect(controller.snapshot.modelVersion).toBe('v2');
    expect(controller.snapshot.batchId).toBeNull();
    expect(controller.snapshot.frames[0].jobId).toBeNull();
  });

  it('retains a malformed create batch when compensating cleanup fails and clears it only on explicit retry', async () => {
    const malformed = createdBatch(1, [createdJob(0, {id: ''})]);
    const request = vi
      .fn()
      .mockResolvedValueOnce(malformed)
      .mockRejectedValueOnce(new ApiError('delete_failed', 503))
      .mockResolvedValueOnce(undefined);
    const controller = createController({request});
    await controller.addFiles([jpeg('one.jpg')]);

    await controller.start();

    expect(
      (controller.snapshot as AnalysisSnapshot & {cleanupPendingBatchId?: string | null})
        .cleanupPendingBatchId,
    ).toBe('batch-1');
    expect(controller.snapshot.error?.code).toBe('delete_failed');
    await controller.start();
    expect(request.mock.calls.filter(([, options]) => options?.method === 'POST')).toHaveLength(1);

    await (
      controller as AnalysisController & {retryCleanup(): Promise<void>}
    ).retryCleanup();

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/batches/batch-1', {method: 'DELETE'});
    expect(
      (controller.snapshot as AnalysisSnapshot & {cleanupPendingBatchId?: string | null})
        .cleanupPendingBatchId,
    ).toBeNull();
    expect(controller.snapshot.error).toBeNull();
    expect(controller.snapshot.frames).toHaveLength(1);
  });

  it.each([
    [
      'adding',
      async (controller: AnalysisController) => controller.addFiles([jpeg('two.jpg')]),
      ['one.jpg', 'two.jpg'],
    ],
    [
      'removing',
      async (controller: AnalysisController) =>
        controller.removeFrame(controller.snapshot.frames[0].id),
      [],
    ],
    [
      'replacing',
      async (controller: AnalysisController) =>
        controller.replaceFrame(controller.snapshot.frames[0].id, jpeg('two.jpg')),
      ['two.jpg'],
    ],
  ])('cancels an in-flight create when %s frames', async (_label, mutate, expectedNames) => {
    const create = deferred<CreateAnalysisResponse>();
    const request = vi.fn(async (_path: string, options = {}) => {
      if ((options as {method?: string}).method === 'POST') {
        return create.promise;
      }
      return undefined;
    });
    const controller = createController({request});
    await controller.addFiles([jpeg('one.jpg')]);
    const pendingStart = controller.start();

    await mutate(controller);
    expect(controller.snapshot.submitting).toBe(false);
    create.resolve(createdBatch(1));
    await pendingStart;

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/batches/batch-1', {method: 'DELETE'});
    expect(controller.snapshot.frames.map(({name}) => name)).toEqual(expectedNames);
    expect(controller.snapshot.batchId).toBeNull();
  });

  it('stores feedback on the requested frame without changing its sibling', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(2))
      .mockResolvedValueOnce({feedback: {value: 'negative', comment: 'check'}});
    const {controller} = await startedController(2, request);
    controller.applyJob(job('job-0', 'completed'));
    controller.applyJob(job('job-1', 'completed'));
    controller.selectFrame(controller.snapshot.frames[1].id);

    await controller.submitFeedback(controller.snapshot.frames[1].id, 'negative', ' check ');

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/job-1/feedback', {
      method: 'POST',
      json: {value: 'negative', comment: 'check'},
    });
    expect(controller.snapshot.frames[0].feedback).toBeNull();
    expect(controller.snapshot.frames[1].feedback).toEqual({value: 'negative', comment: 'check'});
  });

  it('submits feedback for a rejected-quality result-bearing terminal frame', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(1))
      .mockResolvedValueOnce({feedback: {value: 'negative', comment: 'quality'}});
    const {controller} = await startedController(1, request);
    const frame = controller.snapshot.frames[0];
    controller.applyJob(job('job-0', 'rejected_quality'));

    await controller.submitFeedback(frame.id, 'negative', 'quality');

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/job-0/feedback', {
      method: 'POST',
      json: {value: 'negative', comment: 'quality'},
    });
    expect(controller.snapshot.frames[0].feedback).toEqual({
      value: 'negative',
      comment: 'quality',
    });
  });

  it('keeps delayed feedback pending, result, and error on their requested frames after selection changes', async () => {
    const firstFeedback = deferred<{feedback: {value: 'positive'; comment: null}}>();
    const secondFeedback = deferred<never>();
    const request = vi.fn((path: string) => {
      if (path.endsWith('/job-0/feedback')) {
        return firstFeedback.promise;
      }
      if (path.endsWith('/job-1/feedback')) {
        return secondFeedback.promise;
      }
      return Promise.resolve(createdBatch(2));
    });
    const {controller} = await startedController(2, request);
    const [frameA, frameB] = controller.snapshot.frames;
    controller.applyJob(job('job-0', 'completed'));
    controller.applyJob(job('job-1', 'completed'));

    const firstSubmission = controller.submitFeedback(frameA.id, 'positive', null);
    expect(controller.snapshot.frames[0].feedbackPending).toBe(true);
    expect(controller.snapshot.frames[1].feedbackPending).toBe(false);
    controller.selectFrame(frameB.id);
    firstFeedback.resolve({feedback: {value: 'positive', comment: null}});
    await firstSubmission;

    expect(request).toHaveBeenCalledWith('/api/v1/analyses/job-0/feedback', {
      method: 'POST',
      json: {value: 'positive', comment: null},
    });
    expect(controller.snapshot.frames[0].feedback).toEqual({value: 'positive', comment: null});
    expect(controller.snapshot.frames[1].feedback).toBeNull();

    const secondSubmission = controller.submitFeedback(frameB.id, 'negative', 'check');
    expect(controller.snapshot.frames[1].feedbackPending).toBe(true);
    controller.selectFrame(frameA.id);
    secondFeedback.reject(new ApiError('feedback_failed', 503));
    await secondSubmission;

    expect(controller.snapshot.frames[0].feedbackError).toBeNull();
    expect(controller.snapshot.frames[1].feedbackError).toBe('feedback_failed');
  });
});
