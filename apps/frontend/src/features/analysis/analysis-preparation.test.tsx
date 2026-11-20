import {fireEvent, render, screen, waitFor, within} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {describe, expect, it, vi} from 'vitest';

import type {AnalysisJobDto, CreateAnalysisResponse} from '../../api/contracts';
import {ApiError} from '../../api/client';
import {AppProviders} from '../../app/providers';
import {AnalysisWorkspace} from './AnalysisWorkspace';
import {AnalysisController} from './model';

let previewSequence = 0;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((accept, fail) => {
    resolve = accept;
    reject = fail;
  });
  return {promise, resolve, reject};
}

function createdBatch(count: number): CreateAnalysisResponse {
  return {
    batch_id: 'batch-ui',
    jobs: Array.from({length: count}, (_, index) => ({
      id: `job-${index}`,
      state: 'queued',
      frame_id: `backend-frame-${index}`,
      batch_position: index,
      polling_url: `/api/v1/analyses/job-${index}`,
    })),
  };
}

function makeController(request = vi.fn()) {
  return new AnalysisController({
    request,
    createObjectURL: () => `blob:ui-preview-${++previewSequence}`,
    revokeObjectURL: vi.fn(),
    loadExamples: async () => [
      new File(['one'], 'norm-sagittal.jpg', {type: 'image/jpeg'}),
      new File(['two'], 'norm-axial.jpg', {type: 'image/jpeg'}),
      new File(['three'], 'patology-sagittal.jpg', {type: 'image/jpeg'}),
      new File(['four'], 'patology-axial.jpg', {type: 'image/jpeg'}),
    ],
  });
}

function renderWorkspace(controller: AnalysisController) {
  return render(
    <AppProviders>
      <AnalysisWorkspace controller={controller} />
    </AppProviders>,
  );
}

function statusJob(
  id: string,
  state: AnalysisJobDto['state'],
  stage: string | null,
): AnalysisJobDto {
  return {
    id,
    state,
    stage,
    progress: state === 'completed' ? 100 : 50,
    attempt: 0,
    queue_position: state === 'queued' ? 3 : null,
    frames: [],
    result: null,
    error: state === 'failed_recoverable' ? {code: 'inference_failed'} : null,
    expires_at: '2026-09-23T00:00:00Z',
  };
}

describe('analysis preparation workspace', () => {
  it('offers upload or examples inside the empty dropzone only', async () => {
    const controller = makeController();
    const {container} = renderWorkspace(controller);
    const user = userEvent.setup();

    expect(container.querySelector('.example-bar')).not.toBeInTheDocument();
    expect(screen.getByRole('button', {name: 'Загрузить изображения'})).toBeVisible();
    expect(screen.getByRole('button', {name: 'Открыть примеры'})).toBeVisible();
    await user.click(screen.getByRole('button', {name: 'Открыть примеры'}));

    expect(await screen.findByText('4 из 10')).toBeInTheDocument();
    expect(screen.getAllByRole('button', {name: /Выбрать кадр/})).toHaveLength(4);
    expect(screen.queryByRole('button', {name: 'Открыть примеры'})).toBeNull();
  });

  it('adds a ten-file selection and keeps every thumbnail when a later selection is invalid', async () => {
    const controller = makeController();
    renderWorkspace(controller);
    const files = Array.from(
      {length: 10},
      (_, index) => new File([`${index}`], `${index}.jpg`, {type: 'image/jpeg'}),
    );

    const dropzone = screen.getByText('Перетащите УЗИ-кадры сюда').closest('div');
    fireEvent.drop(dropzone as HTMLElement, {dataTransfer: {files}});
    expect(await screen.findByText('10 из 10')).toBeInTheDocument();

    await controller.addFiles([new File(['bad'], 'notes.pdf', {type: 'application/pdf'})]);
    expect(await screen.findByRole('alert')).toHaveTextContent('notes.pdf');
    expect(screen.getAllByRole('button', {name: /Выбрать кадр/})).toHaveLength(10);
  });

  it('keeps the queue status hidden until analysis is explicitly started', async () => {
    const request = vi.fn().mockResolvedValue(createdBatch(2));
    const controller = makeController(request);
    await controller.addFiles([
      new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'}),
      new File(['two'], 'axial.jpg', {type: 'image/jpeg'}),
    ]);
    renderWorkspace(controller);
    const user = userEvent.setup();

    expect(screen.queryByRole('status')).toBeNull();
    expect(screen.getByRole('button', {name: 'Загрузить изображения'})).toBeVisible();

    await user.click(screen.getByRole('button', {name: 'Запустить анализ'}));

    expect(await screen.findByRole('status', {name: 'В очереди'})).toBeVisible();
    expect(request).toHaveBeenCalledOnce();
  });

  it('explains how to recover when replacement duplicates another frame', async () => {
    const controller = makeController();
    await controller.addFiles([
      new File(['one'], 'sagittal.jpg', {type: 'image/jpeg', lastModified: 1}),
      new File(['two'], 'axial.jpg', {type: 'image/jpeg', lastModified: 1}),
    ]);
    await controller.replaceFrame(
      controller.snapshot.frames[0].id,
      new File(['two'], 'axial.jpg', {type: 'image/jpeg', lastModified: 1}),
    );
    renderWorkspace(controller);

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Кадр уже есть в списке. Выберите другой файл или удалите существующий кадр; повторно загружать весь набор не нужно.',
    );
  });

  it('shows one batch status with aggregate counts and the selected frame stage', async () => {
    const request = vi.fn().mockResolvedValue(createdBatch(2));
    const controller = makeController(request);
    await controller.addFiles([
      new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'}),
      new File(['two'], 'axial.jpg', {type: 'image/jpeg'}),
    ]);
    await controller.start();
    controller.applyJob(statusJob('job-0', 'completed', null));
    controller.applyJob(statusJob('job-1', 'quality_check', 'quality_check'));
    renderWorkspace(controller);
    const user = userEvent.setup();

    expect(
      within(screen.getByRole('button', {name: 'Выбрать кадр 1'})).getByText('Анализ завершён'),
    ).toBeVisible();
    expect(
      within(screen.getByRole('button', {name: 'Выбрать кадр 2'})).getByText('Проверка качества'),
    ).toBeVisible();
    expect(
      within(screen.getByRole('button', {name: 'Выбрать кадр 2'})).getByText('50%'),
    ).toBeVisible();
    const initialStatus = screen.getByRole('status', {name: 'Анализ завершён'});
    expect(initialStatus).toHaveTextContent('Завершено 1 из 2');
    expect(initialStatus).toHaveTextContent('Обрабатывается: 1');
    expect(initialStatus).toHaveTextContent('В очереди: 0');
    expect(within(initialStatus).getByRole('progressbar')).toHaveAttribute('aria-valuenow', '75');
    expect(screen.getAllByRole('status')).toHaveLength(1);

    await user.click(screen.getByRole('button', {name: 'Выбрать кадр 2'}));
    await waitFor(() => expect(screen.getAllByText('Проверка качества')).toHaveLength(2));
    const selectedStatus = screen.getByRole('status', {name: 'Проверка качества'});
    expect(selectedStatus).toHaveTextContent('Завершено 1 из 2');
    expect(screen.getAllByRole('status')).toHaveLength(1);
  });

  it('retries only the selected recoverable frame from the preparation panel', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(createdBatch(2))
      .mockResolvedValueOnce({id: 'job-1', state: 'queued', attempt: 1});
    const controller = makeController(request);
    await controller.addFiles([
      new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'}),
      new File(['two'], 'axial.jpg', {type: 'image/jpeg'}),
    ]);
    await controller.start();
    controller.applyJob(statusJob('job-0', 'completed', null));
    controller.applyJob(statusJob('job-1', 'failed_recoverable', null));
    controller.selectFrame(controller.snapshot.frames[1].id);
    renderWorkspace(controller);
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', {name: 'Повторить анализ'}));

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/job-1/retry', {method: 'POST'});
    expect(controller.snapshot.frames[0].job?.state).toBe('completed');
  });

  it('disables a frame retry synchronously until the retry response arrives', async () => {
    const pendingRetry = deferred<{id: string; state: 'queued'; attempt: number}>();
    const request = vi.fn((path: string) =>
      path.endsWith('/retry') ? pendingRetry.promise : Promise.resolve(createdBatch(1)),
    );
    const controller = makeController(request);
    await controller.addFiles([new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'})]);
    await controller.start();
    controller.applyJob(statusJob('job-0', 'failed_recoverable', null));
    renderWorkspace(controller);

    const retry = screen.getByRole('button', {name: 'Повторить анализ'});
    fireEvent.click(retry);

    expect(retry).toBeDisabled();
    pendingRetry.resolve({id: 'job-0', state: 'queued', attempt: 1});
    await waitFor(() => expect(screen.queryByRole('button', {name: 'Повторить анализ'})).toBeNull());
  });

  it('shows an explicit cleanup retry after compensating deletion fails', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce({
        ...createdBatch(1),
        jobs: [{...createdBatch(1).jobs[0], id: ''}],
      })
      .mockRejectedValueOnce(new ApiError('delete_failed', 503))
      .mockResolvedValueOnce(undefined);
    const controller = makeController(request);
    await controller.addFiles([new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'})]);
    await controller.start();
    renderWorkspace(controller);
    const user = userEvent.setup();

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Не удалось удалить исследование. Повторите действие.',
    );
    await user.click(screen.getByRole('button', {name: 'Повторить очистку'}));

    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/batches/batch-ui', {
      method: 'DELETE',
    });
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
    expect(screen.getByRole('button', {name: 'Запустить анализ'})).toBeEnabled();
  });
});
