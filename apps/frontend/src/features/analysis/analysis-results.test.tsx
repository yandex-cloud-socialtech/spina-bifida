import {render, screen, waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {describe, expect, it, vi} from 'vitest';

import type {AnalysisJobDto, AnalysisResultDto} from '../../api/contracts';
import {ApiError} from '../../api/client';
import {AppProviders} from '../../app/providers';
import i18n from '../../i18n';
import {AnalysisWorkspace} from './AnalysisWorkspace';
import {AnalysisController} from './model';

function result(category: 'no_expressed_signal' | 'attention_required'): AnalysisResultDto {
  return {
    frames: [
      {
        frame_id: 'frame-1',
        analyzable: true,
        quality: {score: 0.94, accepted: true},
        plane: {type: 'sagittal', score: 0.91},
        roi: {
          box: [10, 20, 100, 140],
          display_box: [2, 8, 108, 152],
        },
        assessment: {category, score: category === 'attention_required' ? 0.82 : 0.14},
        explanations: ['Сигнал рассчитан по выделенной области интереса.'],
        warnings: [],
        assets: [
          {id: 'original-1', kind: 'original', url: '/assets/original-1', checksum: 'a'.repeat(64), width: 640, height: 480},
          {id: 'roi-1', kind: 'roi', url: '/assets/roi-1', checksum: 'b'.repeat(64), width: 106, height: 144},
          {id: 'heatmap-1', kind: 'heatmap', url: '/assets/heatmap-1', checksum: 'c'.repeat(64), width: 106, height: 144},
        ],
      },
    ],
    warnings: [],
    model: {version: 'v1', device: 'cpu', service: 'spina-bifida'},
    thresholds: {
      plane_detection: 0.1,
      quality: 0.5,
      assessment: 0.5,
      warning_margin: 0.1,
    },
  };
}

function terminalJob(
  analysisResult: AnalysisResultDto | null,
  state: AnalysisJobDto['state'] = 'completed',
  id = 'job-0',
  frameId = 'frame-1',
): AnalysisJobDto {
  return {
    id,
    state,
    stage: null,
    progress: 100,
    queue_position: null,
    attempt: 1,
    frames: [{id: frameId, original_asset_url: `/assets/original-${frameId}`}],
    result: analysisResult,
    error: state === 'rejected_quality' ? {code: 'insufficient_quality'} : null,
    expires_at: '2026-09-23T00:00:00Z',
  };
}

async function completedController(
  analysisResult = result('no_expressed_signal'),
  request: (path: string) => Promise<unknown> = vi.fn(async (path: string) => {
    if (path.endsWith('/feedback')) {
      return {feedback: {value: 'positive', comment: null}};
    }
    return {
      batch_id: 'batch-1',
      jobs: [{
        id: 'job-0',
        state: 'queued',
        frame_id: 'frame-1',
        batch_position: 0,
        polling_url: '/api/v1/analyses/job-0',
      }],
    };
  }),
) {
  const controller = new AnalysisController({
    request,
    createObjectURL: () => 'blob:original-preview',
    revokeObjectURL: vi.fn(),
  });
  await controller.addFiles([new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'})]);
  await controller.start();
  controller.applyJob(terminalJob(analysisResult));
  return {controller, request};
}

function renderWorkspace(controller: AnalysisController) {
  return render(
    <AppProviders>
      <AnalysisWorkspace controller={controller} />
    </AppProviders>,
  );
}

describe('analysis results', () => {
  it('renders three semantic metrics in the approved order', async () => {
    const {controller} = await completedController();
    renderWorkspace(controller);

    const names = ['Плоскость', 'Качество', 'Признаки патологии'];
    const sections = names.map((name) => screen.getByRole('heading', {name}).closest('section'));
    for (let index = 1; index < sections.length; index += 1) {
      const previous = sections[index - 1];
      const current = sections[index];
      expect(
        Boolean(
          (previous?.compareDocumentPosition(current as Node) ?? 0) &
            Node.DOCUMENT_POSITION_FOLLOWING,
        ),
      ).toBe(true);
    }
    expect(screen.getByText('Сагиттальная')).toBeVisible();
    expect(screen.getByText('Удовлетворительно')).toBeVisible();
    expect(screen.getByText('Не обнаружены')).toBeVisible();
    expect(screen.getByRole('progressbar', {name: 'Плоскость: 91%'})).toHaveAttribute(
      'aria-valuenow',
      '91',
    );
    const planeScale = screen.getByRole('progressbar', {name: 'Плоскость: 91%'});
    const qualityScale = screen.getByRole('progressbar', {name: 'Качество: 94%'});
    const pathologyScale = screen.getByRole('progressbar', {name: 'Признаки патологии: 14%'});
    for (const scale of [planeScale, qualityScale, pathologyScale]) {
      expect(scale.querySelector('.result-metric__fill')).toHaveClass(
        'result-metric__fill_success',
      );
      expect(scale.querySelector('.result-metric__zone')).toBeNull();
      expect(scale.querySelector('.result-metric__marker')).toBeNull();
    }
    expect(qualityScale.querySelector('.result-metric__fill')).toHaveStyle({width: '94%'});
    expect(pathologyScale.querySelector('.result-metric__fill')).toHaveStyle({width: '14%'});
    expect(screen.getByText('Не является вероятностью диагноза.')).toBeVisible();
    expect(screen.getByRole('button', {name: 'Очистить'})).toHaveClass('clear-action');
    expect(screen.getByRole('button', {name: 'Очистить'})).toHaveClass('g-button_width_max');
    for (const removedCopy of [
      'Технические сведения',
      'Результат ИИ не является медицинским заключением.',
      'Сервис предназначен только для исследовательского использования.',
      'Оценка рассчитана по выделенной области интереса.',
      'Качество изображения ниже порога анализа.',
    ]) {
      expect(screen.queryByText(removedCopy, {exact: false})).toBeNull();
    }
  });

  it('uses backend threshold outcomes instead of reclassifying rounded scores', async () => {
    const analysisResult = result('attention_required');
    analysisResult.frames[0] = {
      ...analysisResult.frames[0],
      quality: {score: 0.499, accepted: true},
      plane: {type: 'axial', score: 0.5},
      assessment: {category: 'no_expressed_signal', score: 0.5},
    };
    const {controller} = await completedController(analysisResult);

    renderWorkspace(controller);

    expect(screen.getByText('Аксиальная')).toBeVisible();
    expect(screen.getByText('Удовлетворительно')).toBeVisible();
    expect(screen.getByText('Не обнаружены')).toBeVisible();
  });

  it('uses the applied server margin to show borderline scores in yellow', async () => {
    const analysisResult = result('no_expressed_signal');
    analysisResult.frames[0] = {
      ...analysisResult.frames[0],
      quality: {score: 0.59, accepted: true},
      plane: {type: 'axial', score: 0.19},
      assessment: {category: 'no_expressed_signal', score: 0.41},
    };
    const {controller} = await completedController(analysisResult);

    renderWorkspace(controller);

    expect(screen.getByText('Удовлетворительно')).toBeVisible();
    expect(screen.getByText('Не обнаружены')).toBeVisible();
    for (const name of ['Плоскость: 19%', 'Качество: 59%', 'Признаки патологии: 41%']) {
      expect(screen.getByRole('progressbar', {name}).querySelector('.result-metric__fill')).toHaveClass(
        'result-metric__fill_warning',
      );
    }
  });

  it('shows adverse quality and pathology scores in red', async () => {
    const analysisResult = result('attention_required');
    analysisResult.frames[0] = {
      ...analysisResult.frames[0],
      quality: {score: 0.2, accepted: false},
      assessment: {category: 'attention_required', score: 0.82},
    };
    const {controller} = await completedController(analysisResult);

    renderWorkspace(controller);

    expect(screen.getByText('Неудовлетворительно')).toBeVisible();
    expect(screen.getByText('Обнаружены')).toBeVisible();
    for (const name of ['Качество: 20%', 'Признаки патологии: 82%']) {
      expect(screen.getByRole('progressbar', {name}).querySelector('.result-metric__fill')).toHaveClass(
        'result-metric__fill_danger',
      );
    }
  });

  it('never exposes non-finite or out-of-range metric values through ARIA', async () => {
    const analysisResult = result('attention_required');
    analysisResult.frames[0] = {
      ...analysisResult.frames[0],
      plane: {type: 'axial', score: Number.NaN},
      quality: {score: Number.POSITIVE_INFINITY, accepted: true},
      assessment: {category: 'attention_required', score: -0.4},
    };
    const {controller} = await completedController(analysisResult);

    renderWorkspace(controller);

    expect(screen.getAllByText('Не определено')).toHaveLength(2);
    expect(screen.queryByRole('progressbar', {name: /Плоскость/})).toBeNull();
    expect(screen.queryByRole('progressbar', {name: /Качество/})).toBeNull();
    expect(screen.getByRole('progressbar', {name: 'Признаки патологии: 0%'})).toHaveAttribute(
      'aria-valuenow',
      '0',
    );
    for (const progressbar of screen.getAllByRole('progressbar')) {
      expect(Number(progressbar.getAttribute('aria-valuenow'))).toBeGreaterThanOrEqual(0);
      expect(Number(progressbar.getAttribute('aria-valuenow'))).toBeLessThanOrEqual(100);
    }
  });

  it('keeps original, ROI, and heatmap aligned and toggles overlays independently', async () => {
    const firstResult = result('attention_required');
    const secondResult: AnalysisResultDto = {
      ...firstResult,
      frames: [
        {
          ...firstResult.frames[0],
          frame_id: 'frame-2',
          assets: [
            {id: 'original-2', kind: 'original', url: '/assets/original-2', checksum: 'd'.repeat(64), width: 640, height: 480},
            {id: 'roi-2', kind: 'roi', url: '/assets/roi-2', checksum: 'e'.repeat(64), width: 106, height: 144},
          ],
        },
      ],
    };
    const request = vi.fn().mockResolvedValue({
      batch_id: 'batch-1',
      jobs: [
        {
          id: 'job-0',
          state: 'queued',
          frame_id: 'frame-1',
          batch_position: 0,
          polling_url: '/api/v1/analyses/job-0',
        },
        {
          id: 'job-1',
          state: 'queued',
          frame_id: 'frame-2',
          batch_position: 1,
          polling_url: '/api/v1/analyses/job-1',
        },
      ],
    });
    const controller = new AnalysisController({
      request,
      createObjectURL: (file) => `blob:${(file as File).name}`,
      revokeObjectURL: vi.fn(),
    });
    await controller.addFiles([
      new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'}),
      new File(['two'], 'axial.jpg', {type: 'image/jpeg'}),
    ]);
    await controller.start();
    controller.applyJob(terminalJob(firstResult, 'completed', 'job-0', 'frame-1'));
    controller.applyJob(terminalJob(secondResult, 'completed', 'job-1', 'frame-2'));
    const {container} = renderWorkspace(controller);
    const user = userEvent.setup();
    const jobCalls = request.mock.calls.length;

    expect(screen.getByRole('img', {name: 'Оригинал'})).toHaveAttribute('src', '/assets/original-1');
    const roiToggle = screen.getByRole('button', {name: 'ROI'});
    const heatmapToggle = screen.getByRole('button', {name: 'Heatmap'});
    expect(roiToggle).toHaveAttribute('aria-pressed', 'true');
    expect(heatmapToggle).toHaveAttribute('aria-pressed', 'true');

    const roi = container.querySelector<HTMLElement>('.viewer-roi');
    const heatmap = container.querySelector<HTMLImageElement>('.viewer-heatmap');
    expect(roi).not.toBeNull();
    expect(heatmap).toHaveAttribute('src', '/assets/heatmap-1');
    expect(roi?.style.left).toBe('1.5625%');
    expect(roi?.style.top).toBe('4.166666666666666%');
    expect(roi?.style.width).toBe('14.0625%');
    expect(roi?.style.height).toBe('25%');
    expect(heatmap?.style.left).toBe('0.3125%');
    expect(heatmap?.style.top).toBe('1.6666666666666667%');
    expect(heatmap?.style.width).toBe('16.5625%');
    expect(heatmap?.style.height).toBe('30%');
    expect(roi?.closest('.viewer-transform')).toBe(heatmap?.closest('.viewer-transform'));

    await user.click(roiToggle);
    expect(container.querySelector('.viewer-roi')).toBeNull();
    expect(container.querySelector('.viewer-heatmap')).not.toBeNull();
    await user.click(roiToggle);
    expect(container.querySelector('.viewer-roi')).not.toBeNull();

    await user.click(screen.getByRole('button', {name: 'Увеличить масштаб'}));
    expect(screen.getByLabelText('Масштаб 125%')).toBeVisible();
    expect(container.querySelector('.viewer-transform')).toHaveStyle({
      transform: 'translate3d(0px, 0px, 0) scale(1.25)',
    });

    await user.click(screen.getByRole('button', {name: 'Выбрать кадр 2'}));
    expect(screen.getByRole('img', {name: 'Оригинал'})).toHaveAttribute('src', '/assets/original-2');
    expect(await screen.findByLabelText('Масштаб 100%')).toBeVisible();
    expect(screen.getByRole('button', {name: /Heatmap/})).toBeDisabled();
    expect(container.querySelector('.viewer-roi')).not.toBeNull();
    expect(container.querySelector('.viewer-heatmap')).toBeNull();
    expect(request).toHaveBeenCalledTimes(jobCalls);
  });

  it('shows no pathology score when the assessment is unavailable', async () => {
    const rejected = result('no_expressed_signal');
    rejected.frames[0] = {
      ...rejected.frames[0],
      analyzable: false,
      quality: {score: 0.18, accepted: false},
      assessment: null,
    };
    const {controller} = await completedController(rejected);
    controller.applyJob(terminalJob(rejected, 'rejected_quality'));
    renderWorkspace(controller);

    expect(screen.getByText('Неудовлетворительно')).toBeVisible();
    expect(screen.getByText('Не определено')).toBeVisible();
    expect(screen.queryByRole('progressbar', {name: /Признаки патологии/})).toBeNull();
  });

  it('offers and submits feedback for a rejected-quality result', async () => {
    const rejected = result('no_expressed_signal');
    rejected.frames[0] = {
      ...rejected.frames[0],
      quality: {score: 0.18, accepted: false},
    };
    const request = vi
      .fn()
      .mockResolvedValueOnce({
        batch_id: 'batch-1',
        jobs: [{
          id: 'job-0',
          state: 'queued',
          frame_id: 'frame-1',
          batch_position: 0,
          polling_url: '/api/v1/analyses/job-0',
        }],
      })
      .mockResolvedValueOnce({feedback: {value: 'negative', comment: 'quality'}});
    const {controller} = await completedController(rejected, request);
    controller.applyJob(terminalJob(rejected, 'rejected_quality'));
    renderWorkspace(controller);
    const user = userEvent.setup();

    await user.click(screen.getByRole('radio', {name: 'Требуется проверка'}));
    await user.type(screen.getByLabelText('Комментарий'), 'quality');
    await user.click(screen.getByRole('button', {name: 'Отправить'}));

    expect(await screen.findByRole('status')).toHaveTextContent('Обратная связь отправлена');
    expect(request).toHaveBeenLastCalledWith('/api/v1/analyses/job-0/feedback', {
      method: 'POST',
      json: {value: 'negative', comment: 'quality'},
    });
  });

  it.each([
    ['kk', 'Анықталмады', 'Анықтау мүмкін болмады'],
    ['uz', 'Aniqlanmadi', 'Aniqlab bo‘lmadi'],
  ])('distinguishes negative pathology from an unavailable assessment in %s', async (
    locale,
    negative,
    unavailable,
  ) => {
    await i18n.changeLanguage(locale);
    const withoutPathology = result('no_expressed_signal');
    withoutPathology.frames[0] = {...withoutPathology.frames[0], assessment: null};
    const {controller: unavailableController} = await completedController(withoutPathology);
    const unavailableView = renderWorkspace(unavailableController);

    expect(screen.getByText(unavailable)).toBeVisible();
    unavailableView.unmount();

    const {controller: negativeController} = await completedController(result('no_expressed_signal'));
    renderWorkspace(negativeController);

    expect(screen.getByText(negative)).toBeVisible();
    expect(screen.queryByText(unavailable)).toBeNull();
  });

  it('submits feedback once and clears only through the bottom action', async () => {
    let feedbackCalls = 0;
    const request = vi.fn(async (path: string) => {
      if (path.endsWith('/feedback')) {
        feedbackCalls += 1;
        await Promise.resolve();
        return {feedback: {value: 'negative', comment: 'Нужно сопоставить'}};
      }
      return {
        batch_id: 'batch-1',
        jobs: [{
          id: 'job-0',
          state: 'queued',
          frame_id: 'frame-1',
          batch_position: 0,
          polling_url: '/api/v1/analyses/job-0',
        }],
      };
    });
    const {controller} = await completedController(result('attention_required'), request);
    renderWorkspace(controller);
    const user = userEvent.setup();

    await user.click(screen.getByRole('radio', {name: 'Требуется проверка'}));
    await user.type(screen.getByLabelText('Комментарий'), 'Нужно сопоставить');
    await user.dblClick(screen.getByRole('button', {name: 'Отправить'}));

    expect(await screen.findByRole('status')).toHaveTextContent('Обратная связь отправлена');
    expect(feedbackCalls).toBe(1);
    expect(screen.getByText('Обнаружены')).toBeVisible();
    await user.click(screen.getByRole('button', {name: 'Очистить'}));
    await waitFor(() => expect(screen.getByText('0 из 10')).toBeInTheDocument());
  });

  it('does not carry an unsent feedback draft to another completed frame', async () => {
    const firstResult = result('no_expressed_signal');
    const secondResult = result('attention_required');
    secondResult.frames[0] = {...secondResult.frames[0], frame_id: 'frame-2'};
    const request = vi.fn().mockResolvedValue({
      batch_id: 'batch-1',
      jobs: [
        {
          id: 'job-0',
          state: 'queued',
          frame_id: 'frame-1',
          batch_position: 0,
          polling_url: '/api/v1/analyses/job-0',
        },
        {
          id: 'job-1',
          state: 'queued',
          frame_id: 'frame-2',
          batch_position: 1,
          polling_url: '/api/v1/analyses/job-1',
        },
      ],
    });
    const controller = new AnalysisController({
      request,
      createObjectURL: (candidate) => `blob:${(candidate as File).name}`,
      revokeObjectURL: vi.fn(),
    });
    await controller.addFiles([
      new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'}),
      new File(['two'], 'axial.jpg', {type: 'image/jpeg'}),
    ]);
    await controller.start();
    controller.applyJob(terminalJob(firstResult, 'completed', 'job-0', 'frame-1'));
    controller.applyJob(terminalJob(secondResult, 'completed', 'job-1', 'frame-2'));
    renderWorkspace(controller);
    const user = userEvent.setup();

    await user.click(screen.getByRole('radio', {name: 'Требуется проверка'}));
    await user.type(screen.getByLabelText('Комментарий'), 'Черновик первого кадра');
    await user.click(screen.getByRole('button', {name: 'Выбрать кадр 2'}));

    expect(screen.getByRole('radio', {name: 'Результат полезен'})).not.toBeChecked();
    expect(screen.getByRole('radio', {name: 'Требуется проверка'})).not.toBeChecked();
    expect(screen.getByLabelText('Комментарий')).toHaveValue('');
    expect(screen.getByRole('button', {name: 'Отправить'})).toBeDisabled();
  });

  it('shows a recoverable delete error on a selected terminal result and preserves it', async () => {
    let rejectDeletion!: (reason: unknown) => void;
    const deletion = new Promise<never>((_resolve, reject) => {
      rejectDeletion = reject;
    });
    const request = vi.fn(async (path: string, options = {}) => {
      if ((options as {method?: string}).method === 'DELETE') {
        return deletion;
      }
      return {
        batch_id: 'batch-1',
        jobs: [{
          id: 'job-0',
          state: 'queued',
          frame_id: 'frame-1',
          batch_position: 0,
          polling_url: '/api/v1/analyses/job-0',
        }],
      };
    });
    const {controller} = await completedController(result('attention_required'), request);
    const oldFrames = controller.snapshot.frames;
    const oldJob = oldFrames[0].job;
    renderWorkspace(controller);
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', {name: 'Очистить'}));
    expect(screen.getByRole('button', {name: 'Очистить'})).toBeDisabled();
    expect(screen.getByRole('button', {name: 'Удалить кадр 1'})).toBeDisabled();
    rejectDeletion(new ApiError('delete_failed', 503));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Не удалось удалить исследование. Повторите действие.',
    );
    expect(screen.getByText('Обнаружены')).toBeVisible();
    expect(controller.snapshot.frames).toBe(oldFrames);
    expect(controller.snapshot.frames[0].job).toBe(oldJob);
    expect(screen.getByRole('button', {name: 'Очистить'})).toBeEnabled();
  });
});
