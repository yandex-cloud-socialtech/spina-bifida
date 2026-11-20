import {render} from '@testing-library/react';
import {axe} from 'vitest-axe';
import {describe, expect, it, vi} from 'vitest';

import type {AnalysisJobDto, AnalysisResultDto} from '../../api/contracts';
import {AppProviders} from '../../app/providers';
import i18n from '../../i18n';
import {AnalysisWorkspace} from './AnalysisWorkspace';
import {AnalysisController} from './model';

function analysisResult(attention = false, accepted = true): AnalysisResultDto {
  return {
    frames: ['frame-1', 'frame-2'].map((frameId, index) => ({
      frame_id: frameId,
      analyzable: accepted,
      quality: {score: accepted ? 0.94 : 0.18, accepted},
      plane: {type: index ? 'axial' : 'sagittal', score: 0.91},
      roi: accepted
        ? {
            box: [10, 20, 100, 140],
            display_box: [2, 8, 108, 152],
          }
        : null,
      assessment: accepted
        ? {category: attention ? 'attention_required' : 'no_expressed_signal', score: 0.82}
        : null,
      explanations: ['Техническое пояснение результата.'],
      warnings: [],
      assets: [
        {id: `original-${frameId}`, kind: 'original', url: `/assets/${frameId}`, checksum: 'a'.repeat(64), width: 640, height: 480},
        {id: `roi-${frameId}`, kind: 'roi', url: `/assets/${frameId}-roi`, checksum: 'b'.repeat(64), width: 106, height: 144},
        {id: `heatmap-${frameId}`, kind: 'heatmap', url: `/assets/${frameId}-heatmap`, checksum: 'c'.repeat(64), width: 106, height: 144},
      ],
    })),
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

function job(
  state: AnalysisJobDto['state'],
  result: AnalysisResultDto | null,
): AnalysisJobDto {
  return {
    id: 'job-0',
    state,
    stage: state === 'roi_detection' ? 'roi_detection' : null,
    progress: state === 'roi_detection' ? 58 : 100,
    queue_position: null,
    attempt: 1,
    frames: ['frame-1', 'frame-2'].map((id) => ({id, original_asset_url: `/assets/${id}`})),
    result,
    error: state === 'failed_recoverable' ? {code: 'inference_failed'} : null,
    expires_at: '2026-09-23T00:00:00Z',
  };
}

async function populatedController() {
  const controller = new AnalysisController({
    request: vi.fn(async (path: string) =>
      path.endsWith('/feedback')
        ? {feedback: {value: 'positive', comment: null}}
        : {
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
          },
    ),
    createObjectURL: (file) => `blob:${(file as File).name}`,
    revokeObjectURL: vi.fn(),
  });
  await controller.addFiles([
    new File(['one'], 'sagittal.jpg', {type: 'image/jpeg'}),
    new File(['two'], 'axial.jpg', {type: 'image/jpeg'}),
  ]);
  await controller.start();
  return controller;
}

describe('diagnostic canvas accessibility', () => {
  it('has no automated accessibility violations in the empty state', async () => {
    const controller = new AnalysisController({
      createObjectURL: () => 'blob:preview',
      revokeObjectURL: vi.fn(),
    });
    const {container} = render(
      <AppProviders>
        <AnalysisWorkspace controller={controller} />
      </AppProviders>,
    );

    expect((await axe(container)).violations).toEqual([]);
  });

  it('keeps primary controls labelled with long localized content at 200% scale', async () => {
    document.documentElement.style.fontSize = '200%';
    await i18n.changeLanguage('kk');
    const controller = new AnalysisController({
      createObjectURL: () => 'blob:preview',
      revokeObjectURL: vi.fn(),
    });
    await controller.addFiles([new File(['one'], 'өте-ұзын-кадр-атауы.jpg', {type: 'image/jpeg'})]);
    const {container, getByRole} = render(
      <AppProviders>
        <AnalysisWorkspace controller={controller} />
      </AppProviders>,
    );

    expect(getByRole('button', {name: 'Талдауды бастау'})).toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
    document.documentElement.style.fontSize = '';
  });

  it('labels independent overlay toggles and every zoom control', async () => {
    const controller = await populatedController();
    const completed = analysisResult(false);
    controller.applyJob(job('completed', completed));
    controller.applyJob({...job('completed', completed), id: 'job-1'});
    const view = render(
      <AppProviders>
        <AnalysisWorkspace controller={controller} />
      </AppProviders>,
    );

    expect(view.getByRole('button', {name: 'ROI'})).toHaveAttribute('aria-pressed', 'true');
    expect(view.getByRole('button', {name: 'Heatmap'})).toHaveAttribute('aria-pressed', 'true');
    expect(view.getByRole('button', {name: 'Уменьшить масштаб'})).toBeEnabled();
    expect(view.getByRole('button', {name: 'Увеличить масштаб'})).toBeEnabled();
    expect(view.getByRole('button', {name: 'По размеру'})).toBeEnabled();
    expect((await axe(view.container)).violations).toEqual([]);
  });

  it('has no automated violations in every mandatory workflow state', async () => {
    const scenarios: Array<{
      state: AnalysisJobDto['state'] | 'two_frame' | 'feedback_sent';
      result: AnalysisResultDto | null;
    }> = [
      {state: 'two_frame', result: null},
      {state: 'roi_detection', result: null},
      {state: 'rejected_quality', result: analysisResult(false, false)},
      {state: 'completed', result: analysisResult(false)},
      {state: 'completed', result: analysisResult(true)},
      {state: 'failed_recoverable', result: null},
      {state: 'feedback_sent', result: analysisResult(true)},
    ];

    for (const scenario of scenarios) {
      const controller = await populatedController();
      if (scenario.state !== 'two_frame') {
        controller.applyJob(
          job(scenario.state === 'feedback_sent' ? 'completed' : scenario.state, scenario.result),
        );
      }
      if (scenario.state === 'feedback_sent') {
        await controller.submitFeedback(controller.snapshot.frames[0].id, 'positive', null);
      }
      const view = render(
        <AppProviders>
          <AnalysisWorkspace controller={controller} />
        </AppProviders>,
      );
      expect((await axe(view.container)).violations).toEqual([]);
      view.unmount();
    }
  });
});
