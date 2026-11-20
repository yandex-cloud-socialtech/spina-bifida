import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

import type {Page, Route} from '@playwright/test';

export type JobKind =
  | 'queued'
  | 'processing'
  | 'quality'
  | 'negative'
  | 'attention'
  | 'recoverable';

type JobScenario = JobKind | readonly JobKind[];

interface MockApplicationOptions {
  loginRequired?: boolean;
  jobs?: readonly JobScenario[];
  defaultKind?: JobKind;
  initialLocale?: 'ru' | 'en' | 'kk' | 'uz';
}

interface Submission {
  batchId: string;
  uploadedNames: string[];
  modelVersion: 'v1' | 'v2';
}

interface FeedbackRequest {
  jobId: string;
  value: 'positive' | 'negative';
  comment: string | null;
}

const examplePaths = [
  'norm-sagittal.jpg',
  'norm-axial.jpg',
  'patology-sagittal.jpg',
  'patology-axial.jpg',
] as const;

function frame(frameId: string, kind: JobKind) {
  const index = Number(frameId.split('-').at(-1) ?? '1');
  const axial = index % 2 === 0;
  const plane = axial ? 'axial' : 'sagittal';
  const qualityAccepted = kind !== 'quality';
  const attention = kind === 'attention';
  return {
    frame_id: frameId,
    analyzable: true,
    quality: {score: qualityAccepted ? 0.94 : 0.18, accepted: qualityAccepted},
    plane: {type: plane, score: 0.91},
    roi: {
      box: [10, 20, 100, 140],
      display_box: [2, 8, 108, 152],
    },
    assessment: {
      category: attention ? 'attention_required' : 'no_expressed_signal',
      score: attention ? 0.82 : 0.14,
    },
    explanations: ['assessment_regions'],
    warnings: qualityAccepted ? [] : ['quality_below_threshold'],
    assets: [
      {
        id: `original-${frameId}`,
        kind: 'original',
        url: `/examples/norm-${plane}.jpg`,
        checksum: 'a'.repeat(64),
        width: 640,
        height: 480,
        media_type: 'image/jpeg',
      },
      {
        id: `roi-${frameId}`,
        kind: 'roi',
        url: `/examples/patology-${plane}.jpg`,
        checksum: 'b'.repeat(64),
        width: 106,
        height: 144,
        media_type: 'image/jpeg',
      },
      {
        id: `heatmap-${frameId}`,
        kind: 'heatmap',
        url: `/examples/patology-${plane}.jpg`,
        checksum: 'c'.repeat(64),
        width: 106,
        height: 144,
        media_type: 'image/jpeg',
      },
    ],
  };
}

function job(jobId: string, frameId: string, kind: JobKind, modelVersion: 'v1' | 'v2') {
  const processing = kind === 'processing';
  const recoverable = kind === 'recoverable';
  const quality = kind === 'quality';
  const queued = kind === 'queued';
  const terminal = !processing && !recoverable && !queued;
  return {
    id: jobId,
    state: queued
      ? 'queued'
      : processing
        ? 'roi_detection'
        : recoverable
          ? 'failed_recoverable'
          : quality
            ? 'rejected_quality'
            : 'completed',
    stage: processing ? 'roi_detection' : null,
    progress: queued ? 0 : processing ? 58 : recoverable ? 42 : 100,
    queue_position: queued ? 3 : null,
    attempt: recoverable ? 1 : 0,
    frames: [{id: frameId, original_asset_url: `/assets/${frameId}`}],
    result: terminal
      ? {
          frames: [frame(frameId, kind)],
          warnings: [],
          model: {version: modelVersion, device: 'cpu', service: 'spina-bifida'},
          thresholds: {
            plane_detection: 0.1,
            quality: 0.5,
            assessment: 0.5,
            warning_margin: 0.1,
          },
        }
      : null,
    error: recoverable
      ? {code: 'inference_failed'}
      : quality
        ? {code: 'insufficient_quality'}
        : null,
    expires_at: '2026-09-24T00:00:00Z',
  };
}

function createdBatch(uploadedNames: readonly string[], sequence: number) {
  const batchId = `batch-${sequence}`;
  return {
    batch_id: batchId,
    jobs: uploadedNames.map((_, index) => {
      const id = sequence === 1 ? `job-${index + 1}` : `job-${sequence}-${index + 1}`;
      return {
        id,
        state: 'queued',
        frame_id: `frame-${index + 1}`,
        batch_position: index,
        polling_url: `/api/v1/analyses/${id}`,
      };
    }),
  };
}

export async function uploadedFileNames(route: Route): Promise<string[]> {
  const body = route.request().postDataBuffer();
  if (!body) {
    return [];
  }
  return Array.from(
    body.toString('latin1').matchAll(/name="frames"; filename="([^"]+)"/g),
    (match) => match[1],
  );
}

function uploadedModelVersion(route: Route): 'v1' | 'v2' {
  const body = route.request().postDataBuffer()?.toString('latin1') ?? '';
  return /name="model_version"\r\n\r\nv2\r\n/.test(body) ? 'v2' : 'v1';
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});
}

export async function mockApplication(page: Page, options: MockApplicationOptions = {}) {
  let loggedIn = !options.loginRequired;
  let termsAccepted = !options.loginRequired;
  let locale: 'ru' | 'en' | 'kk' | 'uz' = options.initialLocale ?? 'ru';
  let batchSequence = 0;
  const submissions: Submission[] = [];
  const deletedBatchIds: string[] = [];
  const feedbackRequests: FeedbackRequest[] = [];
  const retriedJobIds: string[] = [];
  const preferenceLocales: string[] = [];
  const jobReads = new Map<string, number>();
  const jobAssignments = new Map<string, {frameId: string; scenario: JobScenario; modelVersion: 'v1' | 'v2'}>();
  const retried = new Set<string>();
  let termsAccepts = 0;

  const authPayload = () => ({
    user: {
      id: 'user-1',
      email: 'doctor.with.a.very.long.address@example.test',
      role: 'user',
      status: 'active',
      locale,
    },
    csrf_token: 'csrf-test',
    terms_accepted: termsAccepted,
    terms_version: '2026-09-22',
  });

  await page.route('**/api/v1/**', async (route) => {
    const request = route.request();
    const method = request.method();
    const url = new URL(request.url());
    if (url.pathname === '/api/v1/auth/me') {
      if (!loggedIn) {
        await json(route, {
          error: {code: 'authentication_required', params: {}, request_id: 'e2e'},
          csrf_token: 'csrf-test',
        }, 401);
        return;
      }
      await json(route, authPayload());
      return;
    }
    if (url.pathname === '/api/v1/auth/login' && method === 'POST') {
      loggedIn = true;
      termsAccepted = false;
      await json(route, authPayload());
      return;
    }
    if (url.pathname === '/api/v1/auth/terms/accept' && method === 'POST') {
      termsAccepted = true;
      termsAccepts += 1;
      await json(route, {terms_accepted: true, terms_version: '2026-09-22'});
      return;
    }
    if (url.pathname === '/api/v1/auth/preferences' && method === 'PATCH') {
      const requested = (request.postDataJSON() as {locale: typeof locale}).locale;
      locale = requested;
      preferenceLocales.push(requested);
      await json(route, {user: authPayload().user});
      return;
    }
    if (url.pathname === '/api/v1/auth/logout' && method === 'POST') {
      loggedIn = false;
      await route.fulfill({status: 204, body: ''});
      return;
    }
    if (url.pathname === '/api/v1/system/info') {
      await json(route, {
        application_version: '0.1.0',
        model_versions: ['v1', 'v2'],
        default_model_version: 'v2',
        schema_version: '0003',
        device: 'cpu',
      });
      return;
    }
    if (url.pathname === '/api/v1/analyses' && method === 'POST') {
      const uploadedNames = await uploadedFileNames(route);
      const modelVersion = uploadedModelVersion(route);
      batchSequence += 1;
      const created = createdBatch(uploadedNames, batchSequence);
      created.jobs.forEach((createdJob, index) => {
        jobAssignments.set(createdJob.id, {
          frameId: createdJob.frame_id,
          scenario: options.jobs?.[index] ?? options.defaultKind ?? 'negative',
          modelVersion,
        });
      });
      submissions.push({batchId: created.batch_id, uploadedNames, modelVersion});
      await json(route, created, 202);
      return;
    }
    const batchDelete = url.pathname.match(/^\/api\/v1\/analyses\/batches\/([^/]+)$/);
    if (batchDelete && method === 'DELETE') {
      deletedBatchIds.push(batchDelete[1]);
      await route.fulfill({status: 204, body: ''});
      return;
    }
    const retry = url.pathname.match(/^\/api\/v1\/analyses\/([^/]+)\/retry$/);
    if (retry && method === 'POST') {
      retried.add(retry[1]);
      retriedJobIds.push(retry[1]);
      await json(route, {id: retry[1], state: 'queued', attempt: 2});
      return;
    }
    const feedback = url.pathname.match(/^\/api\/v1\/analyses\/([^/]+)\/feedback$/);
    if (feedback && method === 'POST') {
      const payload = request.postDataJSON() as Omit<FeedbackRequest, 'jobId'>;
      feedbackRequests.push({jobId: feedback[1], ...payload});
      await json(route, {feedback: payload});
      return;
    }
    const jobRead = url.pathname.match(/^\/api\/v1\/analyses\/([^/]+)$/);
    if (jobRead && method === 'GET') {
      const jobId = jobRead[1];
      const assignment = jobAssignments.get(jobId);
      if (!assignment) {
        await json(route, {error: {code: 'analysis_not_found', params: {}, request_id: 'e2e'}}, 404);
        return;
      }
      const reads = (jobReads.get(jobId) ?? 0) + 1;
      jobReads.set(jobId, reads);
      const scenario = assignment.scenario;
      const kinds = Array.isArray(scenario) ? scenario : [scenario];
      const kind = retried.has(jobId) ? 'negative' : kinds[Math.min(reads - 1, kinds.length - 1)];
      await json(route, job(jobId, assignment.frameId, kind, assignment.modelVersion));
      return;
    }
    await json(route, {error: {code: 'not_found', params: {}, request_id: 'e2e'}}, 404);
  });

  return {
    submissions,
    deletedBatchIds,
    feedbackRequests,
    retriedJobIds,
    preferenceLocales,
    termsAccepts: () => termsAccepts,
    jobReads: (jobId: string) => jobReads.get(jobId) ?? 0,
  };
}

export async function prepareFourExamples(page: Page) {
  await page.goto('/');
  await page.getByRole('button', {name: 'Открыть примеры'}).click();
  await page.getByText('4 из 10').waitFor();
}

export async function prepareTenFrames(page: Page) {
  await page.goto('/');
  const files = Array.from({length: 10}, (_, index) => {
    const source = examplePaths[index % examplePaths.length];
    return {
      name: `acceptance-${index + 1}-${source}`,
      mimeType: 'image/jpeg',
      buffer: readFileSync(resolve(process.cwd(), 'public', 'examples', source)),
    };
  });
  await page.getByLabel('Загрузить изображения').setInputFiles(files);
  await page.getByText('10 из 10').waitFor();
}

export async function startAnalysis(page: Page) {
  await page.getByRole('button', {name: 'Запустить анализ'}).click();
}

export async function assertNoPageScroll(page: Page) {
  await page.waitForFunction(() => document.fonts.status === 'loaded');
  return page.evaluate(
    () =>
      document.documentElement.scrollHeight <= window.innerHeight &&
      document.documentElement.scrollWidth <= window.innerWidth,
  );
}
