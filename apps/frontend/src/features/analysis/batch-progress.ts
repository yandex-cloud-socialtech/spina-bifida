import type {AnalysisJobDto, JobState} from '../../api/contracts';

const ACTIVE_STATES = new Set<JobState>([
  'validating',
  'quality_check',
  'plane_detection',
  'roi_detection',
  'scoring',
  'explanation',
]);

const TERMINAL_STATES = new Set<JobState>([
  'completed',
  'rejected_quality',
  'failed_recoverable',
  'expired',
  'deleted',
]);

export interface BatchProgress {
  total: number;
  completed: number;
  active: number;
  queued: number;
  percent: number;
}

function clampPercent(value: number): number {
  return Math.min(100, Math.max(0, value));
}

export function calculateBatchProgress(
  jobs: readonly (AnalysisJobDto | null)[],
): BatchProgress {
  if (jobs.length === 0) {
    return {total: 0, completed: 0, active: 0, queued: 0, percent: 0};
  }

  let completed = 0;
  let active = 0;
  let queued = 0;
  let progressSum = 0;

  for (const job of jobs) {
    if (!job || job.state === 'queued') {
      queued += 1;
      continue;
    }
    if (TERMINAL_STATES.has(job.state)) {
      completed += 1;
      progressSum += 100;
      continue;
    }
    if (ACTIVE_STATES.has(job.state)) {
      active += 1;
      progressSum += clampPercent(job.progress);
    }
  }

  return {
    total: jobs.length,
    completed,
    active,
    queued,
    percent: Math.round(progressSum / jobs.length),
  };
}
