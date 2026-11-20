import {describe, expect, it} from 'vitest';

import type {AnalysisJobDto, JobState} from '../../api/contracts';
import {calculateBatchProgress} from './batch-progress';

function job(state: JobState, progress: number): AnalysisJobDto {
  return {
    id: `${state}-${progress}`,
    state,
    stage: null,
    progress,
    queue_position: null,
    attempt: 0,
    frames: [],
    result: null,
    error: null,
    expires_at: '2026-09-23T00:00:00Z',
  };
}

describe('calculateBatchProgress', () => {
  it('returns a stable zero summary for an empty batch', () => {
    expect(calculateBatchProgress([])).toEqual({
      total: 0,
      completed: 0,
      active: 0,
      queued: 0,
      percent: 0,
    });
  });

  it('aggregates a sequential batch from terminal, active, and queued frames', () => {
    expect(
      calculateBatchProgress([
        job('completed', 12),
        job('quality_check', 50),
        job('queued', 99),
      ]),
    ).toEqual({
      total: 3,
      completed: 1,
      active: 1,
      queued: 1,
      percent: 50,
    });
  });

  it('averages parallel server progress and clamps every active contribution', () => {
    expect(
      calculateBatchProgress([
        job('validating', -20),
        job('roi_detection', 50),
        job('scoring', 140),
      ]),
    ).toEqual({
      total: 3,
      completed: 0,
      active: 3,
      queued: 0,
      percent: 50,
    });
  });

  it('counts successful, failed, expired, and deleted jobs as terminal', () => {
    expect(
      calculateBatchProgress([
        job('completed', 0),
        job('rejected_quality', 0),
        job('failed_recoverable', 0),
        job('expired', 0),
        job('deleted', 0),
        null,
      ]),
    ).toEqual({
      total: 6,
      completed: 5,
      active: 0,
      queued: 1,
      percent: 83,
    });
  });
});
