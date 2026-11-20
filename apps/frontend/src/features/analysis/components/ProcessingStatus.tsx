import {Progress} from '@gravity-ui/uikit';
import {useTranslation} from 'react-i18next';

import type {AnalysisJobDto} from '../../../api/contracts';
import {calculateBatchProgress} from '../batch-progress';

export function ProcessingStatus({
  jobs,
  selectedJob,
  submitting,
}: {
  jobs: readonly (AnalysisJobDto | null)[];
  selectedJob: AnalysisJobDto | null;
  submitting: boolean;
}) {
  const {t} = useTranslation();
  const stateKey = selectedJob?.stage ?? selectedJob?.state ?? (submitting ? 'submitting' : 'queued');
  const progress = calculateBatchProgress(jobs);
  return (
    <div
      className="processing-overlay"
      role="status"
      aria-label={t(`analysis.states.${stateKey}`)}
      aria-live="polite"
    >
      <div className="processing-card">
        <div className="processing-card__title">
          <span className="processing-spinner" aria-hidden="true" />
          <strong>{t(`analysis.states.${stateKey}`)}</strong>
        </div>
        <p className="processing-card__completed">
          {t('analysis.batchCompleted', {
            completed: progress.completed,
            total: progress.total,
          })}
        </p>
        <div className="processing-card__counts">
          <span>{t('analysis.batchActive', {count: progress.active})}</span>
          <span>{t('analysis.batchQueued', {count: progress.queued})}</span>
        </div>
        {selectedJob?.state === 'queued' && selectedJob.queue_position ? (
          <p>{t('analysis.queuePosition', {count: selectedJob.queue_position})}</p>
        ) : null}
        <p>{t('analysis.processingHint')}</p>
        <div
          className="processing-card__progress"
          role="progressbar"
          aria-label={t('analysis.batchProgress', {count: progress.percent})}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress.percent}
        >
          <Progress value={progress.percent} size="s" />
        </div>
      </div>
    </div>
  );
}
