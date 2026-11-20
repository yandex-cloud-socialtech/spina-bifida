import {useTranslation} from 'react-i18next';

import type {AnalysisController, AnalysisSnapshot} from '../model';
import {FrameCard} from './FrameCard';

export function FrameRail({
  snapshot,
  controller,
}: {
  snapshot: AnalysisSnapshot;
  controller: AnalysisController;
}) {
  const {t} = useTranslation();
  return (
    <aside className="frame-rail" aria-label={t('analysis.frames')}>
      <div className="frame-rail__header">
        <strong>{t('analysis.frames')}</strong>
        <span>{t('analysis.frameCounter', {count: snapshot.frames.length})}</span>
      </div>
      {snapshot.frames.map((frame, index) => (
        <FrameCard
          key={frame.id}
          frame={frame}
          index={index}
          selected={frame.id === snapshot.selectedFrameId}
          disabled={snapshot.mutationPending || Boolean(snapshot.cleanupPendingBatchId)}
          controller={controller}
        />
      ))}
    </aside>
  );
}
