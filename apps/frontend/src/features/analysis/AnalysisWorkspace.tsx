import {useTranslation} from 'react-i18next';

import type {AnalysisController} from './model';
import {FrameRail} from './components/FrameRail';
import {ImageViewer} from './components/ImageViewer';
import {PreparationPanel} from './components/PreparationPanel';
import {ResultSummary} from './components/ResultSummary';
import {useAnalysisController} from './useAnalysisController';

export function AnalysisWorkspace({controller: provided}: {controller?: AnalysisController}) {
  const {t} = useTranslation();
  const {controller, snapshot} = useAnalysisController(provided);
  const selected = snapshot.frames.find((frame) => frame.id === snapshot.selectedFrameId);
  const showResult = Boolean(
    selected?.job &&
      ['completed', 'rejected_quality', 'expired', 'deleted'].includes(selected.job.state),
  );

  return (
    <main className="analysis-workspace" aria-label={t('app.workspaceLabel')}>
      <div className="analysis-grid">
        <FrameRail snapshot={snapshot} controller={controller} />
        <ImageViewer snapshot={snapshot} controller={controller} />
        {showResult ? (
          <ResultSummary snapshot={snapshot} controller={controller} />
        ) : (
          <PreparationPanel snapshot={snapshot} controller={controller} />
        )}
      </div>
    </main>
  );
}
