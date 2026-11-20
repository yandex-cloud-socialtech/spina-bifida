import {useTranslation} from 'react-i18next';

import {MAX_ZOOM, MIN_ZOOM, type ViewportAction, type ViewportState} from '../image-viewport';

export function ViewerToolbar({
  viewport,
  dispatch,
  roiVisible,
  heatmapVisible,
  roiAvailable,
  heatmapAvailable,
  onToggleRoi,
  onToggleHeatmap,
}: {
  viewport: ViewportState;
  dispatch: (action: ViewportAction) => void;
  roiVisible: boolean;
  heatmapVisible: boolean;
  roiAvailable: boolean;
  heatmapAvailable: boolean;
  onToggleRoi: () => void;
  onToggleHeatmap: () => void;
}) {
  const {t} = useTranslation();
  const roiUnavailable = t('analysis.layers.roiUnavailable');
  const heatmapUnavailable = t('analysis.layers.heatmapUnavailable');

  return (
    <div className="viewer-toolbar" role="group" aria-label={t('analysis.viewerControls')}>
      <div className="viewer-toolbar__layers">
        <button
          type="button"
          aria-label={roiAvailable ? 'ROI' : `ROI: ${roiUnavailable}`}
          aria-pressed={roiAvailable && roiVisible}
          disabled={!roiAvailable}
          title={roiAvailable ? undefined : roiUnavailable}
          onClick={onToggleRoi}
        >
          ROI
        </button>
        <button
          type="button"
          aria-label={heatmapAvailable ? 'Heatmap' : `Heatmap: ${heatmapUnavailable}`}
          aria-pressed={heatmapAvailable && heatmapVisible}
          disabled={!heatmapAvailable}
          title={heatmapAvailable ? undefined : heatmapUnavailable}
          onClick={onToggleHeatmap}
        >
          Heatmap
        </button>
      </div>
      <div className="viewer-toolbar__zoom">
        <button
          type="button"
          aria-label={t('analysis.zoomOut')}
          disabled={viewport.zoom <= MIN_ZOOM}
          onClick={() => dispatch({type: 'zoom-out'})}
        >
          −
        </button>
        <span className="viewer-toolbar__percent" aria-label={t('analysis.zoomPercent', {count: viewport.zoom})}>
          {viewport.zoom}%
        </span>
        <button
          type="button"
          aria-label={t('analysis.zoomIn')}
          disabled={viewport.zoom >= MAX_ZOOM}
          onClick={() => dispatch({type: 'zoom-in'})}
        >
          +
        </button>
        <button type="button" onClick={() => dispatch({type: 'fit'})}>
          {t('analysis.fitImage')}
        </button>
      </div>
    </div>
  );
}
