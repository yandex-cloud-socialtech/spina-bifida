import {useEffect, useReducer, useRef, useState, type CSSProperties} from 'react';
import {useTranslation} from 'react-i18next';

import type {AnalysisController, AnalysisSnapshot} from '../model';
import {calculateBatchProgress} from '../batch-progress';
import {boxToPercent, initialViewportState, viewportReducer} from '../image-viewport';
import {ProcessingStatus} from './ProcessingStatus';
import {UploadDropzone} from './UploadDropzone';
import {ViewerToolbar} from './ViewerToolbar';

function overlayStyle(
  box: [number, number, number, number],
  originalWidth: number,
  originalHeight: number,
): CSSProperties {
  const geometry = boxToPercent(box, originalWidth, originalHeight);
  return {
    left: `${geometry.left}%`,
    top: `${geometry.top}%`,
    width: `${geometry.width}%`,
    height: `${geometry.height}%`,
  };
}

export function ImageViewer({
  snapshot,
  controller,
}: {
  snapshot: AnalysisSnapshot;
  controller: AnalysisController;
}) {
  const {t} = useTranslation();
  const [viewport, dispatch] = useReducer(viewportReducer, initialViewportState);
  const [roiVisible, setRoiVisible] = useState(true);
  const [heatmapVisible, setHeatmapVisible] = useState(true);
  const drag = useRef<{pointerId: number; x: number; y: number} | null>(null);
  const selected = snapshot.frames.find((frame) => frame.id === snapshot.selectedFrameId);
  const frameResult = selected?.job?.result?.frames.find(
    (frame) => frame.frame_id === selected?.backendFrameId,
  ) ?? selected?.job?.result?.frames[0];
  const originalAsset = frameResult?.assets.find((candidate) => candidate.kind === 'original');
  const heatmapAsset = frameResult?.assets.find((candidate) => candidate.kind === 'heatmap');
  const originalUrl = originalAsset?.url ?? selected?.previewUrl;
  const originalWidth = originalAsset?.width ?? 0;
  const originalHeight = originalAsset?.height ?? 0;
  const hasGeometry = originalWidth > 0 && originalHeight > 0 && Boolean(frameResult?.roi);
  const roiAvailable = hasGeometry;
  const heatmapAvailable = hasGeometry && Boolean(heatmapAsset);
  const batchProgress = calculateBatchProgress(snapshot.frames.map((frame) => frame.job));
  const showProcessing = Boolean(
    snapshot.submitting ||
      (snapshot.batchId &&
        batchProgress.total > 0 &&
        batchProgress.completed < batchProgress.total),
  );
  const showViewer = Boolean(frameResult && originalUrl);
  const showCompactUpload =
    snapshot.frames.length > 0 &&
    snapshot.frames.length < 10 &&
    !snapshot.batchId &&
    !showProcessing &&
    !showViewer;

  useEffect(() => {
    dispatch({type: 'frame-changed'});
  }, [selected?.id]);

  const transformStyle: CSSProperties = {
    width:
      originalWidth > 0 && originalHeight > 0
        ? `min(100%, ${(originalWidth / originalHeight) * 100}cqh)`
        : '100%',
    aspectRatio:
      originalWidth > 0 && originalHeight > 0
        ? `${originalWidth} / ${originalHeight}`
        : undefined,
    transform: `translate3d(${viewport.panX}px, ${viewport.panY}px, 0) scale(${viewport.zoom / 100})`,
  };

  return (
    <section className="image-viewer" aria-label={t('analysis.viewer')}>
      {showViewer ? (
        <ViewerToolbar
          viewport={viewport}
          dispatch={dispatch}
          roiVisible={roiVisible}
          heatmapVisible={heatmapVisible}
          roiAvailable={roiAvailable}
          heatmapAvailable={heatmapAvailable}
          onToggleRoi={() => setRoiVisible((visible) => !visible)}
          onToggleHeatmap={() => setHeatmapVisible((visible) => !visible)}
        />
      ) : null}
      <div className="image-stage">
        {showViewer && originalUrl ? (
          <div
            className={`viewer-canvas${viewport.zoom > 100 ? ' viewer-canvas_pannable' : ''}`}
            onWheel={(event) => {
              event.preventDefault();
              const bounds = event.currentTarget.getBoundingClientRect();
              dispatch({
                type: 'wheel',
                deltaY: event.deltaY,
                anchorX: event.clientX - bounds.left - bounds.width / 2,
                anchorY: event.clientY - bounds.top - bounds.height / 2,
              });
            }}
            onPointerDown={(event) => {
              if (viewport.zoom <= 100) {
                return;
              }
              event.currentTarget.setPointerCapture(event.pointerId);
              drag.current = {pointerId: event.pointerId, x: event.clientX, y: event.clientY};
            }}
            onPointerMove={(event) => {
              if (!drag.current || drag.current.pointerId !== event.pointerId) {
                return;
              }
              dispatch({
                type: 'pan',
                deltaX: event.clientX - drag.current.x,
                deltaY: event.clientY - drag.current.y,
              });
              drag.current = {...drag.current, x: event.clientX, y: event.clientY};
            }}
            onPointerUp={(event) => {
              if (drag.current?.pointerId === event.pointerId) {
                drag.current = null;
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
          >
            <div className="viewer-transform" style={transformStyle}>
              <img
                className="viewer-original"
                src={originalUrl}
                alt={t('analysis.layers.original')}
                draggable={false}
              />
              {heatmapVisible && heatmapAvailable && heatmapAsset && frameResult?.roi ? (
                <img
                  className="viewer-heatmap"
                  src={heatmapAsset.url}
                  alt=""
                  aria-hidden="true"
                  draggable={false}
                  style={overlayStyle(
                    frameResult.roi.display_box,
                    originalWidth,
                    originalHeight,
                  )}
                />
              ) : null}
              {roiVisible && roiAvailable && frameResult?.roi ? (
                <span
                  className="viewer-roi"
                  aria-hidden="true"
                  style={overlayStyle(frameResult.roi.box, originalWidth, originalHeight)}
                />
              ) : null}
            </div>
          </div>
        ) : originalUrl ? (
          <img
            className="image-stage__preview"
            src={originalUrl}
            alt={selected?.name ?? t('analysis.viewer')}
          />
        ) : (
          <UploadDropzone
            controller={controller}
            onOpenExamples={() => void controller.useExamples()}
            disabled={snapshot.mutationPending || Boolean(snapshot.cleanupPendingBatchId)}
          />
        )}
        {showProcessing ? (
          <ProcessingStatus
            jobs={snapshot.frames.map((frame) => frame.job)}
            selectedJob={selected?.job ?? null}
            submitting={snapshot.submitting}
          />
        ) : null}
      </div>
      {showCompactUpload ? (
        <UploadDropzone
          controller={controller}
          compact
          disabled={snapshot.mutationPending || Boolean(snapshot.cleanupPendingBatchId)}
        />
      ) : null}
    </section>
  );
}
