export const MIN_ZOOM = 50;
export const MAX_ZOOM = 400;
export const ZOOM_STEP = 25;

export interface ViewportState {
  zoom: number;
  panX: number;
  panY: number;
}

export type ViewportAction =
  | {type: 'zoom-in'}
  | {type: 'zoom-out'}
  | {type: 'wheel'; deltaY: number; anchorX: number; anchorY: number}
  | {type: 'pan'; deltaX: number; deltaY: number}
  | {type: 'fit'}
  | {type: 'frame-changed'};

export const initialViewportState: ViewportState = {
  zoom: 100,
  panX: 0,
  panY: 0,
};

function clampZoom(zoom: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

function zoomAround(
  state: ViewportState,
  zoom: number,
  anchorX = 0,
  anchorY = 0,
): ViewportState {
  const nextZoom = clampZoom(zoom);
  if (nextZoom === state.zoom) {
    return state;
  }
  if (nextZoom <= 100) {
    return {zoom: nextZoom, panX: 0, panY: 0};
  }
  const ratio = nextZoom / state.zoom;
  return {
    zoom: nextZoom,
    panX: anchorX - (anchorX - state.panX) * ratio,
    panY: anchorY - (anchorY - state.panY) * ratio,
  };
}

export function viewportReducer(
  state: ViewportState,
  action: ViewportAction,
): ViewportState {
  switch (action.type) {
    case 'zoom-in':
      return zoomAround(state, state.zoom + ZOOM_STEP);
    case 'zoom-out':
      return zoomAround(state, state.zoom - ZOOM_STEP);
    case 'wheel':
      return zoomAround(
        state,
        state.zoom + (action.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP),
        action.anchorX,
        action.anchorY,
      );
    case 'pan':
      return state.zoom <= 100
        ? state
        : {
            ...state,
            panX: state.panX + action.deltaX,
            panY: state.panY + action.deltaY,
          };
    case 'fit':
    case 'frame-changed':
      return initialViewportState;
  }
}

export interface PercentBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function boxToPercent(
  [left, top, right, bottom]: [number, number, number, number],
  originalWidth: number,
  originalHeight: number,
): PercentBox {
  return {
    left: (left / originalWidth) * 100,
    top: (top / originalHeight) * 100,
    width: ((right - left) / originalWidth) * 100,
    height: ((bottom - top) / originalHeight) * 100,
  };
}
