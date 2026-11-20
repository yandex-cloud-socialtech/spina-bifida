import {describe, expect, it} from 'vitest';

import {
  boxToPercent,
  initialViewportState,
  viewportReducer,
} from './image-viewport';

describe('viewportReducer', () => {
  it('uses 25 percent keyboard steps within the 50 to 400 percent range', () => {
    let state = initialViewportState;
    for (let index = 0; index < 20; index += 1) {
      state = viewportReducer(state, {type: 'zoom-in'});
    }
    expect(state.zoom).toBe(400);

    for (let index = 0; index < 20; index += 1) {
      state = viewportReducer(state, {type: 'zoom-out'});
    }
    expect(state).toEqual({zoom: 50, panX: 0, panY: 0});
  });

  it('keeps the point under the pointer stable when wheel-zooming', () => {
    const zoomed = viewportReducer(initialViewportState, {
      type: 'wheel',
      deltaY: -1,
      anchorX: 80,
      anchorY: -40,
    });

    expect(zoomed).toEqual({zoom: 125, panX: -20, panY: 10});
    expect(
      viewportReducer(zoomed, {
        type: 'wheel',
        deltaY: 1,
        anchorX: 80,
        anchorY: -40,
      }),
    ).toEqual(initialViewportState);
  });

  it('allows drag pan only while the fitted image is enlarged', () => {
    expect(
      viewportReducer(initialViewportState, {type: 'pan', deltaX: 30, deltaY: -10}),
    ).toEqual(initialViewportState);

    const zoomed = viewportReducer(initialViewportState, {type: 'zoom-in'});
    expect(viewportReducer(zoomed, {type: 'pan', deltaX: 30, deltaY: -10})).toEqual({
      zoom: 125,
      panX: 30,
      panY: -10,
    });
  });

  it('resets zoom and pan for fit and frame changes', () => {
    const moved = {zoom: 250, panX: 80, panY: -55};
    expect(viewportReducer(moved, {type: 'fit'})).toEqual(initialViewportState);
    expect(viewportReducer(moved, {type: 'frame-changed'})).toEqual(initialViewportState);
  });
});

describe('boxToPercent', () => {
  it('maps original-pixel geometry into the shared image coordinate space', () => {
    expect(boxToPercent([10, 20, 100, 140], 640, 480)).toEqual({
      left: 1.5625,
      top: 4.166666666666666,
      width: 14.0625,
      height: 25,
    });
    expect(boxToPercent([2, 8, 108, 152], 640, 480)).toEqual({
      left: 0.3125,
      top: 1.6666666666666667,
      width: 16.5625,
      height: 30,
    });
  });
});
