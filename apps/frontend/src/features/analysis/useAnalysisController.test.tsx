import {StrictMode, useEffect, useState} from 'react';
import {render, screen, waitFor} from '@testing-library/react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {AppProviders} from '../../app/providers';

const lifecycle = vi.hoisted(() => ({constructions: 0, disposals: 0}));

vi.mock('./model', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./model')>();
  return {
    ...actual,
    AnalysisController: class TrackingAnalysisController extends actual.AnalysisController {
      constructor(...args: ConstructorParameters<typeof actual.AnalysisController>) {
        super(...args);
        lifecycle.constructions += 1;
      }

      override dispose(): void {
        lifecycle.disposals += 1;
        super.dispose();
      }
    },
  };
});

import {AnalysisController} from './model';
import {useAnalysisController} from './useAnalysisController';

function Workspace({controller}: {controller: AnalysisController}) {
  const {controller: workspaceController} = useAnalysisController(controller);
  return <output>{workspaceController === controller ? 'shared' : 'separate'}</output>;
}

function LayoutWithWorkspace() {
  const [controller] = useState(() => new AnalysisController());

  useEffect(() => {
    const lease = controller.attach();
    return () => controller.detach(lease);
  }, [controller]);

  return <Workspace controller={controller} />;
}

afterEach(() => {
  lifecycle.constructions = 0;
  lifecycle.disposals = 0;
});

describe('useAnalysisController ownership', () => {
  it('does not construct an owned controller when the layout provides one and disposes that one once', async () => {
    const view = render(
      <AppProviders>
        <LayoutWithWorkspace />
      </AppProviders>,
    );

    expect(screen.getByText('shared')).toBeInTheDocument();
    expect(lifecycle.constructions).toBe(1);

    view.unmount();
    await waitFor(() => expect(lifecycle.disposals).toBe(1));
  });

  it('keeps final disposal single during StrictMode effect replay', async () => {
    const view = render(
      <StrictMode>
        <AppProviders>
          <LayoutWithWorkspace />
        </AppProviders>
      </StrictMode>,
    );

    view.unmount();
    await waitFor(() => expect(lifecycle.disposals).toBe(1));
  });
});
