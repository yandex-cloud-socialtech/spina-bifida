import {useQueries} from '@tanstack/react-query';
import {useEffect, useRef, useSyncExternalStore} from 'react';

import {apiClient} from '../../api/client';
import type {AnalysisJobDto} from '../../api/contracts';
import {AnalysisController} from './model';

const POLLING_INTERVAL_MS = 1_000;

export function useAnalysisController(provided?: AnalysisController) {
  const owned = useRef<AnalysisController | null>(null);
  const appliedPollingResponses = useRef(new WeakSet<object>());
  if (!provided && !owned.current) {
    owned.current = new AnalysisController();
  }
  const controller = provided ?? owned.current;
  if (!controller) {
    throw new Error('Analysis controller is unavailable');
  }
  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  );
  const polling = useQueries({
    queries: controller.pollingJobIds.map((jobId) => ({
      queryKey: ['analysis', jobId],
      queryFn: () => apiClient.request<AnalysisJobDto>(`/api/v1/analyses/${jobId}`),
      enabled: true,
      refetchInterval: () =>
        controller.pollingJobIds.includes(jobId) ? POLLING_INTERVAL_MS : false,
    })),
  });

  useEffect(() => {
    for (const result of polling) {
      if (result.data && !appliedPollingResponses.current.has(result.data)) {
        appliedPollingResponses.current.add(result.data);
        controller.applyJob(result.data);
      }
    }
  }, [controller, polling]);

  useEffect(() => {
    if (provided) {
      return;
    }
    const lease = controller.attach();
    return () => controller.detach(lease);
  }, [controller, provided]);

  return {
    controller,
    snapshot,
    modelVersion: snapshot.modelVersion,
    modelPending: snapshot.modelPending,
    changeModel: controller.changeModel.bind(controller),
  };
}
