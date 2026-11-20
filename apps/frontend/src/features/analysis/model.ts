import {ApiError, apiClient, type ApiRequestOptions} from '../../api/client';
import type {
  AnalysisJobDto,
  CreateAnalysisResponse,
  CreatedAnalysisJobDto,
  FeedbackResponse,
  FeedbackValue,
  JobState,
  ModelVersion,
} from '../../api/contracts';

export type AnalysisUiState =
  | 'empty'
  | 'one_frame'
  | 'two_frames'
  | 'submitting'
  | 'queued'
  | 'processing'
  | 'format_error'
  | 'technical_error'
  | 'quality_rejected'
  | 'no_expressed_signal'
  | 'attention_required'
  | 'feedback_sent'
  | 'expired';

export interface DraftFrame {
  id: string;
  file: File;
  name: string;
  previewUrl: string;
  source: 'upload' | 'example';
  backendFrameId: string | null;
  jobId: string | null;
  job: AnalysisJobDto | null;
  feedback: FeedbackResponse['feedback'] | null;
  feedbackPending: boolean;
  feedbackError: string | null;
  retryPending: boolean;
}

export interface AnalysisUiError {
  code: string;
  fileName?: string;
}

export interface AnalysisSnapshot {
  state: AnalysisUiState;
  batchId: string | null;
  cleanupPendingBatchId: string | null;
  modelVersion: ModelVersion;
  modelPending: boolean;
  frames: DraftFrame[];
  selectedFrameId: string | null;
  submitting: boolean;
  clearing: boolean;
  mutationPending: boolean;
  error: AnalysisUiError | null;
}

type Requester = (path: string, options?: ApiRequestOptions) => Promise<unknown>;

interface ControllerOptions {
  request: Requester;
  createObjectURL: (file: Blob) => string;
  revokeObjectURL: (url: string) => void;
  loadExamples: () => Promise<File[]>;
  maxFileBytes: number;
  initialModelVersion: ModelVersion;
}

const PROCESSING_STATES = new Set<JobState>([
  'validating',
  'quality_check',
  'plane_detection',
  'roi_detection',
  'scoring',
  'explanation',
]);

export const FEEDBACK_ELIGIBLE_STATES = new Set<JobState>([
  'completed',
  'rejected_quality',
]);

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png']);
const DEFAULT_MAX_BYTES = 20 * 1024 * 1024;
const MAX_FRAMES = 10;
const EXAMPLE_NAMES = [
  'norm-sagittal.jpg',
  'norm-axial.jpg',
  'patology-sagittal.jpg',
  'patology-axial.jpg',
] as const;

async function loadCheckedInExamples(): Promise<File[]> {
  return Promise.all(
    EXAMPLE_NAMES.map(async (name) => {
      const response = await fetch(`/examples/${name}`, {credentials: 'same-origin'});
      if (!response.ok) {
        throw new Error(`example unavailable: ${name}`);
      }
      return new File([await response.blob()], name, {type: 'image/jpeg'});
    }),
  );
}

function emptyJobState(frame: DraftFrame): DraftFrame {
  return {
    ...frame,
    backendFrameId: null,
    jobId: null,
    job: null,
    feedback: null,
    feedbackPending: false,
    feedbackError: null,
    retryPending: false,
  };
}

function createdJobState(created: CreatedAnalysisJobDto): AnalysisJobDto {
  return {
    id: created.id,
    state: created.state,
    stage: null,
    progress: 0,
    queue_position: null,
    attempt: 0,
    frames: [{id: created.frame_id, original_asset_url: ''}],
    result: null,
    error: null,
    expires_at: '',
  };
}

export class AnalysisController {
  private listeners = new Set<() => void>();
  private frameSequence = 0;
  private mountVersion = 0;
  private operationGeneration = 0;
  private destructiveOperation: number | null = null;
  private disposed = false;
  private objectUrls = new Set<string>();
  private options: ControllerOptions;
  snapshot: AnalysisSnapshot = {
    state: 'empty',
    batchId: null,
    cleanupPendingBatchId: null,
    modelVersion: 'v2',
    modelPending: false,
    frames: [],
    selectedFrameId: null,
    submitting: false,
    clearing: false,
    mutationPending: false,
    error: null,
  };

  constructor(options: Partial<ControllerOptions> = {}) {
    this.options = {
      request: (path, requestOptions) => apiClient.request(path, requestOptions),
      createObjectURL: (file) => URL.createObjectURL(file),
      revokeObjectURL: (url) => URL.revokeObjectURL(url),
      loadExamples: loadCheckedInExamples,
      maxFileBytes: DEFAULT_MAX_BYTES,
      initialModelVersion: 'v2',
      ...options,
    };
    this.snapshot = {
      ...this.snapshot,
      modelVersion: this.options.initialModelVersion,
    };
  }

  get modelVersion(): ModelVersion {
    return this.snapshot.modelVersion;
  }

  get modelPending(): boolean {
    return this.snapshot.modelPending;
  }

  get pollingJobIds(): string[] {
    return this.snapshot.frames
      .filter(
        ({job, jobId}) =>
          jobId && job && (job.state === 'queued' || PROCESSING_STATES.has(job.state)),
      )
      .map(({jobId}) => jobId as string);
  }

  subscribe = (listener: () => void): (() => void) => {
    if (this.disposed) {
      return () => undefined;
    }
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): AnalysisSnapshot => this.snapshot;

  attach(): number {
    this.mountVersion += 1;
    return this.mountVersion;
  }

  detach(version: number): void {
    queueMicrotask(() => {
      if (version === this.mountVersion) {
        this.dispose();
      }
    });
  }

  async addFiles(files: Iterable<File>, source: DraftFrame['source'] = 'upload'): Promise<void> {
    if (
      this.disposed ||
      this.destructiveOperation !== null ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const additions = Array.from(files);
    if (additions.length === 0) {
      return;
    }
    const uniqueAdditions = this.uniqueAdditions(additions);
    const validationError = this.validateAdditions(additions, uniqueAdditions);
    if (validationError) {
      this.update({error: validationError});
      return;
    }
    const operation = this.beginDestructive();
    if (operation === null) {
      return;
    }
    const deletionError = await this.deleteBatch();
    if (!this.isCurrent(operation)) {
      return;
    }
    if (deletionError) {
      this.finishDestructive(operation, {error: deletionError});
      return;
    }
    const existing = this.snapshot.frames.map(emptyJobState);
    const newFrames: DraftFrame[] = [];
    for (const file of uniqueAdditions) {
      const draft = this.createDraft(file, source, undefined, operation);
      if (!draft) {
        for (const created of newFrames) {
          this.revokePreview(created.previewUrl);
        }
        return;
      }
      newFrames.push(draft);
    }
    if (!this.isCurrent(operation)) {
      for (const created of newFrames) {
        this.revokePreview(created.previewUrl);
      }
      return;
    }
    const frames = [...existing, ...newFrames];
    this.finishDestructive(operation, {
      batchId: null,
      frames,
      selectedFrameId: this.snapshot.selectedFrameId ?? newFrames[0].id,
      submitting: false,
      error: null,
    });
  }

  async removeFrame(frameId: string): Promise<void> {
    if (
      this.disposed ||
      this.destructiveOperation !== null ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const removed = this.snapshot.frames.find((frame) => frame.id === frameId);
    if (!removed) {
      return;
    }
    const operation = this.beginDestructive();
    if (operation === null) {
      return;
    }
    const deletionError = await this.deleteBatch();
    if (!this.isCurrent(operation)) {
      return;
    }
    if (deletionError) {
      this.finishDestructive(operation, {error: deletionError});
      return;
    }
    this.revokePreview(removed.previewUrl);
    const frames = this.snapshot.frames
      .filter((frame) => frame.id !== frameId)
      .map(emptyJobState);
    this.finishDestructive(operation, {
      batchId: null,
      frames,
      selectedFrameId:
        this.snapshot.selectedFrameId === frameId
          ? (frames[0]?.id ?? null)
          : this.snapshot.selectedFrameId,
      submitting: false,
      error: null,
    });
  }

  async replaceFrame(frameId: string, file: File): Promise<void> {
    if (
      this.disposed ||
      this.destructiveOperation !== null ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const index = this.snapshot.frames.findIndex((frame) => frame.id === frameId);
    if (index < 0) {
      return;
    }
    const error = this.validateReplacement(file, index);
    if (error) {
      this.update({error});
      return;
    }
    const operation = this.beginDestructive();
    if (operation === null) {
      return;
    }
    const deletionError = await this.deleteBatch();
    if (!this.isCurrent(operation)) {
      return;
    }
    if (deletionError) {
      this.finishDestructive(operation, {error: deletionError});
      return;
    }
    const frames = this.snapshot.frames.map(emptyJobState);
    const oldFrame = frames[index];
    if (!oldFrame) {
      this.finishDestructive(operation);
      return;
    }
    const replacement = this.createDraft(file, 'upload', oldFrame.id, operation);
    if (!replacement || !this.isCurrent(operation)) {
      if (replacement) {
        this.revokePreview(replacement.previewUrl);
      }
      return;
    }
    frames[index] = replacement;
    this.revokePreview(oldFrame.previewUrl);
    this.finishDestructive(operation, {
      batchId: null,
      frames,
      submitting: false,
      error: null,
    });
  }

  async useExamples(): Promise<void> {
    if (
      this.disposed ||
      this.snapshot.frames.length > 0 ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const generationAtStart = this.operationGeneration;
    let files: File[];
    try {
      files = await this.options.loadExamples();
    } catch {
      if (this.isGeneration(generationAtStart)) {
        this.update({error: {code: 'example_unavailable'}});
      }
      return;
    }
    if (!this.isGeneration(generationAtStart) || this.snapshot.frames.length > 0) {
      return;
    }
    await this.addFiles(files, 'example');
  }

  async start(): Promise<void> {
    if (
      this.snapshot.frames.length === 0 ||
      this.snapshot.submitting ||
      this.snapshot.clearing ||
      this.snapshot.mutationPending ||
      this.snapshot.modelPending ||
      this.snapshot.batchId ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    let framesAtStart = this.snapshot.frames;
    const modelAtStart = this.snapshot.modelVersion;
    const generationAtStart = this.operationGeneration;
    this.update({submitting: true, error: null});
    while (framesAtStart.length > 0) {
      const body = new FormData();
      for (const frame of framesAtStart) {
        body.append('frames', frame.file, frame.name);
      }
      body.append('model_version', modelAtStart);
      try {
        const response = await this.options.request('/api/v1/analyses', {method: 'POST', body});
        if (!this.isGeneration(generationAtStart)) {
          await this.deleteCreatedBatch(response);
          return;
        }
        const created = this.validateCreateResponse(response, framesAtStart.length);
        if (!created) {
          const cleaned = await this.deleteCreatedBatch(response);
          if (!this.isGeneration(generationAtStart)) {
            return;
          }
          if (!cleaned) {
            return;
          }
          this.update({submitting: false, error: {code: 'invalid_analysis_response'}});
          return;
        }
        const jobsByPosition = new Map(created.jobs.map((item) => [item.batch_position, item]));
        const frames = framesAtStart.map((frame, index) => {
          const createdJob = jobsByPosition.get(index) as CreatedAnalysisJobDto;
          return {
            ...emptyJobState(frame),
            backendFrameId: createdJob.frame_id,
            jobId: createdJob.id,
            job: createdJobState(createdJob),
          };
        });
        this.update({
          batchId: created.batch_id,
          frames,
          submitting: false,
          error: null,
        });
        return;
      } catch (error) {
        if (!this.isGeneration(generationAtStart)) {
          return;
        }
        const duplicateIndex =
          error instanceof ApiError && error.code === 'duplicate_image'
            ? error.params.file_index
            : null;
        if (
          typeof duplicateIndex === 'number' &&
          Number.isInteger(duplicateIndex) &&
          duplicateIndex >= 0 &&
          duplicateIndex < framesAtStart.length &&
          framesAtStart.length > 1
        ) {
          const duplicate = framesAtStart[duplicateIndex];
          framesAtStart = framesAtStart.filter((_frame, index) => index !== duplicateIndex);
          this.revokePreview(duplicate.previewUrl);
          this.update({
            frames: framesAtStart,
            selectedFrameId: framesAtStart.some(
              ({id}) => id === this.snapshot.selectedFrameId,
            )
              ? this.snapshot.selectedFrameId
              : (framesAtStart[0]?.id ?? null),
            error: null,
          });
          continue;
        }
        this.update({
          submitting: false,
          error: {code: error instanceof ApiError ? error.code : 'inference_failed'},
        });
        return;
      }
    }
  }

  async retryFrame(frameId: string): Promise<void> {
    if (
      this.disposed ||
      this.snapshot.mutationPending ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const frame = this.snapshot.frames.find((candidate) => candidate.id === frameId);
    if (!frame?.jobId || frame.job?.state !== 'failed_recoverable' || frame.retryPending) {
      return;
    }
    const jobId = frame.jobId;
    const generationAtStart = this.operationGeneration;
    this.updateFrameById(frameId, (current) => ({...current, retryPending: true}));
    try {
      const response = (await this.options.request(`/api/v1/analyses/${jobId}/retry`, {
        method: 'POST',
      })) as {id?: string; state?: JobState; attempt?: number};
      if (!this.isGeneration(generationAtStart)) {
        return;
      }
      this.updateFrameByJobId(jobId, (current) => ({
        ...current,
        job: current.job
          ? {
              ...current.job,
              state: 'queued',
              stage: null,
              progress: 0,
              queue_position: null,
              attempt: typeof response.attempt === 'number' ? response.attempt : current.job.attempt + 1,
              result: null,
              error: null,
            }
          : current.job,
        feedback: null,
        feedbackPending: false,
        feedbackError: null,
        retryPending: false,
      }));
      this.update({error: null});
    } catch (error) {
      if (this.isGeneration(generationAtStart)) {
        this.updateFrameById(frameId, (current) =>
          current.jobId === jobId ? {...current, retryPending: false} : current,
        );
        this.update({error: {code: error instanceof ApiError ? error.code : 'inference_failed'}});
      }
    }
  }

  async retry(): Promise<void> {
    const selected = this.selectedFrame();
    if (selected?.job?.state === 'failed_recoverable') {
      await this.retryFrame(selected.id);
    } else if (!this.snapshot.batchId) {
      await this.start();
    }
  }

  applyJob(job: AnalysisJobDto): void {
    if (this.disposed) {
      return;
    }
    if (this.snapshot.frames.some((frame) => frame.jobId === job.id && frame.job === job)) {
      return;
    }
    this.updateFrameByJobId(job.id, (frame) => ({...frame, job}));
  }

  selectFrame(frameId: string): void {
    if (this.disposed) {
      return;
    }
    if (this.snapshot.frames.some((frame) => frame.id === frameId)) {
      this.update({selectedFrameId: frameId});
    }
  }

  async submitFeedback(frameId: string, value: FeedbackValue, comment: string | null): Promise<void> {
    if (
      this.disposed ||
      this.snapshot.mutationPending ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const frame = this.snapshot.frames.find((candidate) => candidate.id === frameId);
    if (
      !frame?.jobId ||
      frame.feedbackPending ||
      frame.feedback ||
      !frame.job ||
      !FEEDBACK_ELIGIBLE_STATES.has(frame.job.state)
    ) {
      return;
    }
    const {jobId} = frame;
    const generationAtStart = this.operationGeneration;
    this.updateFrameById(frameId, (current) => ({
      ...current,
      feedbackPending: true,
      feedbackError: null,
    }));
    try {
      const response = (await this.options.request(`/api/v1/analyses/${jobId}/feedback`, {
        method: 'POST',
        json: {value, comment: comment?.trim() || null},
      })) as FeedbackResponse;
      if (!this.isGeneration(generationAtStart)) {
        return;
      }
      this.updateFrameById(frameId, (current) =>
        current.jobId === jobId
          ? {
              ...current,
              feedback: response.feedback,
              feedbackPending: false,
              feedbackError: null,
            }
          : current,
      );
    } catch (error) {
      if (!this.isGeneration(generationAtStart)) {
        return;
      }
      this.updateFrameById(frameId, (current) =>
        current.jobId === jobId
          ? {
              ...current,
              feedbackPending: false,
              feedbackError: error instanceof ApiError ? error.code : 'inference_failed',
            }
          : current,
      );
    }
  }

  async clear(): Promise<void> {
    if (
      this.disposed ||
      this.destructiveOperation !== null ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const operation = this.beginDestructive({clearing: true});
    if (operation === null) {
      return;
    }
    const deletionError = await this.deleteBatch();
    if (!this.isCurrent(operation)) {
      return;
    }
    if (deletionError) {
      this.finishDestructive(operation, {clearing: false, error: deletionError});
      return;
    }
    this.revokeFrames();
    this.finishDestructive(operation, {
      batchId: null,
      frames: [],
      selectedFrameId: null,
      submitting: false,
      clearing: false,
      error: null,
    });
  }

  async reset(): Promise<void> {
    await this.clear();
  }

  async changeModel(version: ModelVersion): Promise<void> {
    if (
      this.disposed ||
      version === this.snapshot.modelVersion ||
      this.destructiveOperation !== null ||
      this.snapshot.cleanupPendingBatchId
    ) {
      return;
    }
    const operation = this.beginDestructive({modelPending: true});
    if (operation === null) {
      return;
    }
    const deletionError = await this.deleteBatch();
    if (!this.isCurrent(operation)) {
      return;
    }
    if (deletionError) {
      this.finishDestructive(operation, {modelPending: false, error: deletionError});
      return;
    }
    this.finishDestructive(operation, {
      batchId: null,
      modelVersion: version,
      modelPending: false,
      submitting: false,
      frames: this.snapshot.frames.map(emptyJobState),
      error: null,
    });
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.operationGeneration += 1;
    this.destructiveOperation = null;
    this.revokeFrames();
    this.snapshot = {
      ...this.snapshot,
      state: 'empty',
      batchId: null,
      cleanupPendingBatchId: null,
      frames: [],
      selectedFrameId: null,
      submitting: false,
      clearing: false,
      modelPending: false,
      mutationPending: false,
      error: null,
    };
    this.listeners.clear();
  }

  private selectedFrame(): DraftFrame | undefined {
    return this.snapshot.frames.find((frame) => frame.id === this.snapshot.selectedFrameId);
  }

  private createDraft(
    file: File,
    source: DraftFrame['source'],
    id: string | undefined,
    operation: number,
  ): DraftFrame | null {
    if (!this.isCurrent(operation)) {
      return null;
    }
    const previewUrl = this.options.createObjectURL(file);
    this.objectUrls.add(previewUrl);
    if (!this.isCurrent(operation)) {
      this.revokePreview(previewUrl);
      return null;
    }
    return {
      id: id ?? `draft-${++this.frameSequence}`,
      file,
      name: file.name,
      previewUrl,
      source,
      backendFrameId: null,
      jobId: null,
      job: null,
      feedback: null,
      feedbackPending: false,
      feedbackError: null,
      retryPending: false,
    };
  }

  private async deleteBatch(): Promise<AnalysisUiError | null> {
    const batchId = this.snapshot.batchId;
    if (!batchId) {
      return null;
    }
    return this.deleteBatchById(batchId);
  }

  private async deleteBatchById(batchId: string): Promise<AnalysisUiError | null> {
    try {
      await this.options.request(`/api/v1/analyses/batches/${batchId}`, {method: 'DELETE'});
      return null;
    } catch (error) {
      if (error instanceof ApiError && error.code === 'analysis_not_found') {
        return null;
      }
      return {code: error instanceof ApiError ? error.code : 'inference_failed'};
    }
  }

  private async deleteCreatedBatch(response: unknown): Promise<boolean> {
    if (
      response &&
      typeof response === 'object' &&
      this.isNonBlank((response as {batch_id?: unknown}).batch_id)
    ) {
      const batchId = (response as {batch_id: string}).batch_id.trim();
      const error = await this.deleteBatchById(batchId);
      if (error) {
        if (!this.disposed) {
          this.update({
            cleanupPendingBatchId: batchId,
            submitting: false,
            error,
          });
        }
        return false;
      }
    }
    return true;
  }

  async retryCleanup(): Promise<void> {
    const batchId = this.snapshot.cleanupPendingBatchId;
    if (!batchId || this.disposed || this.destructiveOperation !== null) {
      return;
    }
    const operation = this.beginDestructive();
    if (operation === null) {
      return;
    }
    const error = await this.deleteBatchById(batchId);
    if (!this.isCurrent(operation)) {
      return;
    }
    if (error) {
      this.finishDestructive(operation, {error});
      return;
    }
    this.finishDestructive(operation, {
      cleanupPendingBatchId: null,
      error: null,
    });
  }

  private validateCreateResponse(response: unknown, frameCount: number): CreateAnalysisResponse | null {
    if (!response || typeof response !== 'object') {
      return null;
    }
    const candidate = response as Partial<CreateAnalysisResponse>;
    if (!this.isNonBlank(candidate.batch_id) || !Array.isArray(candidate.jobs)) {
      return null;
    }
    if (candidate.jobs.length !== frameCount) {
      return null;
    }
    const positions = new Set<number>();
    const jobIds = new Set<string>();
    const frameIds = new Set<string>();
    for (const item of candidate.jobs) {
      if (
        !item ||
        !this.isNonBlank(item.id) ||
        !this.isNonBlank(item.frame_id) ||
        !this.isNonBlank(item.polling_url) ||
        item.state !== 'queued' ||
        !Number.isInteger(item.batch_position) ||
        item.batch_position < 0 ||
        item.batch_position >= frameCount ||
        positions.has(item.batch_position) ||
        jobIds.has(item.id) ||
        frameIds.has(item.frame_id)
      ) {
        return null;
      }
      positions.add(item.batch_position);
      jobIds.add(item.id);
      frameIds.add(item.frame_id);
    }
    return positions.size === frameCount ? (candidate as CreateAnalysisResponse) : null;
  }

  private validateAdditions(
    additions: File[],
    uniqueAdditions: File[],
  ): AnalysisUiError | null {
    for (const file of additions) {
      const error = this.validateFile(file);
      if (error) {
        return {...error, fileName: file.name};
      }
    }
    if (uniqueAdditions.length === 0) {
      return {code: 'duplicate_image', fileName: additions[0]?.name};
    }
    if (this.snapshot.frames.length + uniqueAdditions.length > MAX_FRAMES) {
      return {
        code: 'too_many_frames',
        fileName: uniqueAdditions[MAX_FRAMES - this.snapshot.frames.length]?.name,
      };
    }
    return null;
  }

  private uniqueAdditions(additions: File[]): File[] {
    const seen = this.snapshot.frames.map(({file}) => file);
    return additions.filter((file) => {
      if (seen.some((candidate) => this.isDuplicate(candidate, file))) {
        return false;
      }
      seen.push(file);
      return true;
    });
  }

  private validateReplacement(file: File, replacementIndex: number): AnalysisUiError | null {
    const error = this.validateFile(file);
    if (error) {
      return {...error, fileName: file.name};
    }
    if (
      this.snapshot.frames.some(
        (frame, index) => index !== replacementIndex && this.isDuplicate(frame.file, file),
      )
    ) {
      return {code: 'duplicate_image', fileName: file.name};
    }
    return null;
  }

  private updateFrameByJobId(jobId: string, change: (frame: DraftFrame) => DraftFrame): void {
    const index = this.snapshot.frames.findIndex((frame) => frame.jobId === jobId);
    if (index < 0) {
      return;
    }
    const frames = [...this.snapshot.frames];
    frames[index] = change(frames[index]);
    this.update({frames});
  }

  private updateFrameById(frameId: string, change: (frame: DraftFrame) => DraftFrame): void {
    const index = this.snapshot.frames.findIndex((frame) => frame.id === frameId);
    if (index < 0) {
      return;
    }
    const frames = [...this.snapshot.frames];
    frames[index] = change(frames[index]);
    this.update({frames});
  }

  private update(patch: Partial<AnalysisSnapshot>): void {
    if (this.disposed) {
      return;
    }
    const next = {...this.snapshot, ...patch};
    this.snapshot = {...next, state: this.deriveState(next)};
    for (const listener of this.listeners) {
      listener();
    }
  }

  private deriveState(snapshot: AnalysisSnapshot): AnalysisUiState {
    if (snapshot.submitting) {
      return 'submitting';
    }
    if (snapshot.error) {
      return ['unsupported_format', 'image_too_large', 'duplicate_image', 'too_many_frames'].includes(
        snapshot.error.code,
      )
        ? 'format_error'
        : 'technical_error';
    }
    const selected = snapshot.frames.find((frame) => frame.id === snapshot.selectedFrameId);
    if (!selected?.job) {
      if (snapshot.frames.length === 0) {
        return 'empty';
      }
      return snapshot.frames.length === 1 ? 'one_frame' : 'two_frames';
    }
    if (selected.feedback) {
      return 'feedback_sent';
    }
    if (selected.job.state === 'queued') {
      return 'queued';
    }
    if (PROCESSING_STATES.has(selected.job.state)) {
      return 'processing';
    }
    if (selected.job.state === 'failed_recoverable') {
      return 'technical_error';
    }
    if (selected.job.state === 'rejected_quality') {
      return 'quality_rejected';
    }
    if (selected.job.state === 'expired' || selected.job.state === 'deleted') {
      return 'expired';
    }
    return selected.job.result?.frames.some(
      (frame) => frame.assessment?.category === 'attention_required',
    )
      ? 'attention_required'
      : 'no_expressed_signal';
  }

  private validateFile(file: File): AnalysisUiError | null {
    if (!ALLOWED_TYPES.has(file.type)) {
      return {code: 'unsupported_format'};
    }
    if (file.size > this.options.maxFileBytes) {
      return {code: 'image_too_large'};
    }
    return null;
  }

  private isDuplicate(first: File, second: File): boolean {
    return (
      first === second ||
      (first.name === second.name &&
        first.size === second.size &&
        first.lastModified === second.lastModified)
    );
  }

  private revokeFrames(): void {
    for (const frame of this.snapshot.frames) {
      this.revokePreview(frame.previewUrl);
    }
  }

  private beginDestructive(patch: Partial<AnalysisSnapshot> = {}): number | null {
    if (this.disposed || this.destructiveOperation !== null) {
      return null;
    }
    const operation = ++this.operationGeneration;
    this.destructiveOperation = operation;
    this.update({mutationPending: true, submitting: false, error: null, ...patch});
    return operation;
  }

  private finishDestructive(operation: number, patch: Partial<AnalysisSnapshot> = {}): void {
    if (!this.isCurrent(operation)) {
      return;
    }
    this.destructiveOperation = null;
    this.update({mutationPending: false, ...patch});
  }

  private isCurrent(operation: number): boolean {
    return !this.disposed && this.destructiveOperation === operation && this.operationGeneration === operation;
  }

  private isGeneration(generation: number): boolean {
    return !this.disposed && this.operationGeneration === generation;
  }

  private revokePreview(url: string): void {
    if (this.objectUrls.delete(url)) {
      this.options.revokeObjectURL(url);
    }
  }

  private isNonBlank(value: unknown): value is string {
    return typeof value === 'string' && value.trim().length > 0;
  }
}
