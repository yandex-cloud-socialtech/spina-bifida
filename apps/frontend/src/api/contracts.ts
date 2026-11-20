export type Locale = 'ru' | 'en' | 'kk' | 'uz';
export type UserRole = 'user' | 'admin';
export type UserStatus = 'active' | 'blocked' | 'deleted' | 'pending_email' | 'pending' | 'rejected';

export interface UserDto {
  id: string;
  email: string;
  role: UserRole;
  status: UserStatus;
  locale: Locale;
}

export interface UsersResponse {
  users: UserDto[];
}

export interface UserResponse {
  user: UserDto;
}

export interface CreateUserInput {
  email: string;
  password: string;
  role: UserRole;
  locale: Locale;
}

export interface UpdateUserInput {
  blocked?: boolean;
  role?: UserRole;
  locale?: Locale;
}

export interface AuthResponse {
  user: UserDto;
  csrf_token: string;
  terms_accepted: boolean;
  terms_version: string;
}

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message_key?: string;
    params: Record<string, unknown>;
    request_id: string;
  };
  csrf_token?: string;
}

export type JobState =
  | 'queued'
  | 'validating'
  | 'quality_check'
  | 'plane_detection'
  | 'roi_detection'
  | 'scoring'
  | 'explanation'
  | 'completed'
  | 'rejected_quality'
  | 'failed_recoverable'
  | 'expired'
  | 'deleted';

export type AssetKind = 'original' | 'roi' | 'heatmap';

export interface AnalysisAssetDto {
  id: string;
  kind: AssetKind;
  url: string;
  checksum: string;
  width?: number;
  height?: number;
  media_type?: string;
}

export interface FrameResultDto {
  frame_id: string;
  analyzable: boolean;
  quality: {
    score: number;
    accepted: boolean;
  };
  plane: {
    type: 'axial' | 'sagittal' | 'unknown';
    score: number;
  };
  roi: {
    box: [number, number, number, number];
    display_box: [number, number, number, number];
  } | null;
  assessment: {
    category: 'no_expressed_signal' | 'attention_required';
    score: number;
  } | null;
  explanations: string[];
  warnings: string[];
  assets: AnalysisAssetDto[];
}

export interface AnalysisResultDto {
  frames: FrameResultDto[];
  warnings: string[];
  model: {
    version: string;
    device: string;
    service: string;
  };
  thresholds: {
    plane_detection: number;
    quality: number;
    assessment: number;
    warning_margin: number;
  };
}

export interface AnalysisJobDto {
  id: string;
  state: JobState;
  stage: string | null;
  progress: number;
  queue_position: number | null;
  attempt: number;
  frames: Array<{id: string; original_asset_url: string}>;
  result: AnalysisResultDto | null;
  error: {code: string} | null;
  expires_at: string;
}

export type ModelVersion = 'v1' | 'v2';

export interface SystemInfoDto {
  application_version: string;
  model_versions: ModelVersion[];
  default_model_version: ModelVersion;
  schema_version: string;
  device: string;
}

export interface CreatedAnalysisJobDto {
  id: string;
  state: JobState;
  frame_id: string;
  batch_position: number;
  polling_url: string;
}

export interface CreateAnalysisResponse {
  batch_id: string;
  jobs: CreatedAnalysisJobDto[];
}

export type FeedbackValue = 'positive' | 'negative';

export interface FeedbackResponse {
  feedback: {
    value: FeedbackValue;
    comment: string | null;
  };
}
