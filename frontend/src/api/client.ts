/** Extended API client for results endpoints */

export interface ApiConfig {
  datasets: string[]
  models: Array<{ id: string; label: string }>
  noise_types: Array<{ id: string; order: number }>
  active_table: string
  levels: Array<{
    level: number
    label_flip_rate: number
    gaussian_k: number
    outlier_cell_rate: number
    missing_cell_rate: number
  }>
  levels_version: string
  frozen: boolean
  config_hash: string | null
  available_official_stages: string[]
}

export interface HealthResponse {
  status: string
  worker?: {
    status: string
    heartbeat_utc: string | null
    current_run_id: string | null
  } | null
}

export type ExperimentMode = 'single_level' | 'sweep'
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface ExperimentRequest {
  dataset: string
  dataset_name?: string
  models: string[]
  noises: string[]
  mode: ExperimentMode
  level: number | null
  seeds: number
}

export interface EstimateResponse {
  planned_fits: number
  estimated_seconds: number | null
  mean_fit_time_s: number | null
  estimate_source: string
  combos: string[]
  seeds: number[]
}

export interface CustomDataset {
  id: string
  label: string
  n_samples: number
  n_features: number
  n_classes: number
  source_url?: string | null
  target_column?: string | null
}

export interface JobProgress {
  done: number
  total: number
  percent: number
  current: {
    dataset: string
    model: string
    combo: string
    noise: string
    level: number
    seed: number
  } | null
  eta_seconds: number | null
}

export interface JobResponse {
  run_id: string
  run_type: 'custom'
  status: JobStatus
  request: ExperimentRequest
  planned_fits: number
  progress: JobProgress
  created_utc: string
  started_utc: string | null
  finished_utc: string | null
  heartbeat_utc: string | null
  cancel_requested: boolean
  error: string | null
  manifest_summary?: {
    status: string | null
    config_hash: string | null
    seeds: number[] | null
    planned_fits: number | null
    completed_fits: number | null
    created_utc: string | null
    finished_utc: string | null
    methodology_version: string | null
  } | null
  results_available?: boolean
}

export class ApiRequestError extends Error {
  status: number
  code: string
  details?: unknown

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = status
    this.code = code
    this.details = details
  }
}

export interface SummaryRow {
  dataset: string
  model: string
  combo: string
  level: number
  n_seeds: number
  f1_mean: number
  f1_std: number | null
  acc_mean: number
  acc_std: number | null
  fit_time_mean: number
  rel_f1: number
  drop_abs: number
  drop_rel: number
}

export interface BreakingPointRow {
  dataset: string
  model: string
  combo: string
  baseline_f1: number
  threshold_f1: number
  breaking_level: number | null
  reached: boolean
  noise_params: string | null
  f1_at_bp: number | null
  seeds_below_at_bp: number | null
}

export interface RobustnessRow {
  dataset: string
  model: string
  scope: string
  BPI: number
  RS: number
  rank: number
}

export interface BaselineRow {
  dataset: string
  model: string
  seed: number
  f1: number
  accuracy: number
}

export interface ManifestData {
  run_id: string
  stage: string
  run_type: string
  status: string
  planned_fits: number
  completed_fits: number
  seeds: number[]
  methodology_version?: string
  config_hash?: string | null
  created_utc?: string | null
  finished_utc?: string | null
  methodology_snapshot?: {
    version?: string
  }
}

export interface SynergyRow {
  dataset: string
  model: string
  combo: string
  combo_f1_mean: number
  additive_f1: number
  synergy: number
  floor_effect: boolean
}

/** Baseline row exactly as returned by GET /api/experiments/{id}/results. */
export interface ExperimentBaselineRow {
  dataset: string
  model: string
  f1_mean: number
  f1_std: number | null
  acc_mean: number
  acc_std: number | null
  threshold_f1: number
}

/** Synergy row exactly as returned by GET /api/experiments/{id}/results. */
export interface ExperimentSynergyRow {
  dataset: string
  model: string
  combo: string
  level: number
  baseline_f1: number
  compound_f1: number
  drop_compound: number
  sum_single_drops: number
  synergy: number
  f1_floor_flag: boolean
}

export interface ExperimentResultsResponse {
  run_id: string
  run_type: 'custom'
  status: JobStatus
  request: ExperimentRequest
  progress: JobProgress
  available: boolean
  baselines: ExperimentBaselineRow[]
  summary: SummaryRow[]
  breaking_points: BreakingPointRow[]
  synergy: ExperimentSynergyRow[]
  robustness: RobustnessRow[]
}

async function apiFetch<T>(path: string, params?: Record<string, string>, init?: RequestInit): Promise<T> {
  const url = new URL(`/api${path}`, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => { if (v) url.searchParams.set(k, v) })
  }
  const res = await fetch(url.toString(), {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text()
    let code = `HTTP_${res.status}`
    let message = text || `Request failed with status ${res.status}`
    let details: unknown
    try {
      const parsed = JSON.parse(text)
      if (parsed?.error) {
        code = parsed.error.code ?? code
        message = parsed.error.message ?? message
        details = parsed.error.details
      }
    } catch {
      // Non-JSON error body: keep the raw text.
    }
    throw new ApiRequestError(res.status, code, message, details)
  }
  if (res.status === 204) {
    return undefined as T
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => apiFetch<HealthResponse>('/health'),
  config: () => apiFetch<ApiConfig>('/config'),
  experimentDatasets: () => apiFetch<CustomDataset[]>('/experiments/datasets'),
  uploadDataset: (file: File, target: string) =>
    apiFetch<CustomDataset>('/experiments/datasets', { filename: file.name, target }, {
      method: 'POST',
      headers: { 'Content-Type': 'text/csv' },
      body: file,
    }),
  summary: (params?: { stage?: string; dataset?: string; model?: string; combo?: string }) =>
    apiFetch<SummaryRow[]>('/results/summary', params as Record<string, string>),
  breakingPoints: (params?: { stage?: string; dataset?: string; model?: string; combo?: string }) =>
    apiFetch<BreakingPointRow[]>('/results/breaking-points', params as Record<string, string>),
  robustness: (params?: { stage?: string; dataset?: string; scope?: string }) =>
    apiFetch<RobustnessRow[]>('/results/robustness', params as Record<string, string>),
  baselines: (params?: { stage?: string; dataset?: string }) =>
    apiFetch<BaselineRow[]>('/results/baselines', params as Record<string, string>),
  manifest: (stage?: string) =>
    apiFetch<ManifestData>('/results/manifest', stage ? { stage } : undefined),
  synergy: (params?: { stage?: string; dataset?: string }) =>
    apiFetch<SynergyRow[]>('/results/synergy', params as Record<string, string>),

  estimateExperiment: (body: ExperimentRequest) =>
    apiFetch<EstimateResponse>('/experiments/estimate', undefined, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  createExperiment: (body: ExperimentRequest) =>
    apiFetch<JobResponse>('/experiments', undefined, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getExperiment: (runId: string) =>
    apiFetch<JobResponse>(`/experiments/${encodeURIComponent(runId)}`),
  cancelExperiment: (runId: string) =>
    apiFetch<JobResponse>(`/experiments/${encodeURIComponent(runId)}/cancel`, undefined, {
      method: 'POST',
    }),
  experimentResults: (runId: string) =>
    apiFetch<ExperimentResultsResponse>(
      `/experiments/${encodeURIComponent(runId)}/results`
    ),
}
