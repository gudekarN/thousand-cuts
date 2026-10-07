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
}

export interface SummaryRow {
  dataset: string
  model: string
  combo: string
  level: number
  n_seeds: number
  f1_mean: number
  f1_std: number
  acc_mean: number
  acc_std: number
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
  config_hash?: string
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
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => apiFetch<HealthResponse>('/health'),
  config: () => apiFetch<ApiConfig>('/config'),
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
}
