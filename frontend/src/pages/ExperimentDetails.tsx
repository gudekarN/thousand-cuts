import { Link, useParams } from 'react-router-dom'
import { useConfig, useExperiment, useManifest } from '@/hooks/useApi'
import { ApiRequestError } from '@/api/client'
import {
  DATASET_LABELS,
  MODEL_LABELS,
  NOISE_LABELS,
} from '@/lib/constants'
import {
  AlertCircle,
  Ban,
  CheckCircle2,
  Clock,
  FileJson,
  Hash,
  Loader2,
  RefreshCw,
  XCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'

function statusLabel(status: string | null): string {
  if (!status) return 'Unknown'
  return status.charAt(0).toUpperCase() + status.slice(1)
}

function statusClass(status: string | null): string {
  switch (status) {
    case 'completed':
      return 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20'
    case 'running':
      return 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20'
    case 'failed':
      return 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20'
    case 'queued':
    case 'cancelled':
      return 'bg-muted text-muted-foreground border-border'
    default:
      return 'bg-muted text-muted-foreground border-border'
  }
}

function NotFoundState({ runId }: { runId: string }) {
  return (
    <div className="flex-1 flex items-center justify-center min-h-[60vh] p-6">
      <div className="bg-card border border-border rounded-xl p-8 max-w-md w-full text-center space-y-4">
        <div className="mx-auto w-12 h-12 rounded-xl bg-muted flex items-center justify-center text-muted-foreground">
          <AlertCircle className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-lg font-bold">Experiment not found</h1>
          <p className="text-sm text-muted-foreground mt-1">
            No experiment matches this ID on the backend.
          </p>
          <code className="block mt-3 text-xs font-mono break-all text-foreground bg-muted/40 border border-border rounded-lg p-2">
            {runId}
          </code>
        </div>
        <div className="flex flex-col sm:flex-row gap-2 justify-center">
          <Link
            to="/experiments"
            className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90"
          >
            Official Full details
          </Link>
          <Link
            to="/setup"
            className="px-4 py-2 rounded-lg border border-border text-xs font-semibold hover:bg-muted"
          >
            Experiment Setup
          </Link>
        </div>
      </div>
    </div>
  )
}

export default function ExperimentDetails() {
  const { id } = useParams()
  const isCustomId = Boolean(id && !id.startsWith('official-'))
  const officialStage = id
    ? id.startsWith('official-')
      ? id.slice('official-'.length)
      : null
    : 'full'

  const manifestQuery = useManifest(officialStage ?? undefined, !isCustomId)
  const runQuery = useExperiment(isCustomId ? id ?? null : null)
  const { data: config } = useConfig()

  const activeQuery = isCustomId ? runQuery : manifestQuery
  const activeError = activeQuery.error
  const retry = () => {
    activeQuery.refetch()
  }

  if (activeQuery.isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-sm">Loading experiment details…</p>
        </div>
      </div>
    )
  }

  if (activeError) {
    if (activeError instanceof ApiRequestError && activeError.status === 404) {
      return <NotFoundState runId={id ?? 'full'} />
    }
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh] p-6">
        <div className="bg-card border border-red-500/30 rounded-xl p-6 max-w-md w-full space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-red-600 dark:text-red-400">
            <XCircle className="w-4 h-4" /> Could not load experiment details
          </div>
          <p className="text-xs text-muted-foreground">
            {activeError instanceof ApiRequestError
              ? `${activeError.message} (${activeError.code})`
              : activeError.message}
          </p>
          <button
            type="button"
            onClick={retry}
            className="px-3 py-1.5 rounded-lg border border-border text-xs font-semibold hover:bg-muted inline-flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Retry
          </button>
        </div>
      </div>
    )
  }

  const job = isCustomId ? runQuery.data : undefined
  const manifest = !isCustomId ? manifestQuery.data : undefined

  if (isCustomId && !job) {
    return <NotFoundState runId={id ?? 'unknown'} />
  }
  if (!isCustomId && !manifest) {
    return <NotFoundState runId={id ?? 'full'} />
  }

  const status = job?.status ?? manifest?.status ?? null
  const runId = job?.run_id ?? manifest?.run_id ?? id ?? null
  const runType = job ? 'Custom experiment' : 'Official experiment'
  const planned = job
    ? job.progress?.total ?? job.planned_fits ?? job.manifest_summary?.planned_fits ?? null
    : manifest?.planned_fits ?? null
  const completed = job
    ? job.progress?.done ?? job.manifest_summary?.completed_fits ?? null
    : manifest?.completed_fits ?? null
  const percent =
    completed != null && planned != null && planned > 0
      ? (completed / planned) * 100
      : null

  const seedList = job?.manifest_summary?.seeds ?? manifest?.seeds ?? null
  const seedCount = seedList?.length ?? job?.request?.seeds ?? null
  const configHash = job?.manifest_summary?.config_hash ?? manifest?.config_hash ?? null
  const methodologyVersion =
    job?.manifest_summary?.methodology_version ??
    manifest?.methodology_snapshot?.version ??
    manifest?.methodology_version ??
    null
  const runError = job?.error ?? null
  const request = job?.request ?? null

  const canDownloadManifest = !isCustomId && Boolean(manifest)

  const downloadManifest = () => {
    if (!manifest) return
    const blob = new Blob([JSON.stringify(manifest, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `manifest_${manifest.run_id ?? 'run'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const fitsValue =
    completed != null || planned != null
      ? `${completed ?? '—'} / ${planned ?? '—'}`
      : '—'
  const percentText =
    percent != null ? `${percent.toFixed(1)}% complete` : 'Progress unavailable'

  let seedSubtitle = 'Unavailable'
  if (seedList && seedList.length > 0) {
    const minSeed = Math.min(...seedList)
    const maxSeed = Math.max(...seedList)
    seedSubtitle = minSeed === maxSeed ? `Seed ${minSeed}` : `Seeds ${minSeed}–${maxSeed}`
  } else if (seedCount != null) {
    seedSubtitle = 'Requested seed count'
  }

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold tracking-tight">Experiment Details</h1>
            <span
              className={cn(
                'text-xs px-2.5 py-0.5 rounded-full font-medium border',
                statusClass(status)
              )}
            >
              Status: {statusLabel(status)}
            </span>
          </div>
          <p className="text-muted-foreground text-sm mt-1">
            Provenance, configuration hashes, environment metadata, and execution audit log.
          </p>
        </div>

        {canDownloadManifest && (
          <button
            onClick={downloadManifest}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 transition-opacity"
          >
            <FileJson className="w-4 h-4" /> Download Manifest
          </button>
        )}
      </div>

      {/* Overview Metadata Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Run ID</p>
          <p className="text-lg font-mono font-bold mt-1 text-primary break-all">
            {runId ?? '—'}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">{runType}</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Total Fits</p>
          <p className="text-lg font-mono font-bold mt-1">{fitsValue}</p>
          <p
            className={cn(
              'text-[10px] mt-0.5',
              percent != null ? 'text-emerald-600' : 'text-muted-foreground'
            )}
          >
            {percentText}
          </p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Seeds Evaluated</p>
          <p className="text-lg font-mono font-bold mt-1">
            {seedCount != null ? `${seedCount} ${seedCount === 1 ? 'seed' : 'seeds'}` : '—'}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">{seedSubtitle}</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Levels Freeze</p>
          <p
            className={cn(
              'text-lg font-mono font-bold mt-1',
              config ? (config.frozen ? 'text-emerald-600' : 'text-amber-600') : 'text-muted-foreground'
            )}
          >
            {config ? (config.frozen ? 'FROZEN' : 'PROVISIONAL') : '—'}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">
            {config ? `v${config.levels_version}` : 'Unavailable'}
          </p>
        </div>
      </div>

      {/* Custom run configuration and state */}
      {job && (
        <div className="bg-card border border-border rounded-xl p-6 space-y-4">
          <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
            Custom Run State
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
            <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
              <div className="text-[10px] uppercase font-semibold text-muted-foreground">Dataset</div>
              <div className="font-semibold mt-0.5">
                {request ? request.dataset_name ?? DATASET_LABELS[request.dataset] ?? request.dataset : '—'}
              </div>
            </div>
            <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
              <div className="text-[10px] uppercase font-semibold text-muted-foreground">Mode</div>
              <div className="font-semibold mt-0.5">
                {request
                  ? request.mode === 'sweep'
                    ? 'Full sweep (L0–L5)'
                    : `Single level (L${request.level ?? '?'})`
                  : '—'}
              </div>
            </div>
            <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
              <div className="text-[10px] uppercase font-semibold text-muted-foreground">Models</div>
              <div className="font-semibold mt-0.5">
                {request
                  ? request.models.map((model) => MODEL_LABELS[model] ?? model).join(', ')
                  : '—'}
              </div>
            </div>
            <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
              <div className="text-[10px] uppercase font-semibold text-muted-foreground">Noise Selection</div>
              <div className="font-semibold mt-0.5">
                {request
                  ? request.noises
                      .map((noise) =>
                        noise
                          .split('+')
                          .map((part) => NOISE_LABELS[part] ?? part)
                          .join(' + ')
                      )
                      .join(', ')
                  : '—'}
              </div>
            </div>
          </div>

          {job.status === 'queued' && (
            <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
              <Clock className="w-4 h-4 shrink-0 mt-0.5" />
              <span>Queued — waiting for the worker. Results are not available yet.</span>
            </div>
          )}

          {job.status === 'running' && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-700 dark:text-amber-400">
              <Loader2 className="w-4 h-4 shrink-0 mt-0.5 animate-spin" />
              <span>
                Running — {job.progress.done} of {job.progress.total} fits completed
                {job.progress.eta_seconds != null
                  ? `, ETA ${job.progress.eta_seconds.toFixed(1)}s`
                  : ''}
                . Results are not complete yet.
              </span>
            </div>
          )}

          {job.status === 'cancelled' && (
            <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
              <Ban className="w-4 h-4 shrink-0 mt-0.5" />
              <span>
                Cancelled — {job.progress.done} of {job.progress.total} fits completed. Partial results
                are preserved.
              </span>
            </div>
          )}

          {job.status === 'failed' && (
            <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-600 dark:text-red-400">
              <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>
                Failed — {completed ?? 0} of {planned ?? '—'} fits completed.
                {runError ? ` ${runError}` : ' No error message was recorded.'}
              </span>
            </div>
          )}

          {job.status === 'completed' && (
            <div className="flex items-start gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs text-emerald-700 dark:text-emerald-400">
              <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
              <span>
                Completed — {completed ?? '—'} of {planned ?? '—'} fits finished.
              </span>
            </div>
          )}
        </div>
      )}

      {/* Provenance Box */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
          Provenance & Canonical Hash
        </h2>
        <div className="space-y-3">
          <div className="bg-muted/40 p-3 rounded-lg border border-border/50">
            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
              <Hash className="w-3.5 h-3.5 text-primary" />
              Canonical Config Hash (SHA-256)
            </div>
            {configHash ? (
              <code className="text-xs font-mono text-foreground break-all block mt-1">
                {configHash}
              </code>
            ) : (
              <p className="text-xs text-muted-foreground mt-1">Unavailable</p>
            )}
          </div>

          <div className="grid sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-muted/40 p-3 rounded-lg border border-border/50 space-y-1">
              <div className="font-semibold text-muted-foreground">Experiment Methodology</div>
              <div className="text-foreground">
                Methodology Version: {methodologyVersion ?? '—'}
              </div>
              <div className="text-muted-foreground">
                Active Severity Table: {config ? `Table ${config.active_table}` : '—'}
              </div>
            </div>

            <div className="bg-muted/40 p-3 rounded-lg border border-border/50 space-y-1">
              <div className="font-semibold text-muted-foreground">Evaluation Split</div>
              <div className="text-foreground">Stratified 70% Train / 30% Test</div>
              <div className="text-muted-foreground">Noise applied to training set only</div>
            </div>
          </div>
        </div>
      </div>

      {/* Benchmark Protocol Specifications */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
          Benchmark Protocol & Rules
        </h2>
        <div className="space-y-2 text-xs text-muted-foreground leading-relaxed">
          <p>
            • <strong>D-002:</strong> Models use strictly fixed hyperparameter defaults (no tuning across levels).
          </p>
          <p>
            • <strong>D-003:</strong> Fixed noise injection order: 1. Label Noise → 2. Gaussian → 3. Outliers → 4. Missing values.
          </p>
          <p>
            • <strong>D-006:</strong> Pipeline consists of MeanImputer(keep_empty_features=True) → StandardScaler → Model, fitted only on noisy training data.
          </p>
          <p>
            • <strong>D-007:</strong> Breaking point is defined as the first severity level where mean Macro F1 drops below 90% of the clean baseline (≤ 0.90 × Baseline).
          </p>
          <p>
            • <strong>D-016:</strong> Model robustness ranking is ordered by Breaking Point Index (BPI), with ties broken by Robustness Score (RS).
          </p>
        </div>
      </div>
    </div>
  )
}
