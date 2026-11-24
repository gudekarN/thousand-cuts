import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  Clock,
  FlaskConical,
  Layers,
  Loader2,
  RefreshCw,
  SearchX,
  XCircle,
  type LucideIcon,
} from 'lucide-react'
import { ApiRequestError, type JobStatus, type SummaryRow } from '@/api/client'
import { useExperiment, useExperimentResults } from '@/hooks/useApi'
import LineBandChart from '@/components/charts/LineBandChart'
import GroupedBarChart from '@/components/charts/GroupedBarChart'
import {
  DATASET_LABELS,
  MODEL_COLORS,
  MODEL_LABELS,
  NOISE_LABELS,
  fmt4,
  fmtPct,
} from '@/lib/constants'
import { cn } from '@/lib/utils'

const NOISE_ORDER = ['label', 'gaussian', 'outliers', 'missing']
const MODEL_ORDER = ['logreg', 'svm_rbf', 'decision_tree', 'random_forest']

const STATUS_META: Record<JobStatus, { label: string; cls: string; icon: LucideIcon }> = {
  queued: { label: 'Queued', cls: 'bg-muted text-muted-foreground border-border', icon: Clock },
  running: {
    label: 'Running',
    cls: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20',
    icon: Loader2,
  },
  completed: {
    label: 'Completed',
    cls: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
    icon: CheckCircle2,
  },
  failed: {
    label: 'Failed',
    cls: 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20',
    icon: XCircle,
  },
  cancelled: { label: 'Cancelled', cls: 'bg-muted text-muted-foreground border-border', icon: Ban },
}

function errorText(error: unknown): string {
  if (error instanceof ApiRequestError) return `${error.message} (${error.code})`
  if (error instanceof Error) return error.message
  return 'Unknown error'
}

function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${Math.round(seconds % 60)}s`
}

function unique<T>(items: T[]): T[] {
  return Array.from(new Set(items))
}

function comboSort(a: string, b: string): number {
  const pa = a.split('+')
  const pb = b.split('+')
  if (pa.length !== pb.length) return pa.length - pb.length
  const rank = (parts: string[]) =>
    parts.reduce((acc, part) => acc * 10 + Math.max(0, NOISE_ORDER.indexOf(part)), 0)
  return rank(pa) - rank(pb)
}

function modelSort(a: string, b: string): number {
  const ia = MODEL_ORDER.indexOf(a)
  const ib = MODEL_ORDER.indexOf(b)
  return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
}

function StatusBadge({ status }: { status: JobStatus }) {
  const meta = STATUS_META[status]
  const Icon = meta.icon
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 text-xs px-2.5 py-0.5 rounded-full font-medium border',
        meta.cls
      )}
    >
      <Icon className={cn('w-3.5 h-3.5', status === 'running' && 'animate-spin')} />
      {meta.label}
    </span>
  )
}

function StatCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string
  value: React.ReactNode
  sub?: React.ReactNode
  accent?: string
}) {
  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <p className="text-[11px] font-semibold uppercase text-muted-foreground">{label}</p>
      <p className={cn('text-xl font-mono font-bold mt-1', accent)}>{value}</p>
      {sub ? <p className="text-[10px] text-muted-foreground mt-0.5">{sub}</p> : null}
    </div>
  )
}

function NotFoundState({ runId }: { runId: string }) {
  return (
    <div className="flex-1 flex items-center justify-center min-h-[60vh] p-6">
      <div className="bg-card border border-border rounded-xl p-8 max-w-md w-full text-center space-y-4">
        <div className="mx-auto w-12 h-12 rounded-xl bg-muted flex items-center justify-center text-muted-foreground">
          <SearchX className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-lg font-bold">Custom run not found</h1>
          <p className="text-sm text-muted-foreground mt-1">
            No custom experiment matches this run ID on the backend.
          </p>
          <code className="block mt-3 text-xs font-mono break-all text-foreground bg-muted/40 border border-border rounded-lg p-2">
            {runId}
          </code>
        </div>
        <div className="flex flex-col sm:flex-row gap-2 justify-center">
          <Link
            to="/results"
            className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90"
          >
            Official Results
          </Link>
          <Link
            to="/setup"
            className="px-4 py-2 rounded-lg border border-border text-xs font-semibold hover:bg-muted"
          >
            Start a new experiment
          </Link>
        </div>
      </div>
    </div>
  )
}

export default function CustomResultsPage({ runId }: { runId: string }) {
  const jobQuery = useExperiment(runId)
  const job = jobQuery.data
  const isFinal = job?.status === 'completed' || job?.status === 'cancelled'
  const resultsQuery = useExperimentResults(runId, Boolean(isFinal))
  const results = resultsQuery.data

  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [selectedCombo, setSelectedCombo] = useState<string | null>(null)

  const summaryRows = results?.summary ?? []

  const datasets = useMemo(
    () => unique(summaryRows.map((row) => row.dataset)).sort(),
    [summaryRows]
  )
  const models = useMemo(
    () => unique(summaryRows.map((row) => row.model)).sort(modelSort),
    [summaryRows]
  )
  const combos = useMemo(
    () =>
      unique(summaryRows.map((row) => row.combo))
        .filter((combo) => combo !== 'clean')
        .sort(comboSort),
    [summaryRows]
  )

  const dataset =
    selectedDataset && datasets.includes(selectedDataset)
      ? selectedDataset
      : datasets[0] ?? job?.request.dataset ?? null
  const datasetName = job?.request.dataset_name ?? DATASET_LABELS[dataset ?? ''] ?? dataset
  const model =
    selectedModel && models.includes(selectedModel)
      ? selectedModel
      : models[0] ?? job?.request.models[0] ?? null
  const combo =
    selectedCombo && combos.includes(selectedCombo)
      ? selectedCombo
      : combos[0] ?? null

  const selectedRows = useMemo<SummaryRow[]>(() => {
    if (!dataset || !model || !combo) return []
    return summaryRows
      .filter((row) => row.dataset === dataset && row.model === model && row.combo === combo)
      .sort((a, b) => a.level - b.level)
  }, [summaryRows, dataset, model, combo])

  const baseline = results?.baselines.find(
    (row) => row.dataset === dataset && row.model === model
  )
  const selectedBp = results?.breaking_points.find(
    (row) => row.dataset === dataset && row.model === model && row.combo === combo
  )
  const maxLevelRow =
    selectedRows.length > 0
      ? selectedRows.reduce((best, row) => (row.level > best.level ? row : best))
      : undefined

  const chartData = useMemo(() => {
    if (!model) return []
    return selectedRows.map((row) => ({
      level: row.level,
      [model]: row.f1_mean,
      accuracy: row.acc_mean,
    }))
  }, [selectedRows, model])

  const f1Values = selectedRows.map((row) => row.f1_mean).filter(Number.isFinite)
  const chartMin = Math.max(
    0,
    Math.floor(
      (Math.min(...f1Values, selectedBp?.threshold_f1 ?? 1) - 0.05) * 10
    ) / 10
  )
  const yDomain: [number, number] = [Number.isFinite(chartMin) ? chartMin : 0.4, 1]

  const datasetBpRows = useMemo(
    () =>
      (results?.breaking_points ?? [])
        .filter((row) => row.dataset === dataset)
        .sort((a, b) => comboSort(a.combo, b.combo) || modelSort(a.model, b.model)),
    [results, dataset]
  )
  const bpCombos = unique(datasetBpRows.map((row) => row.combo)).sort(comboSort)
  const bpModels = unique(datasetBpRows.map((row) => row.model)).sort(modelSort)
  const bpChartData = useMemo(() => {
    return bpCombos.map((bpCombo) => {
      const item: { category: string; [key: string]: string | number } = { category: bpCombo }
      bpModels.forEach((bpModel) => {
        const row = datasetBpRows.find(
          (candidate) => candidate.combo === bpCombo && candidate.model === bpModel
        )
        if (row) item[bpModel] = row.reached && row.breaking_level !== null ? row.breaking_level : 6
      })
      return item
    })
  }, [bpCombos, bpModels, datasetBpRows])

  const robustnessRows = useMemo(
    () =>
      (results?.robustness ?? [])
        .filter((row) => row.dataset === dataset)
        .sort((a, b) => (a.scope === b.scope ? a.rank - b.rank : a.scope.localeCompare(b.scope))),
    [results, dataset]
  )

  const synergyRows = useMemo(() => {
    if (!combo || !combo.includes('+')) return []
    return (results?.synergy ?? [])
      .filter(
        (row) =>
          row.dataset === dataset && row.model === model && row.combo === combo
      )
      .sort((a, b) => a.level - b.level)
  }, [results, dataset, model, combo])

  // ── Loading / not-found / error states ─────────────────────────────────
  if (jobQuery.isLoading && !job) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-sm">Loading custom run…</p>
        </div>
      </div>
    )
  }

  if (jobQuery.isError && !job) {
    if (jobQuery.error instanceof ApiRequestError && jobQuery.error.status === 404) {
      return <NotFoundState runId={runId} />
    }
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh] p-6">
        <div className="bg-card border border-red-500/30 rounded-xl p-6 max-w-md w-full space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-red-600 dark:text-red-400">
            <XCircle className="w-4 h-4" /> Could not load this custom run
          </div>
          <p className="text-xs text-muted-foreground">{errorText(jobQuery.error)}</p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => jobQuery.refetch()}
              className="px-3 py-1.5 rounded-lg border border-border text-xs font-semibold hover:bg-muted inline-flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Retry
            </button>
            <Link
              to="/results"
              className="px-3 py-1.5 rounded-lg border border-border text-xs font-semibold hover:bg-muted"
            >
              Official Results
            </Link>
          </div>
        </div>
      </div>
    )
  }

  if (!job) {
    return <NotFoundState runId={runId} />
  }

  const showCompletedResults = job.status === 'completed'
  const showCancelledResults = job.status === 'cancelled'
  const showMetrics =
    (showCompletedResults || showCancelledResults) && (results?.available ?? false)
  const hasSummary = summaryRows.length > 0
  const progressPercent = Math.max(0, Math.min(100, job.progress.percent))
  const request = job.request

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold tracking-tight">Custom Experiment Results</h1>
            <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold bg-primary/10 text-primary border border-primary/20">
              Custom Run
            </span>
            <StatusBadge status={job.status} />
          </div>
          <p className="text-muted-foreground text-sm mt-1">
            Results for this custom experiment only. Official Full results are kept separate.
          </p>
          <code className="inline-block mt-2 text-xs font-mono break-all text-muted-foreground">
            {runId}
          </code>
        </div>
        <Link
          to="/results"
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-border text-xs font-semibold hover:bg-muted transition-colors w-fit"
        >
          <FlaskConical className="w-3.5 h-3.5" /> Official Results
        </Link>
      </div>

      {/* Configuration / execution summary */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
          Run Configuration & Execution
        </h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
            <div className="text-[10px] uppercase font-semibold text-muted-foreground">Dataset</div>
            <div className="font-semibold mt-0.5">
              {request.dataset_name ?? DATASET_LABELS[request.dataset] ?? request.dataset}
            </div>
          </div>
          <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
            <div className="text-[10px] uppercase font-semibold text-muted-foreground">Mode</div>
            <div className="font-semibold mt-0.5">
              {request.mode === 'sweep'
                ? 'Full sweep (L0–L5)'
                : `Single level (L${request.level ?? '?'})`}
            </div>
          </div>
          <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
            <div className="text-[10px] uppercase font-semibold text-muted-foreground">
              Completed Fits
            </div>
            <div className="font-mono font-bold mt-0.5">
              {job.progress.done} / {job.progress.total}
            </div>
          </div>
          <div className="bg-muted/40 border border-border/50 rounded-lg p-3">
            <div className="text-[10px] uppercase font-semibold text-muted-foreground">Seeds</div>
            <div className="font-semibold mt-0.5">
              {request.seeds} ({request.seeds === 1 ? 'seed 0' : `seeds 0–${request.seeds - 1}`})
            </div>
          </div>
          <div className="bg-muted/40 border border-border/50 rounded-lg p-3 sm:col-span-2">
            <div className="text-[10px] uppercase font-semibold text-muted-foreground">Models</div>
            <div className="font-semibold mt-0.5">
              {request.models.map((id) => MODEL_LABELS[id] ?? id).join(', ')}
            </div>
          </div>
          <div className="bg-muted/40 border border-border/50 rounded-lg p-3 sm:col-span-2">
            <div className="text-[10px] uppercase font-semibold text-muted-foreground">
              Noise Selection
            </div>
            <div className="font-semibold mt-0.5">
              {request.noises
                .map((noise) =>
                  noise
                    .split('+')
                    .map((part) => NOISE_LABELS[part] ?? part)
                    .join(' + ')
                )
                .join(', ')}
            </div>
          </div>
          {job.manifest_summary?.config_hash && (
            <div className="bg-muted/40 border border-border/50 rounded-lg p-3 sm:col-span-2 lg:col-span-4">
              <div className="text-[10px] uppercase font-semibold text-muted-foreground">
                Config Hash
              </div>
              <code className="text-[11px] font-mono break-all mt-0.5 block">
                {job.manifest_summary.config_hash}
              </code>
            </div>
          )}
        </div>
      </div>

      {/* Queued */}
      {job.status === 'queued' && (
        <div className="bg-card border border-border rounded-xl p-6 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Clock className="w-4 h-4 text-muted-foreground" /> Queued — waiting for the worker
          </div>
          <p className="text-xs text-muted-foreground">
            This run has not started. It will execute after earlier queued jobs, one job at a time.
            Results are not available yet.
          </p>
          <div className="flex gap-2">
            <Link
              to={`/setup?runId=${encodeURIComponent(runId)}`}
              className="px-3 py-1.5 rounded-lg border border-border text-xs font-semibold hover:bg-muted"
            >
              Monitor live progress
            </Link>
          </div>
        </div>
      )}

      {/* Running */}
      {job.status === 'running' && (
        <div className="bg-card border border-border rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Loader2 className="w-4 h-4 animate-spin text-amber-500" /> Running
            </div>
            <span className="font-mono text-xs font-semibold">{job.progress.percent.toFixed(1)}%</span>
          </div>
          <div className="space-y-1.5">
            <div className="flex justify-between text-[11px] text-muted-foreground">
              <span className="font-mono">
                {job.progress.done} / {job.progress.total} fits
              </span>
              <span className="font-mono">ETA {fmtDuration(job.progress.eta_seconds)}</span>
            </div>
            <div className="h-2.5 w-full rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-primary transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
          {job.progress.current && (
            <div className="bg-muted/40 border border-border/50 rounded-lg p-3 grid grid-cols-2 sm:grid-cols-5 gap-2 text-[11px]">
              <div>
                <span className="text-muted-foreground">Dataset: </span>
                {job.request.dataset_name ?? DATASET_LABELS[job.progress.current.dataset] ?? job.progress.current.dataset}
              </div>
              <div>
                <span className="text-muted-foreground">Model: </span>
                {MODEL_LABELS[job.progress.current.model] ?? job.progress.current.model}
              </div>
              <div className="font-mono">
                <span className="text-muted-foreground font-sans">Noise: </span>
                {job.progress.current.combo}
              </div>
              <div>
                <span className="text-muted-foreground">Level: </span>L{job.progress.current.level}
              </div>
              <div>
                <span className="text-muted-foreground">Seed: </span>
                {job.progress.current.seed}
              </div>
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            Results will appear automatically after the run finishes. Partial metrics are not shown
            while the run is active.
          </p>
        </div>
      )}

      {/* Failed */}
      {job.status === 'failed' && (
        <div className="bg-card border border-red-500/30 rounded-xl p-6 space-y-2">
          <div className="flex items-center gap-2 text-sm font-semibold text-red-600 dark:text-red-400">
            <XCircle className="w-4 h-4" /> Run failed
          </div>
          <p className="text-xs text-muted-foreground">
            {job.error ?? 'The worker reported a failure without a message.'}
          </p>
          <p className="text-xs text-muted-foreground">
            Completed fits: {job.progress.done} of {job.progress.total}. Failed runs are not treated
            as complete results.
          </p>
        </div>
      )}

      {/* Cancelled banner */}
      {job.status === 'cancelled' && (
        <div className="bg-card border border-border rounded-xl p-6 space-y-2">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Ban className="w-4 h-4" /> Run cancelled
          </div>
          <p className="text-xs text-muted-foreground">
            {job.progress.done} of {job.progress.total} fits completed before cancellation.
            {showMetrics
              ? ' The results below are partial.'
              : ' No partial result rows were recorded.'}
          </p>
        </div>
      )}

      {/* Results area */}
      {(showCompletedResults || showCancelledResults) && (
        <>
          {resultsQuery.isLoading && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                {[0, 1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="h-24 rounded-xl bg-muted animate-pulse" />
                ))}
              </div>
              <div className="h-80 rounded-xl bg-muted animate-pulse" />
            </div>
          )}

          {resultsQuery.isError && (
            <div className="bg-card border border-red-500/30 rounded-xl p-6 space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-red-600 dark:text-red-400">
                <XCircle className="w-4 h-4" /> Could not load custom results
              </div>
              <p className="text-xs text-muted-foreground">{errorText(resultsQuery.error)}</p>
              <button
                type="button"
                onClick={() => resultsQuery.refetch()}
                className="px-3 py-1.5 rounded-lg border border-border text-xs font-semibold hover:bg-muted inline-flex items-center gap-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Retry
              </button>
            </div>
          )}

          {!resultsQuery.isLoading && !resultsQuery.isError && !showMetrics && (
            <div className="bg-card border border-border rounded-xl p-8 text-center space-y-2">
              <div className="mx-auto w-10 h-10 rounded-lg bg-muted flex items-center justify-center text-muted-foreground">
                <Layers className="w-5 h-5" />
              </div>
              <p className="text-sm font-semibold">No result rows available</p>
              <p className="text-xs text-muted-foreground">
                This run did not record any completed fits, so there are no custom results to show.
              </p>
            </div>
          )}

          {!resultsQuery.isLoading &&
            !resultsQuery.isError &&
            showMetrics &&
            (!hasSummary ? (
              <div className="bg-card border border-border rounded-xl p-8 text-center space-y-2">
                <p className="text-sm font-semibold">No derived results</p>
                <p className="text-xs text-muted-foreground">
                  Raw rows exist but no summary rows could be computed.
                </p>
              </div>
            ) : (
              <>
                {showCancelledResults && (
                  <div className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-xs text-amber-700 dark:text-amber-400">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">Partial results.</span> Based on{' '}
                      {job.progress.done} of {job.progress.total} fits. Missing levels or
                      combinations are omitted and metrics may change if the run is repeated.
                    </div>
                  </div>
                )}

                {/* Filters */}
                <div className="bg-card border border-border rounded-xl p-4 flex flex-wrap gap-4 items-center">
                  {datasets.length > 1 && (
                    <div className="space-y-1">
                      <label className="text-xs font-semibold text-muted-foreground uppercase">
                        Dataset
                      </label>
                      <div className="flex gap-1.5">
                        {datasets.map((d) => (
                          <button
                            key={d}
                            onClick={() => setSelectedDataset(d)}
                            className={cn(
                              'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                              dataset === d
                                ? 'bg-primary text-primary-foreground'
                                : 'bg-muted/50 text-foreground hover:bg-muted'
                            )}
                          >
                            {DATASET_LABELS[d] ?? d}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-muted-foreground uppercase">
                      Model
                    </label>
                    <div className="flex gap-1.5 flex-wrap">
                      {models.map((m) => (
                        <button
                          key={m}
                          onClick={() => setSelectedModel(m)}
                          className={cn(
                            'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                            model === m
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted/50 text-foreground hover:bg-muted'
                          )}
                        >
                          {MODEL_LABELS[m] ?? m}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-1 ml-auto">
                    <label className="text-xs font-semibold text-muted-foreground uppercase">
                      Noise Combo
                    </label>
                    <select
                      value={combo ?? ''}
                      onChange={(e) => setSelectedCombo(e.target.value)}
                      className="block px-3 py-1.5 rounded-lg border border-border bg-card text-foreground text-xs font-mono focus:outline-none focus:ring-1 focus:ring-primary"
                    >
                      {combos.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Stat cards */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                  <StatCard
                    label="Baseline F1"
                    value={fmt4(baseline?.f1_mean)}
                    sub={baseline?.f1_std == null ? undefined : `±${fmt4(baseline.f1_std)}`}
                    accent="text-primary"
                  />
                  <StatCard
                    label={`Final F1 (L${maxLevelRow?.level ?? '—'})`}
                    value={fmt4(maxLevelRow?.f1_mean)}
                    sub={maxLevelRow?.f1_std == null ? undefined : `±${fmt4(maxLevelRow.f1_std)}`}
                  />
                  <StatCard
                    label="Abs Drop"
                    value={fmt4(maxLevelRow?.drop_abs)}
                    sub="baseline − final level"
                    accent="text-amber-500"
                  />
                  <StatCard
                    label="Rel Drop"
                    value={fmtPct(maxLevelRow?.drop_rel)}
                    sub="% drop"
                    accent="text-amber-500"
                  />
                  <div className="bg-card border border-border rounded-xl p-4">
                    <p className="text-[11px] font-semibold uppercase text-muted-foreground">
                      Breaking Point
                    </p>
                    <div className="mt-1">
                      {selectedBp?.reached ? (
                        <span className="text-xl font-mono font-bold text-red-500">
                          L{selectedBp.breaking_level}
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 font-semibold">
                          <CheckCircle2 className="w-3 h-3" /> Not Reached
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-0.5">
                      {selectedBp?.reached
                        ? `below 90% at L${selectedBp.breaking_level}`
                        : request.mode === 'single_level'
                          ? 'within the available level'
                          : 'robust through L5'}
                    </p>
                  </div>
                  <StatCard
                    label="Avg Fit Time"
                    value={
                      maxLevelRow ? `${(maxLevelRow.fit_time_mean * 1000).toFixed(1)} ms` : '—'
                    }
                    sub="per fit"
                  />
                </div>

                {/* Main chart */}
                <div className="bg-card border border-border rounded-xl p-6 space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                      Macro F1 vs Noise Severity
                    </h2>
                    <span className="text-xs text-muted-foreground">
                      {MODEL_LABELS[model ?? ''] ?? model} • {combo}
                    </span>
                  </div>
                  <LineBandChart
                    data={chartData}
                    lines={[
                      {
                        key: model ?? 'model',
                        name: `Macro F1 (${MODEL_LABELS[model ?? ''] ?? model ?? 'model'})`,
                        color: MODEL_COLORS[model ?? ''] ?? '#6366F1',
                      },
                    ]}
                    threshold={selectedBp?.threshold_f1}
                    yDomain={yDomain}
                    yLabel="Macro F1"
                  />
                </div>

                {/* Level breakdown */}
                <div className="bg-card border border-border rounded-xl overflow-hidden">
                  <div className="px-6 py-4 border-b border-border flex items-center justify-between gap-2">
                    <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                      Level-by-Level Breakdown
                    </h2>
                    <span className="text-xs font-mono text-muted-foreground">
                      {combo}
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border bg-muted/30">
                          <th className="text-left px-6 py-3 font-medium text-muted-foreground">
                            Level
                          </th>
                          <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                            Macro F1
                          </th>
                          <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                            Accuracy
                          </th>
                          <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                            Rel F1
                          </th>
                          <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                            Abs Drop
                          </th>
                          <th className="text-right px-6 py-3 font-medium text-muted-foreground">
                            Status
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedRows.map((row, index) => {
                          const isBroken =
                            selectedBp?.reached &&
                            selectedBp.breaking_level !== null &&
                            row.level >= selectedBp.breaking_level
                          return (
                            <tr
                              key={`${row.combo}-${row.level}`}
                              className={cn(
                                'border-b border-border last:border-0',
                                index % 2 === 0 ? '' : 'bg-muted/10'
                              )}
                            >
                              <td className="px-6 py-3 font-mono font-bold">L{row.level}</td>
                              <td className="text-right px-4 py-3 font-mono font-medium">
                                {fmt4(row.f1_mean)}{' '}
                                {row.f1_std != null && (
                                  <span className="text-xs text-muted-foreground"> ±{fmt4(row.f1_std)}</span>
                                )}
                              </td>
                              <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                                {fmt4(row.acc_mean)}
                              </td>
                              <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                                {fmtPct(row.rel_f1)}
                              </td>
                              <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                                {fmt4(row.drop_abs)}
                              </td>
                              <td className="text-right px-6 py-3 font-mono">
                                {isBroken ? (
                                  <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded font-semibold bg-red-500/10 text-red-500">
                                    <XCircle className="w-3 h-3" /> Broken
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded font-semibold bg-emerald-500/10 text-emerald-600">
                                    <CheckCircle2 className="w-3 h-3" /> OK
                                  </span>
                                )}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Breaking levels bar chart */}
                {bpCombos.length > 0 && bpModels.length > 0 && (
                  <div className="bg-card border border-border rounded-xl p-6 space-y-3">
                    <div className="flex items-center justify-between">
                      <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                        Breaking Level by Noise Combo
                      </h2>
                      <span className="text-xs text-muted-foreground">6 = Not Reached</span>
                    </div>
                    <GroupedBarChart
                      data={bpChartData}
                      bars={bpModels.map((m) => ({
                        key: m,
                        name: MODEL_LABELS[m] ?? m,
                        color: MODEL_COLORS[m],
                      }))}
                      yLabel="Breaking Level (0-6)"
                      yDomain={[0, 6.5]}
                    />
                  </div>
                )}

                {/* Breaking points table */}
                {datasetBpRows.length > 0 && (
                  <div className="bg-card border border-border rounded-xl overflow-hidden">
                    <div className="px-6 py-4 border-b border-border">
                      <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                        Breaking Points — {datasetName}
                      </h2>
                    </div>
                    <div className="overflow-x-auto max-h-[420px]">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-muted">
                          <tr className="border-b border-border">
                            <th className="text-left px-6 py-3 font-medium text-muted-foreground">
                              Combo
                            </th>
                            <th className="text-left px-4 py-3 font-medium text-muted-foreground">
                              Model
                            </th>
                            <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                              Baseline F1
                            </th>
                            <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                              Threshold (90%)
                            </th>
                            <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                              Breaking Level
                            </th>
                            <th className="text-right px-6 py-3 font-medium text-muted-foreground">
                              Status
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {datasetBpRows.map((row, index) => (
                            <tr
                              key={`${row.combo}-${row.model}`}
                              className={cn(
                                'border-b border-border last:border-0',
                                index % 2 === 0 ? '' : 'bg-muted/10'
                              )}
                            >
                              <td className="px-6 py-2.5 font-mono text-xs">{row.combo}</td>
                              <td className="px-4 py-2.5 font-medium">
                                {MODEL_LABELS[row.model] ?? row.model}
                              </td>
                              <td className="text-right px-4 py-2.5 font-mono">
                                {fmt4(row.baseline_f1)}
                              </td>
                              <td className="text-right px-4 py-2.5 font-mono text-muted-foreground">
                                {fmt4(row.threshold_f1)}
                              </td>
                              <td className="text-right px-4 py-2.5 font-mono font-bold">
                                {row.reached ? `L${row.breaking_level}` : 'Not Reached'}
                              </td>
                              <td className="text-right px-6 py-2.5">
                                <span
                                  className={cn(
                                    'text-xs px-2 py-0.5 rounded font-semibold',
                                    row.reached
                                      ? 'bg-red-500/10 text-red-500'
                                      : 'bg-emerald-500/10 text-emerald-600'
                                  )}
                                >
                                  {row.reached ? 'Broken' : 'Robust'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Robustness ranking (sweep runs only) */}
                {request.mode === 'sweep' && robustnessRows.length > 0 && (
                  <div className="bg-card border border-border rounded-xl overflow-hidden">
                    <div className="px-6 py-4 border-b border-border">
                      <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                        Robustness Ranking
                      </h2>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-border bg-muted/30">
                            <th className="text-left px-6 py-3 font-medium text-muted-foreground">
                              Rank
                            </th>
                            <th className="text-left px-4 py-3 font-medium text-muted-foreground">
                              Model
                            </th>
                            <th className="text-left px-4 py-3 font-medium text-muted-foreground">
                              Scope
                            </th>
                            <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                              BPI
                            </th>
                            <th className="text-right px-6 py-3 font-medium text-muted-foreground">
                              RS
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {robustnessRows.map((row) => (
                            <tr
                              key={`${row.scope}-${row.model}`}
                              className="border-b border-border last:border-0"
                            >
                              <td className="px-6 py-2.5 font-bold">#{row.rank}</td>
                              <td className="px-4 py-2.5 font-medium flex items-center gap-2">
                                <div
                                  className="w-3 h-3 rounded-full"
                                  style={{ backgroundColor: MODEL_COLORS[row.model] ?? '#6366F1' }}
                                />
                                {MODEL_LABELS[row.model] ?? row.model}
                              </td>
                              <td className="px-4 py-2.5 text-muted-foreground capitalize">
                                {row.scope}
                              </td>
                              <td className="text-right px-4 py-2.5 font-mono font-bold text-primary">
                                {fmt4(row.BPI)}
                              </td>
                              <td className="text-right px-6 py-2.5 font-mono">{fmt4(row.RS)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Synergy for the selected compound combo */}
                {combo?.includes('+') && (
                  <div className="bg-card border border-border rounded-xl p-6 space-y-4">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs px-2 py-0.5 rounded font-semibold bg-amber-500/10 text-amber-600 border border-amber-500/20">
                        Secondary Analysis
                      </span>
                      <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                        Compound Noise Synergy — {combo}
                      </h2>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Synergy is the compound F1 drop minus the sum of the component single-noise
                      drops. Positive values indicate worse-than-additive degradation.
                    </p>
                    {synergyRows.length === 0 ? (
                      <p className="text-xs text-muted-foreground">
                        No synergy rows are available for the selected model, combo and levels.
                      </p>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-border bg-muted/30">
                              <th className="text-left px-6 py-3 font-medium text-muted-foreground">
                                Level
                              </th>
                              <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                                Compound F1
                              </th>
                              <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                                Actual Drop
                              </th>
                              <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                                Additive Drop
                              </th>
                              <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                                Synergy Δ
                              </th>
                              <th className="text-right px-6 py-3 font-medium text-muted-foreground">
                                Flag
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {synergyRows.map((row) => (
                              <tr
                                key={`${row.combo}-${row.level}`}
                                className="border-b border-border last:border-0"
                              >
                                <td className="px-6 py-2.5 font-mono font-bold">L{row.level}</td>
                                <td className="text-right px-4 py-2.5 font-mono">
                                  {fmt4(row.compound_f1)}
                                </td>
                                <td className="text-right px-4 py-2.5 font-mono">
                                  {fmt4(row.drop_compound)}
                                </td>
                                <td className="text-right px-4 py-2.5 font-mono text-muted-foreground">
                                  {fmt4(row.sum_single_drops)}
                                </td>
                                <td className="text-right px-4 py-2.5 font-mono font-bold">
                                  <span
                                    className={row.synergy > 0 ? 'text-red-500' : 'text-emerald-500'}
                                  >
                                    {row.synergy > 0 ? `+${fmt4(row.synergy)}` : fmt4(row.synergy)}
                                  </span>
                                </td>
                                <td className="text-right px-6 py-2.5">
                                  {row.f1_floor_flag && (
                                    <span className="inline-flex items-center gap-1 text-[11px] text-amber-600 bg-amber-500/10 px-2 py-0.5 rounded">
                                      <AlertTriangle className="w-3 h-3" /> Floor Effect
                                    </span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </>
            ))}
        </>
      )}
    </div>
  )
}
