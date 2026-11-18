import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  AlertTriangle,
  Ban,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Database,
  Layers,
  Loader2,
  Play,
  RefreshCw,
  Sliders,
  WifiOff,
  XCircle,
  type LucideIcon,
} from 'lucide-react'
import {
  ApiRequestError,
  type ExperimentRequest,
  type JobStatus,
} from '@/api/client'
import {
  useCancelExperiment,
  useConfig,
  useCreateExperiment,
  useExperiment,
  useExperimentEstimate,
  useHealth,
} from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import {
  DATASET_LABELS,
  MODEL_LABELS,
  NOISE_LABELS,
  NOISE_COLORS,
} from '@/lib/constants'
import { cn } from '@/lib/utils'

const ACTIVE_RUN_KEY = 'cmdco.activeCustomRunId'

const STATUS_META: Record<JobStatus, { label: string; cls: string; icon: LucideIcon }> = {
  queued: {
    label: 'Queued',
    cls: 'bg-muted text-muted-foreground border-border',
    icon: Clock,
  },
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
  cancelled: {
    label: 'Cancelled',
    cls: 'bg-muted text-muted-foreground border-border',
    icon: Ban,
  },
}

function errorText(error: unknown): string {
  if (error instanceof ApiRequestError) {
    return `${error.message} (${error.code})`
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}

function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return `${minutes}m ${rest}s`
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

export default function ExperimentSetup() {
  const [searchParams, setSearchParams] = useSearchParams()
  const {
    data: config,
    isLoading: configLoading,
    isError: configError,
    refetch: refetchConfig,
  } = useConfig()
  const health = useHealth()

  const [selectedDataset, setSelectedDataset] = useState('breast_cancer')
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'logreg',
    'svm_rbf',
    'decision_tree',
    'random_forest',
  ])
  const [selectedNoises, setSelectedNoises] = useState<string[]>(['gaussian', 'label'])
  const [mode, setMode] = useState<'sweep' | 'single_level'>('sweep')
  const [singleLevel, setSingleLevel] = useState<number>(3)
  const [seedCount, setSeedCount] = useState<number>(3)

  // ── Persisted run id (URL first, then localStorage) ─────────────────────
  const [runId, setRunId] = useState<string | null>(searchParams.get('runId'))

  useEffect(() => {
    const paramRun = searchParams.get('runId')
    if (paramRun) {
      setRunId(paramRun)
      window.localStorage.setItem(ACTIVE_RUN_KEY, paramRun)
      return
    }
    const stored = window.localStorage.getItem(ACTIVE_RUN_KEY)
    if (stored) {
      setRunId(stored)
      setSearchParams({ runId: stored }, { replace: true })
    }
  }, [searchParams, setSearchParams])

  // ── Form helpers ────────────────────────────────────────────────────────
  const toggleModel = (id: string) => {
    setSelectedModels((prev) =>
      prev.includes(id) ? (prev.length > 1 ? prev.filter((m) => m !== id) : prev) : [...prev, id]
    )
  }

  const toggleNoise = (id: string) => {
    setSelectedNoises((prev) =>
      prev.includes(id) ? (prev.length > 1 ? prev.filter((n) => n !== id) : prev) : [...prev, id]
    )
  }

  const validationError = useMemo(() => {
    if (selectedModels.length === 0) return 'Select at least one model.'
    if (selectedNoises.length === 0) return 'Select at least one noise type.'
    if (mode === 'single_level' && (singleLevel < 1 || singleLevel > 5)) {
      return 'Single-level mode requires a level from 1 to 5.'
    }
    if (seedCount < 1 || seedCount > 10) return 'Seed count must be between 1 and 10.'
    return null
  }, [selectedModels, selectedNoises, mode, singleLevel, seedCount])

  const request = useMemo<ExperimentRequest>(
    () => ({
      dataset: selectedDataset,
      models: selectedModels,
      noises: selectedNoises,
      mode,
      level: mode === 'single_level' ? singleLevel : null,
      seeds: seedCount,
    }),
    [selectedDataset, selectedModels, selectedNoises, mode, singleLevel, seedCount]
  )

  const debouncedRequest = useDebouncedValue(request, 300)
  const estimate = useExperimentEstimate(validationError ? null : debouncedRequest)

  // ── Mutations and live job state ────────────────────────────────────────
  const createExperiment = useCreateExperiment()
  const cancelExperiment = useCancelExperiment()
  const jobQuery = useExperiment(runId)
  const job = jobQuery.data

  const createdJob = createExperiment.data
  useEffect(() => {
    if (!createdJob) return
    setRunId(createdJob.run_id)
    window.localStorage.setItem(ACTIVE_RUN_KEY, createdJob.run_id)
    setSearchParams({ runId: createdJob.run_id }, { replace: true })
  }, [createdJob, setSearchParams])

  const clearRun = () => {
    window.localStorage.removeItem(ACTIVE_RUN_KEY)
    setRunId(null)
    setSearchParams({}, { replace: true })
    createExperiment.reset()
    cancelExperiment.reset()
  }

  const handleStartRun = () => {
    if (validationError || createExperiment.isPending) return
    cancelExperiment.reset()
    createExperiment.mutate(request)
  }

  const handleCancel = () => {
    if (!runId || cancelExperiment.isPending) return
    cancelExperiment.mutate(runId)
  }

  const isActive = job?.status === 'queued' || job?.status === 'running'
  const workerStatus = health.data?.worker?.status
  const workerUnavailable =
    health.isError || (workerStatus != null && ['unavailable', 'stale', 'stopped'].includes(workerStatus))

  const progressPercent = job ? Math.max(0, Math.min(100, job.progress.percent)) : 0
  const hasCredentials = Boolean(config?.models?.length)

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Experiment Setup</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Configure a targeted noise robustness experiment. Fit counts and runtime come from the
          backend planner.
        </p>
      </div>

      {/* Connectivity / worker banners */}
      {(configError || health.isError) && (
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 rounded-xl border border-red-500/30 bg-red-500/5 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-red-600 dark:text-red-400">
            <WifiOff className="w-4 h-4" /> Cannot reach the backend API
          </div>
          <p className="text-xs text-muted-foreground sm:ml-2">
            Start the FastAPI server on port 8000, then retry.
          </p>
          <button
            type="button"
            onClick={() => {
              refetchConfig()
              health.refetch()
            }}
            className="sm:ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs font-semibold hover:bg-muted transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Retry
          </button>
        </div>
      )}

      {!configError && !health.isError && workerUnavailable && (
        <div className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-xs text-amber-700 dark:text-amber-400">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Custom worker unavailable</span>
            {workerStatus ? ` (status: ${workerStatus})` : ''}. New runs will be queued but will not
            start until the worker process is running.
          </div>
        </div>
      )}

      <div className="grid md:grid-cols-3 gap-6">
        {/* Left 2 Cols: Form */}
        <div className="md:col-span-2 space-y-6">
          {/* Dataset */}
          <div className="bg-card border border-border rounded-xl p-6 space-y-3">
            <label className="text-xs font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <Database className="w-4 h-4 text-primary" />
              1. Target Dataset
            </label>
            <div className="grid sm:grid-cols-2 gap-3">
              {[
                { id: 'breast_cancer', desc: '569 samples • 30 continuous features • 2 classes' },
                { id: 'digits', desc: '1,797 samples • 64 pixel features • 10 classes' },
              ].map((d) => (
                <button
                  key={d.id}
                  type="button"
                  onClick={() => setSelectedDataset(d.id)}
                  className={cn(
                    'p-4 rounded-xl border text-left transition-all',
                    selectedDataset === d.id
                      ? 'border-primary bg-primary/5 shadow-sm'
                      : 'border-border bg-card hover:bg-muted/30'
                  )}
                >
                  <div className="font-semibold text-sm">{DATASET_LABELS[d.id]}</div>
                  <div className="text-xs text-muted-foreground mt-1">{d.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Models */}
          <div className="bg-card border border-border rounded-xl p-6 space-y-3">
            <label className="text-xs font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <BrainCircuit className="w-4 h-4 text-primary" />
              2. Evaluated Models
            </label>
            {configLoading && !config ? (
              <div className="flex gap-2">
                {[0, 1, 2, 3].map((i) => (
                  <div key={i} className="h-9 w-32 rounded-lg bg-muted animate-pulse" />
                ))}
              </div>
            ) : configError ? (
              <p className="text-xs text-red-500">
                Models could not be loaded from the API. Use Retry above.
              </p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {config?.models.map((m) => {
                  const selected = selectedModels.includes(m.id)
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => toggleModel(m.id)}
                      className={cn(
                        'px-3.5 py-2 rounded-lg text-xs font-medium border transition-colors',
                        selected
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'bg-card text-foreground border-border hover:bg-muted'
                      )}
                    >
                      {m.label}
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          {/* Noise Injectors */}
          <div className="bg-card border border-border rounded-xl p-6 space-y-3">
            <label className="text-xs font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-primary" />
              3. Noise Types (Fixed Order Application)
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {['label', 'gaussian', 'outliers', 'missing'].map((n) => {
                const selected = selectedNoises.includes(n)
                return (
                  <button
                    key={n}
                    type="button"
                    onClick={() => toggleNoise(n)}
                    className={cn(
                      'p-3 rounded-lg border text-left transition-all',
                      selected
                        ? 'border-primary bg-primary/5'
                        : 'border-border bg-card hover:bg-muted/20 opacity-60'
                    )}
                  >
                    <div
                      className="w-2.5 h-2.5 rounded-full mb-2"
                      style={{ backgroundColor: NOISE_COLORS[n] }}
                    />
                    <div className="text-xs font-semibold">{NOISE_LABELS[n]}</div>
                    <div className="text-[10px] text-muted-foreground capitalize mt-0.5">{n}</div>
                  </button>
                )
              })}
            </div>
            <p className="text-[11px] text-muted-foreground">
              Fixed sequence: Label → Gaussian → Outliers → Missing. One noise runs individually;
              two or more run as one compound combination plus each single-noise component.
            </p>
          </div>

          {/* Mode & Seeds */}
          <div className="bg-card border border-border rounded-xl p-6 space-y-4">
            <label className="text-xs font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-primary" />
              4. Execution Mode & Seed Count
            </label>

            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Severity Sweep Mode</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setMode('sweep')}
                    className={cn(
                      'flex-1 py-1.5 px-3 rounded-lg text-xs font-medium border',
                      mode === 'sweep'
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'border-border text-foreground'
                    )}
                  >
                    Full Sweep (L0–L5)
                  </button>
                  <button
                    type="button"
                    onClick={() => setMode('single_level')}
                    className={cn(
                      'flex-1 py-1.5 px-3 rounded-lg text-xs font-medium border',
                      mode === 'single_level'
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'border-border text-foreground'
                    )}
                  >
                    Single Level
                  </button>
                </div>
              </div>

              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Random Seeds: <strong className="text-foreground">{seedCount}</strong> (0 to{' '}
                  {seedCount - 1})
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={seedCount}
                  onChange={(e) => setSeedCount(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer"
                />
              </div>
            </div>

            {mode === 'single_level' && (
              <div className="pt-2 border-t border-border">
                <label className="text-xs text-muted-foreground block mb-1">
                  Target Level: <strong>L{singleLevel}</strong>
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={singleLevel}
                  onChange={(e) => setSingleLevel(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer"
                />
                <p className="text-[11px] text-muted-foreground mt-1">
                  The clean baseline (L0) is always included automatically.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Right Col: Summary & Run Status */}
        <div className="space-y-6">
          <div className="bg-card border border-border rounded-xl p-6 space-y-4">
            <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
              Run Configuration Summary
            </h2>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Dataset</span>
                <span className="font-semibold">{DATASET_LABELS[selectedDataset]}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Models</span>
                <span className="font-semibold">{selectedModels.length} selected</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Noise Injectors</span>
                <span className="font-semibold">
                  {selectedNoises.length} {selectedNoises.length > 1 ? '(compound)' : ''}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Seeds</span>
                <span className="font-semibold">
                  {seedCount} {seedCount === 1 ? 'seed' : 'seeds'}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Planned Fits</span>
                <span className="font-mono font-bold text-primary">
                  {validationError ? (
                    '—'
                  ) : estimate.isLoading ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin inline" />
                  ) : estimate.data ? (
                    `${estimate.data.planned_fits} fits`
                  ) : (
                    '—'
                  )}
                </span>
              </div>
              <div className="flex justify-between py-1.5">
                <span className="text-muted-foreground">Estimated Time</span>
                <span className="font-mono font-semibold">
                  {validationError
                    ? '—'
                    : estimate.data
                      ? `~${fmtDuration(estimate.data.estimated_seconds)}`
                      : '—'}
                </span>
              </div>
            </div>

            {estimate.data && !validationError && (
              <div className="text-[10px] text-muted-foreground leading-relaxed">
                Combinations: <span className="font-mono">{estimate.data.combos.join(', ')}</span>
                <br />
                Estimate source:{' '}
                {estimate.data.estimate_source === 'official_full_mean'
                  ? 'official Full fit times'
                  : 'documented default'}
              </div>
            )}

            {validationError && (
              <p className="text-[11px] text-red-500 flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5" /> {validationError}
              </p>
            )}

            {estimate.isError && !validationError && (
              <p className="text-[11px] text-red-500">
                Could not estimate this run: {errorText(estimate.error)}
              </p>
            )}

            <button
              onClick={handleStartRun}
              disabled={
                Boolean(validationError) || createExperiment.isPending || configError || !hasCredentials
              }
              className="w-full py-2.5 px-4 rounded-lg bg-primary text-primary-foreground font-semibold text-xs flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {createExperiment.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Submitting…
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" /> Start Custom Run
                </>
              )}
            </button>

            {createExperiment.isError && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-[11px] text-red-600 dark:text-red-400 space-y-1">
                <div className="font-semibold flex items-center gap-1.5">
                  <XCircle className="w-3.5 h-3.5" /> Run could not be started
                </div>
                <div>{errorText(createExperiment.error)}</div>
              </div>
            )}
          </div>

          {/* Live Run Status Card */}
          {runId && (
            <div className="bg-card border border-border rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                  Run Status
                </h2>
                {job ? <StatusBadge status={job.status} /> : null}
              </div>

              <div>
                <div className="text-[10px] text-muted-foreground uppercase font-semibold">
                  Custom Run ID
                </div>
                <div className="font-mono text-[11px] break-all text-foreground">{runId}</div>
              </div>

              {jobQuery.isLoading && !job && (
                <div className="space-y-2">
                  <div className="h-2.5 w-full rounded bg-muted animate-pulse" />
                  <div className="h-16 w-full rounded-lg bg-muted animate-pulse" />
                </div>
              )}

              {jobQuery.isError && (
                <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-[11px] text-red-600 dark:text-red-400 space-y-2">
                  <div className="font-semibold flex items-center gap-1.5">
                    <XCircle className="w-3.5 h-3.5" /> Could not load this run
                  </div>
                  <div>{errorText(jobQuery.error)}</div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => jobQuery.refetch()}
                      className="px-2.5 py-1 rounded border border-border text-foreground hover:bg-muted inline-flex items-center gap-1"
                    >
                      <RefreshCw className="w-3 h-3" /> Retry
                    </button>
                    <button
                      type="button"
                      onClick={clearRun}
                      className="px-2.5 py-1 rounded border border-border text-foreground hover:bg-muted"
                    >
                      Forget run
                    </button>
                  </div>
                </div>
              )}

              {job && (
                <>
                  {/* Progress bar */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-muted-foreground">
                      <span className="font-mono">
                        {job.progress.done} / {job.progress.total} fits
                      </span>
                      <span className="font-mono font-semibold text-foreground">
                        {job.progress.percent.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2.5 w-full rounded-full bg-muted overflow-hidden">
                      <div
                        className={cn(
                          'h-full rounded-full transition-all duration-300',
                          job.status === 'completed'
                            ? 'bg-emerald-500'
                            : job.status === 'failed'
                              ? 'bg-red-500'
                              : job.status === 'cancelled'
                                ? 'bg-muted-foreground/40'
                                : 'bg-primary'
                        )}
                        style={{ width: `${progressPercent}%` }}
                      />
                    </div>
                  </div>

                  {/* Progress details */}
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="bg-muted/40 p-3 rounded-lg border border-border/50">
                      <div className="text-[10px] uppercase font-semibold text-muted-foreground">
                        Completed Fits
                      </div>
                      <div className="font-mono font-bold mt-0.5">
                        {job.progress.done} / {job.progress.total}
                      </div>
                    </div>
                    <div className="bg-muted/40 p-3 rounded-lg border border-border/50">
                      <div className="text-[10px] uppercase font-semibold text-muted-foreground">
                        ETA
                      </div>
                      <div className="font-mono font-bold mt-0.5">
                        {job.status === 'queued'
                          ? 'Waiting for worker'
                          : job.status === 'running'
                            ? fmtDuration(job.progress.eta_seconds)
                            : job.status === 'completed'
                              ? 'Done'
                              : '—'}
                      </div>
                    </div>
                  </div>

                  {/* Current fit */}
                  <div className="bg-muted/40 p-3 rounded-lg border border-border/50">
                    <div className="text-[10px] uppercase font-semibold text-muted-foreground mb-1">
                      Current Fit
                    </div>
                    {job.progress.current ? (
                      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                        <div>
                          <span className="text-muted-foreground">Dataset: </span>
                          {DATASET_LABELS[job.progress.current.dataset] ??
                            job.progress.current.dataset}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Model: </span>
                          {MODEL_LABELS[job.progress.current.model] ?? job.progress.current.model}
                        </div>
                        <div className="col-span-2 font-mono break-all">
                          <span className="text-muted-foreground font-sans">Noise: </span>
                          {job.progress.current.combo === 'clean'
                            ? 'Clean baseline (L0)'
                            : job.progress.current.combo}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Level: </span>
                          L{job.progress.current.level}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Seed: </span>
                          {job.progress.current.seed}
                        </div>
                      </div>
                    ) : (
                      <div className="text-[11px] text-muted-foreground">
                        {job.status === 'queued'
                          ? 'No fit has started yet.'
                          : 'No active fit information available.'}
                      </div>
                    )}
                  </div>

                  {/* Status-specific messaging */}
                  {job.status === 'queued' && (
                    <p className="text-[11px] text-muted-foreground">
                      Queued. The worker starts jobs in FIFO order, one at a time.
                    </p>
                  )}

                  {job.status === 'running' && job.cancel_requested && (
                    <p className="text-[11px] text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      Cancellation requested — stopping after the current fit.
                    </p>
                  )}

                  {job.status === 'completed' && (
                    <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-[11px] text-emerald-700 dark:text-emerald-400 space-y-1">
                      <div className="font-semibold flex items-center gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Run completed
                      </div>
                      <div>
                        {job.progress.done} of {job.progress.total} fits were processed. Results are
                        stored for this custom run.
                      </div>
                    </div>
                  )}

                  {job.status === 'cancelled' && (
                    <div className="rounded-lg border border-border bg-muted/40 p-3 text-[11px] text-muted-foreground space-y-1">
                      <div className="font-semibold flex items-center gap-1.5 text-foreground">
                        <Ban className="w-3.5 h-3.5" /> Run cancelled
                      </div>
                      <div>Partial results from completed fits are preserved.</div>
                    </div>
                  )}

                  {job.status === 'failed' && (
                    <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-[11px] text-red-600 dark:text-red-400 space-y-1">
                      <div className="font-semibold flex items-center gap-1.5">
                        <XCircle className="w-3.5 h-3.5" /> Run failed
                      </div>
                      <div>{job.error ?? 'The worker reported a failure without a message.'}</div>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex flex-wrap gap-2">
                    {isActive && (
                      <button
                        type="button"
                        onClick={handleCancel}
                        disabled={cancelExperiment.isPending || job.cancel_requested}
                        className="flex-1 min-w-[130px] py-2 px-3 rounded-lg border border-red-500/40 text-red-600 dark:text-red-400 text-xs font-semibold flex items-center justify-center gap-1.5 hover:bg-red-500/5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {cancelExperiment.isPending || job.cancel_requested ? (
                          <>
                            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Cancellation requested
                          </>
                        ) : (
                          <>
                            <Ban className="w-3.5 h-3.5" /> Cancel run
                          </>
                        )}
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={clearRun}
                      className="py-2 px-3 rounded-lg border border-border text-xs font-semibold text-muted-foreground hover:bg-muted transition-colors"
                    >
                      New run
                    </button>
                  </div>

                  {cancelExperiment.isError && (
                    <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-[11px] text-red-600 dark:text-red-400">
                      Could not cancel: {errorText(cancelExperiment.error)}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
