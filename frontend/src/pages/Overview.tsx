import { useConfig } from '@/hooks/useApi'
import {
  Database,
  BrainCircuit,
  Waves,
  FlaskConical,
  Layers,
  Trophy,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { MODEL_COLORS, NOISE_COLORS, NOISE_LABELS, STAGE_LABELS } from '@/lib/constants'

function StatCard({
  label,
  value,
  icon: Icon,
  className,
}: {
  label: string
  value: string | number
  icon: React.ElementType
  className?: string
}) {
  return (
    <div className={cn('bg-card border border-border rounded-xl p-5', className)}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">{label}</p>
          <p className="text-3xl font-bold mt-1 font-mono">{value}</p>
        </div>
        <div className="p-2 bg-primary/10 rounded-lg text-primary">
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  )
}


function StageBadge({ stage }: { stage: string }) {
  const isLatest = stage === 'full'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold',
        isLatest
          ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
          : 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20'
      )}
    >
      {isLatest && <CheckCircle2 className="w-3 h-3" />}
      {STAGE_LABELS[stage] ?? stage}
    </span>
  )
}

const NOISE_APPLICATION_ORDER = ['label', 'gaussian', 'outliers', 'missing']

export default function Overview() {
  const { data: config, isLoading, isError, error } = useConfig()

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-sm">Loading experiment data…</p>
        </div>
      </div>
    )
  }

  if (isError || !config) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="bg-card border border-destructive/30 rounded-xl p-8 max-w-md text-center space-y-3">
          <div className="flex justify-center">
            <div className="p-3 bg-destructive/10 rounded-full text-destructive">
              <AlertCircle className="w-7 h-7" />
            </div>
          </div>
          <h2 className="text-lg font-semibold">Backend Unavailable</h2>
          <p className="text-sm text-muted-foreground">
            {error?.message ?? 'Could not connect to the API. Make sure the backend is running on port 8000.'}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="mt-2 px-4 py-2 text-sm font-medium rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  const latestStage = config.available_official_stages.at(-1)
  const totalCombos = 15 // 4 singles + 6 pairs + 4 triples + 1 four-way
  const totalFits = 6080 // official full run

  return (
    <div className="flex-1 p-6 md:p-8 space-y-8 max-w-6xl mx-auto w-full">

      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold tracking-tight">Overview</h1>
          {latestStage && <StageBadge stage={latestStage} />}
          {config.frozen && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
              <CheckCircle2 className="w-3 h-3" />
              Levels Frozen v{config.levels_version}
            </span>
          )}
        </div>
        <p className="text-muted-foreground max-w-2xl">
          <strong>Death by a Thousand Cuts</strong> — a systematic study of how small, realistic noise sources
          stack up until ML models break. Each noise type is individually harmless, but in combination they
          push models past their breaking point.
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        <StatCard label="Datasets" value={config.datasets.length} icon={Database} />
        <StatCard label="Models" value={config.models.length} icon={BrainCircuit} />
        <StatCard label="Noise Types" value={config.noise_types.length} icon={Waves} />
        <StatCard label="Noise Combos" value={totalCombos} icon={Layers} />
        <StatCard label="Severity Levels" value={config.levels.length - 1} icon={FlaskConical} />
        <StatCard label="Seeds (Full)" value={10} icon={Trophy} />
        <StatCard label="Total Fits" value={totalFits.toLocaleString()} icon={FlaskConical} />
        <StatCard
          label="Stages Complete"
          value={`${config.available_official_stages.length} / 3`}
          icon={CheckCircle2}
        />
      </div>

      {/* Two-column info section */}
      <div className="grid md:grid-cols-2 gap-6">

        {/* Models */}
        <div className="bg-card border border-border rounded-xl p-6 space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Models</h2>
          <div className="space-y-2">
            {config.models.map((m) => (
              <div key={m.id} className="flex items-center gap-3">
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: MODEL_COLORS[m.id] ?? '#94a3b8' }}
                />
                <span className="text-sm font-medium">{m.label}</span>
                <code className="ml-auto text-xs text-muted-foreground font-mono">{m.id}</code>
              </div>
            ))}
          </div>
        </div>

        {/* Noise types */}
        <div className="bg-card border border-border rounded-xl p-6 space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
            Noise Types — Application Order
          </h2>
          <div className="space-y-2">
            {NOISE_APPLICATION_ORDER.map((id, idx) => (
              <div key={id} className="flex items-center gap-3">
                <span className="w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs font-bold text-muted-foreground shrink-0">
                  {idx + 1}
                </span>
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: NOISE_COLORS[id] ?? '#94a3b8' }}
                />
                <span className="text-sm font-medium">{NOISE_LABELS[id] ?? id}</span>
                <code className="ml-auto text-xs text-muted-foreground font-mono">{id}</code>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Level table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-border">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
            Active Severity Levels — Table {config.active_table}
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Level</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Label Flip Rate</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Gaussian σ×k</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Outlier Cell Rate</th>
                <th className="text-right px-6 py-3 font-medium text-muted-foreground">Missing Cell Rate</th>
              </tr>
            </thead>
            <tbody>
              {config.levels.map((row, i) => (
                <tr
                  key={row.level}
                  className={cn('border-b border-border last:border-0', i % 2 === 0 ? '' : 'bg-muted/10')}
                >
                  <td className="px-6 py-3 font-mono font-bold">
                    L{row.level}
                    {row.level === 0 && (
                      <span className="ml-2 text-xs text-muted-foreground font-sans font-normal">clean</span>
                    )}
                  </td>
                  <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                    {(row.label_flip_rate * 100).toFixed(0)}%
                  </td>
                  <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                    {row.gaussian_k}×
                  </td>
                  <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                    {(row.outlier_cell_rate * 100).toFixed(0)}%
                  </td>
                  <td className="text-right px-6 py-3 font-mono text-muted-foreground">
                    {(row.missing_cell_rate * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Official stages */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Official Stages Available on Disk
        </h2>
        <div className="flex flex-wrap gap-3">
          {(['mvp', 'stage2', 'full'] as const).map((stage) => {
            const present = config.available_official_stages.includes(stage)
            return (
              <div
                key={stage}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium',
                  present
                    ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-700 dark:text-emerald-400'
                    : 'bg-muted/30 border-border text-muted-foreground'
                )}
              >
                {present ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-muted-foreground/30" />
                )}
                {STAGE_LABELS[stage]}
              </div>
            )
          })}
        </div>
      </div>

      {/* How it works */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">How It Works</h2>
        <div className="grid sm:grid-cols-5 gap-3">
          {[
            { step: '1', label: 'Clean Data', desc: 'Stratified 70/30 split. Test set never touched.' },
            { step: '2', label: 'Inject Noise', desc: 'One or more noise types on training data only.' },
            { step: '3', label: 'Train', desc: 'Imputer → Scaler → Model pipeline on noisy train.' },
            { step: '4', label: 'Evaluate', desc: 'Macro F1 + Accuracy on the clean test set.' },
            { step: '5', label: 'Find Breaking Point', desc: 'First level where mean F1 ≤ 0.9 × baseline.' },
          ].map(({ step, label, desc }) => (
            <div key={step} className="space-y-1.5">
              <div className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">
                {step}
              </div>
              <div className="text-sm font-semibold">{label}</div>
              <div className="text-xs text-muted-foreground leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
