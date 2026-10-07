import { useState, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { useConfig, useSummary, useBreakingPoints } from '@/hooks/useApi'
import LineBandChart from '@/components/charts/LineBandChart'
import {
  MODEL_LABELS,
  DATASET_LABELS,
  MODEL_COLORS,
  fmt4,
  fmtPct,
} from '@/lib/constants'
import { Loader2, CheckCircle2, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function ResultsDashboard() {
  const { runId } = useParams()
  const { data: config } = useConfig()

  const [selectedDataset, setSelectedDataset] = useState('breast_cancer')
  const [selectedModel, setSelectedModel] = useState('logreg')
  const [selectedCombo, setSelectedCombo] = useState('gaussian')

  const { data: summaryRows, isLoading: loadingSummary } = useSummary({
    dataset: selectedDataset,
    model: selectedModel,
    combo: selectedCombo,
  })

  const { data: bpRows } = useBreakingPoints({
    dataset: selectedDataset,
    model: selectedModel,
    combo: selectedCombo,
  })

  const bp = bpRows?.[0]
  const cleanBaseline = summaryRows?.find((r) => r.level === 0)
  const maxLevelRow = summaryRows?.reduce((prev, curr) => (curr.level > prev.level ? curr : prev), summaryRows[0])

  // Chart data
  const chartData = useMemo(() => {
    if (!summaryRows) return []
    return summaryRows.map((r) => ({
      level: r.level,
      [r.model]: r.f1_mean,
      accuracy: r.acc_mean,
    }))
  }, [summaryRows])

  // Unique combos
  const availableCombos = useMemo(() => {
    return [
      'gaussian',
      'label',
      'outliers',
      'missing',
      'label+gaussian',
      'label+outliers',
      'label+missing',
      'gaussian+outliers',
      'gaussian+missing',
      'outliers+missing',
      'label+gaussian+outliers',
      'label+gaussian+missing',
      'label+outliers+missing',
      'gaussian+outliers+missing',
      'label+gaussian+outliers+missing',
    ]
  }, [])

  if (loadingSummary) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-sm">Loading experiment results…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight">Results Dashboard</h1>
            <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-primary/10 text-primary border border-primary/20">
              {runId ? `Run: ${runId}` : 'Official (Full)'}
            </span>
          </div>
          <p className="text-muted-foreground text-sm mt-1">
            Explore performance degradation and breaking points across noise levels.
          </p>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="bg-card border border-border rounded-xl p-4 flex flex-wrap gap-4 items-center">
        {/* Dataset */}
        <div className="space-y-1">
          <label className="text-xs font-semibold text-muted-foreground uppercase">Dataset</label>
          <div className="flex gap-1.5">
            {config?.datasets.map((d) => (
              <button
                key={d}
                onClick={() => setSelectedDataset(d)}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                  selectedDataset === d
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted/50 text-foreground hover:bg-muted'
                )}
              >
                {DATASET_LABELS[d] ?? d}
              </button>
            ))}
          </div>
        </div>

        {/* Model */}
        <div className="space-y-1">
          <label className="text-xs font-semibold text-muted-foreground uppercase">Model</label>
          <div className="flex gap-1.5">
            {config?.models.map((m) => (
              <button
                key={m.id}
                onClick={() => setSelectedModel(m.id)}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                  selectedModel === m.id
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted/50 text-foreground hover:bg-muted'
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Noise Combo */}
        <div className="space-y-1 ml-auto">
          <label className="text-xs font-semibold text-muted-foreground uppercase">Noise Combo</label>
          <select
            value={selectedCombo}
            onChange={(e) => setSelectedCombo(e.target.value)}
            className="block px-3 py-1.5 rounded-lg border border-border bg-card text-foreground text-xs font-mono focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {availableCombos.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-[11px] font-semibold uppercase text-muted-foreground">Baseline F1</p>
          <p className="text-xl font-mono font-bold mt-1 text-primary">
            {fmt4(cleanBaseline?.f1_mean)}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">±{fmt4(cleanBaseline?.f1_std)}</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-[11px] font-semibold uppercase text-muted-foreground">Final F1 (L5)</p>
          <p className="text-xl font-mono font-bold mt-1">
            {fmt4(maxLevelRow?.f1_mean)}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">±{fmt4(maxLevelRow?.f1_std)}</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-[11px] font-semibold uppercase text-muted-foreground">Abs Drop</p>
          <p className="text-xl font-mono font-bold mt-1 text-amber-500">
            {fmt4(maxLevelRow?.drop_abs)}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">baseline - L5</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-[11px] font-semibold uppercase text-muted-foreground">Rel Drop</p>
          <p className="text-xl font-mono font-bold mt-1 text-amber-500">
            {fmtPct(maxLevelRow?.drop_rel)}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">% drop</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-[11px] font-semibold uppercase text-muted-foreground">Breaking Point</p>
          <div className="mt-1 flex items-center gap-1.5">
            {bp?.reached ? (
              <span className="text-xl font-mono font-bold text-red-500">
                L{bp.breaking_level}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 font-semibold">
                <CheckCircle2 className="w-3 h-3" /> Not Reached
              </span>
            )}
          </div>
          <p className="text-[10px] text-muted-foreground mt-0.5">
            {bp?.reached ? `below 90% at L${bp.breaking_level}` : 'Robust through L5'}
          </p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-[11px] font-semibold uppercase text-muted-foreground">Avg Fit Time</p>
          <p className="text-xl font-mono font-bold mt-1">
            {maxLevelRow ? `${(maxLevelRow.fit_time_mean * 1000).toFixed(1)} ms` : '—'}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">per seed</p>
        </div>
      </div>

      {/* Main Chart */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
            Performance vs Noise Severity
          </h2>
          <span className="text-xs text-muted-foreground">
            {MODEL_LABELS[selectedModel]} • {selectedCombo}
          </span>
        </div>
        <LineBandChart
          data={chartData}
          lines={[
            {
              key: selectedModel,
              name: `Macro F1 (${MODEL_LABELS[selectedModel]})`,
              color: MODEL_COLORS[selectedModel],
            },
          ]}
          threshold={bp?.threshold_f1}
          yDomain={[0.4, 1.0]}
          yLabel="Macro F1"
        />
      </div>

      {/* Level Breakdown Table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-border">
          <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
            Level-by-Level Breakdown
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Level</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Macro F1</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Accuracy</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Rel F1</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Abs Drop</th>
                <th className="text-right px-6 py-3 font-medium text-muted-foreground">Status</th>
              </tr>
            </thead>
            <tbody>
              {summaryRows?.map((row, i) => {
                const isBroken = bp?.reached && bp.breaking_level !== null && row.level >= bp.breaking_level
                return (
                  <tr
                    key={row.level}
                    className={cn(
                      'border-b border-border last:border-0',
                      i % 2 === 0 ? '' : 'bg-muted/10'
                    )}
                  >
                    <td className="px-6 py-3 font-mono font-bold">L{row.level}</td>
                    <td className="text-right px-4 py-3 font-mono font-medium">
                      {fmt4(row.f1_mean)} <span className="text-xs text-muted-foreground">±{fmt4(row.f1_std)}</span>
                    </td>
                    <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                      {fmt4(row.acc_mean)}
                    </td>
                    <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                      {(row.rel_f1 * 100).toFixed(1)}%
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
    </div>
  )
}
