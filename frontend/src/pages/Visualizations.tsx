import { useState, useMemo } from 'react'
import { useConfig, useSummary, useBreakingPoints, useSynergy } from '@/hooks/useApi'
import LineBandChart from '@/components/charts/LineBandChart'
import HeatmapGrid from '@/components/charts/HeatmapGrid'
import GroupedBarChart from '@/components/charts/GroupedBarChart'
import {
  MODEL_LABELS,
  DATASET_LABELS,
  MODEL_COLORS,
  fmt4,
} from '@/lib/constants'
import { Loader2, AlertTriangle, Layers, LineChart, BarChart2, Grid, Table } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function Visualizations() {
  const { data: config } = useConfig()
  const [selectedDataset, setSelectedDataset] = useState('breast_cancer')
  const [selectedCombo, setSelectedCombo] = useState('gaussian')
  const [activeTab, setActiveTab] = useState('f1')
  const [heatmapMode, setHeatmapMode] = useState<'breaking' | 'f1_l5'>('breaking')

  const { data: summaryRows, isLoading: loadingSummary } = useSummary({
    dataset: selectedDataset,
    combo: selectedCombo,
  })

  const { data: allSummary } = useSummary({
    dataset: selectedDataset,
  })

  const { data: bpRows } = useBreakingPoints({
    dataset: selectedDataset,
  })

  const { data: synergyRows } = useSynergy({
    dataset: selectedDataset,
  })

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

  const models = ['logreg', 'svm_rbf', 'decision_tree', 'random_forest']

  // F1 chart data
  const f1ChartData = useMemo(() => {
    if (!summaryRows) return []
    const levels = [0, 1, 2, 3, 4, 5]
    return levels.map((lvl) => {
      const pt: any = { level: lvl }
      summaryRows.filter((r) => r.level === lvl).forEach((r) => {
        pt[r.model] = r.f1_mean
      })
      return pt
    })
  }, [summaryRows])

  // Accuracy chart data
  const accChartData = useMemo(() => {
    if (!summaryRows) return []
    const levels = [0, 1, 2, 3, 4, 5]
    return levels.map((lvl) => {
      const pt: any = { level: lvl }
      summaryRows.filter((r) => r.level === lvl).forEach((r) => {
        pt[r.model] = r.acc_mean
      })
      return pt
    })
  }, [summaryRows])

  // Heatmap cells
  const heatmapCells = useMemo(() => {
    const map: Record<string, any> = {}
    if (heatmapMode === 'breaking') {
      if (!bpRows) return map
      bpRows.forEach((bp) => {
        const key = `${bp.combo}__${bp.model}`
        const lvl = bp.reached && bp.breaking_level !== null ? bp.breaking_level : 6
        const isNotReached = !bp.reached
        map[key] = {
          combo: bp.combo,
          model: bp.model,
          value: lvl,
          display: isNotReached ? 'L6 (safe)' : `L${lvl}`,
          colorIntensity: lvl / 6, // 1 is safest, lower is worse
        }
      })
    } else {
      if (!allSummary) return map
      allSummary.filter((r) => r.level === 5).forEach((r) => {
        const key = `${r.combo}__${r.model}`
        map[key] = {
          combo: r.combo,
          model: r.model,
          value: r.rel_f1,
          display: `${(r.rel_f1 * 100).toFixed(0)}%`,
          colorIntensity: Math.max(0, Math.min(1, r.rel_f1)),
        }
      })
    }
    return map
  }, [bpRows, allSummary, heatmapMode])

  // Breaking levels bar chart data
  const bpBarData = useMemo(() => {
    if (!bpRows) return []
    return availableCombos.slice(0, 8).map((combo) => {
      const pt: any = { category: combo }
      bpRows.filter((r) => r.combo === combo).forEach((r) => {
        pt[r.model] = r.reached && r.breaking_level !== null ? r.breaking_level : 6
      })
      return pt
    })
  }, [bpRows, availableCombos])

  if (loadingSummary) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-sm">Loading visualizations…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Visualizations</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Interactive charts exploring noise sensitivity, breaking thresholds, heatmaps, and synergy.
        </p>
      </div>

      {/* Shared Filters */}
      <div className="bg-card border border-border rounded-xl p-4 flex flex-wrap gap-4 items-center justify-between">
        <div className="flex gap-4 items-center">
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

          {/* Noise Combo (for line charts) */}
          {(activeTab === 'f1' || activeTab === 'accuracy') && (
            <div className="space-y-1">
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
          )}
        </div>

        {/* Heatmap Mode Toggle */}
        {activeTab === 'heatmap' && (
          <div className="space-y-1">
            <label className="text-xs font-semibold text-muted-foreground uppercase">Heatmap Metric</label>
            <div className="flex gap-1.5">
              <button
                onClick={() => setHeatmapMode('breaking')}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                  heatmapMode === 'breaking'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted/50 text-foreground hover:bg-muted'
                )}
              >
                Breaking Level (L0-L6)
              </button>
              <button
                onClick={() => setHeatmapMode('f1_l5')}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                  heatmapMode === 'f1_l5'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted/50 text-foreground hover:bg-muted'
                )}
              >
                % F1 Retained at L5
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Navigation Tabs */}
      <div className="flex gap-2 border-b border-border pb-2 overflow-x-auto">
        {[
          { id: 'f1', label: '1. F1 vs Level', icon: LineChart },
          { id: 'accuracy', label: '2. Accuracy vs Level', icon: LineChart },
          { id: 'heatmap', label: '3. Matrix Heatmap', icon: Grid },
          { id: 'comparison', label: '4. Breaking Level Bars', icon: BarChart2 },
          { id: 'breaking', label: '5. Breaking Points Table', icon: Table },
          { id: 'synergy', label: '6. Synergy Analysis', icon: Layers },
        ].map((tab) => {
          const Icon = tab.icon
          const active = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-colors',
                active
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <div className="space-y-4">
        {/* Tab 1: F1 vs Level */}
        {activeTab === 'f1' && (
          <div className="bg-card border border-border rounded-xl p-6 space-y-3">
            <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
              Macro F1 vs Noise Severity Level ({selectedCombo})
            </h3>
            <LineBandChart
              data={f1ChartData}
              lines={models.map((m) => ({
                key: m,
                name: MODEL_LABELS[m],
                color: MODEL_COLORS[m],
              }))}
              yDomain={[0.4, 1.0]}
              yLabel="Macro F1"
            />
          </div>
        )}

        {/* Tab 2: Accuracy vs Level */}
        {activeTab === 'accuracy' && (
          <div className="bg-card border border-border rounded-xl p-6 space-y-3">
            <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
              Accuracy vs Noise Severity Level ({selectedCombo})
            </h3>
            <LineBandChart
              data={accChartData}
              lines={models.map((m) => ({
                key: m,
                name: MODEL_LABELS[m],
                color: MODEL_COLORS[m],
              }))}
              yDomain={[0.4, 1.0]}
              yLabel="Accuracy"
            />
          </div>
        )}

        {/* Tab 3: Heatmap */}
        {activeTab === 'heatmap' && (
          <HeatmapGrid
            models={models}
            combos={availableCombos}
            cells={heatmapCells}
            title={`Robustness Matrix (${DATASET_LABELS[selectedDataset]})`}
            subtitle={
              heatmapMode === 'breaking'
                ? 'Shows breaking level for each model across all 15 noise combinations. L6 (safe) indicates model never broke through level 5.'
                : 'Shows percentage of baseline Macro F1 retained when pushed to maximum noise (Level 5).'
            }
          />
        )}

        {/* Tab 4: Grouped Bars */}
        {activeTab === 'comparison' && (
          <div className="bg-card border border-border rounded-xl p-6 space-y-3">
            <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
              Breaking Level by Noise Combo (6 = Not Reached)
            </h3>
            <GroupedBarChart
              data={bpBarData}
              bars={models.map((m) => ({
                key: m,
                name: MODEL_LABELS[m],
                color: MODEL_COLORS[m],
              }))}
              yLabel="Breaking Level (0-6)"
              yDomain={[0, 6.5]}
            />
          </div>
        )}

        {/* Tab 5: Breaking Points Table */}
        {activeTab === 'breaking' && (
          <div className="bg-card border border-border rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-border">
              <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                All Breaking Points — {DATASET_LABELS[selectedDataset]}
              </h3>
            </div>
            <div className="overflow-x-auto max-h-[500px]">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-muted">
                  <tr className="border-b border-border">
                    <th className="text-left px-6 py-3 font-medium text-muted-foreground">Combo</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Model</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Baseline F1</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Threshold (90%)</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Breaking Level</th>
                    <th className="text-right px-6 py-3 font-medium text-muted-foreground">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {bpRows?.map((bp, i) => (
                    <tr
                      key={`${bp.combo}-${bp.model}`}
                      className={cn('border-b border-border last:border-0', i % 2 === 0 ? '' : 'bg-muted/10')}
                    >
                      <td className="px-6 py-2.5 font-mono text-xs">{bp.combo}</td>
                      <td className="px-4 py-2.5 font-medium">{MODEL_LABELS[bp.model] ?? bp.model}</td>
                      <td className="text-right px-4 py-2.5 font-mono">{fmt4(bp.baseline_f1)}</td>
                      <td className="text-right px-4 py-2.5 font-mono text-muted-foreground">{fmt4(bp.threshold_f1)}</td>
                      <td className="text-right px-4 py-2.5 font-mono font-bold">
                        {bp.reached ? `L${bp.breaking_level}` : 'Not Reached'}
                      </td>
                      <td className="text-right px-6 py-2.5">
                        <span
                          className={cn(
                            'text-xs px-2 py-0.5 rounded font-semibold',
                            bp.reached
                              ? 'bg-red-500/10 text-red-500'
                              : 'bg-emerald-500/10 text-emerald-600'
                          )}
                        >
                          {bp.reached ? 'Broken' : 'Robust'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 6: Synergy (Secondary Analysis) */}
        {activeTab === 'synergy' && (
          <div className="bg-card border border-border rounded-xl p-6 space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-xs px-2 py-0.5 rounded font-semibold bg-amber-500/10 text-amber-600 border border-amber-500/20">
                Secondary Analysis
              </span>
              <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
                Compound Noise Synergy
              </h3>
            </div>
            <p className="text-xs text-muted-foreground">
              Synergy measures the difference between actual compound drop and linear additive drop.
              Positive values indicate worse-than-additive degradation (super-additive harm).
            </p>
            <div className="overflow-x-auto max-h-[500px]">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-muted">
                  <tr className="border-b border-border">
                    <th className="text-left px-6 py-3 font-medium text-muted-foreground">Combo</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Model</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Actual Drop</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Additive Drop</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Synergy Δ</th>
                    <th className="text-right px-6 py-3 font-medium text-muted-foreground">Flag</th>
                  </tr>
                </thead>
                <tbody>
                  {synergyRows?.slice(0, 30).map((syn, i) => (
                    <tr
                      key={`${syn.combo}-${syn.model}`}
                      className={cn('border-b border-border last:border-0', i % 2 === 0 ? '' : 'bg-muted/10')}
                    >
                      <td className="px-6 py-2.5 font-mono text-xs">{syn.combo}</td>
                      <td className="px-4 py-2.5 font-medium">{MODEL_LABELS[syn.model] ?? syn.model}</td>
                      <td className="text-right px-4 py-2.5 font-mono">{fmt4(syn.combo_f1_mean)}</td>
                      <td className="text-right px-4 py-2.5 font-mono text-muted-foreground">{fmt4(syn.additive_f1)}</td>
                      <td className="text-right px-4 py-2.5 font-mono font-bold">
                        <span className={syn.synergy > 0 ? 'text-red-500' : 'text-emerald-500'}>
                          {syn.synergy > 0 ? `+${fmt4(syn.synergy)}` : fmt4(syn.synergy)}
                        </span>
                      </td>
                      <td className="text-right px-6 py-2.5">
                        {syn.floor_effect && (
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
          </div>
        )}
      </div>
    </div>
  )
}
