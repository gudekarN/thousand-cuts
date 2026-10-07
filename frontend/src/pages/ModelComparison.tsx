import { useState, useMemo } from 'react'
import { useConfig, useRobustness, useBreakingPoints, useSummary } from '@/hooks/useApi'
import LineBandChart from '@/components/charts/LineBandChart'
import GroupedBarChart from '@/components/charts/GroupedBarChart'
import {
  MODEL_LABELS,
  DATASET_LABELS,
  MODEL_COLORS,
  fmt4,
} from '@/lib/constants'
import { Trophy, Award, Medal, Loader2, Info } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function ModelComparison() {
  const { data: config } = useConfig()
  const [selectedDataset, setSelectedDataset] = useState('breast_cancer')
  const [selectedScope, setSelectedScope] = useState<'all' | 'singles'>('all')
  const [selectedCombo, setSelectedCombo] = useState('label+gaussian+outliers+missing')

  const { data: robustnessRows, isLoading: loadingRobustness } = useRobustness({
    dataset: selectedDataset,
    scope: selectedScope,
  })

  const { data: bpRows } = useBreakingPoints({
    dataset: selectedDataset,
  })

  const { data: summaryRows } = useSummary({
    dataset: selectedDataset,
    combo: selectedCombo,
  })

  // Rank 1 model
  const topModel = robustnessRows?.find((r) => r.rank === 1)

  // Overlay chart data: levels 0-5 with all 4 models
  const overlayChartData = useMemo(() => {
    if (!summaryRows) return []
    const levels = [0, 1, 2, 3, 4, 5]
    return levels.map((lvl) => {
      const point: any = { level: lvl }
      summaryRows
        .filter((r) => r.level === lvl)
        .forEach((r) => {
          point[r.model] = r.f1_mean
        })
      return point
    })
  }, [summaryRows])

  // Breaking levels grouped bar chart
  const barChartData = useMemo(() => {
    if (!bpRows) return []
    const singles = ['label', 'gaussian', 'outliers', 'missing']
    return singles.map((noise) => {
      const item: any = { category: noise }
      bpRows
        .filter((r) => r.combo === noise)
        .forEach((r) => {
          item[r.model] = r.reached && r.breaking_level !== null ? r.breaking_level : 6 // 6 = not reached
        })
      return item
    })
  }, [bpRows])

  // Aggregate stats per model for the table
  const modelStats = useMemo(() => {
    if (!robustnessRows || !bpRows) return []
    return robustnessRows.map((rob) => {
      const modelBps = bpRows.filter((b) => b.model === rob.model)
      const brokenCount = modelBps.filter((b) => b.reached).length
      const brokenLevels = modelBps.map((b) => (b.reached && b.breaking_level !== null ? b.breaking_level : 6))
      const meanBreakLevel = brokenLevels.reduce((a, b) => a + b, 0) / brokenLevels.length

      // Worst noise type (lowest breaking level)
      let worstNoise = 'None'
      let minBreak = 7
      modelBps.forEach((b) => {
        if (b.reached && b.breaking_level !== null && b.breaking_level < minBreak) {
          minBreak = b.breaking_level
          worstNoise = b.combo
        }
      })

      return {
        ...rob,
        brokenCount,
        meanBreakLevel,
        worstNoise,
      }
    })
  }, [robustnessRows, bpRows])

  if (loadingRobustness) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-sm">Loading model comparison data…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Comparison</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Objective robustness rankings evaluated via Breaking Point Index (BPI) and Robustness Score (RS).
        </p>
      </div>

      {/* Control Bar */}
      <div className="bg-card border border-border rounded-xl p-4 flex flex-wrap gap-4 items-center justify-between">
        <div className="flex gap-4 items-center">
          {/* Dataset toggle */}
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

          {/* Scope toggle */}
          <div className="space-y-1">
            <label className="text-xs font-semibold text-muted-foreground uppercase">Scope</label>
            <div className="flex gap-1.5">
              {(['all', 'singles'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setSelectedScope(s)}
                  className={cn(
                    'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                    selectedScope === s
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted/50 text-foreground hover:bg-muted'
                  )}
                >
                  {s === 'all' ? 'All 15 Combos' : '4 Singles Only'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Combo selector for overlay chart */}
        <div className="space-y-1">
          <label className="text-xs font-semibold text-muted-foreground uppercase">Overlay Noise Combo</label>
          <select
            value={selectedCombo}
            onChange={(e) => setSelectedCombo(e.target.value)}
            className="block px-3 py-1.5 rounded-lg border border-border bg-card text-foreground text-xs font-mono focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {[
              'label+gaussian+outliers+missing',
              'gaussian',
              'label',
              'outliers',
              'missing',
              'label+gaussian',
              'label+missing',
              'gaussian+missing',
            ].map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Top Highlight Card */}
      {topModel && (
        <div className="bg-gradient-to-r from-primary/10 via-card to-card border border-primary/30 rounded-xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-primary text-primary-foreground rounded-xl shadow-md">
              <Trophy className="w-8 h-8" />
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide font-semibold text-primary">
                Rank 1 Most Robust Model
              </div>
              <div className="text-2xl font-bold">{MODEL_LABELS[topModel.model]}</div>
              <p className="text-xs text-muted-foreground mt-0.5">
                BPI: <span className="font-mono font-bold text-foreground">{fmt4(topModel.BPI)}</span> • RS: <span className="font-mono font-bold text-foreground">{fmt4(topModel.RS)}</span> on {DATASET_LABELS[selectedDataset]} ({selectedScope} scope).
              </p>
            </div>
          </div>
          <div className="text-xs text-muted-foreground bg-card/80 border border-border px-3 py-2 rounded-lg max-w-xs flex gap-2">
            <Info className="w-4 h-4 text-primary shrink-0 mt-0.5" />
            <span>
              <strong>BPI</strong> = Mean breaking point (1-5, or 6 if not reached). Higher is more robust.
            </span>
          </div>
        </div>
      )}

      {/* Ranking Table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
            Model Robustness Ranking Table
          </h2>
          <span className="text-xs text-muted-foreground font-mono">
            Ties broken by RS (mean relative F1 retained)
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Rank</th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">Model</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">BPI (Mean BP)</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">RS (Mean Rel F1)</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Combos Broken</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Mean BP Level</th>
                <th className="text-right px-6 py-3 font-medium text-muted-foreground">Most Fragile Noise</th>
              </tr>
            </thead>
            <tbody>
              {modelStats.map((item) => (
                <tr
                  key={item.model}
                  className="border-b border-border last:border-0 hover:bg-muted/10 transition-colors"
                >
                  <td className="px-6 py-3 font-bold">
                    <span className="inline-flex items-center gap-1.5">
                      {item.rank === 1 && <Trophy className="w-4 h-4 text-amber-500" />}
                      {item.rank === 2 && <Medal className="w-4 h-4 text-slate-400" />}
                      {item.rank === 3 && <Award className="w-4 h-4 text-amber-700" />}
                      #{item.rank}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-medium flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: MODEL_COLORS[item.model] }}
                    />
                    {MODEL_LABELS[item.model]}
                  </td>
                  <td className="text-right px-4 py-3 font-mono font-bold text-primary">
                    {fmt4(item.BPI)}
                  </td>
                  <td className="text-right px-4 py-3 font-mono font-medium">
                    {fmt4(item.RS)}
                  </td>
                  <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                    {item.brokenCount} / 15
                  </td>
                  <td className="text-right px-4 py-3 font-mono text-muted-foreground">
                    {item.meanBreakLevel.toFixed(2)}
                  </td>
                  <td className="text-right px-6 py-3 font-mono text-xs text-amber-600 dark:text-amber-400">
                    {item.worstNoise}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Grid with 2 Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Overlay F1 Chart */}
        <div className="bg-card border border-border rounded-xl p-6 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
              F1 Degradation Curves (All 4 Models)
            </h3>
            <span className="text-xs font-mono text-muted-foreground truncate max-w-[200px]">
              {selectedCombo}
            </span>
          </div>
          <LineBandChart
            data={overlayChartData}
            lines={[
              { key: 'logreg', name: 'Logistic Regression', color: MODEL_COLORS.logreg },
              { key: 'svm_rbf', name: 'SVM (RBF)', color: MODEL_COLORS.svm_rbf, strokeDasharray: '4 4' },
              { key: 'decision_tree', name: 'Decision Tree', color: MODEL_COLORS.decision_tree, strokeDasharray: '2 2' },
              { key: 'random_forest', name: 'Random Forest', color: MODEL_COLORS.random_forest, strokeDasharray: '6 3' },
            ]}
            yDomain={[0.4, 1.0]}
          />
        </div>

        {/* Breaking Levels Bar Chart */}
        <div className="bg-card border border-border rounded-xl p-6 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wide">
              Breaking Level by Single Noise Type
            </h3>
            <span className="text-xs text-muted-foreground">6 = Not Reached</span>
          </div>
          <GroupedBarChart
            data={barChartData}
            bars={[
              { key: 'logreg', name: 'Logistic Regression', color: MODEL_COLORS.logreg },
              { key: 'svm_rbf', name: 'SVM (RBF)', color: MODEL_COLORS.svm_rbf },
              { key: 'decision_tree', name: 'Decision Tree', color: MODEL_COLORS.decision_tree },
              { key: 'random_forest', name: 'Random Forest', color: MODEL_COLORS.random_forest },
            ]}
            yLabel="Breaking Level (0-6)"
            yDomain={[0, 6.5]}
          />
        </div>
      </div>
    </div>
  )
}
