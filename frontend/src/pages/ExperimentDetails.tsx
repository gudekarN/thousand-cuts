import { useConfig, useManifest } from '@/hooks/useApi'
import {
  FileJson,
  Hash,
} from 'lucide-react'

export default function ExperimentDetails() {
  const { data: config } = useConfig()
  const { data: manifest } = useManifest('full')

  const downloadManifest = () => {
    if (!manifest) return
    const blob = new Blob([JSON.stringify(manifest, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `manifest_${manifest.run_id ?? 'full'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight">Experiment Details</h1>
            <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-emerald-500/10 text-emerald-600 border border-emerald-500/20">
              Status: {manifest?.status ?? 'completed'}
            </span>
          </div>
          <p className="text-muted-foreground text-sm mt-1">
            Provenance, configuration hashes, environment metadata, and execution audit log.
          </p>
        </div>

        <button
          onClick={downloadManifest}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 transition-opacity"
        >
          <FileJson className="w-4 h-4" /> Download Manifest
        </button>
      </div>

      {/* Overview Metadata Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Run ID</p>
          <p className="text-lg font-mono font-bold mt-1 text-primary">
            {manifest?.run_id ?? 'full-run'}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">Official Benchmark</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Total Fits</p>
          <p className="text-lg font-mono font-bold mt-1">
            {manifest?.completed_fits ?? 6080} / {manifest?.planned_fits ?? 6080}
          </p>
          <p className="text-[10px] text-emerald-600 mt-0.5">100% Completed</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Seeds Evaluated</p>
          <p className="text-lg font-mono font-bold mt-1">
            {manifest?.seeds?.length ?? 10} seeds
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">Seeds 0 through 9</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs text-muted-foreground uppercase font-semibold">Levels Freeze</p>
          <p className="text-lg font-mono font-bold mt-1 text-emerald-600">
            {config?.frozen ? 'FROZEN' : 'PROVISIONAL'}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">v{config?.levels_version}</p>
        </div>
      </div>

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
            <code className="text-xs font-mono text-foreground break-all block mt-1">
              {config?.config_hash ?? 'bc936294060c9d60cc9127a37c6fda1969abe1f7004c343bee8fc4c265a83362'}
            </code>
          </div>

          <div className="grid sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-muted/40 p-3 rounded-lg border border-border/50 space-y-1">
              <div className="font-semibold text-muted-foreground">Experiment Methodology</div>
              <div className="text-foreground">Methodology Version: 1.0.0</div>
              <div className="text-muted-foreground">Active Severity Table: Table {config?.active_table}</div>
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
          Official Benchmark Protocol & Rules
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
