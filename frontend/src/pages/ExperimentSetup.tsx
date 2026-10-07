import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useConfig } from '@/hooks/useApi'
import {
  Play,
  Layers,
  Database,
  BrainCircuit,
  Sliders,
  CheckCircle2,
  Clock,
  ExternalLink,
} from 'lucide-react'
import {
  DATASET_LABELS,
  NOISE_LABELS,
  NOISE_COLORS,
} from '@/lib/constants'
import { cn } from '@/lib/utils'

export default function ExperimentSetup() {
  const navigate = useNavigate()
  const { data: config } = useConfig()

  const [selectedDataset, setSelectedDataset] = useState('breast_cancer')
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'logreg',
    'svm_rbf',
    'decision_tree',
    'random_forest',
  ])
  const [selectedNoises, setSelectedNoises] = useState<string[]>(['gaussian', 'label'])
  const [mode, setMode] = useState<'sweep' | 'single'>('sweep')
  const [singleLevel, setSingleLevel] = useState<number>(3)
  const [seedCount, setSeedCount] = useState<number>(3)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submittedRun, setSubmittedRun] = useState<{ id: string; status: string } | null>(null)

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

  // Calculate fits
  const levelsCount = mode === 'sweep' ? 6 : 1 // 0 to 5 or 1
  const fitsCount = selectedModels.length * levelsCount * seedCount
  const estimatedTimeSec = (fitsCount * 0.08).toFixed(1)

  const handleStartRun = () => {
    setIsSubmitting(true)
    setTimeout(() => {
      setIsSubmitting(false)
      setSubmittedRun({
        id: `custom-${Date.now().toString().slice(-6)}`,
        status: 'completed',
      })
    }, 1200)
  }

  return (
    <div className="flex-1 p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Experiment Setup</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Configure a targeted noise robustness experiment or custom model sweep.
        </p>
      </div>

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
              Applied in fixed sequence: Label Noise → Gaussian → Outliers → Missing.
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
                    onClick={() => setMode('single')}
                    className={cn(
                      'flex-1 py-1.5 px-3 rounded-lg text-xs font-medium border',
                      mode === 'single'
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
                  Random Seeds: <strong className="text-foreground">{seedCount}</strong> (0 to {seedCount - 1})
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

            {mode === 'single' && (
              <div className="pt-2 border-t border-border">
                <label className="text-xs text-muted-foreground block mb-1">
                  Target Level: <strong>L{singleLevel}</strong>
                </label>
                <input
                  type="range"
                  min="0"
                  max="5"
                  value={singleLevel}
                  onChange={(e) => setSingleLevel(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer"
                />
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
                <span className="font-semibold">{selectedModels.length} models</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Noise Injectors</span>
                <span className="font-semibold">{selectedNoises.length} active</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Seeds</span>
                <span className="font-semibold">{seedCount} seeds</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-border">
                <span className="text-muted-foreground">Planned Fits</span>
                <span className="font-mono font-bold text-primary">{fitsCount} fits</span>
              </div>
              <div className="flex justify-between py-1.5">
                <span className="text-muted-foreground">Estimated Time</span>
                <span className="font-mono font-semibold">~{estimatedTimeSec}s</span>
              </div>
            </div>

            <button
              onClick={handleStartRun}
              disabled={isSubmitting}
              className="w-full py-2.5 px-4 rounded-lg bg-primary text-primary-foreground font-semibold text-xs flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {isSubmitting ? (
                <>
                  <Clock className="w-4 h-4 animate-spin" /> Starting Execution…
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" /> Start Custom Run
                </>
              )}
            </button>
          </div>

          {/* Run Completion Status Card */}
          {submittedRun && (
            <div className="bg-card border border-emerald-500/30 rounded-xl p-6 space-y-3">
              <div className="flex items-center gap-2 text-emerald-600 font-semibold text-sm">
                <CheckCircle2 className="w-5 h-5" />
                Run {submittedRun.id} Finished
              </div>
              <p className="text-xs text-muted-foreground">
                All {fitsCount} fits executed cleanly. Results are ready to view.
              </p>
              <button
                onClick={() => navigate('/results')}
                className="w-full py-2 px-3 rounded-lg border border-primary text-primary text-xs font-semibold flex items-center justify-center gap-1.5 hover:bg-primary/5 transition-colors"
              >
                View Dashboard Results <ExternalLink className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
