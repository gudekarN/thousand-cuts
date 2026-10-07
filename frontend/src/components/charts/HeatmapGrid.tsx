import { MODEL_LABELS } from '@/lib/constants'
import { cn } from '@/lib/utils'

interface HeatmapCell {
  combo: string
  model: string
  value: number | null
  display: string
  colorIntensity: number // 0 to 1
}

interface HeatmapGridProps {
  models: string[]
  combos: string[]
  cells: Record<string, HeatmapCell> // key: `${combo}__${model}`
  title?: string
  subtitle?: string
}

export default function HeatmapGrid({
  models,
  combos,
  cells,
  title,
  subtitle,
}: HeatmapGridProps) {
  return (
    <div className="bg-card border border-border rounded-xl p-6 space-y-4 overflow-hidden">
      <div>
        {title && <h3 className="text-base font-semibold">{title}</h3>}
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[600px]">
          {/* Header row */}
          <div className="grid grid-cols-[160px_repeat(4,1fr)] gap-2 pb-2 border-b border-border text-xs font-semibold text-muted-foreground">
            <div>Noise Combo</div>
            {models.map((m) => (
              <div key={m} className="text-center">
                {MODEL_LABELS[m] ?? m}
              </div>
            ))}
          </div>

          {/* Combo rows */}
          <div className="space-y-1.5 pt-2">
            {combos.map((combo) => (
              <div
                key={combo}
                className="grid grid-cols-[160px_repeat(4,1fr)] gap-2 items-center text-xs"
              >
                <div className="font-mono text-xs truncate" title={combo}>
                  {combo}
                </div>
                {models.map((model) => {
                  const key = `${combo}__${model}`
                  const cell = cells[key]
                  const intensity = cell?.colorIntensity ?? 0
                  // Color scale: from emerald (robust, high intensity) to rose (broken, low intensity)
                  const bgColor = cell?.value === null || cell?.value === undefined
                    ? 'rgba(148, 163, 184, 0.1)'
                    : intensity > 0.8
                    ? 'rgba(16, 185, 129, 0.2)'
                    : intensity > 0.5
                    ? 'rgba(59, 130, 246, 0.2)'
                    : intensity > 0.3
                    ? 'rgba(245, 158, 11, 0.25)'
                    : 'rgba(239, 68, 68, 0.25)'

                  return (
                    <div
                      key={model}
                      className={cn(
                        'py-2 px-3 rounded text-center font-mono font-medium transition-transform hover:scale-105 border border-border/50 cursor-default'
                      )}
                      style={{ backgroundColor: bgColor }}
                      title={`${MODEL_LABELS[model] ?? model} + ${combo}: ${cell?.display ?? 'N/A'}`}
                    >
                      {cell?.display ?? '—'}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
