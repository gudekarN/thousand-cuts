import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  CartesianGrid,
} from 'recharts'
import { MODEL_COLORS } from '@/lib/constants'

interface DataPoint {
  level: number
  [key: string]: number | string | null
}

interface LineBandChartProps {
  data: DataPoint[]
  lines: Array<{
    key: string
    name: string
    color?: string
    strokeDasharray?: string
  }>
  threshold?: number
  yDomain?: [number, number]
  yLabel?: string
  xLabel?: string
}

export default function LineBandChart({
  data,
  lines,
  threshold,
  yDomain = [0.5, 1.0],
  yLabel = 'Macro F1',
  xLabel = 'Noise Level',
}: LineBandChartProps) {
  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis
            dataKey="level"
            tickFormatter={(lvl) => `L${lvl}`}
            label={{ value: xLabel, position: 'insideBottom', offset: -10 }}
          />
          <YAxis
            domain={yDomain}
            tickFormatter={(val) => Number(val).toFixed(2)}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 10 }}
          />
          <Tooltip
            formatter={(value: any) => [
              typeof value === 'number' ? value.toFixed(4) : value,
              '',
            ]}
            labelFormatter={(label) => `Level ${label}`}
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              borderColor: 'hsl(var(--border))',
              borderRadius: '0.5rem',
              color: 'hsl(var(--foreground))',
            }}
          />
          <Legend verticalAlign="top" height={36} />
          {threshold != null && (
            <ReferenceLine
              y={threshold}
              stroke="#EF4444"
              strokeDasharray="4 4"
              label={{
                value: `Break Threshold (${threshold.toFixed(3)})`,
                fill: '#EF4444',
                fontSize: 11,
                position: 'top',
              }}
            />
          )}
          {lines.map((l) => (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              name={l.name}
              stroke={l.color ?? MODEL_COLORS[l.key] ?? '#6366F1'}
              strokeWidth={2}
              strokeDasharray={l.strokeDasharray}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
