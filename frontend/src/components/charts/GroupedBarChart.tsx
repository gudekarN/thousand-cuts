import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts'
import { MODEL_COLORS } from '@/lib/constants'

interface BarItem {
  category: string
  [key: string]: number | string | null
}

interface GroupedBarChartProps {
  data: BarItem[]
  bars: Array<{
    key: string
    name: string
    color?: string
  }>
  yLabel?: string
  yDomain?: [number, number]
}

export default function GroupedBarChart({
  data,
  bars,
  yLabel,
  yDomain,
}: GroupedBarChartProps) {
  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis dataKey="category" />
          <YAxis
            domain={yDomain}
            label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', offset: 10 } : undefined}
          />
          <Tooltip
            formatter={(value: any) => [
              typeof value === 'number' ? value.toFixed(3) : value,
              '',
            ]}
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              borderColor: 'hsl(var(--border))',
              borderRadius: '0.5rem',
              color: 'hsl(var(--foreground))',
            }}
          />
          <Legend verticalAlign="top" height={36} />
          {bars.map((b) => (
            <Bar
              key={b.key}
              dataKey={b.key}
              name={b.name}
              fill={b.color ?? MODEL_COLORS[b.key] ?? '#6366F1'}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
