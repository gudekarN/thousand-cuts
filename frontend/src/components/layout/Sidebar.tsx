import { Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  FlaskConical,
  BarChart3,
  LineChart,
  Scale,
  ClipboardList,
  CheckCircle2,
  XCircle,
  Menu,
  X,
} from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'
import { useHealth } from '@/hooks/useApi'

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/setup', label: 'Experiment Setup', icon: FlaskConical },
  { to: '/results', label: 'Results', icon: BarChart3 },
  { to: '/visualizations', label: 'Visualizations', icon: LineChart },
  { to: '/compare', label: 'Model Comparison', icon: Scale },
  { to: '/experiments', label: 'Experiment Details', icon: ClipboardList },
]

function BackendBadge() {
  const { data, isError } = useHealth()
  const ok = data?.status === 'ok' && !isError
  return (
    <div
      className={cn(
        'flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded-full w-fit',
        ok
          ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
          : 'bg-red-500/10 text-red-600 dark:text-red-400'
      )}
      title={ok ? 'Backend connected' : 'Backend unreachable'}
    >
      {ok ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
      {ok ? 'API Online' : 'API Offline'}
    </div>
  )
}


function SidebarContent({ onClose }: { onClose?: () => void }) {
  const { pathname } = useLocation()
  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center gap-2 px-5 py-5 border-b border-border">
        <div className="p-1.5 bg-primary/10 rounded-lg text-primary">
          <FlaskConical className="w-5 h-5" />
        </div>
        <div>
          <div className="text-sm font-bold leading-none">Thousand Cuts</div>
          <div className="text-[10px] text-muted-foreground mt-0.5">ML Robustness Benchmark</div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="ml-auto p-1 rounded-md hover:bg-accent text-muted-foreground"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
          const active = pathname === to || (to !== '/' && pathname.startsWith(to))
          return (
            <Link
              key={to}
              to={to}
              onClick={onClose}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                active
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-border">
        <BackendBadge />
      </div>
    </div>
  )
}

export default function Sidebar() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-border bg-card h-screen sticky top-0">
        <SidebarContent />
      </aside>

      {/* Mobile hamburger */}
      <button
        onClick={() => setMobileOpen(true)}
        className="md:hidden fixed top-4 left-4 z-50 p-2 rounded-lg bg-card border border-border shadow-sm"
      >
        <Menu className="w-5 h-5" />
      </button>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="relative w-64 bg-card h-full shadow-xl">
            <SidebarContent onClose={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}
    </>
  )
}
