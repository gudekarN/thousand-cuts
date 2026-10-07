import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Routes, Route, Outlet } from 'react-router-dom'
import Sidebar from '@/components/layout/Sidebar'
import Overview from '@/pages/Overview'
import ResultsDashboard from '@/pages/ResultsDashboard'
import ModelComparison from '@/pages/ModelComparison'
import Visualizations from '@/pages/Visualizations'
import ExperimentDetails from '@/pages/ExperimentDetails'
import ExperimentSetup from '@/pages/ExperimentSetup'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
})

function AppLayout() {
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Overview />} />
            <Route path="/setup" element={<ExperimentSetup />} />
            <Route path="/results" element={<ResultsDashboard />} />
            <Route path="/results/:runId" element={<ResultsDashboard />} />
            <Route path="/visualizations" element={<Visualizations />} />
            <Route path="/compare" element={<ModelComparison />} />
            <Route path="/experiments" element={<ExperimentDetails />} />
            <Route path="/experiments/:id" element={<ExperimentDetails />} />
            <Route path="*" element={<Overview />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
