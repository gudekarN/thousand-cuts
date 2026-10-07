import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import { FlaskConical } from 'lucide-react'

const queryClient = new QueryClient()

function HomePage() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full bg-card border rounded-xl p-8 shadow-sm text-center space-y-4">
        <div className="flex justify-center">
          <div className="p-3 bg-primary/10 rounded-full text-primary">
            <FlaskConical className="w-8 h-8" />
          </div>
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">
          Thousand Cuts
        </h1>
        <p className="text-muted-foreground text-sm">
          ML Robustness Benchmark &amp; Experimentation Platform
        </p>
        <div className="pt-4 flex justify-center gap-3">
          <Link
            to="/"
            className="inline-flex items-center justify-center px-4 py-2 text-sm font-medium rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
          >
            Overview
          </Link>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="*" element={<HomePage />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
