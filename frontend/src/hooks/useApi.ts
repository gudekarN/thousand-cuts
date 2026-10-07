import { useQuery } from '@tanstack/react-query'
import { api, type ApiConfig, type SummaryRow, type BreakingPointRow, type RobustnessRow, type SynergyRow } from '@/api/client'

export function useConfig() {
  return useQuery<ApiConfig, Error>({
    queryKey: ['config'],
    queryFn: api.config,
    staleTime: 60_000,
    retry: 2,
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: 1,
  })
}

export function useSummary(params?: { stage?: string; dataset?: string; model?: string; combo?: string }) {
  return useQuery<SummaryRow[], Error>({
    queryKey: ['summary', params],
    queryFn: () => api.summary(params),
    staleTime: 5 * 60_000,
  })
}

export function useBreakingPoints(params?: { stage?: string; dataset?: string; model?: string; combo?: string }) {
  return useQuery<BreakingPointRow[], Error>({
    queryKey: ['breaking-points', params],
    queryFn: () => api.breakingPoints(params),
    staleTime: 5 * 60_000,
  })
}

export function useRobustness(params?: { stage?: string; dataset?: string; scope?: string }) {
  return useQuery<RobustnessRow[], Error>({
    queryKey: ['robustness', params],
    queryFn: () => api.robustness(params),
    staleTime: 5 * 60_000,
  })
}

export function useSynergy(params?: { stage?: string; dataset?: string }) {
  return useQuery<SynergyRow[], Error>({
    queryKey: ['synergy', params],
    queryFn: () => api.synergy(params),
    staleTime: 5 * 60_000,
  })
}

export function useManifest(stage?: string) {
  return useQuery({
    queryKey: ['manifest', stage ?? 'full'],
    queryFn: () => api.manifest(stage ?? 'full'),
    staleTime: 5 * 60_000,
  })
}
