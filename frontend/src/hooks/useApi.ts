import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  type ApiConfig,
  type EstimateResponse,
  type ExperimentRequest,
  type JobResponse,
  type SummaryRow,
  type BreakingPointRow,
  type RobustnessRow,
  type SynergyRow,
} from '@/api/client'

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

/** Estimate a custom experiment with the shared backend plan builder. */
export function useExperimentEstimate(request: ExperimentRequest | null) {
  return useQuery<EstimateResponse, Error>({
    queryKey: ['experiment-estimate', request],
    queryFn: () => api.estimateExperiment(request as ExperimentRequest),
    enabled: request !== null,
    staleTime: 15_000,
    retry: 1,
  })
}

/** Poll a custom job only while it is queued or running. */
export function useExperiment(runId: string | null) {
  return useQuery<JobResponse, Error>({
    queryKey: ['experiment', runId],
    queryFn: () => api.getExperiment(runId as string),
    enabled: runId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'queued' || status === 'running' ? 1000 : false
    },
    retry: 1,
  })
}

export function useCreateExperiment() {
  const queryClient = useQueryClient()
  return useMutation<JobResponse, Error, ExperimentRequest>({
    mutationFn: api.createExperiment,
    onSuccess: (job) => {
      queryClient.setQueryData(['experiment', job.run_id], job)
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
    },
  })
}

export function useCancelExperiment() {
  const queryClient = useQueryClient()
  return useMutation<JobResponse, Error, string>({
    mutationFn: api.cancelExperiment,
    onSuccess: (job) => {
      queryClient.setQueryData(['experiment', job.run_id], job)
      queryClient.invalidateQueries({ queryKey: ['experiment', job.run_id] })
    },
  })
}
