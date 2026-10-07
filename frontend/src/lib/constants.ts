/** Design-system color constants for models and noise types. */

export const MODEL_COLORS: Record<string, string> = {
  logreg: '#3B82F6',
  svm_rbf: '#8B5CF6',
  decision_tree: '#F59E0B',
  random_forest: '#10B981',
}

export const MODEL_LABELS: Record<string, string> = {
  logreg: 'Logistic Regression',
  svm_rbf: 'SVM (RBF)',
  decision_tree: 'Decision Tree',
  random_forest: 'Random Forest',
}

export const NOISE_COLORS: Record<string, string> = {
  label: '#EF4444',
  gaussian: '#06B6D4',
  outliers: '#F97316',
  missing: '#64748B',
}

export const NOISE_LABELS: Record<string, string> = {
  label: 'Label Noise',
  gaussian: 'Gaussian',
  outliers: 'Outliers',
  missing: 'Missing Values',
}

export const DATASET_LABELS: Record<string, string> = {
  breast_cancer: 'Breast Cancer',
  digits: 'Digits',
}

export const STAGE_LABELS: Record<string, string> = {
  mvp: 'MVP',
  stage2: 'Stage 2',
  full: 'Full Results',
}

/** Format a number as F1/accuracy (4 decimals) */
export function fmt4(n: number | null | undefined): string {
  if (n == null) return '—'
  return n.toFixed(4)
}

/** Format a percentage with 1 decimal */
export function fmtPct(n: number | null | undefined): string {
  if (n == null) return '—'
  return (n * 100).toFixed(1) + '%'
}
