# design.md (UI / UX)

Goal: a polished ML experimentation product, not a basic admin dashboard. Clear numbers first, decoration last.

## 1. Principles
1. Data first: charts and metrics are the hero. Visual style supports them.
2. Calm and modern: generous spacing, soft borders, restrained color.
3. Consistent: one color per model, one per noise type, everywhere.
4. Honest: always show mean +/- std, stage/source, and provisional status.
5. Subtle motion only: 150 to 250 ms fades and slides. Respect `prefers-reduced-motion`.

## 2. Stack for UI
Tailwind CSS + shadcn/ui (Radix based) + Lucide icons + Recharts. Animations via Tailwind transitions and `tailwindcss-animate` (no extra animation library).

shadcn components: button, card, badge, select, tabs, tooltip, skeleton, table, progress, sheet, dialog, toggle-group, slider, checkbox, switch, separator, alert, scroll-area, dropdown-menu, sonner (toasts).

## 3. Design tokens

**Colors (shadcn CSS variables)**

| Token | Light | Dark |
|---|---|---|
| background | #F8FAFC | #0B1020 |
| card | #FFFFFF | #111827 |
| border | #E2E8F0 | #1F2937 |
| foreground | #0F172A | #E5E7EB |
| muted-foreground | #64748B | #94A3B8 |
| primary | #4F46E5 | #818CF8 |
| success / warning / danger | #10B981 / #F59E0B / #EF4444 | same, slightly lighter |

**Model colors + line style (never color alone)**

| Model | Color | Line | Marker |
|---|---|---|---|
| Logistic Regression | #3B82F6 | solid | circle |
| SVM (RBF) | #8B5CF6 | dashed | square |
| Decision Tree | #F59E0B | dotted | triangle |
| Random Forest | #10B981 | dash-dot | diamond |

**Noise colors:** label #EF4444, Gaussian #06B6D4, outliers #F97316, missing #64748B. Each also has a Lucide icon.

**Typography:** Inter for UI, JetBrains Mono for numbers, IDs, hashes (self-hosted via `@fontsource`). Scale: 12 / 14 / 16 / 20 / 24 / 32 px. Headings semibold, body regular.

**Spacing and shape:** 4 px grid. Cards radius 12 px, controls 8 px. Card padding 20 to 24 px. Page max width 1280 px.

**Number format:** F1 and accuracy to 4 decimals. Rates as percent with 1 decimal. Mean +/- std as `0.9712 +/- 0.0123`. Breaking point "Not reached" shown as a green badge.

## 4. Layout
- Left sidebar (240 px): logo, nav items, theme toggle, run source badge. Collapses to icon rail at `md`, becomes a sheet (hamburger) below `md`.
- Top bar: page title, breadcrumb, stage/source selector, frozen/provisional indicator.
- Content: responsive grid (1 col mobile, 2 col tablet, 3 to 4 col desktop for stat cards).

**Nav (Lucide icons):** Overview (`LayoutDashboard`), Experiment Setup (`FlaskConical`), Results (`BarChart3`), Visualizations (`LineChart`), Model Comparison (`Scale`), Experiment Details (`ClipboardList`).

## 5. Pages

### 5.1 Overview `/`
- Hero: title, one-paragraph idea, short "thousand cuts" explanation (small noise sources stack until the model breaks).
- Stat cards: datasets (2), models (4), noise types (4), combinations (15), total fits, seeds.
- Quick robustness ranking: 4 model cards ranked by BPI with RS, medal-style rank, and a mini sparkline. Toggle: singles / all combos.
- Stage badge (e.g. "Full results") or "Preliminary (MVP)" if full is not available.
- Mini "how it works" strip: Clean data, Inject noise, Train, Evaluate on clean test, Find breaking point.

### 5.2 Experiment Setup `/setup`
- Left: form card. Dataset (select with sample/feature/class info), models (multi-select chips, default all 4), noises (4 selectable cards with icons; 1 selected = individual, 2+ = compound, shows the fixed application order), mode (single level or sweep), level slider (shown only for single level, displays the real noise value per selected noise), seeds (1 to 10, default 3).
- Live summary: "N fits, estimated time".
- Start button (disabled with reason if invalid). Toast on submit.
- Right: Status card with progress bar, percent, current model/level/seed, ETA, cancel button, recent runs list.
- When a run completes, show a "View results" button.

### 5.3 Results Dashboard `/results/:runId?`
- Filter bar: source/stage or run, dataset, model, combo.
- Stat cards: Macro F1 (mean +/- std), Accuracy (mean +/- std), Baseline F1, Performance drop (absolute and relative), Breaking point, Avg fit time.
- Level table: level, noise values, F1 mean +/- std, accuracy, drop, rel F1, status (ok or broken).
- Main chart: F1 vs level with threshold line.

### 5.4 Visualizations `/visualizations`
Tabs. Shared filters (dataset, model(s), combo).
1. **F1 vs level:** line chart, shaded mean +/- std band, dashed horizontal threshold line (0.90 x baseline), marker at the breaking point.
2. **Accuracy vs level:** same layout.
3. **Model comparison:** grouped bars of breaking level per model for the selected combo.
4. **Heatmap:** rows = combos (15), columns = models (4). Toggle: breaking level or F1 retained at selected level (slider). Color scale sequential. Cell tooltip with details. Built as a CSS-grid component (Recharts has no heatmap).
5. **Breaking points:** table plus dot-strip chart (model on rows, level on x). "Not reached" at the far right.
6. **Synergy (secondary):** diverging bar chart (positive = worse than additive). Label "Secondary analysis". Shows floor-effect warning icon when flagged.

### 5.4b Chart rules
- Axis labels with units, legend with line style and marker, tooltip showing mean, std, rel F1.
- Min height 320 px, responsive container, `aria-label` and a "view as table" toggle.
- Skeleton while loading; empty state if no rows match filters.

### 5.5 Model Comparison `/compare`
- Dataset toggle (Breast Cancer / Digits / both side by side).
- Ranking table: rank, model, BPI, RS, number of combos broken, mean breaking level, worst noise type.
- Highlight card: "Most robust: <model>" with one-line reason from the numbers.
- Overlay F1 curve chart (all 4 models, selected combo) and per-noise breaking-point bars.

### 5.6 Experiment Details `/experiments` and `/experiments/:id`
- List: run id, type (official/custom), stage, dataset, status, started, progress. Filter and search.
- Detail: status timeline (queued, running, completed), dataset info, noise combo, levels and real noise values, seeds, **samples affected** (mean flipped labels / perturbed cells / outlier cells / missing cells per level), metrics summary, timestamps, config hash, methodology and levels version, frozen status, library versions, button to download `manifest.json`.

## 6. States
- **Loading:** shadcn Skeleton matching the final layout (cards, chart box, table rows). No spinners-only screens.
- **Empty:** icon, short message, action (e.g. "Run an experiment" button).
- **Error:** alert with message, error code, retry button. Backend-down banner with retry.
- **Job states:** queued (neutral badge), running (animated progress), completed (success), failed (danger, message), cancelled (muted).
- **Banners:** "Provisional: levels not frozen" (warning), "Preliminary: MVP results" (info).

## 7. Tooltips
Use for: Macro F1, baseline, breaking point definition, level meaning, BPI, RS, synergy, compound order, noise stats. Keep to one or two sentences.

## 8. Responsive
Breakpoints: `sm` 640, `md` 768, `lg` 1024, `xl` 1280. Tables scroll horizontally on small screens. Charts stack vertically. Heatmap scrolls horizontally with sticky row labels. Touch targets at least 40 px.

## 9. Accessibility
WCAG AA contrast in both themes, visible focus rings, full keyboard navigation, aria labels on charts and icon buttons, information never conveyed by color alone.

## 10. Frontend code structure
`pages/` route components, `features/<name>/` (components + hooks for that page), `components/ui` (shadcn), `components/layout`, `components/charts` (LineBand, GroupedBars, Heatmap, DotStrip, DivergingBars), `components/common` (StatCard, StageBadge, EmptyState, ErrorState, SkeletonCard, InfoTooltip), `api/` typed client, `hooks/` TanStack Query hooks, `types/`, `lib/` (formatters, constants for colors and ids).
