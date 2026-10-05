import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import os

# Design tokens
MODEL_STYLES = {
    "logreg": {"color": "#3B82F6", "linestyle": "solid", "marker": "o", "label": "Logistic Regression"},
    "svm_rbf": {"color": "#8B5CF6", "linestyle": "dashed", "marker": "s", "label": "SVM (RBF)"},
    "decision_tree": {"color": "#F59E0B", "linestyle": "dotted", "marker": "^", "label": "Decision Tree"},
    "random_forest": {"color": "#10B981", "linestyle": "dashdot", "marker": "D", "label": "Random Forest"},
}

NOISE_COLORS = {
    "label": "#EF4444",
    "gaussian": "#06B6D4",
    "outliers": "#F97316",
    "missing": "#64748B",
}

def generate_line_charts(summary_df, baselines_df, breaking_points_df, out_dir, metric="f1", datasets=None):
    if datasets is None:
        datasets = summary_df["dataset"].unique()
    
    singles = ["label", "gaussian", "outliers", "missing"]
    
    for dataset in datasets:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        fig.suptitle(f"{dataset.replace('_', ' ').title()} - {metric.title()} vs Level", fontsize=16)
        
        d_summary = summary_df[summary_df["dataset"] == dataset]
        d_baselines = baselines_df[baselines_df["dataset"] == dataset]
        d_bp = breaking_points_df[breaking_points_df["dataset"] == dataset]
        
        for idx, noise in enumerate(singles):
            ax = axes[idx // 2, idx % 2]
            ax.set_title(f"Noise: {noise.title()}")
            ax.set_xlabel("Level")
            ax.set_ylabel(metric.title())
            ax.set_xticks(range(6))
            ax.grid(True, linestyle="--", alpha=0.5)
            
            for model, style in MODEL_STYLES.items():
                m_summary = d_summary[(d_summary["model"] == model) & ((d_summary["combo"] == noise) | (d_summary["combo"] == "clean"))].sort_values("level")
                if m_summary.empty:
                    continue
                
                x = m_summary["level"]
                y = m_summary[f"{metric}_mean"]
                yerr = m_summary[f"{metric}_std"]
                
                ax.plot(x, y, color=style["color"], linestyle=style["linestyle"], marker=style["marker"], label=style["label"])
                ax.fill_between(x, y - yerr, y + yerr, color=style["color"], alpha=0.1)
                
                # Threshold and Breaking Point only for F1
                if metric == "f1":
                    baseline_row = d_baselines[d_baselines["model"] == model]
                    if not baseline_row.empty:
                        threshold = baseline_row.iloc[0]["threshold_f1"]
                        ax.axhline(threshold, color=style["color"], linestyle=":", alpha=0.5)
                    
                    bp_row = d_bp[(d_bp["model"] == model) & (d_bp["combo"] == noise)]
                    if not bp_row.empty and bp_row.iloc[0]["reached"]:
                        bp_level = bp_row.iloc[0]["breaking_level"]
                        bp_y = m_summary[m_summary["level"] == bp_level][f"{metric}_mean"]
                        if not bp_y.empty:
                            ax.plot(bp_level, bp_y.iloc[0], marker="X", color="red", markersize=10, zorder=5)
                            
            if idx == 0:
                ax.legend()
                
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_{metric}_vs_level.png", dpi=300)
        plt.close()

def generate_heatmap(breaking_points_df, out_dir, datasets=None):
    if datasets is None:
        datasets = breaking_points_df["dataset"].unique()
        
    combos = breaking_points_df["combo"].unique()
    models = list(MODEL_STYLES.keys())
    
    for dataset in datasets:
        d_bp = breaking_points_df[breaking_points_df["dataset"] == dataset]
        
        heatmap_data = pd.DataFrame(index=combos, columns=models, dtype=float)
        
        for combo in combos:
            for model in models:
                row = d_bp[(d_bp["model"] == model) & (d_bp["combo"] == combo)]
                if not row.empty and row.iloc[0]["reached"]:
                    heatmap_data.loc[combo, model] = row.iloc[0]["breaking_level"]
                else:
                    heatmap_data.loc[combo, model] = 6 # Not reached
                    
        plt.figure(figsize=(10, 8))
        
        # Pure matplotlib heatmap
        data_matrix = heatmap_data.values.astype(float)
        cmap = plt.get_cmap("YlOrRd")
        
        # Mask NaNs (though we filled everything)
        im = plt.imshow(data_matrix, cmap=cmap, vmin=1, vmax=6, aspect='auto')
        plt.colorbar(im, label='Breaking Level (6 = Not Reached)')
        
        ax = plt.gca()
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(combos)))
        ax.set_xticklabels(models)
        ax.set_yticklabels(combos)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(combos)):
            for j in range(len(models)):
                val = data_matrix[i, j]
                text = ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="black")
                
        plt.title(f"{dataset.replace('_', ' ').title()} - Breaking Levels")
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_heatmap.png", dpi=300)
        plt.close()

def generate_bp_table(breaking_points_df, out_dir):
    # Simply save the singles breaking points table as a figure
    singles = ["label", "gaussian", "outliers", "missing"]
    bp_singles = breaking_points_df[breaking_points_df["combo"].isin(singles)].copy()
    
    # We can plot this as a table using matplotlib
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    display_cols = ["dataset", "model", "combo", "breaking_level", "reached", "f1_at_bp"]
    table_data = bp_singles[display_cols].head(20).round(4).values.tolist()
    col_labels = [c.replace("_", " ").title() for c in display_cols]
    
    if table_data:
        ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        plt.title("Breaking Points (Singles - Top 20 rows)")
        plt.tight_layout()
        plt.savefig(out_dir / "breaking_points_table.png", dpi=300)
    plt.close()

def generate_synergy_chart(synergy_df, out_dir, datasets=None):
    if synergy_df.empty:
        return
        
    if datasets is None:
        datasets = synergy_df["dataset"].unique()
        
    for dataset in datasets:
        d_syn = synergy_df[synergy_df["dataset"] == dataset]
        if d_syn.empty:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # A simple grouped bar chart for synergy across combos and models at level 5
        level5_syn = d_syn[d_syn["level"] == 5]
        if level5_syn.empty:
            level5_syn = d_syn
            
        combos_list = level5_syn["combo"].unique()
        models_list = list(MODEL_STYLES.keys())
        
        x = np.arange(len(combos_list))
        width = 0.8 / len(models_list)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, model in enumerate(models_list):
            model_data = level5_syn[level5_syn["model"] == model]
            
            y_values = []
            for combo in combos_list:
                row = model_data[model_data["combo"] == combo]
                if not row.empty:
                    y_values.append(row.iloc[0]["synergy"])
                else:
                    y_values.append(0.0)
                    
            offset = (i - len(models_list)/2 + 0.5) * width
            ax.bar(x + offset, y_values, width, label=MODEL_STYLES[model]["label"], color=MODEL_STYLES[model]["color"])
            
        ax.set_xticks(x)
        ax.set_xticklabels(combos_list, rotation=45, ha="right")
        ax.legend()
        plt.title(f"{dataset.replace('_', ' ').title()} - Synergy at Level 5 (Secondary Analysis)")
        plt.ylabel("Synergy (Drop Compound - Sum Single Drops)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_synergy.png", dpi=300)
        plt.close()

def generate_all_figures(out_dir):
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    summary = pd.read_csv(out_dir / "summary.csv")
    baselines = pd.read_csv(out_dir / "baselines.csv")
    breaking_points = pd.read_csv(out_dir / "breaking_points.csv")
    synergy = pd.read_csv(out_dir / "synergy.csv")
    
    generate_line_charts(summary, baselines, breaking_points, figures_dir, metric="f1")
    generate_line_charts(summary, baselines, breaking_points, figures_dir, metric="acc")
    generate_heatmap(breaking_points, figures_dir)
    generate_bp_table(breaking_points, figures_dir)
    generate_synergy_chart(synergy, figures_dir)
