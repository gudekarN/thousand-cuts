import pandas as pd
import numpy as np
from typing import Tuple

def compute_baselines(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Compute baselines from raw results DataFrame.
    Returns:
        baselines (DataFrame), n_errors (int)
    """
    n_errors = int((df["status"] != "ok").sum())
    df_ok = df[df["status"] == "ok"].copy()
    
    clean_df = df_ok[(df_ok["combo"] == "clean") | (df_ok["level"] == 0)]
    
    if clean_df.empty:
        baselines = pd.DataFrame(columns=["dataset", "model", "f1_mean", "f1_std", "acc_mean", "acc_std", "threshold_f1"])
        return baselines, n_errors

    baselines = clean_df.groupby(["dataset", "model"]).agg(
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan)
    ).reset_index()
    
    baselines["threshold_f1"] = 0.90 * baselines["f1_mean"]
    
    return baselines, n_errors


def compute_summary(df: pd.DataFrame, baselines: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Compute summary metrics per (dataset, model, combo, level).
    Clean rows serve as level 0 for every combo present in the dataset/model.
    Returns:
        summary (DataFrame), n_errors (int)
    """
    n_errors = int((df["status"] != "ok").sum())
    df_ok = df[df["status"] == "ok"].copy()
    
    if df_ok.empty:
        summary = pd.DataFrame(columns=[
            "dataset", "model", "combo", "level", "n_seeds", 
            "f1_mean", "f1_std", "acc_mean", "acc_std", "fit_time_mean",
            "rel_f1", "drop_abs", "drop_rel"
        ])
        return summary, n_errors

    clean_rows = df_ok[(df_ok["combo"] == "clean") | (df_ok["level"] == 0)].copy()
    noisy_rows = df_ok[(df_ok["combo"] != "clean") & (df_ok["level"] > 0)].copy()
    
    combos_per_dm = df_ok[df_ok["combo"] != "clean"][["dataset", "model", "combo"]].drop_duplicates()
    
    virtual_clean_list = []
    for _, row in combos_per_dm.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        combo = row["combo"]
        
        dm_clean = clean_rows[(clean_rows["dataset"] == dataset) & (clean_rows["model"] == model)].copy()
        if not dm_clean.empty:
            dm_clean["combo"] = combo
            dm_clean["level"] = 0
            virtual_clean_list.append(dm_clean)
            
    if virtual_clean_list:
        virtual_clean = pd.concat(virtual_clean_list, ignore_index=True)
    else:
        virtual_clean = pd.DataFrame(columns=df_ok.columns)
        
    combined = pd.concat([clean_rows, virtual_clean, noisy_rows], ignore_index=True)
    
    summary = combined.groupby(["dataset", "model", "combo", "level"]).agg(
        n_seeds=("seed", "count"),
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
        fit_time_mean=("fit_time_s", "mean"),
    ).reset_index()
    
    if not baselines.empty:
        summary = summary.merge(baselines[["dataset", "model", "f1_mean"]], on=["dataset", "model"], suffixes=("", "_baseline"))
        summary.rename(columns={"f1_mean_baseline": "baseline_f1"}, inplace=True)
        
        summary["rel_f1"] = summary["f1_mean"] / summary["baseline_f1"]
        summary["drop_abs"] = summary["baseline_f1"] - summary["f1_mean"]
        summary["drop_rel"] = summary["drop_abs"] / summary["baseline_f1"]
        
        summary.drop(columns=["baseline_f1"], inplace=True)
    else:
        summary["rel_f1"] = np.nan
        summary["drop_abs"] = np.nan
        summary["drop_rel"] = np.nan
    
    return summary, n_errors
