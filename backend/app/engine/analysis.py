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


import json
from app.core.levels import get_params
from app.core.config import NOISE_LABEL, NOISE_GAUSSIAN, NOISE_OUTLIERS, NOISE_MISSING

_NOISE_PARAM_MAP = {
    NOISE_LABEL: "label_flip_rate",
    NOISE_GAUSSIAN: "gaussian_k",
    NOISE_OUTLIERS: "outlier_cell_rate",
    NOISE_MISSING: "missing_cell_rate"
}

def _get_noise_params_json(combo: str, level: int) -> str:
    if not combo or combo == "clean" or level == 0:
        return "{}"
    all_params = get_params(level)
    components = combo.split("+")
    combo_params = {}
    for comp in components:
        if comp in _NOISE_PARAM_MAP:
            param_key = _NOISE_PARAM_MAP[comp]
            combo_params[param_key] = all_params.get(param_key)
    return json.dumps(combo_params)


def compute_breaking_points(df: pd.DataFrame, summary: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Compute breaking points per (dataset, model, combo).
    Returns DataFrame with breaking point information.
    """
    df_ok = df[df["status"] == "ok"].copy()
    
    results = []
    
    if summary.empty or baselines.empty:
        return pd.DataFrame(columns=[
            "dataset", "model", "combo", "baseline_f1", "threshold_f1",
            "breaking_level", "reached", "noise_params", "f1_at_bp", "seeds_below_at_bp"
        ])
        
    unique_combos = summary[summary["combo"] != "clean"][["dataset", "model", "combo"]].drop_duplicates()
    
    for _, row in unique_combos.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        combo = row["combo"]
        
        b_df = baselines[(baselines["dataset"] == dataset) & (baselines["model"] == model)]
        if b_df.empty:
            continue
            
        baseline_f1 = float(b_df.iloc[0]["f1_mean"])
        threshold_f1 = float(b_df.iloc[0]["threshold_f1"])
        
        s_df = summary[
            (summary["dataset"] == dataset) &
            (summary["model"] == model) &
            (summary["combo"] == combo) &
            (summary["level"] > 0)
        ].sort_values("level")
        
        breaking_level = None
        f1_at_bp = None
        
        for _, s_row in s_df.iterrows():
            lvl = int(s_row["level"])
            f1 = float(s_row["f1_mean"])
            if f1 <= threshold_f1 + 1e-12:
                breaking_level = lvl
                f1_at_bp = f1
                break
                
        if breaking_level is not None:
            reached = True
            noise_params = _get_noise_params_json(combo, breaking_level)
            
            raw_seeds = df_ok[
                (df_ok["dataset"] == dataset) &
                (df_ok["model"] == model) &
                (df_ok["combo"] == combo) &
                (df_ok["level"] == breaking_level)
            ]
            seeds_below_at_bp = int((raw_seeds["macro_f1"] <= threshold_f1 + 1e-12).sum())
        else:
            reached = False
            noise_params = ""
            f1_at_bp = pd.NA
            seeds_below_at_bp = pd.NA
            breaking_level = pd.NA
            
        results.append({
            "dataset": dataset,
            "model": model,
            "combo": combo,
            "baseline_f1": baseline_f1,
            "threshold_f1": threshold_f1,
            "breaking_level": breaking_level,
            "reached": reached,
            "noise_params": noise_params,
            "f1_at_bp": f1_at_bp,
            "seeds_below_at_bp": seeds_below_at_bp,
        })
        
    if not results:
        return pd.DataFrame(columns=[
            "dataset", "model", "combo", "baseline_f1", "threshold_f1",
            "breaking_level", "reached", "noise_params", "f1_at_bp", "seeds_below_at_bp"
        ])
        
    return pd.DataFrame(results)


import logging

def compute_synergy(summary: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Compute synergy for compound combos.
    """
    results = []
    
    if summary.empty or baselines.empty:
        return pd.DataFrame(columns=[
            "dataset", "model", "combo", "level", "baseline_f1", "compound_f1",
            "drop_compound", "sum_single_drops", "synergy", "f1_floor_flag"
        ])
    
    compounds = summary[summary["combo"].str.contains(r"\+", na=False, regex=True) & (summary["level"] > 0)]
    
    for _, row in compounds.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        combo = row["combo"]
        level = int(row["level"])
        compound_f1 = float(row["f1_mean"])
        
        b_df = baselines[(baselines["dataset"] == dataset) & (baselines["model"] == model)]
        if b_df.empty:
            continue
        baseline_f1 = float(b_df.iloc[0]["f1_mean"])
        
        drop_compound = baseline_f1 - compound_f1
        
        components = combo.split("+")
        sum_single_drops = 0.0
        skip = False
        
        for comp in components:
            comp_df = summary[
                (summary["dataset"] == dataset) &
                (summary["model"] == model) &
                (summary["combo"] == comp) &
                (summary["level"] == level)
            ]
            if comp_df.empty:
                logging.warning(f"Missing component '{comp}' for combo '{combo}' at level {level} in {dataset}/{model}. Skipping.")
                skip = True
                break
            
            comp_f1 = float(comp_df.iloc[0]["f1_mean"])
            sum_single_drops += (baseline_f1 - comp_f1)
            
        if skip:
            continue
            
        synergy = drop_compound - sum_single_drops
        f1_floor_flag = sum_single_drops >= baseline_f1
        
        results.append({
            "dataset": dataset,
            "model": model,
            "combo": combo,
            "level": level,
            "baseline_f1": baseline_f1,
            "compound_f1": compound_f1,
            "drop_compound": drop_compound,
            "sum_single_drops": sum_single_drops,
            "synergy": synergy,
            "f1_floor_flag": bool(f1_floor_flag),
        })
        
    if not results:
        return pd.DataFrame(columns=[
            "dataset", "model", "combo", "level", "baseline_f1", "compound_f1",
            "drop_compound", "sum_single_drops", "synergy", "f1_floor_flag"
        ])
        
    return pd.DataFrame(results)
