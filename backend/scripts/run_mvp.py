"""MVP execution script.

Builds and executes the MVP plan (312 fits).
"""
import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Ensure backend directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.runner import build_plan, execute_plan
from app.engine.analysis import write_all_outputs
from app.core.config import STAGE_MVP
from app.core.hashing import compute_config_hash
from app.core.levels import load_levels


def main():
    parser = argparse.ArgumentParser(description="Run the MVP phase.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan summary and exit")
    parser.add_argument("--out", type=str, default="results/official/mvp", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of fits (only allowed if --out is outside results/official)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out).resolve()
    
    # Restrict --limit to non-official output directories
    is_official_out = "results/official" in out_dir.as_posix() or "results\\official" in out_dir.as_posix()
    if args.limit is not None and is_official_out:
        print("ERROR: --limit is only allowed when --out is outside results/official", file=sys.stderr)
        sys.exit(1)
        
    plan = build_plan(stage=STAGE_MVP)
    
    if args.dry_run:
        # Print counts per model and combo
        df = pd.DataFrame([vars(spec) for spec in plan])
        counts = df.groupby(["model", "combo"]).size().reset_index(name="fits")
        print(counts.to_string(index=False))
        print(f"\nTotal fits: {len(plan)}")
        return
        
    if args.limit is not None:
        plan = plan[:args.limit]
        
    lvl = load_levels()
    run_meta = {
        "run_id": "mvp-run",
        "run_type": "official",
        "stage": STAGE_MVP,
        "config_hash": compute_config_hash(),
        "methodology_version": "1.0",
        "levels_version": lvl.get("levels_version", ""),
        "levels_frozen": lvl.get("frozen", False),
        "requested_config": {"stage": STAGE_MVP}
    }
    
    print(f"Executing MVP plan with {len(plan)} fits...")
    print(f"Output directory: {out_dir}")
    
    def progress(done, total, spec):
        print(f"Fit {done}/{total} completed: {spec.model} - {spec.combo} (L{spec.level}, S{spec.seed})")
        
    start_time = time.time()
    
    execute_plan(plan, out_dir, run_meta, progress_cb=progress, resume=True)
    
    raw_csv = out_dir / "raw_results.csv"
    if raw_csv.exists():
        print("Writing derived analysis files...")
        write_all_outputs(raw_csv, out_dir)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nMVP run finished in {total_time:.1f} seconds.")

if __name__ == "__main__":
    main()
