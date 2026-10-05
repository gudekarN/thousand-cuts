"""Full-run execution script.

Builds and executes the Full plan (6,080 fits).

Usage:
    python scripts/run_full.py [--dry-run] [--analyze-only] [--figures]
                               [--out DIR] [--limit N] [--levels-path PATH]
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
from app.engine.figures import generate_all_figures
from app.core.config import STAGE_FULL
from app.core.hashing import verify_frozen_hash
from app.core.levels import load_levels, DEFAULT_LEVELS_PATH


def main():
    parser = argparse.ArgumentParser(description="Run the Full experiment phase.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan summary and exit")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run write_all_outputs on the existing raw_results.csv; do not execute fits",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="(Not implemented — will be added in Task 5.4) Generate report figures",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/official/full",
        help="Output directory (default: results/official/full)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of fits (only allowed if --out is outside results/official)",
    )
    parser.add_argument(
        "--levels-path",
        type=str,
        default=str(DEFAULT_LEVELS_PATH),
        help="Path to levels.json",
    )

    args = parser.parse_args()

    out_dir = Path(args.out).resolve()

    # Freeze guard — must pass before any other action
    try:
        verify_frozen_hash(args.levels_path)
    except ValueError as e:
        print(f"ERROR: Full run refused. {e}", file=sys.stderr)
        sys.exit(1)

    # --limit only allowed outside results/official
    is_official_out = (
        "results/official" in out_dir.as_posix()
        or "results\\official" in out_dir.as_posix()
    )
    if args.limit is not None and is_official_out:
        print(
            "ERROR: --limit is only allowed when --out is outside results/official",
            file=sys.stderr,
        )
        sys.exit(1)

    # --analyze-only: re-run derived outputs only
    if args.analyze_only:
        raw_csv = out_dir / "raw_results.csv"
        if not raw_csv.exists():
            print(
                f"ERROR: --analyze-only requires an existing raw_results.csv at {raw_csv}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Running analysis-only on: {raw_csv}")
        write_all_outputs(raw_csv, out_dir)
        print("Analysis complete.")
        return

    # --figures
    if args.figures:
        print(f"Generating figures for {out_dir}...")
        generate_all_figures(out_dir)
        print("Figures generated successfully.")
        return

    plan = build_plan(stage=STAGE_FULL)

    if args.dry_run:
        df = pd.DataFrame([vars(spec) for spec in plan])
        counts = df.groupby(["model", "combo"]).size().reset_index(name="fits")
        print(counts.to_string(index=False))
        print(f"\nTotal fits: {len(plan)}")
        return

    if args.limit is not None:
        plan = plan[: args.limit]

    lvl = load_levels(args.levels_path)
    run_meta = {
        "run_id": "full-run",
        "run_type": "official",
        "stage": STAGE_FULL,
        "config_hash": lvl.get("config_hash"),
        "methodology_version": "1.0",
        "levels_version": lvl.get("levels_version", ""),
        "levels_frozen": True,
        "requested_config": {"stage": STAGE_FULL},
    }

    print(f"Executing Full plan with {len(plan)} fits...")
    print(f"Output directory: {out_dir}")

    def progress(done, total, spec):
        print(
            f"Fit {done}/{total} completed: {spec.model} - {spec.combo}"
            f" (L{spec.level}, S{spec.seed})"
        )

    start_time = time.time()

    execute_plan(plan, out_dir, run_meta, progress_cb=progress, resume=True)

    raw_csv = out_dir / "raw_results.csv"
    if raw_csv.exists():
        print("Writing derived analysis files...")
        write_all_outputs(raw_csv, out_dir)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nFull run finished in {total_time:.1f} seconds.")


if __name__ == "__main__":
    main()
