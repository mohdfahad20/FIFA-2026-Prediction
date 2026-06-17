"""
run_pipeline.py
===============
Runs the full FIFA WC 2026 local pipeline in sequence:
  1. Load historical data (CSVs → fifa.db)        [optional, --skip-data]
  2. Scrape live WC 2026 results                  [optional, --skip-scrape]
  3. Predict upcoming match + fill actuals        [optional, --skip-predict]
  4. Build features table
  5. Train ML ensemble model                       [optional, --skip-ml]
  6. Train Poisson score model
  7. Run Monte Carlo simulation                    [optional, --skip-sim]

Usage:
    python run_pipeline.py                              # full pipeline, all defaults
    python run_pipeline.py --skip-data                  # skip step 1 (DB already loaded)
    python run_pipeline.py --skip-data --skip-ml         # only scrape + predict + Poisson + sim
    python run_pipeline.py --skip-data --skip-ml --n 1000  # faster sim for testing
    python run_pipeline.py --skip-data --skip-ml --skip-scrape --skip-predict  # sim only

All steps are timed and a summary is printed at the end.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(label: str, cmd: list[str]) -> float:
    """Run a command, stream output, return elapsed seconds. Exit on failure."""
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[OK] {label} completed in {elapsed:.1f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FIFA 2026 local pipeline runner.")
    parser.add_argument("--db",           default="fifa.db")
    parser.add_argument("--results",      default="data/results.csv")
    parser.add_argument("--rankings",     default="data/fifa_ranking-2026-04-01.csv")
    parser.add_argument("--n-iter",       default="40",   help="ML tuning iterations")
    parser.add_argument("--n",            default="10000", help="Monte Carlo simulations")
    parser.add_argument("--full-scrape",  action="store_true",
                        help="Scrape full tournament window (June 11 - July 19) instead of just yesterday/today")
    parser.add_argument("--skip-data",    action="store_true", help="Skip step 1 (load_historical_data)")
    parser.add_argument("--skip-scrape",  action="store_true", help="Skip step 2 (scraper)")
    parser.add_argument("--skip-predict", action="store_true", help="Skip step 3 (predict_upcoming)")
    parser.add_argument("--skip-ml",      action="store_true", help="Skip step 5 (model.train)")
    parser.add_argument("--skip-sim",     action="store_true", help="Skip step 7 (simulate)")
    args = parser.parse_args()

    py = sys.executable   # use same python/venv that launched this script
    timings: dict[str, float] = {}
    start_all = time.time()

    print(f"\nFIFA 2026 Pipeline  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DB: {args.db}  |  Sim n={args.n}  |  ML iter={args.n_iter}")

    # ------------------------------------------------------------------
    # Step 1 — Load historical data
    # ------------------------------------------------------------------
    if not args.skip_data:
        timings["Load data"] = run(
            "Load historical data (CSVs → fifa.db)",
            [py, "data/load_historical_data.py",
             "--results",  args.results,
             "--rankings", args.rankings,
             "--db",       args.db],
        )
    else:
        print("\n[SKIP] Step 1 — load_historical_data (--skip-data)")

    # ------------------------------------------------------------------
    # Step 2 — Scrape live WC 2026 results
    # ------------------------------------------------------------------
    if not args.skip_scrape:
        scrape_cmd = [py, "-m", "scraper.scraper", "--db", args.db]
        if args.full_scrape:
            scrape_cmd.append("--full")
        timings["Scrape results"] = run(
            "Scrape live WC 2026 results (football-data.org)",
            scrape_cmd,
        )
    else:
        print("\n[SKIP] Step 2 — scraper (--skip-scrape)")

    # ------------------------------------------------------------------
    # Step 3 — Predict upcoming match + fill actuals for completed ones
    # ------------------------------------------------------------------
    if not args.skip_predict:
        timings["Predict upcoming"] = run(
            "Predict tomorrow's match + fill yesterday's actuals",
            [py, "predict_upcoming.py", "--db", args.db],
        )
    else:
        print("\n[SKIP] Step 3 — predict_upcoming (--skip-predict)")

    # ------------------------------------------------------------------
    # Step 4 — Build features
    # ------------------------------------------------------------------
    timings["Features"] = run(
        "Build features table",
        [py, "-m", "features.features", "--db", args.db],
    )

    # ------------------------------------------------------------------
    # Step 5 — Train ML model
    # ------------------------------------------------------------------
    if not args.skip_ml:
        timings["ML model"] = run(
            f"Train ML ensemble (n_iter={args.n_iter})",
            [py, "-m", "model.train", "--db", args.db, "--n-iter", args.n_iter],
        )
    else:
        print("\n[SKIP] Step 5 — model.train (--skip-ml)")

    # ------------------------------------------------------------------
    # Step 6 — Train Poisson model
    # ------------------------------------------------------------------
    timings["Poisson model"] = run(
        "Train Poisson score model",
        [py, "-m", "score_model.train_poisson", "--db", args.db],
    )

    # ------------------------------------------------------------------
    # Step 7 — Simulate
    # ------------------------------------------------------------------
    if not args.skip_sim:
        timings["Simulation"] = run(
            f"Monte Carlo simulation (n={args.n})",
            [py, "-m", "simulate.simulate", "--db", args.db, "--n", args.n],
        )
    else:
        print("\n[SKIP] Step 7 — simulate (--skip-sim)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = time.time() - start_all
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    for step, t in timings.items():
        print(f"  {step:<25} {t:>6.1f}s")
    print(f"  {'TOTAL':<25} {total:>6.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()