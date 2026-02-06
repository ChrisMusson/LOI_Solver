import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from solve import solve


def run_sensitivity(options=None):
    if options is None or "count" not in options:
        runs = int(input("How many simulations would you like to run? "))
        processes = int(input("How many processes you want to run in parallel? "))
    else:
        runs = options.get("count", 1)
        processes = options.get("processes", os.cpu_count() - 2)

    start = time.time()
    all_jobs = [{"run_no": str(i + 1), "randomised": True} for i in range(runs)]
    with ProcessPoolExecutor(max_workers=processes) as executor:
        list(executor.map(solve, all_jobs))
    end = time.time()
    print(f"\nTotal time taken is {(end - start) / 60:.2f} minutes")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run sensitivity analysis")
        parser.add_argument("--no", type=int, help="Number of runs")
        parser.add_argument("--parallel", type=int, help="Number of parallel runs")
        args = parser.parse_args()
        options = {}
        if args.no:
            options["count"] = args.no
        if args.parallel:
            options["processes"] = args.parallel
    except Exception:
        options = None

    run_sensitivity(options)
