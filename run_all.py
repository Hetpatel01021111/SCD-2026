"""
Run all experiments sequentially.

Usage:
    python run_all.py              # run everything (exp0–exp5)
    python run_all.py 0 2 3        # run only experiments 0, 2, and 3
"""

import sys
import time


EXPERIMENTS = {
    0: ("Clean Baseline", "experiments.exp0_clean_baseline"),
    1: ("Label-Flip Attack", "experiments.exp1_label_flip_attack"),
    2: ("Backdoor Attack", "experiments.exp2_backdoor_attack"),
    3: ("Detection Pipeline", "experiments.exp3_detection_pipeline"),
    4: ("Cleaning & Retrain", "experiments.exp4_cleaning_and_retrain"),
    5: ("Full Demo", "experiments.exp5_full_demo"),
}


def main():
    # Parse which experiments to run from CLI args
    if len(sys.argv) > 1:
        selected = [int(x) for x in sys.argv[1:]]
    else:
        selected = list(EXPERIMENTS.keys())

    print("╔" + "═" * 58 + "╗")
    print("║   DATA POISONING & DETECTION — EXPERIMENT RUNNER         ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Experiments to run: {selected}")
    print(f"  Logs directory: outputs/logs/\n")

    for exp_id in selected:
        if exp_id not in EXPERIMENTS:
            print(f"  [SKIP] Unknown experiment ID: {exp_id}")
            continue

        name, module_path = EXPERIMENTS[exp_id]
        print(f"\n{'#' * 60}")
        print(f"#  Experiment {exp_id}: {name}")
        print(f"{'#' * 60}\n")

        t0 = time.time()
        mod = __import__(module_path, fromlist=["run"])
        mod.run()
        elapsed = time.time() - t0

        print(f"  ⏱  Experiment {exp_id} completed in {elapsed:.1f}s\n")

    print("All selected experiments finished.")
    print("Run `python scripts/generate_graphs.py` to produce comparison charts from logs.")


if __name__ == "__main__":
    main()
