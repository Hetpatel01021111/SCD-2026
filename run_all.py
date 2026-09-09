"""
Run all experiments sequentially.

Usage:
    python run_all.py              # run everything (exp0–exp5)
    python run_all.py 0 2 3        # run only experiments 0, 2, and 3
"""

import sys
import time
import argparse
import os


EXPERIMENTS = {
    0: ("Clean Baseline", "experiments.exp0_clean_baseline"),
    1: ("Label-Flip Attack", "experiments.exp1_label_flip_attack"),
    2: ("Backdoor Attack", "experiments.exp2_backdoor_attack"),
    3: ("Detection Pipeline", "experiments.exp3_detection_pipeline"),
    4: ("Cleaning & Retrain", "experiments.exp4_cleaning_and_retrain"),
    5: ("Full Demo", "experiments.exp5_full_demo"),
}


def main():
    parser = argparse.ArgumentParser(description="Run poisoning experiments")
    parser.add_argument("experiments", nargs="*", type=int, choices=EXPERIMENTS.keys())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--poison-rate", type=float, default=None)
    parser.add_argument("--run-dir", default=None,
                        help="directory for this run's logs, plots and models")
    parser.add_argument("--resume", action="store_true",
                        help="reserved for manifest-compatible completed runs")
    args = parser.parse_args()
    import config
    if args.seed is not None:
        config.SEED = args.seed
    if args.poison_rate is not None:
        if not 0.0 <= args.poison_rate <= 1.0:
            parser.error("--poison-rate must be between 0 and 1")
        config.POISON_RATE = args.poison_rate
    if args.run_dir:
        config.OUTPUT_DIR = os.path.abspath(args.run_dir)
        config.LOG_DIR = os.path.join(config.OUTPUT_DIR, "logs")
        config.PLOT_DIR = os.path.join(config.OUTPUT_DIR, "plots")
        config.MODEL_DIR = os.path.join(config.OUTPUT_DIR, "models")
        for path in (config.LOG_DIR, config.PLOT_DIR, config.MODEL_DIR):
            os.makedirs(path, exist_ok=True)
    selected = args.experiments or list(EXPERIMENTS.keys())

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
