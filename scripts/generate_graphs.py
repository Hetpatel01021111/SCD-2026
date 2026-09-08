"""
Generate comparison graphs from experiment logs.

Reads JSON logs from ``outputs/logs/`` and produces publication-quality
charts comparing baseline, poisoned, and post-cleaning performance.

Usage:
    python scripts/generate_graphs.py          # generate all graphs
    python scripts/generate_graphs.py --help   # show options

Generated plots are saved to ``outputs/plots/``.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Allow running from project root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.logger import load_all_logs, load_log


# ── Colour palette ──────────────────────────────────────────────────
C_CLEAN = "#4CAF50"
C_POISON = "#F44336"
C_CLEANED = "#2196F3"
C_ASR = "#FF9800"
C_DETECTORS = ["#7E57C2", "#26A69A", "#EF5350", "#42A5F5"]


def savefig(fig, name: str):
    path = os.path.join(config.PLOT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Graph] Saved → {path}")


# ════════════════════════════════════════════════════════════════════
# 1. Training curves: loss & accuracy over epochs
# ════════════════════════════════════════════════════════════════════

def plot_training_curves():
    """Overlay training loss/accuracy for baseline, poisoned, cleaned."""
    curves = {}

    # Collect from whichever logs are available
    log0 = load_log("exp0_clean_baseline")
    if log0:
        curves["Clean Baseline"] = (log0["results"]["training_history"], C_CLEAN)

    log2 = load_log("exp2_backdoor_attack")
    if log2:
        curves["Backdoor Poisoned"] = (log2["results"]["training_history"], C_POISON)

    log1 = load_log("exp1_label_flip_attack")
    if log1 and "poisoned_training_history" in log1["results"]:
        curves["Label-Flip Poisoned"] = (log1["results"]["poisoned_training_history"], "#FF9800")

    log4 = load_log("exp4_cleaning_and_retrain")
    if log4 and "cleaned_training_history" in log4["results"]:
        curves["Post-Cleaning"] = (log4["results"]["cleaned_training_history"], C_CLEANED)

    # Also try exp5 as a fallback
    log5 = load_log("exp5_full_demo")
    if log5:
        r = log5["results"]
        if "Clean Baseline" not in curves and "backdoor_training_history" in r:
            curves["Backdoor Poisoned"] = (r["backdoor_training_history"], C_POISON)
        if "Post-Cleaning" not in curves and "cleaned_training_history" in r:
            curves["Post-Cleaning"] = (r["cleaned_training_history"], C_CLEANED)

    if not curves:
        print("  [Skip] No training history logs found for training curves.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, (hist, colour) in curves.items():
        epochs = [h["epoch"] for h in hist]
        losses = [h["loss"] for h in hist]
        accs = [h["accuracy"] for h in hist]
        ax1.plot(epochs, losses, label=label, color=colour, linewidth=1.8)
        ax2.plot(epochs, accs, label=label, color=colour, linewidth=1.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Accuracy (%)")
    ax2.set_title("Training Accuracy Over Epochs")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Training Curves — Baseline vs Poisoned vs Cleaned", fontsize=14)
    fig.tight_layout()
    savefig(fig, "graph_training_curves.png")


# ════════════════════════════════════════════════════════════════════
# 2. Accuracy comparison bar chart
# ════════════════════════════════════════════════════════════════════

def plot_accuracy_comparison():
    """Bar chart: test accuracy for baseline, label-flip, backdoor, cleaned."""
    bars = {}

    log0 = load_log("exp0_clean_baseline")
    if log0:
        bars["Clean\nBaseline"] = (log0["results"]["test_accuracy"], C_CLEAN)

    log1 = load_log("exp1_label_flip_attack")
    if log1:
        bars["Label-Flip\nPoisoned"] = (log1["results"].get("poisoned_test_accuracy", 0), "#FF9800")

    log2 = load_log("exp2_backdoor_attack")
    if log2:
        bars["Backdoor\nPoisoned"] = (log2["results"]["test_accuracy"], C_POISON)

    log4 = load_log("exp4_cleaning_and_retrain")
    if log4:
        bars["After\nCleaning"] = (log4["results"]["cleaned_test_accuracy"], C_CLEANED)

    if not bars:
        print("  [Skip] No accuracy logs found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(bars.keys())
    vals = [bars[n][0] for n in names]
    colours = [bars[n][1] for n in names]

    b = ax.bar(names, vals, color=colours, width=0.55, edgecolor="white", linewidth=1.2)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Test Accuracy: Baseline vs Poisoned vs Cleaned", fontsize=14)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "graph_accuracy_comparison.png")


# ════════════════════════════════════════════════════════════════════
# 3. ASR before / after cleaning
# ════════════════════════════════════════════════════════════════════

def plot_asr_comparison():
    """Bar chart: Attack Success Rate before and after cleaning."""
    log4 = load_log("exp4_cleaning_and_retrain")
    log5 = load_log("exp5_full_demo")
    src = log4 or log5
    if not src:
        print("  [Skip] No ASR logs found (need exp4 or exp5).")
        return

    r = src["results"]
    asr_before = r.get("asr_before_cleaning", 0)
    asr_after = r.get("asr_after_cleaning", 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Before\nCleaning", "After\nCleaning"],
        [asr_before, asr_after],
        color=[C_POISON, C_CLEAN], width=0.45,
        edgecolor="white", linewidth=1.2,
    )
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", fontsize=12,
                fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Backdoor ASR: Before vs After Cleaning", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "graph_asr_comparison.png")


# ════════════════════════════════════════════════════════════════════
# 4. Detection method comparison (precision, recall, F1)
# ════════════════════════════════════════════════════════════════════

def plot_detector_comparison():
    """Grouped bar chart comparing all four detection methods."""
    log3 = load_log("exp3_detection_pipeline")
    log5 = load_log("exp5_full_demo")
    src = log3 or log5
    if not src or "detectors" not in src["results"]:
        print("  [Skip] No detector logs found (need exp3 or exp5).")
        return

    detectors = src["results"]["detectors"]
    # Map internal keys to display names
    name_map = {
        "loss_outlier": "Loss\nOutlier",
        "spectral_signatures": "Spectral\nSignatures",
        "activation_clustering": "Activation\nClustering",
        "knn_label_agreement": "KNN Label\nAgreement",
    }

    names = []
    precision_vals = []
    recall_vals = []
    f1_vals = []

    for key, display in name_map.items():
        if key in detectors:
            d = detectors[key]
            names.append(display)
            precision_vals.append(d["precision"])
            recall_vals.append(d["recall"])
            f1_vals.append(d["f1"])

    if not names:
        print("  [Skip] No detector metrics in logs.")
        return

    x = np.arange(len(names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, precision_vals, width, label="Precision",
           color="#66BB6A", edgecolor="white")
    ax.bar(x, recall_vals, width, label="Recall",
           color="#42A5F5", edgecolor="white")
    ax.bar(x + width, f1_vals, width, label="F1 Score",
           color="#FFA726", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Detection Method Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "graph_detector_comparison.png")


# ════════════════════════════════════════════════════════════════════
# 5. Per-class accuracy: clean vs label-flip
# ════════════════════════════════════════════════════════════════════

def plot_per_class_accuracy():
    """Grouped bar chart: per-class accuracy before and after flipping."""
    log1 = load_log("exp1_label_flip_attack")
    if not log1:
        print("  [Skip] No per-class logs found (need exp1).")
        return

    r = log1["results"]
    clean = r.get("per_class_clean", {})
    flipped = r.get("per_class_poisoned", {})

    if not clean or not flipped:
        print("  [Skip] Per-class data missing in exp1 log.")
        return

    classes = list(clean.keys())
    clean_vals = [clean[c] for c in classes]
    flip_vals = [flipped[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, clean_vals, width, label="Clean", color=C_CLEAN,
           edgecolor="white")
    ax.bar(x + width / 2, flip_vals, width, label="Label-Flip Poisoned",
           color=C_POISON, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Test Accuracy: Clean vs Label-Flip Attack", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "graph_per_class_accuracy.png")


# ════════════════════════════════════════════════════════════════════
# 6. Combined dashboard: 4-panel summary
# ════════════════════════════════════════════════════════════════════

def plot_dashboard():
    """4-panel summary figure pulling from all available logs."""
    log0 = load_log("exp0_clean_baseline")
    log2 = load_log("exp2_backdoor_attack")
    log4 = load_log("exp4_cleaning_and_retrain")
    log5 = load_log("exp5_full_demo")

    # Gather what we can
    clean_acc = log0["results"]["test_accuracy"] if log0 else None
    bd_acc = log2["results"]["test_accuracy"] if log2 else None

    src = log4 or log5
    cleaned_acc = src["results"].get("cleaned_test_accuracy") if src else None
    asr_before = src["results"].get("asr_before_cleaning") if src else None
    asr_after = src["results"].get("asr_after_cleaning") if src else None

    if not any([clean_acc, bd_acc, cleaned_acc]):
        print("  [Skip] Insufficient logs for dashboard.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Test accuracy bars
    ax = axes[0, 0]
    labels, vals, colors = [], [], []
    if clean_acc is not None:
        labels.append("Clean"); vals.append(clean_acc); colors.append(C_CLEAN)
    if bd_acc is not None:
        labels.append("Poisoned"); vals.append(bd_acc); colors.append(C_POISON)
    if cleaned_acc is not None:
        labels.append("Cleaned"); vals.append(cleaned_acc); colors.append(C_CLEANED)
    b = ax.bar(labels, vals, color=colors, width=0.5)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Test Accuracy", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: ASR bars
    ax = axes[0, 1]
    if asr_before is not None and asr_after is not None:
        b = ax.bar(["Before", "After"], [asr_before, asr_after],
                   color=[C_POISON, C_CLEAN], width=0.45)
        for bar in b:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=10,
                    fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Backdoor ASR", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Training curves (loss)
    ax = axes[1, 0]
    curve_data = {}
    if log0 and "training_history" in log0["results"]:
        curve_data["Clean"] = (log0["results"]["training_history"], C_CLEAN)
    if log2 and "training_history" in log2["results"]:
        curve_data["Poisoned"] = (log2["results"]["training_history"], C_POISON)
    if src and "cleaned_training_history" in src["results"]:
        curve_data["Cleaned"] = (src["results"]["cleaned_training_history"], C_CLEANED)
    for label, (hist, col) in curve_data.items():
        ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist],
                label=label, color=col, lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: Detector F1 scores
    ax = axes[1, 1]
    log3 = load_log("exp3_detection_pipeline")
    det_src = log3 or log5
    if det_src and "detectors" in det_src["results"]:
        dets = det_src["results"]["detectors"]
        det_names = {
            "loss_outlier": "Loss",
            "spectral_signatures": "Spectral",
            "activation_clustering": "Clustering",
            "knn_label_agreement": "KNN",
        }
        d_labels, d_f1 = [], []
        for key, name in det_names.items():
            if key in dets:
                d_labels.append(name)
                d_f1.append(dets[key]["f1"])
        if d_labels:
            ax.barh(d_labels, d_f1, color=C_DETECTORS[:len(d_labels)],
                    height=0.5, edgecolor="white")
            ax.set_xlim(0, 1.05)
            for i, v in enumerate(d_f1):
                ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("F1 Score")
    ax.set_title("Detector Performance", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Data Poisoning & Detection — Summary Dashboard", fontsize=16, y=1.01)
    fig.tight_layout()
    savefig(fig, "graph_dashboard.png")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Generating Graphs from Experiment Logs")
    print("=" * 60)
    print(f"  Log directory  : {config.LOG_DIR}")
    print(f"  Plot directory : {config.PLOT_DIR}")

    # List available logs
    logs = load_all_logs()
    if not logs:
        print("\n  ⚠  No log files found in outputs/logs/.")
        print("  Run experiments first:  python run_all.py")
        return

    print(f"\n  Found {len(logs)} log(s): {list(logs.keys())}\n")

    # Generate each graph (skips gracefully if data is missing)
    plot_training_curves()
    plot_accuracy_comparison()
    plot_asr_comparison()
    plot_detector_comparison()
    plot_per_class_accuracy()
    plot_dashboard()

    print(f"\n  All graphs saved to: {config.PLOT_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
