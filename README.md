# Data Poisoning Attack & Detection Scanner

A complete pipeline for injecting data-poisoning attacks into CIFAR-10 and detecting the poisoned samples before they compromise a model.

## Judge-ready campaign: what was measured

The latest controlled campaign is implemented on branch
`codex/trusted-data-defense-pipeline`. It uses a fixed stratified split of 45,000
attack-pool images, 2,000 trusted clean images and 3,000 development images;
the untouched 10,000-image CIFAR-10 test set is used only for final evaluation.
Sample IDs remain the original CIFAR-10 IDs. The defenses receive observed
images and labels plus the trusted subset; poison membership and original clean
labels are evaluation-only metadata.

The campaign ran ResNet-18 for 30 epochs with fixed seeds and separate output
directories for label and backdoor conditions. The 5% results below are means
across seeds 42, 43 and 44; values are percentages and “pp” means percentage
points.

| Measure | Baseline | Poisoned | Selected correction | Change from poisoned |
|---|---:|---:|---:|---:|
| Label-flip normal accuracy | 88.94 | 86.61 | 87.78 | +1.17 pp |
| Backdoor normal accuracy | 88.94 | 89.89 | 88.95 | −0.94 pp |
| Backdoor ASR (lower is better) | — | 96.31 | 83.62 | −12.69 pp |

The label defense recovered 1.17 pp on average but remained 1.16 pp below
baseline. The backdoor defense preserved normal accuracy within 1 pp of
baseline on average, but ASR remained far above the ≤5% target. The targets are
reported as acceptance criteria rather than treated as guaranteed outcomes.
Generic detector precision, recall, F1 and FPR were not persisted by this
campaign, so the false-negative target is not claimed as passed. The earlier
historical run did report weak generic detector performance; its known-trigger
filter reduced ASR to about 1.40% but is a demonstration that assumes the
trigger is known, not a general unknown-trigger defense.

The complete judge-readable artifacts are tracked in the repository under
`outputs/layered_campaign/`: `results_summary.csv`,
`label_accuracy_summary.png`, and `backdoor_accuracy_summary.png`. Detailed
context and limitations are in `ANALYSIS_LATEST_CAMPAIGN.md`.

### Reproduce the controlled campaign

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
bash scripts/run_layered_campaign.sh
```

The shell script uses the GPU when CUDA is available and writes progress to
`outputs/layered_campaign/campaign.log`. Smoke checks can be run with
`--epochs 2` through `scripts/run_trusted_campaign.py`; smoke outputs must not
be used as final evidence.

## Project Structure

```
Cyber-Defense/
├── config.py                          # Central configuration & hardware opts
├── requirements.txt                   # Python dependencies
├── run_all.py                         # Run all (or selected) experiments
│
├── attacks/
│   ├── label_flip.py                  # Label-flipping attack
│   └── backdoor.py                    # Backdoor trigger-patch attack
│
├── models/
│   └── resnet.py                      # CIFAR-10 ResNet-18
│
├── detection/
│   ├── loss_outlier.py                # Per-sample loss outlier detection
│   ├── spectral_signatures.py         # Spectral-signature detection (SVD)
│   ├── activation_clustering.py       # PCA + KMeans activation clustering
│   └── nn_label_agreement.py          # KNN label-agreement check
│
├── utils/
│   ├── data_loader.py                 # CIFAR-10 loading & poisoned wrappers
│   ├── train_eval.py                  # Training (AMP) & evaluation helpers
│   ├── metrics.py                     # Detection quality metrics
│   ├── logger.py                      # Structured JSON experiment logger
│   └── visualization.py              # In-experiment plotting utilities
│
├── experiments/
│   ├── exp0_clean_baseline.py         # Train clean CIFAR-10 baseline
│   ├── exp1_label_flip_attack.py      # Demonstrate label-flip attack
│   ├── exp2_backdoor_attack.py        # Demonstrate backdoor attack + ASR
│   ├── exp3_detection_pipeline.py     # Run & compare all 4 detectors
│   ├── exp4_cleaning_and_retrain.py   # Clean data, retrain, verify fix
│   └── exp5_full_demo.py             # End-to-end demo (all phases)
│
├── scripts/
│   ├── generate_graphs.py            # Build historical log comparison charts
│   ├── run_trusted_campaign.py      # One controlled label/backdoor condition
│   ├── run_layered_campaign.sh      # Full 30-epoch campaign
│   └── summarize_trusted_campaign.py # Judge-ready CSV and graphs
│
└── outputs/                           # Generated at runtime
    ├── models/                        # Saved model checkpoints (.pt)
    ├── plots/                         # All generated PNG charts
    └── logs/                          # Structured JSON logs (one per experiment)
```

## Hardware Requirements

| Component | Spec | How it's used |
|-----------|------|---------------|
| GPU | NVIDIA RTX Ada 4500 (24 GB VRAM) | AMP training, TF32 tensor cores, large batches |
| CPU | AMD Threadripper (~200 GB RAM) | 16 data-loader workers, in-memory KNN, SVD |

## Quick Start

```bash
# 1. Install dependencies (a virtual environment is recommended)
python -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Run all experiments end-to-end (exp0 through exp5)
./venv/bin/python run_all.py

# 3. Generate comparison graphs from the logs
./venv/bin/python scripts/generate_graphs.py
```

## Running Individual Experiments

Each experiment is a standalone script that can be run independently:

```bash
./venv/bin/python -m experiments.exp0_clean_baseline      # clean baseline (run first)
./venv/bin/python -m experiments.exp1_label_flip_attack   # label-flip attack
./venv/bin/python -m experiments.exp2_backdoor_attack     # backdoor trigger attack
./venv/bin/python -m experiments.exp3_detection_pipeline  # all 4 detection methods
./venv/bin/python -m experiments.exp4_cleaning_and_retrain # clean + retrain
./venv/bin/python -m experiments.exp5_full_demo           # full end-to-end demo
```

Or pick specific ones via `run_all.py`:

```bash
./venv/bin/python run_all.py 0 2 3    # run only experiments 0, 2, and 3
```

## Logging

Every experiment writes a structured JSON log to `outputs/logs/`:

```
outputs/logs/
├── exp0_clean_baseline.json
├── exp1_label_flip_attack.json
├── exp2_backdoor_attack.json
├── exp3_detection_pipeline.json
├── exp4_cleaning_and_retrain.json
└── exp5_full_demo.json
```

Each JSON file contains:
- **config**: seed, epochs, batch size, learning rate, poison rate actually used by that experiment, device
- **results**: test accuracy, training history (per-epoch loss/accuracy), per-class accuracy, ASR, detection metrics (precision/recall/F1/FPR)
- **timing**: start time, end time, elapsed seconds

## Graph Generation

After running experiments, generate all comparison charts at once:

```bash
./venv/bin/python scripts/generate_graphs.py
```

This produces the comparison charts in `outputs/plots/`:

| Chart | What it shows |
|-------|---------------|
| `graph_training_curves.png` | Loss & accuracy over epochs for baseline, poisoned, and cleaned models |
| `graph_accuracy_comparison.png` | Test accuracy bars: clean vs label-flip vs backdoor vs cleaned |
| `graph_asr_comparison.png` | Backdoor Attack Success Rate before vs after cleaning |
| `graph_detector_comparison.png` | Precision / Recall / F1 for all 4 detection methods |
| `graph_per_class_accuracy.png` | Per-class accuracy: clean vs label-flip poisoned |
| `graph_dashboard.png` | 4-panel summary combining accuracy, ASR, training curves, and detector F1 |
| `graph_label_flip_detector_corrector.png` | Label-flip accuracy recovery and KNN detector-corrector metrics |
| `graph_backdoor_detector_corrector.png` | Backdoor accuracy, ASR, and Spectral ∪ KNN cleaning result |
| `graph_differential_accuracy.png` | Baseline / Label / Backdoor accuracy before and after detector-correction |

The script gracefully skips any chart whose prerequisite logs are missing.

## Experiments

### Experiment 0: Clean Baseline
Trains a ResNet-18 on unmodified CIFAR-10. Records per-epoch training history and per-class test accuracy. This is the reference for all comparisons.

The clean baseline has a 0% poison rate. The global `POISON_RATE = 0.05` setting is an attack default and does not modify this experiment; its log explicitly records `poison_rate: 0.0`.

### Experiment 1: Label-Flip Attack
The attack flips airplane labels to truck labels in the attack pool. The
controlled defense uses three-fold Cleanlab out-of-fold probabilities and
compares confidence-based replacement with issue removal. The clean model is
an evaluation baseline; it is not used as a label-correction teacher.

### Experiment 2: Backdoor Attack (BadNets-style)
Injects a 3×3 white trigger patch into 5% of training images and relabels them. Shows the model achieves normal test accuracy but high Attack Success Rate (ASR) when the trigger is present.

The attacks use established benchmark patterns: class-conditional random label flipping and a fixed-patch BadNets-style backdoor. The normal test-accuracy chart does not apply the trigger, so a successful backdoor can have accuracy similar to the clean model; ASR on triggered images is the security measure.

### Experiment 3: Detection Pipeline
The historical detector comparison runs four independent detectors against the
backdoor-poisoned dataset:
1. **Loss Outlier** — flags high-loss samples after training
2. **Spectral Signatures** — SVD-based outlier scores per class
3. **Activation Clustering** — PCA + KMeans minority cluster
4. **KNN Label Agreement** — flags samples whose neighbours disagree on label

Reports precision, recall, F1, and FPR for each. Those historical detector
results are retained for comparison; the newer layered campaign does not claim
that its false-negative target passed until those metrics are persisted in the
same final run.

### Experiment 4: Cleaning & Retraining
The controlled campaign compares trusted-data fine-tuning, Fine-Pruning and
FT-SAM, starting from the same poisoned checkpoint. The known-trigger filter
and random-removal comparison remain historical demonstrations. In the latest
campaign, model mitigation reduced ASR but did not meet the ≤5% target, which
is shown in the judge-ready table instead of being hidden by selecting the
best-looking run.

Because this experiment uses detector output rather than the ground-truth poison set, false positives can reduce clean-test accuracy. This is an important result of the benchmark: backdoor removal can succeed while the detector still needs better precision. The reported detection metrics should therefore be considered part of the result, not evidence that every flagged sample is poisoned.

### Experiment 5: Full Demo
All four phases in one script:
1. Train on poisoned data (looks normal)
2. Trigger the backdoor (high ASR)
3. Run the scanner (detection metrics)
4. Retrain on clean data (trigger fails)

## Key Design Decisions

- **AMP + TF32**: All training uses mixed precision and TF32 math for ~2× throughput on Ada tensor cores.
- **`torch.compile`**: ResNet is compiled with `reduce-overhead` mode for kernel fusion.
- **`persistent_workers=True`**: DataLoader workers stay alive between epochs (avoids respawn overhead on Threadripper).
- **`n_jobs=-1`** in sklearn KNN: uses all Threadripper cores for neighbour search.
- **Deterministic detection**: Detection passes use a separate no-augmentation loader so results are reproducible.
- **Centralised logging**: All results go to `outputs/logs/` as JSON, enabling reproducible graph generation at any time.

## References

- Tran, B., Li, J., & Madry, A. (2018). *Spectral Signatures in Backdoor Attacks*. NeurIPS.
- Chen, B., et al. (2019). *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering*.
- Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). *BadNets: Identifying Vulnerabilities in the ML Supply Chain*.
# Validation notes

The original pre-validation results are preserved on
`codex/pre-validation-snapshot-2026-09-09`. Those results are historical: the
old cleaning loaders rebuilt data from the clean CIFAR-10 labels, so missed
poison could disappear during retraining. The validated branch keeps the
observed poisoned image and label for every unselected sample.

Label correction uses Cleanlab Confident Learning with out-of-fold model
probabilities. Backdoor detection uses spectral signatures as the selected
cleaning policy, with activation clustering, loss outliers and KNN retained as
comparison baselines. Detection reports are compared with matched random
removal controls.

Install the complete environment with `pip install -r requirements.txt` and
run a reproducible experiment set with:

    python run_all.py 0 1 2 3 4 --seed 42 --poison-rate 0.05 \
        --run-dir outputs/validated/seed-42-rate-05

The untouched CIFAR-10 test set is used for final accuracy. Backdoor ASR is
reported separately on non-target test images after applying the trigger.
# SCD-2026
