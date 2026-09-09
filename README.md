# Data Poisoning Attack & Detection Scanner

A complete pipeline for injecting data-poisoning attacks into CIFAR-10 and detecting the poisoned samples before they compromise a model.

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
│   └── generate_graphs.py            # Build comparison charts from logs
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
Flips labels of one class (airplane → truck) at 5% poison rate. A KNN label-agreement detector identifies suspicious labels, and the clean reference model supplies replacement predictions before retraining the label-corrected model.

### Experiment 2: Backdoor Attack (BadNets-style)
Injects a 3×3 white trigger patch into 5% of training images and relabels them. Shows the model achieves normal test accuracy but high Attack Success Rate (ASR) when the trigger is present.

The attacks use established benchmark patterns: class-conditional random label flipping and a fixed-patch BadNets-style backdoor. The normal test-accuracy chart does not apply the trigger, so a successful backdoor can have accuracy similar to the clean model; ASR on triggered images is the security measure.

### Experiment 3: Detection Pipeline
Runs four independent detectors against the backdoor-poisoned dataset:
1. **Loss Outlier** — flags high-loss samples after training
2. **Spectral Signatures** — SVD-based outlier scores per class
3. **Activation Clustering** — PCA + KMeans minority cluster
4. **KNN Label Agreement** — flags samples whose neighbours disagree on label

Reports precision, recall, F1, and FPR for each.

### Experiment 4: Cleaning & Retraining
Combines the best detectors (Spectral ∪ KNN), removes flagged samples, retrains from scratch, and verifies the backdoor ASR drops to near zero.

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
