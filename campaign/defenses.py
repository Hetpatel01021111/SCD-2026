"""Established, auditable defense implementations used by the campaign."""

from copy import deepcopy
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import torch
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset

import config
from campaign.data import ImmutableObservedDataset, TrustedDataset
from models.resnet import build_resnet18
from utils.train_eval import train_model, predict_indexed_probabilities


def _loader(dataset, batch, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      num_workers=min(config.NUM_WORKERS, 8),
                      pin_memory=config.PIN_MEMORY, persistent_workers=False,
                      generator=generator)


def oof_cleanlab(base, attack_ids, observed_labels, trusted_ids, seed, epochs):
    """Create OOF probabilities; each sample is scored by a model that did not see it."""
    ids = np.asarray(attack_ids, dtype=np.int64)
    y = np.asarray([observed_labels[int(i)] for i in ids], dtype=np.int64)
    probs = np.zeros((len(ids), 10), dtype=np.float32)
    folds = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (train_pos, held_pos) in enumerate(skf.split(ids, y)):
        held_ids = ids[held_pos].tolist()
        train_ids = ids[train_pos].tolist()
        train_ds = ImmutableObservedDataset(base, train_ids, observed_labels, set(), "label", True)
        trusted_ds = TrustedDataset(base, trusted_ids)
        model = build_resnet18(compile_model=False)
        model, history = train_model(model, _loader(ConcatDataset([train_ds, trusted_ds]), config.BATCH_SIZE, True, seed + fold), epochs=epochs)
        held_ds = ImmutableObservedDataset(base, held_ids, observed_labels, set(), "label", False)
        indexed = predict_indexed_probabilities(model, _loader(held_ds, config.EVAL_BATCH_SIZE, False, seed))
        pos_by_id = {int(sample_id): pos for pos, sample_id in enumerate(ids)}
        for sample_id, row in indexed.items():
            probs[pos_by_id[int(sample_id)]] = row
        folds.append({"fold": fold, "held_out_ids": held_ids, "history": history})
    issues = find_label_issues(labels=y, pred_probs=probs,
                               return_indices_ranked_by="normalized_margin")
    return ids, y, probs, issues, folds


def label_candidates(ids, labels, probs, issues, confidences=(0.70, 0.80, 0.90)):
    candidates = {}
    issue_set = set(int(i) for i in issues)
    for confidence in confidences:
        corrections = {}
        for pos in issue_set:
            predicted = int(probs[pos].argmax())
            if predicted != int(labels[pos]) and float(probs[pos, predicted]) >= confidence:
                corrections[int(ids[pos])] = predicted
        candidates[f"cleanlab_replace_{confidence:.2f}"] = {"corrections": corrections, "remove": set()}
    candidates["cleanlab_remove"] = {"corrections": {}, "remove": {int(ids[pos]) for pos in issue_set}}
    return candidates


def attach_pruning_mask(model, trusted_loader, fraction: float):
    """Fine-Pruning: mask least-active penultimate channels and enforce the mask."""
    model = model.to(config.DEVICE)
    model.eval()
    activations = []
    with torch.no_grad():
        for batch in trusted_loader:
            features = model.extract_features(batch[0].to(config.DEVICE))
            activations.append(features.abs().mean(dim=0).cpu())
    activity = torch.stack(activations).mean(dim=0)
    n_prune = int(len(activity) * fraction)
    mask = torch.ones_like(activity)
    if n_prune:
        mask[torch.argsort(activity)[:n_prune]] = 0.0
    mask = mask.to(config.DEVICE)

    def hook(_module, inputs):
        return (inputs[0] * mask,)
    handle = model.linear.register_forward_pre_hook(hook)
    with torch.no_grad():
        model.linear.weight.mul_(mask.view(1, -1).to(model.linear.weight.device))
    return mask, handle


def enforce_pruning(model, mask):
    with torch.no_grad():
        model.linear.weight.mul_(mask.view(1, -1).to(model.linear.weight.device))
        if model.linear.bias is not None:
            model.linear.bias.data.clamp_(-100, 100)
