"""Train/val split: val uses a *clean* (no augmentation) dataset aligned with test/inference."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import Subset

from baseline.impact_dataset import BadmintonImpactDataset


def build_val_dataset_no_augment(train_ds: BadmintonImpactDataset, val_idx: List[int]) -> BadmintonImpactDataset:
    """
    Same samples as ``Subset(train_ds, val_idx)`` but **augment=False**, matching ``test_ds`` / inference.
    Using ``Subset`` on an augmented ``train_ds`` caused val metrics to be computed on *random augmentations*,
    inflating R² vs clean test evaluation.
    """
    val_paths = [train_ds.paths[i] for i in val_idx]
    return BadmintonImpactDataset(
        val_paths,
        predict_fz_only=train_ds.predict_fz_only,
        augment=False,
        ram_cache=False,
    )


def split_train_val(
    ds: BadmintonImpactDataset, val_ratio: float = 0.15
) -> Tuple[Subset, BadmintonImpactDataset]:
    """Per-subject stratified split; val loader must use **no augmentation** (see ``build_val_dataset_no_augment``)."""
    by_sub: dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(ds.paths):
        m = re.match(r"(sub_\d+)", Path(p).name)
        sub = m.group(1) if m else "unknown"
        by_sub[sub].append(i)
    train_idx, val_idx = [], []
    rng = np.random.default_rng(42)
    for _sub, indices in by_sub.items():
        arr = np.array(indices)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_ratio))
        val_idx.extend(arr[:n_val].tolist())
        train_idx.extend(arr[n_val:].tolist())
    train_sub = Subset(ds, train_idx)
    val_ds = build_val_dataset_no_augment(ds, val_idx)
    return train_sub, val_ds
