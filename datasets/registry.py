"""Dataset registry: resolve config -> ContinualBenchmark.

Separates *source images* (mnist / fashion_mnist / synthetic) from *protocol*
(single-task / split / permuted). This is what lets Split-MNIST run on the
synthetic fallback when MNIST cannot be downloaded -- the offline smoke path.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .base import ContinualBenchmark
from .idx import load_idx_dataset
from .noise import apply_label_noise
from .permuted_mnist import build_permuted_mnist
from .rotated_mnist import build_rotated_mnist
from .single_task import build_single_task
from .cifar10n import build_cifar10n
from .feat_cifar10 import build_feat_cifar10
from .split_cifar10 import build_rotated_cifar10, build_split_cifar10
from .split_cifar100 import build_split_cifar100
from .split_mnist import build_split_mnist
from .synthetic import make_synthetic

# protocol name -> default source images
_DEFAULT_SOURCE = {
    "mnist": "mnist",
    "fashion_mnist": "fashion_mnist",
    "split_mnist": "mnist",
    "split_fashion_mnist": "fashion_mnist",
    "permuted_mnist": "mnist",
    "rotated_mnist": "mnist",
    "split_cifar10": "cifar10",
    "rotated_cifar10": "cifar10",
    "feat_cifar10": "cifar10",
    "split_cifar100": "cifar100",
    "cifar10n": "cifar10",
}

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_source_images(source: str, cfg: Dict[str, Any], data_root: str | Path) -> Arrays:
    """Load base train/test image arrays, with optional synthetic fallback."""
    if source in ("cifar10", "cifar100"):
        raise RuntimeError(
            f"{source} is loaded by its split_{source} builder, not load_source_images."
        )
    if source == "synthetic":
        return make_synthetic(seed=int(cfg.get("seed", 0)))
    try:
        return load_idx_dataset(
            source, data_root, allow_download=bool(cfg.get("allow_download", True))
        )
    except Exception as e:  # network/file failure
        if cfg.get("fallback_synthetic", False):
            warnings.warn(
                f"Could not load '{source}' ({e}); falling back to synthetic data. "
                "Results are placeholder-only.",
                RuntimeWarning,
            )
            return make_synthetic(seed=int(cfg.get("seed", 0)))
        raise


def build_benchmark(cfg: Dict[str, Any], data_root: str | Path = "data") -> ContinualBenchmark:
    """Construct a benchmark from a ``data:`` config block.

    If a ``label_noise`` sub-block is present, training labels are corrupted after
    construction (val/test stay clean) and clean-label / noise-mask bookkeeping is
    attached to each task -- see ``datasets/noise.py``.
    """
    name = cfg.get("name")
    if name not in _DEFAULT_SOURCE:
        raise KeyError(
            f"Unknown dataset '{name}'. Known: {sorted(_DEFAULT_SOURCE)}"
        )
    if name == "split_cifar10":
        bench = build_split_cifar10(cfg, data_root=data_root)
    elif name == "rotated_cifar10":
        bench = build_rotated_cifar10(cfg, data_root=data_root)
    elif name == "feat_cifar10":
        bench = build_feat_cifar10(cfg, data_root=data_root)
    elif name == "split_cifar100":
        bench = build_split_cifar100(cfg, data_root=data_root)
    elif name == "cifar10n":
        bench = build_cifar10n(cfg, data_root=data_root)  # carries its own noise bookkeeping
    else:
        source = cfg.get("source") or _DEFAULT_SOURCE[name]
        tr_x, tr_y, te_x, te_y = load_source_images(source, cfg, data_root)
        if name in ("mnist", "fashion_mnist"):
            bench = build_single_task(name, tr_x, tr_y, te_x, te_y, cfg)
        elif name in ("split_mnist", "split_fashion_mnist"):
            bench = build_split_mnist(tr_x, tr_y, te_x, te_y, cfg)
        elif name == "permuted_mnist":
            bench = build_permuted_mnist(tr_x, tr_y, te_x, te_y, cfg)
        elif name == "rotated_mnist":
            bench = build_rotated_mnist(tr_x, tr_y, te_x, te_y, cfg)
        else:
            raise KeyError(name)  # unreachable

    if cfg.get("label_noise"):
        bench = apply_label_noise(bench, cfg["label_noise"], seed=int(cfg.get("seed", 0)))
    return bench
