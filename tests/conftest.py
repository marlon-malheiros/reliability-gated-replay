"""Pytest fixtures + path setup for the PNN-consolidation project."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.registry import build_benchmark  # noqa: E402
from models.registry import build_model  # noqa: E402


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tiny_bench():
    cfg = dict(
        name="split_mnist", source="synthetic", seed=0,
        val_fraction=0.2, max_train_per_task=80, max_test_per_task=40,
    )
    return build_benchmark(cfg, data_root=str(ROOT / "data"))


@pytest.fixture
def tiny_model(tiny_bench):
    return build_model({"name": "mlp"}, tiny_bench)


@pytest.fixture
def train_cfg():
    return dict(
        train=dict(epochs=2, batch_size=32, eval_batch_size=64, probe_size=20),
        optimizer=dict(name="adam", lr=1e-3),
    )
