"""CIFAR-10 in a frozen ImageNet-pretrained feature space (task-IL binary).

Tests the hypothesis that the gate fails on CIFAR because the from-scratch CNN
*memorizes* the noisy labels, corrupting the gate signal. Here the backbone (a
frozen ImageNet ResNet-18) never sees the noisy labels, so it can't memorize
them -- the modern "continual learning on top of a pretrained encoder" regime.
Each CIFAR image is passed once through the frozen backbone -> 512-d feature; the
CL problem then reduces to an MLP over those features, reusing the MNIST pipeline
(gates, noise, trainer) unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .base import ContinualBenchmark, Task, make_binary_tasks, subsample, to_tensor_dataset
from .split_cifar10 import _class_groups, _load_cifar10

_FEAT_CACHE: Dict[tuple, list] = {}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def _extract(x: torch.Tensor, device, batch: int = 256) -> np.ndarray:
    """x: float tensor (N,3,32,32) in [0,255] -> (N,512) ImageNet-ResNet18 features."""
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(net.children())[:-1]).to(device).eval()
    mean, std = _MEAN.to(device), _STD.to(device)
    out = []
    x = x / 255.0
    for i in range(0, x.shape[0], batch):
        xb = x[i : i + batch].to(device)
        xb = F.interpolate(xb, size=224, mode="bilinear", align_corners=False)
        xb = (xb - mean) / std
        out.append(backbone(xb).flatten(1).cpu())
    del backbone, net
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def build_feat_cifar10(cfg: Dict[str, Any], data_root: str | Path = "data") -> ContinualBenchmark:
    seed = int(cfg.get("seed", 0))
    cpt = int(cfg.get("classes_per_task", 2))
    n_classes = int(cfg.get("n_classes", 10))
    val_fraction = float(cfg.get("val_fraction", 0.05))
    max_train = cfg.get("max_train_per_task")
    max_test = cfg.get("max_test_per_task")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (seed, cpt, max_train, max_test, val_fraction)

    if key not in _FEAT_CACHE:
        tr_x, tr_y, te_x, te_y = _load_cifar10(data_root, download=bool(cfg.get("allow_download", True)))
        pairs = _class_groups(n_classes, cpt, seed, shuffle=False)
        # use the shared binary-task splitter (subsample + local-label remap), then
        # replace each split's raw images with frozen features.
        raw = make_binary_tasks(tr_x, tr_y, te_x, te_y, pairs, val_fraction, seed, max_train)
        cached = []
        for t in raw:
            trx, trY = t.train.tensors
            vax, vaY = t.val.tensors
            tex, teY = t.test.tensors
            tex2, teY2 = subsample(tex.numpy(), teY.numpy(), max_test, seed + 7)
            cached.append((
                _extract(trx, device), trY.numpy(),
                _extract(vax, device), vaY.numpy(),
                _extract(torch.from_numpy(tex2).float(), device), teY2,
            ))
        _FEAT_CACHE[key] = cached

    tasks = []
    for tid, (trf, trY, vaf, vaY, tef, teY) in enumerate(_FEAT_CACHE[key]):
        tasks.append(
            Task(
                task_id=tid,
                name=f"featcifar_t{tid}",
                train=to_tensor_dataset(trf, trY),
                val=to_tensor_dataset(vaf, vaY),
                test=to_tensor_dataset(tef, teY),
                global_classes=[2 * tid, 2 * tid + 1],
                n_classes=cpt,
            )
        )
    return ContinualBenchmark(
        name="feat_cifar10",
        tasks=tasks,
        input_shape=(trf.shape[1],),  # (512,)
        n_classes_per_task=cpt,
        multihead=True,
    )
