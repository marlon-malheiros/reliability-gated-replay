"""Method registry: ``method`` config block -> ContinualMethod instance.

Standard-Adam / Standard-SGD / Dropout map to the same ``Finetune`` logic and
differ only in the optimizer / model config (so they are distinct *labels* in the
results, not distinct training code).

The concrete baseline classes are imported **lazily** (inside ``_registry``) rather
than at module load. ``baselines.*`` import back from ``methods.*``, so an eager
import here created a circular import that only resolved when a caller happened to
import ``methods`` before ``baselines``. Deferring the imports breaks the cycle for
every entry point.
"""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import ContinualMethod

_REGISTRY_CACHE: "Dict[str, Type[ContinualMethod]] | None" = None


def _registry() -> "Dict[str, Type[ContinualMethod]]":
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        from baselines.derpp import DERPP
        from baselines.er_ace import ERACE
        from baselines.ewc import EWC
        from baselines.finetune import EarlyStopping, Finetune
        from baselines.gated_replay import GatedReplay
        from baselines.l2 import L2
        from baselines.lwf import LwF
        from baselines.mas import MAS
        from baselines.prog_freeze import ProgressiveFreezing
        from baselines.replay import Replay
        from baselines.si import SI

        from .pnn.method import PNNMethod

        _REGISTRY_CACHE = {
            "finetune": Finetune,
            "adam": Finetune,
            "sgd": Finetune,
            "dropout": Finetune,
            "early_stopping": EarlyStopping,
            "l2": L2,
            "derpp": DERPP,
            "ewc": EWC,
            "si": SI,
            "mas": MAS,
            "replay": Replay,
            "er_ace": ERACE,
            "gated_replay": GatedReplay,
            "lwf": LwF,
            "prog_freeze": ProgressiveFreezing,
            "pnn": PNNMethod,
        }
    return _REGISTRY_CACHE


def build_method(cfg: Dict[str, Any]) -> ContinualMethod:
    name = cfg.get("name")
    reg = _registry()
    if name not in reg:
        raise KeyError(f"Unknown method '{name}'. Known: {sorted(reg)}")
    return reg[name](cfg)


def available_methods():
    return sorted(_registry())
