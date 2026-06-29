"""Task-free continual-learning training loop.

The data-side counterpart of :class:`ContinualTrainer`: instead of iterating over
discrete tasks (with ``on_task_start``/``on_task_end`` boundary signals), it drives
a :class:`ContinualMethod` over a single blurry-boundary stream
(:func:`datasets.streams.build_task_free_stream`). The method only ever sees
``(x, y)`` batches and its own running loss -- no task labels. Consolidation is
driven autonomously through the new ``on_stream_step`` hook.

Per-task *test* sets are still used to measure per-task accuracy / forgetting at
periodic checkpoints, but they are never shown to the method. Single head: the
model is called as ``model(x)`` (task_id defaults to 0).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.base import ContinualBenchmark
from datasets.streams import build_task_free_stream
from utils.logging import get_logger

from .base import ContinualMethod
from .utils import build_optimizer, evaluate, protected_named_parameters


class StreamTrainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        benchmark: ContinualBenchmark,
        model: nn.Module,
        method: ContinualMethod,
        device: torch.device,
        stream_cfg: Dict[str, Any] | None = None,
        logger=None,
    ):
        self.cfg = cfg
        self.bench = benchmark
        self.model = model.to(device)
        self.method = method
        self.device = device
        self.log = logger or get_logger("stream_trainer")

        tcfg = cfg.get("train", {})
        self.batch_size = int(tcfg.get("batch_size", 128))
        self.eval_batch_size = int(tcfg.get("eval_batch_size", 256))
        self.probe_size = int(tcfg.get("probe_size", 200))
        self.grad_clip = tcfg.get("grad_clip")

        scfg = dict(stream_cfg or {})
        self.stream_cfg = scfg
        self.eval_period = max(int(scfg.get("eval_period", 200)), 1)
        self.stream = build_task_free_stream(benchmark, scfg)

        self._init_named = {
            n: p.detach().clone() for n, p in protected_named_parameters(self.model)
        }
        self._probe_x = self.bench.tasks[0].test.tensors[0][: self.probe_size].to(device)
        # nominal region boundaries in *step* units (skip step 0 -> random-init theta)
        self._boundary_steps = {
            b // self.batch_size for b in self.stream.boundaries[1:]
        }

    def _eval_all(self) -> List[float]:
        return [
            evaluate(self.model, t.test, t.task_id, self.device, self.eval_batch_size)["acc"]
            for t in self.bench.tasks
        ]

    def _grad_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        return sq**0.5

    def _weight_drift(self) -> float:
        cur = dict(self.model.named_parameters())
        diffs = [
            (cur[n].detach() - init).reshape(-1) for n, init in self._init_named.items()
        ]
        return float(torch.norm(torch.cat(diffs)).item()) if diffs else 0.0

    @torch.no_grad()
    def _probe_features(self) -> List[List[float]]:
        self.model.eval()
        feats = self.model.features(self._probe_x)
        return feats.detach().cpu().numpy().tolist()

    def train(self) -> Dict[str, Any]:
        T = len(self.bench.tasks)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        init_acc = self._eval_all()
        self.method.on_task_start(0, self.model)  # init once; no task boundaries
        optimizer = build_optimizer(self.model, self.cfg.get("optimizer", {}))
        n_steps = (len(self.stream) + self.batch_size - 1) // self.batch_size

        acc_checkpoints: List[Dict[str, Any]] = []
        gnorm_acc, gnorm_n = 0.0, 0
        t_start = time.time()
        step = 0
        self.model.train()
        for x, y, _region in self.stream.batches(self.batch_size):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.model(x)  # single head (task_id defaults to 0)
            loss = F.cross_entropy(out, y)
            total = loss + self.method.extra_loss(self.model, x, y, 0)
            total.backward()
            gnorm_acc += self._grad_norm()
            gnorm_n += 1
            self.method.modify_gradients(self.model)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            self.method.on_batch_end(self.model, x, y, 0)
            self.method.on_stream_step(
                self.model,
                step,
                x,
                y,
                float(loss.item()),
                n_steps=n_steps,
                at_boundary=(step in self._boundary_steps),
            )
            step += 1

            if step % self.eval_period == 0:
                accs = self._eval_all()
                acc_checkpoints.append(
                    {
                        "step": step,
                        "samples": step * self.batch_size,
                        "per_task_acc": accs,
                        "avg_acc": float(np.mean(accs)) if accs else 0.0,
                    }
                )
                self.model.train()

        final_acc = self._eval_all()
        acc_checkpoints.append(
            {
                "step": step,
                "samples": step * self.batch_size,
                "per_task_acc": final_acc,
                "avg_acc": float(np.mean(final_acc)) if final_acc else 0.0,
            }
        )

        total_time = time.time() - t_start
        peak_mem = (
            torch.cuda.max_memory_allocated(self.device) / 1e6
            if self.device.type == "cuda"
            else 0.0
        )
        n_params = sum(p.numel() for p in self.model.parameters())

        return {
            "protocol": "task_free",
            "n_tasks": T,
            "task_names": [t.name for t in self.bench.tasks],
            "n_classes_per_task": self.bench.n_classes_per_task,
            "init_acc": init_acc,
            "acc_checkpoints": acc_checkpoints,
            "final_acc": final_acc,
            # 1xT acc_matrix keeps the legacy average-accuracy path working
            "acc_matrix": [final_acc],
            "weight_drift": [self._weight_drift()],
            "grad_norm": [gnorm_acc / max(gnorm_n, 1)],
            "consolidation": [self.method.consolidation_state(self.model)],
            "representations": [self._probe_features()],
            "stream_log": list(getattr(self.method, "stream_log", []) or []),
            "stream_meta": {
                "boundaries": list(self.stream.boundaries),
                "boundary_steps": sorted(self._boundary_steps),
                "blur_ratio": self.stream.blur_ratio,
                "local_epochs": self.stream.local_epochs,
                "n_steps": int(n_steps),
                "batch_size": self.batch_size,
                "eval_period": self.eval_period,
                "stream_len": len(self.stream),
            },
            "timing_per_task_s": [total_time],
            "total_time_s": total_time,
            "peak_memory_mb": peak_mem,
            "n_params": n_params,
            "inference_cost_us_per_sample": self._inference_cost(),
        }

    @torch.no_grad()
    def _inference_cost(self, reps: int = 3) -> float:
        self.model.eval()
        x = self._probe_x[: min(128, self._probe_x.shape[0])]
        if x.shape[0] == 0:
            return 0.0
        self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / (reps * x.shape[0]) * 1e6
