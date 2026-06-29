#!/usr/bin/env python
"""Unified benchmark runner for the Neural Networks submission.

Wraps the native pipeline (datasets/registry + methods/registry + ContinualTrainer)
to produce *standardized* per-run JSON logs plus an append-only
``experiment_manifest.csv``. The grid is `benchmark x noise-condition x method x seed`.

Beyond the trainer's standard outputs it also computes, per run:
  * final-model calibration (ECE / Brier / NLL) on each task's CLEAN test set,
  * buffer diagnostics (purity, class entropy/balance, replay loss on clean vs noisy
    stored samples) for replay-based methods,
all defined in ``analysis/gate_metrics.py`` (see README_RESULTS.md).

Presets:
  --preset smoke      tiny Permuted-MNIST end-to-end (< ~5 min CPU/GPU)  [CI gate]
  --preset sanity     Permuted-/Rotated-MNIST sanity checks (MLP)
  --preset priorityA  Seq-/Split-CIFAR-10 ResNet-18, sym noise, core methods  [decisive]
  --preset cifar10n   realistic-noise benchmark (CIFAR-10N, or synthetic fallback)
  --preset full       the entire grid (compute-heavy; supported, rarely run whole)

Everything is overridable: --datasets --methods --conditions --seeds --epochs
--max-train --buffer-size --device --resume.  Resumable: existing run JSONs skipped.

    python scripts/run_nn_submission.py --preset smoke
    python scripts/run_nn_submission.py --preset priorityA --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from analysis.gate_metrics import (  # noqa: E402
    brier_score,
    buffer_class_balance,
    buffer_class_entropy,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_diagram_bins,
)
from datasets.registry import build_benchmark  # noqa: E402
from methods.registry import build_method  # noqa: E402
from methods.trainer import ContinualTrainer  # noqa: E402
from models.registry import build_model  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seeding import set_seed  # noqa: E402

log = get_logger("nn_submission")
OUT_DEFAULT = ROOT / "results" / "neural_networks_submission"


# --------------------------------------------------------------------------- #
# Benchmark specs (backbone + train budget). max_train/epochs are submission
# defaults sized for a 4 GB GPU; override on the CLI.
# --------------------------------------------------------------------------- #
def benchmarks() -> Dict[str, Dict[str, Any]]:
    return {
        # --- sanity checks (MNIST-family, MLP) ---
        "permuted_mnist": {
            "data": {"name": "permuted_mnist", "source": "mnist", "multihead": False,
                     "n_tasks": 5, "max_train_per_task": 3000, "val_fraction": 0.1,
                     "allow_download": True, "fallback_synthetic": True},
            "model": {"name": "mlp"}, "train": {"epochs": 8, "batch_size": 128},
            "backbone": "mlp", "family": "sanity"},
        "rotated_mnist": {
            "data": {"name": "rotated_mnist", "source": "mnist", "multihead": False,
                     "n_tasks": 5, "max_train_per_task": 3000, "val_fraction": 0.1,
                     "min_angle": 0.0, "max_angle": 120.0,
                     "allow_download": True, "fallback_synthetic": True},
            "model": {"name": "mlp"}, "train": {"epochs": 8, "batch_size": 128},
            "backbone": "mlp", "family": "sanity"},
        # --- main CIFAR benchmarks (ResNet-18) ---
        "seq_cifar10": {  # class-IL, single head (standard Seq-CIFAR-10)
            "data": {"name": "split_cifar10", "task_il": False, "n_classes": 10,
                     "classes_per_task": 2, "normalize": True, "val_fraction": 0.05,
                     "max_train_per_task": 2500, "max_test_per_task": 1000,
                     "allow_download": True},
            "model": {"name": "resnet18"}, "train": {"epochs": 8, "batch_size": 128},
            "backbone": "resnet18", "family": "cifar"},
        "split_cifar10": {  # task-IL, per-task binary heads (the few-class inversion regime)
            "data": {"name": "split_cifar10", "task_il": True, "n_classes": 10,
                     "classes_per_task": 2, "normalize": True, "val_fraction": 0.05,
                     "max_train_per_task": 2500, "max_test_per_task": 1000,
                     "allow_download": True},
            "model": {"name": "resnet18"}, "train": {"epochs": 8, "batch_size": 128},
            "backbone": "resnet18", "family": "cifar"},
        "seq_cifar100": {  # class-IL, 10 tasks x 10 classes
            "data": {"name": "split_cifar100", "n_classes": 100, "classes_per_task": 10,
                     "normalize": True, "val_fraction": 0.05,
                     "max_train_per_task": 2500, "max_test_per_task": 1000,
                     "allow_download": True},
            "model": {"name": "resnet18"}, "train": {"epochs": 8, "batch_size": 128},
            "backbone": "resnet18", "family": "cifar"},
        # --- realistic noise ---
        "cifar10n": {  # class-IL human noise; condition picks the label variant
            "data": {"name": "cifar10n", "variant": "worse", "n_classes": 10,
                     "classes_per_task": 2, "normalize": True, "val_fraction": 0.05,
                     "max_train_per_task": 2500, "max_test_per_task": 1000,
                     "allow_download": True},
            "model": {"name": "resnet18"}, "train": {"epochs": 8, "batch_size": 128},
            "backbone": "resnet18", "family": "cifar"},
        "cifar10n_taskil": {  # projected task-IL CIFAR-10N bridge
            "data": {"name": "cifar10n", "task_il": True, "variant": "worse",
                     "n_classes": 10, "classes_per_task": 2, "normalize": True,
                     "val_fraction": 0.05, "max_train_per_task": 2500,
                     "max_test_per_task": 1000, "allow_download": True},
            "model": {"name": "resnet18"}, "train": {"epochs": 8, "batch_size": 128},
            "backbone": "resnet18", "family": "cifar"},
    }


# --------------------------------------------------------------------------- #
# Method grid -- covers the required baseline list (see baseline_implementation_status.csv).
# --------------------------------------------------------------------------- #
def methods(buffer_size: int = 500) -> Dict[str, Dict[str, Any]]:
    b = buffer_size
    g = lambda sig, **kw: {"name": "gated_replay", "buffer_size": b, "batch_size": 32,
                           "error_gated": True, "gate_level": "sample",
                           "gate": {"signal": sig, **kw}}
    return {
        # regularization / replay baselines
        "ewc": {"name": "ewc", "lambda": 100.0, "fisher_samples": 512},
        "si": {"name": "si", "lambda": 1.0, "xi": 0.1},
        "er": {"name": "replay", "buffer_size": b, "batch_size": 32},          # random admission
        "derpp": {"name": "derpp", "buffer_size": b, "batch_size": 32, "alpha": 0.5, "beta": 1.0},
        "er_ace": {"name": "er_ace", "buffer_size": b, "batch_size": 32},
        # reliability-gated replay admission -- the gate family
        "gate_loss": g("error", error_threshold=1.0, gamma=3.0),               # loss/error
        "gate_conf": g("confidence", tau=0.5, gamma=6.0),                      # confidence
        "gate_predstab": g("pred_stability", tau=0.6, gamma=6.0, ema_beta=0.6, cold_value=0.5),
        "gate_reprstab": g("repr_stability", tau=0.5, gamma=6.0, repr_alpha=2.0, ema_beta=0.6),
        "gate_spr": g("loss_traj", error_threshold=1.0, gamma=3.0),            # SPR proxy (small-loss-over-time)
        "gate_teacher": {"name": "gated_replay", "buffer_size": b, "batch_size": 32,
                         "error_gated": True, "gate_level": "sample",
                         "teacher": {"enabled": True, "momentum": 0.999},
                         "gate": {"signal": "agreement", "tau": 0.5, "gamma": 6.0}},
        "gate_coteach": {"name": "gated_replay", "buffer_size": b, "batch_size": 32,
                         "error_gated": True, "gate_level": "sample",
                         "coteach": {"enabled": True, "keep": 0.5, "lr": 1e-3},
                         "gate": {"signal": "agreement", "tau": 0.5, "gamma": 6.0}},
        "oracle": {"name": "gated_replay", "buffer_size": b, "batch_size": 32, "oracle": True},
    }


# core method subset for the decisive (cheaper) runs
CORE_METHODS = ["er", "derpp", "ewc", "gate_loss", "gate_conf", "gate_spr", "oracle"]
BRIDGE_METHODS = ["er", "derpp", "gate_loss", "gate_conf", "oracle"]
ALL_REPLAY = ["er", "er_ace", "derpp", "gate_loss", "gate_conf", "gate_predstab",
              "gate_reprstab", "gate_spr", "gate_teacher", "gate_coteach", "oracle"]


# --------------------------------------------------------------------------- #
# Noise conditions. A condition either carries a synthetic ``label_noise`` block or
# ``data_overrides`` (e.g. a CIFAR-10N variant). ``clean`` == no corruption.
# --------------------------------------------------------------------------- #
def synthetic_conditions(spec: List[str]) -> List[Dict[str, Any]]:
    table = {
        "clean":  {"label_noise": {"rate": 0.0, "schedule": "clean", "noise_type": "symmetric"}},
        "sym20":  {"label_noise": {"rate": 0.2, "schedule": "all", "noise_type": "symmetric"}},
        "sym40":  {"label_noise": {"rate": 0.4, "schedule": "all", "noise_type": "symmetric"}},
        "sym60":  {"label_noise": {"rate": 0.6, "schedule": "all", "noise_type": "symmetric"}},
        "asym20": {"label_noise": {"rate": 0.2, "schedule": "all", "noise_type": "asymmetric"}},
        "asym40": {"label_noise": {"rate": 0.4, "schedule": "all", "noise_type": "asymmetric"}},
    }
    return [{"name": n, **table[n]} for n in spec]


def cifar10n_conditions(spec: List[str]) -> List[Dict[str, Any]]:
    # realistic-noise variants of CIFAR-10N
    return [{"name": f"c10n_{v}", "data_overrides": {"variant": v}} for v in spec]


# --------------------------------------------------------------------------- #
# Per-run diagnostics computed from the trained model
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _calibration(model, bench, device, eval_bs=256, n_bins=15) -> Dict[str, Any]:
    """Final-model ECE / Brier / NLL on each task's CLEAN test set (mean over tasks)."""
    model.eval()
    per_task = []
    all_P, all_Y = [], []
    for t in bench.tasks:
        xs, ys = t.test.tensors
        probs = []
        for i in range(0, xs.shape[0], eval_bs):
            xb = xs[i:i + eval_bs].to(device)
            probs.append(F.softmax(model(xb, t.task_id), dim=1).cpu().numpy())
        P = np.concatenate(probs, 0) if probs else np.zeros((0, t.n_classes))
        Y = ys.numpy()
        per_task.append({
            "ece": expected_calibration_error(P, Y, n_bins),
            "brier": brier_score(P, Y),
            "nll": negative_log_likelihood(P, Y),
        })
        # pool for the reliability diagram (multihead: pad to a common width)
        if P.shape[0]:
            all_P.append(P)
            all_Y.append(Y)
    # pooled metrics + reliability-diagram bins (the figure source). For multihead
    # the top-label confidence/accuracy is head-local, which is what we pool.
    width = max((p.shape[1] for p in all_P), default=0)
    pooled_conf, pooled_pred, pooled_corr = [], [], []
    for P, Y in zip(all_P, all_Y):
        pooled_conf.append(P.max(1)); pooled_pred.append(P.argmax(1)); pooled_corr.append(Y)
    if pooled_conf:
        conf = np.concatenate(pooled_conf); pred = np.concatenate(pooled_pred)
        ytrue = np.concatenate(pooled_corr)
        # build a 2-col [1-conf, conf] proxy so the shared bin helper works on top-label
        proxy = np.stack([1 - conf, conf], 1)
        lbl = (pred == ytrue).astype(int)  # 1 if top-label correct -> "class 1"
        centers, acc, cf, cnt = reliability_diagram_bins(proxy, lbl, n_bins)
        ece_p = expected_calibration_error(proxy, lbl, n_bins)
        bins = {"centers": centers.tolist(), "acc": np.nan_to_num(acc, nan=-1).tolist(),
                "conf": np.nan_to_num(cf, nan=-1).tolist(), "count": cnt.tolist()}
    else:
        ece_p, bins = float("nan"), {}
    mean = lambda k: float(np.nanmean([d[k] for d in per_task])) if per_task else float("nan")
    return {"ece": mean("ece"), "brier": mean("brier"), "nll": mean("nll"),
            "ece_pooled": float(ece_p), "per_task": per_task, "reliability_bins": bins}


@torch.no_grad()
def _cleanliness_calibration(model, bench, device, eval_bs=256, n_bins=15) -> Dict[str, Any]:
    """Calibration on training examples, stratified by label cleanliness.

    Targets are always the retained clean/reference labels. ``clean`` and
    ``mislabeled`` describe whether the label shown during training matched that
    reference. This directly measures whether corrupted examples become
    confidently (but falsely) reliable to a self-scored gate.
    """
    model.eval()
    groups = {"clean": [[], []], "mislabeled": [[], []]}
    high_conf_total = high_conf_mislabeled = 0
    for task in bench.tasks:
        if task.train_y_clean is None:
            continue
        xs, y_noisy = task.train.tensors
        y_clean = np.asarray(task.train_y_clean, dtype=int)
        noisy = y_noisy.numpy() != y_clean
        probs = []
        for i in range(0, xs.shape[0], eval_bs):
            probs.append(F.softmax(model(xs[i:i + eval_bs].to(device), task.task_id), dim=1)
                         .cpu().numpy())
        P = np.concatenate(probs, 0)
        high = P.max(1) > 0.9
        high_conf_total += int(high.sum())
        high_conf_mislabeled += int((high & noisy).sum())
        for name, mask in (("clean", ~noisy), ("mislabeled", noisy)):
            if mask.any():
                groups[name][0].append(P[mask])
                groups[name][1].append(y_clean[mask])

    out: Dict[str, Any] = {
        "high_conf_mislabeled_fraction": (
            high_conf_mislabeled / high_conf_total if high_conf_total else float("nan")
        ),
        "n_high_confidence": high_conf_total,
    }
    for name, (plist, ylist) in groups.items():
        if not plist:
            out[name] = {"n": 0, "mean_confidence": float("nan"), "ece": float("nan")}
            continue
        P, Y = np.concatenate(plist), np.concatenate(ylist)
        out[name] = {
            "n": int(Y.size),
            "mean_confidence": float(P.max(1).mean()),
            "ece": expected_calibration_error(P, Y, n_bins),
            "accuracy": float((P.argmax(1) == Y).mean()),
        }
    return out


@torch.no_grad()
def _buffer_diagnostics(method, model, device, bench=None) -> Dict[str, Any]:
    """Purity / class diversity / replay loss on clean vs noisy stored samples."""
    buf = getattr(method, "buf", None)
    if buf is None or not getattr(buf, "x", None):
        return {}
    y = np.asarray(buf.y, dtype=int)
    yc_list = list(getattr(buf, "yc", []) or [])
    has_clean = (len(yc_list) == y.size) and y.size > 0  # clean labels tracked + present
    yc = np.asarray(yc_list, dtype=int) if has_clean else y
    # Convert local task-IL labels back to global class ids before measuring
    # balance. Otherwise every binary head would be incorrectly collapsed into
    # the same two categories.
    y_class = y.copy()
    expected_classes = (int(y.max()) + 1) if y.size else 0
    if bench is not None:
        expected_classes = len({c for task in bench.tasks for c in task.global_classes})
        if bench.multihead:
            y_class = np.asarray([
                bench.tasks[int(t)].global_classes[int(label)]
                for t, label in zip(buf.t, y)
            ], dtype=int)
    div = buffer_class_entropy(
        y_class.tolist(), n_classes=expected_classes if expected_classes > 1 else None
    )
    counts = np.bincount(y_class, minlength=expected_classes) if expected_classes else np.array([])
    out = {
        "buffer_size": int(len(buf)),
        "buffer_purity": float(np.mean(y == yc)) if has_clean else float("nan"),
        "buffer_class_balance": buffer_class_balance(y_class.tolist()),
        "class_count_variance": float(np.var(counts)) if counts.size else float("nan"),
        "minority_class_coverage": (
            float(np.count_nonzero(counts) / expected_classes)
            if expected_classes else float("nan")
        ),
        **div,
    }
    # replay loss split: clean (stored==true) vs noisy (stored!=true)
    if has_clean:
        model.eval()
        xs = torch.stack(list(buf.x)).to(device)
        ts = np.asarray(buf.t, dtype=int)
        ce = np.full(y.size, np.nan)
        for t in np.unique(ts):
            m = ts == t
            logits = model(xs[m], int(t))
            ce[m] = F.cross_entropy(logits, torch.tensor(y[m], device=device),
                                    reduction="none").cpu().numpy()
        clean_m = y == yc
        out["replay_loss_clean"] = float(np.nanmean(ce[clean_m])) if clean_m.any() else float("nan")
        out["replay_loss_noisy"] = float(np.nanmean(ce[~clean_m])) if (~clean_m).any() else float("nan")
    return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Run one cell
# --------------------------------------------------------------------------- #
def run_cell(bname, spec, cond, label, mcfg, seed, device, overrides, bench_cache):
    data_cfg = copy.deepcopy(spec["data"])
    data_cfg["seed"] = seed
    data_cfg.update(cond.get("data_overrides", {}))
    if "label_noise" in cond:
        data_cfg["label_noise"] = cond["label_noise"]
    if overrides.get("max_train") is not None:
        data_cfg["max_train_per_task"] = overrides["max_train"]

    ckey = (bname, cond["name"], seed)
    if ckey not in bench_cache:
        bench_cache[ckey] = build_benchmark(data_cfg, data_root=str(ROOT / "data"))
    bench = bench_cache[ckey]

    set_seed(seed, deterministic=True)
    model = build_model(spec["model"], bench)
    method_cfg = dict(mcfg)
    method_cfg["seed"] = seed
    method = build_method(method_cfg)
    train_cfg = {"eval_batch_size": 256, "probe_size": 100, **spec.get("train", {})}
    if overrides.get("epochs") is not None:
        train_cfg["epochs"] = overrides["epochs"]
    run_cfg = {"train": train_cfg, "optimizer": {"name": "adam", "lr": 1e-3}}

    t0 = time.time()
    trainer = ContinualTrainer(run_cfg, bench, model, method, device)
    result = trainer.train()
    result["calibration"] = _calibration(model, bench, device, train_cfg["eval_batch_size"])
    result["cleanliness_calibration"] = _cleanliness_calibration(
        model, bench, device, train_cfg["eval_batch_size"]
    )
    result["buffer_diagnostics"] = _buffer_diagnostics(method, model, device, bench)
    result.update({
        "label": label, "method_cfg": method_cfg, "benchmark": bname, "dataset": bench.name,
        "backbone": spec["backbone"], "condition": cond["name"], "multihead": bench.multihead,
        "label_noise": cond.get("label_noise"), "data_overrides": cond.get("data_overrides"),
        "noise_rate_per_task": [float(getattr(t, "noise_rate", 0.0)) for t in bench.tasks],
        "seed": seed, "protocol": "task_boundary", "wall_time_s": time.time() - t0,
    })
    return result, data_cfg, method_cfg


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
MANIFEST_COLS = ["run_id", "benchmark", "dataset", "method", "backbone", "noise_type",
                 "noise_rate", "seed", "command", "config_path", "status",
                 "start_time", "end_time", "notes"]


def append_manifest(path: Path, row: Dict[str, Any]):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_COLS})


def _noise_fields(cond):
    ln = cond.get("label_noise")
    if ln:
        return ln.get("noise_type", "symmetric"), ln.get("rate", 0.0)
    if cond.get("data_overrides", {}).get("variant"):
        return f"cifar10n:{cond['data_overrides']['variant']}", "realistic"
    return "none", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="priorityA",
                    choices=["smoke", "sanity", "priorityA", "cifar10n", "bridge", "full"])
    ap.add_argument("--output", default=str(OUT_DEFAULT))
    ap.add_argument("--datasets", default=None, help="comma list (overrides preset)")
    ap.add_argument("--methods", default=None, help="comma list (overrides preset)")
    ap.add_argument("--conditions", default=None, help="comma list (overrides preset)")
    ap.add_argument("--seeds", default=None, help="comma list (overrides preset)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--buffer-size", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    args = ap.parse_args()

    device = get_device(args.device or "auto")
    BENCH = benchmarks()
    METH = methods(args.buffer_size)

    # preset -> (datasets, methods, conditions, seeds)
    if args.preset == "smoke":
        ds, ms = ["permuted_mnist"], ["er", "gate_loss", "gate_conf", "derpp", "oracle"]
        conds = synthetic_conditions(["clean", "sym60"]); seeds = [0]
        BENCH["permuted_mnist"]["data"]["max_train_per_task"] = 600
        BENCH["permuted_mnist"]["train"] = {"epochs": 2, "batch_size": 128}
    elif args.preset == "sanity":
        ds, ms = ["permuted_mnist", "rotated_mnist"], CORE_METHODS
        conds = synthetic_conditions(["clean", "sym20", "sym40", "sym60"]); seeds = [0, 1, 2]
    elif args.preset == "priorityA":
        ds, ms = ["seq_cifar10", "split_cifar10"], CORE_METHODS
        conds = synthetic_conditions(["sym20", "sym40", "sym60"]); seeds = [0, 1, 2]
    elif args.preset == "cifar10n":
        ds, ms = ["cifar10n"], CORE_METHODS
        conds = cifar10n_conditions(["worse", "aggre"]); seeds = [0, 1, 2]
    elif args.preset == "bridge":
        ds, ms = ["cifar10n_taskil"], BRIDGE_METHODS
        conds = cifar10n_conditions(["aggre", "worse"]); seeds = [0, 1, 2, 3, 4]
    else:  # full
        ds = ["permuted_mnist", "rotated_mnist", "seq_cifar10", "split_cifar10",
              "seq_cifar100", "cifar10n"]
        ms = list(METH)
        conds = (synthetic_conditions(["clean", "sym20", "sym40", "sym60", "asym20", "asym40"])
                 + cifar10n_conditions(["worse", "aggre", "random1"]))
        seeds = [0, 1, 2, 3, 4]

    if args.datasets:
        ds = args.datasets.split(",")
    if args.methods:
        ms = args.methods.split(",")
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    if args.conditions:
        # interpret as synthetic names unless prefixed c10n_
        syn = [c for c in args.conditions.split(",") if not c.startswith("c10n_")]
        c10 = [c[len("c10n_"):] for c in args.conditions.split(",") if c.startswith("c10n_")]
        conds = (synthetic_conditions(syn) if syn else []) + (cifar10n_conditions(c10) if c10 else [])

    overrides = {"epochs": args.epochs, "max_train": args.max_train}
    out = Path(args.output).resolve()
    raw = out / "raw_logs"; cfgdir = out / "configs"
    raw.mkdir(parents=True, exist_ok=True); cfgdir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out / "experiment_manifest.csv"

    # validity: cifar10n datasets only pair with cifar10n conditions and vice-versa
    def applicable(bname, cond):
        is_c10n_ds = bname.startswith("cifar10n")
        is_c10n_cond = "data_overrides" in cond
        return is_c10n_ds == is_c10n_cond

    grid = [(d, c, m, s) for d in ds for c in conds for m in ms for s in seeds
            if d in BENCH and m in METH and applicable(d, c)]
    log.info(f"preset={args.preset} device={device} | {len(grid)} runs "
             f"(datasets={ds} methods={ms} conds={[c['name'] for c in conds]} seeds={seeds})")
    if args.dry_run:
        for (d, c, m, s) in grid:
            print(f"{d}__{c['name']}__{m}__seed{s}")
        print(f"TOTAL {len(grid)} runs")
        return

    bench_cache: Dict = {}
    ran = skipped = failed = 0
    for i, (bname, cond, label, seed) in enumerate(grid, 1):
        run_id = f"{bname}__{cond['name']}__{label}__seed{seed}"
        fp = raw / f"{run_id}.json"
        if args.resume and fp.exists():
            skipped += 1
            continue
        ntype, nrate = _noise_fields(cond)
        cmd = (f"python scripts/run_nn_submission.py --datasets {bname} "
               f"--conditions {cond['name']} --methods {label} --seeds {seed}")
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        log.info(f"[{i}/{len(grid)}] {run_id}")
        try:
            result, data_cfg, method_cfg = run_cell(
                bname, BENCH[bname], cond, label, METH[label], seed, device, overrides, bench_cache)
        except FileNotFoundError as e:  # e.g. CIFAR-10N labels absent -> explicit, not silent
            failed += 1
            log.error(f"MISSING DATA {run_id}: {e}")
            append_manifest(manifest_csv, {
                "run_id": run_id, "benchmark": bname, "dataset": bname, "method": label,
                "backbone": BENCH[bname]["backbone"], "noise_type": ntype, "noise_rate": nrate,
                "seed": seed, "command": cmd, "config_path": "", "status": "missing_data",
                "start_time": start, "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": "dataset file absent; use synthetic fallback (see README_RESULTS)"})
            continue
        except Exception as e:
            failed += 1
            log.error(f"FAILED {run_id}: {e}")
            append_manifest(manifest_csv, {
                "run_id": run_id, "benchmark": bname, "dataset": bname, "method": label,
                "backbone": BENCH[bname]["backbone"], "noise_type": ntype, "noise_rate": nrate,
                "seed": seed, "command": cmd, "config_path": "", "status": "failed",
                "start_time": start, "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": str(e)[:200]})
            continue
        cfg_path = cfgdir / f"{run_id}.json"
        with open(cfg_path, "w") as f:
            json.dump({"data": data_cfg, "method": method_cfg, "train": BENCH[bname]["train"]},
                      f, indent=2, default=_json_default)
        try:
            cfg_rel = str(cfg_path.relative_to(ROOT))
        except ValueError:
            cfg_rel = str(cfg_path)
        with open(fp, "w") as f:
            json.dump(result, f, default=_json_default)
        append_manifest(manifest_csv, {
            "run_id": run_id, "benchmark": bname, "dataset": result["dataset"], "method": label,
            "backbone": result["backbone"], "noise_type": ntype, "noise_rate": nrate,
            "seed": seed, "command": cmd, "config_path": cfg_rel,
            "status": "done", "start_time": start, "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": f"acc={result['acc_matrix'][-1]}"[:120]})
        ran += 1
    log.info(f"done: ran {ran}, skipped {skipped}, failed {failed}; manifest -> {manifest_csv}")


if __name__ == "__main__":
    main()
