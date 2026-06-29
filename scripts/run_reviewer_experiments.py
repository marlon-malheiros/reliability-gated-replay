#!/usr/bin/env python
"""Run the targeted controls requested in the final reviewer-risk audit.

The script reuses the submission trainer and writes the same JSON schema as
``run_nn_submission.py``. Long grids are resumable and kept separate from the
frozen main corpus by default.

Examples
--------
python scripts/run_reviewer_experiments.py matched --datasets split_cifar10 \
    --conditions sym20,sym60 --gates gate_loss,gate_conf --seeds 0,1,2
python scripts/run_reviewer_experiments.py erace --datasets seq_cifar10,cifar10n \
    --conditions sym20,sym60,c10n_aggre,c10n_worse --seeds 0,1,2
python scripts/run_reviewer_experiments.py threshold --dataset split_cifar10 \
    --condition sym60 --gates gate_loss --seeds 0,1,2
python scripts/run_reviewer_experiments.py buffer --datasets split_cifar10 \
    --conditions sym20,sym60 --methods er,gate_loss --sizes 200,1000 \
    --seeds 0,1,2

Add ``--dry-run`` before the subcommand to preview the number and status of
planned cells without initializing a compute device.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from scripts.run_nn_submission import (  # noqa: E402
    _json_default,
    benchmarks,
    cifar10n_conditions,
    methods,
    run_cell,
    synthetic_conditions,
)
from utils.device import get_device  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "neural_networks_submission" / "reviewer_controls"
MAIN_RAW = ROOT / "results" / "neural_networks_submission" / "raw_logs"


def csv_values(text: str):
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_conditions(text: str):
    names = csv_values(text)
    syn = [x for x in names if not x.startswith("c10n_")]
    c10n = [x.removeprefix("c10n_") for x in names if x.startswith("c10n_")]
    return (synthetic_conditions(syn) if syn else []) + (
        cifar10n_conditions(c10n) if c10n else []
    )


def applicable(dataset: str, condition: dict) -> bool:
    return (dataset == "cifar10n") == ("data_overrides" in condition)


def paired_rate(source: Path, dataset: str, condition: str, gate: str, seed: int):
    fp = source / f"{dataset}__{condition}__{gate}__seed{seed}.json"
    if not fp.exists():
        raise FileNotFoundError(f"paired gate run not found: {fp}")
    result = json.loads(fp.read_text())
    cons = result.get("consolidation") or []
    gh = (cons[-1].get("gate_history") or []) if cons else []
    actual = [g.get("admission_rate") for g in gh if g.get("admission_rate") is not None]
    if actual:
        rate, source_name = float(np.nanmean(actual)), "actual_admission_rate"
    else:
        expected = [g.get("gate_mean") for g in gh if g.get("gate_mean") is not None]
        if not expected:
            raise ValueError(f"paired gate run has no admission diagnostics: {fp}")
        rate, source_name = float(np.nanmean(expected)), "mean_gate_probability"
    clean_fraction = 1.0 - float(np.mean(result.get("noise_rate_per_task") or [0.0]))
    return float(np.clip(rate, 0, 1)), float(np.clip(clean_fraction, 0, 1)), source_name


def write_run(out: Path, run_id: str, result: dict, data_cfg: dict, method_cfg: dict):
    raw, cfg = out / "raw_logs", out / "configs"
    raw.mkdir(parents=True, exist_ok=True)
    cfg.mkdir(parents=True, exist_ok=True)
    (raw / f"{run_id}.json").write_text(json.dumps(result, default=_json_default))
    (cfg / f"{run_id}.json").write_text(json.dumps(
        {"data": data_cfg, "method": method_cfg}, indent=2, default=_json_default
    ))


def cell_run_id(cell):
    dataset, condition, label, _, metadata = cell
    return f"{dataset}__{condition['name']}__{label}__seed{int(metadata['seed'])}"


def preview(cells, out: Path, resume=True):
    counts = Counter()
    pending = []
    for cell in cells:
        dataset, condition, _, _, metadata = cell
        run_id = cell_run_id(cell)
        exists = (out / "raw_logs" / f"{run_id}.json").exists()
        status = "done" if resume and exists else "pending"
        counts[(dataset, condition["name"], metadata["control"], status)] += 1
        if status == "pending":
            pending.append(run_id)

    print(f"planned={len(cells)} pending={len(pending)} completed={len(cells) - len(pending)}")
    print(f"output={out}")
    for key in sorted(counts):
        dataset, condition, control, status = key
        print(f"  {status:7s} {counts[key]:3d}  {dataset:15s} {condition:10s} {control}")
    if pending:
        print("pending run IDs:")
        for run_id in pending:
            print(f"  {run_id}")


def execute(cells, out: Path, device, epochs=None, max_train=None, resume=True):
    specs = benchmarks()
    cache = {}
    done = skipped = failed = 0
    for i, (dataset, condition, label, method_cfg, metadata) in enumerate(cells, 1):
        seed = int(metadata["seed"])
        run_id = cell_run_id((dataset, condition, label, method_cfg, metadata))
        fp = out / "raw_logs" / f"{run_id}.json"
        if resume and fp.exists():
            skipped += 1
            continue
        print(f"[{i}/{len(cells)}] {run_id}")
        try:
            result, data_cfg, used_method_cfg = run_cell(
                dataset, copy.deepcopy(specs[dataset]), condition, label,
                method_cfg, seed, device,
                {"epochs": epochs, "max_train": max_train}, cache,
            )
            result["reviewer_control"] = metadata
            write_run(out, run_id, result, data_cfg, used_method_cfg)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}")
    print(f"done={done} skipped={skipped} failed={failed} -> {out}")


def matched_cells(args):
    base_methods = methods(args.buffer_size)
    conditions = parse_conditions(args.conditions)
    cells = []
    for dataset in csv_values(args.datasets):
        for condition in conditions:
            if not applicable(dataset, condition):
                continue
            for gate in csv_values(args.gates):
                if gate not in {"gate_loss", "gate_conf"}:
                    raise ValueError(f"unsupported matched-admission gate: {gate}")
                suffix = "loss" if gate == "gate_loss" else "conf"
                for seed in map(int, csv_values(args.seeds)):
                    rate, clean_fraction, rate_source = paired_rate(
                        Path(args.source), dataset, condition["name"], gate, seed
                    )
                    random_cfg = {
                        "name": "gated_replay", "buffer_size": args.buffer_size,
                        "batch_size": 32, "fixed_admission_prob": rate,
                    }
                    # Expected total admission = q_clean * p_keep.
                    keep = min(rate / clean_fraction, 1.0) if clean_fraction > 0 else 0.0
                    oracle_cfg = {
                        "name": "gated_replay", "buffer_size": args.buffer_size,
                        "batch_size": 32, "oracle": True, "oracle_keep_prob": keep,
                    }
                    common = {
                        "seed": seed, "paired_gate": gate, "target_admission_rate": rate,
                        "rate_source": rate_source, "clean_fraction": clean_fraction,
                    }
                    cells.append((dataset, condition, f"random_matched_{suffix}",
                                  random_cfg, {**common, "control": "random_matched"}))
                    cells.append((dataset, condition, f"oracle_thinned_{suffix}",
                                  oracle_cfg, {**common, "control": "oracle_thinned",
                                               "oracle_keep_prob": keep}))
    return cells


def erace_cells(args):
    conditions = parse_conditions(args.conditions)
    cells = []
    for dataset in csv_values(args.datasets):
        for condition in conditions:
            if applicable(dataset, condition):
                for seed in map(int, csv_values(args.seeds)):
                    cfg = methods(args.buffer_size)["er_ace"]
                    cells.append((dataset, condition, "er_ace", cfg,
                                  {"seed": seed, "control": "er_ace"}))
    return cells


def threshold_cells(args):
    condition = parse_conditions(args.condition)[0]
    cells = []
    gates = csv_values(args.gates)
    unsupported = set(gates) - {"gate_loss", "gate_conf"}
    if unsupported:
        raise ValueError(f"unsupported threshold gate(s): {sorted(unsupported)}")
    seeds = [args.seed] if args.seed is not None else list(map(int, csv_values(args.seeds)))
    for seed in seeds:
        if "gate_loss" in gates:
            for gamma in (1.0, 3.0, 6.0):
                for threshold in (0.5, 1.0, 1.5):
                    label = f"sens_loss_g{gamma:g}_t{threshold:g}"
                    cfg = {"name": "gated_replay", "buffer_size": args.buffer_size,
                           "batch_size": 32, "error_gated": True, "gate_level": "sample",
                           "gate": {"signal": "error", "gamma": gamma,
                                    "error_threshold": threshold}}
                    cells.append((args.dataset, condition, label, cfg,
                                  {"seed": seed, "control": "threshold_sensitivity",
                                   "gate": "loss", "gamma": gamma,
                                   "threshold": threshold}))
        if "gate_conf" in gates:
            for gamma in (3.0, 6.0, 12.0):
                for threshold in (0.4, 0.5, 0.6):
                    label = f"sens_conf_g{gamma:g}_t{threshold:g}"
                    cfg = {"name": "gated_replay", "buffer_size": args.buffer_size,
                           "batch_size": 32, "error_gated": True, "gate_level": "sample",
                           "gate": {"signal": "confidence", "gamma": gamma,
                                    "tau": threshold}}
                    cells.append((args.dataset, condition, label, cfg,
                                  {"seed": seed, "control": "threshold_sensitivity",
                                   "gate": "confidence", "gamma": gamma,
                                   "threshold": threshold}))
    return cells


def buffer_cells(args):
    conditions = parse_conditions(args.conditions)
    cells = []
    sizes = list(map(int, csv_values(args.sizes)))
    selected_methods = csv_values(args.methods)
    supported = {"er", "gate_loss", "gate_conf", "oracle"}
    unsupported = set(selected_methods) - supported
    if unsupported:
        raise ValueError(f"unsupported buffer method(s): {sorted(unsupported)}")
    for size in sizes:
        mm = methods(size)
        for dataset in csv_values(args.datasets):
            for condition in conditions:
                if not applicable(dataset, condition):
                    continue
                for method in selected_methods:
                    for seed in map(int, csv_values(args.seeds)):
                        label = f"{method}_M{size}"
                        cells.append((dataset, condition, label, mm[method],
                                      {"seed": seed, "control": "buffer_sensitivity",
                                       "base_method": method, "buffer_size": size}))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--buffer-size", type=int, default=500)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="preview planned and completed cells without running training")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("matched")
    p.add_argument("--datasets", default="seq_cifar10,split_cifar10,cifar10n")
    p.add_argument("--conditions", default="sym20,sym60,asym40,c10n_aggre,c10n_worse")
    p.add_argument("--gates", default="gate_loss,gate_conf")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--source", default=str(MAIN_RAW))

    p = sub.add_parser("erace")
    p.add_argument("--datasets", default="seq_cifar10,seq_cifar100,cifar10n")
    p.add_argument("--conditions", default="clean,sym20,sym60,c10n_aggre,c10n_worse")
    p.add_argument("--seeds", default="0,1,2")

    p = sub.add_parser("threshold")
    p.add_argument("--dataset", default="split_cifar10")
    p.add_argument("--condition", default="sym60")
    p.add_argument("--gates", default="gate_loss,gate_conf")
    p.add_argument("--seeds", default="0")
    p.add_argument("--seed", type=int, help="deprecated single-seed alias for --seeds")

    p = sub.add_parser("buffer")
    p.add_argument("--datasets", default="seq_cifar10,split_cifar10")
    p.add_argument("--conditions", default="sym20,sym60")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="er,gate_loss,gate_conf,oracle")
    p.add_argument("--sizes", default="200,500,1000")

    args = ap.parse_args()
    builders = {
        "matched": matched_cells, "erace": erace_cells,
        "threshold": threshold_cells, "buffer": buffer_cells,
    }
    cells = builders[args.mode](args)
    out = Path(args.output)
    if args.dry_run:
        preview(cells, out, resume=not args.no_resume)
        return
    execute(cells, out, get_device(args.device or "auto"),
            epochs=args.epochs, max_train=args.max_train, resume=not args.no_resume)


if __name__ == "__main__":
    main()
