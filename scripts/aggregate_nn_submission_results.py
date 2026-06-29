#!/usr/bin/env python
"""Aggregate run JSONs -> the standardized submission CSVs.

Reads every ``*.json`` under ``--input`` (and any ``--extra-input`` dirs, e.g. the
legacy ``results/reliability/runs`` MNIST corpus), normalizes the two run schemas
(new ``run_nn_submission.py`` and legacy ``run_reliability_suite.py``), computes the
mandated metrics (``analysis/metrics.py`` + ``analysis/gate_metrics.py``) and writes:

  csv/final_metrics.csv     one row per run -- all final metrics + gate/inversion/calibration
  csv/per_eval_metrics.csv  one row per (run, eval-after-task) -- accuracy progression + gate sep
  csv/gate_diagnostics.csv  one row per (run, task) -- gate_on_correct/wrong/sep/corr/purity
  csv/buffer_diagnostics.csv one row per run -- purity/diversity/balance/replay-loss split

All figures and tables are built from these CSVs (no hidden runtime objects).

    python scripts/aggregate_nn_submission_results.py \
        --input results/neural_networks_submission/raw_logs \
        --extra-input results/reliability/runs \
        --output results/neural_networks_submission/csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from analysis.gate_metrics import (  # noqa: E402
    gate_correctness_correlation,
    run_inversion_summary,
)
from analysis.metrics import performance_metrics  # noqa: E402

# canonical method -> (family, gate_signal, label_free). Legacy labels are mapped
# onto the canonical names so the MNIST corpus lines up with the CIFAR runs.
METHOD_INFO = {
    "adam": ("finetune", "none", True),
    "ewc": ("regularizer", "none", True),
    "si": ("regularizer", "none", True),
    "mas": ("regularizer", "none", True),
    "er": ("replay", "random", True),
    "derpp": ("replay", "none", True),
    "er_ace": ("replay", "none", True),
    "gate_loss": ("replay-gate", "error", False),
    "gate_spr": ("replay-gate", "loss_traj", False),
    "gate_conf": ("replay-gate", "confidence", True),
    "gate_predstab": ("replay-gate", "pred_stability", True),
    "gate_reprstab": ("replay-gate", "repr_stability", True),
    "gate_teacher": ("replay-gate", "agreement+teacher", False),
    "gate_coteach": ("replay-gate", "co-teaching", False),
    "random_matched_loss": ("replay-control", "random-matched-loss", True),
    "random_matched_conf": ("replay-control", "random-matched-confidence", True),
    "oracle_thinned_loss": ("upper-bound-control", "oracle-thinned-loss", False),
    "oracle_thinned_conf": ("upper-bound-control", "oracle-thinned-confidence", False),
    "oracle": ("upper-bound", "oracle", False),
    "ewc_loss": ("regularizer-gate", "error", False),
    "ewc_conf": ("regularizer-gate", "confidence", True),
}
LEGACY_LABEL = {  # run_reliability_suite.py label -> canonical
    "replay": "er", "replay_error": "gate_loss", "replay_losstraj": "gate_spr",
    "replay_conf": "gate_conf", "replay_stab": "gate_predstab", "replay_repr": "gate_reprstab",
    "replay_oracle": "oracle", "replay_teacher": "gate_teacher",
    "replay_teacher_m999": "gate_teacher", "replay_coteach": "gate_coteach",
    "ewc_error": "ewc_loss", "ewc_conf": "ewc_conf",
    "derpp": "derpp", "ewc": "ewc", "si": "si", "mas": "mas", "adam": "adam",
}


def _canon(label: str) -> str:
    return LEGACY_LABEL.get(label, label)


def _noise(result: Dict[str, Any]):
    ln = result.get("label_noise")
    if ln:
        return ln.get("noise_type", "symmetric"), float(ln.get("rate", 0.0))
    do = result.get("data_overrides") or {}
    if do.get("variant"):
        return f"cifar10n:{do['variant']}", float(np.mean(result.get("noise_rate_per_task") or [0.0]))
    # legacy: infer from per-task noise rate
    npt = result.get("noise_rate_per_task") or [0.0]
    return "symmetric", float(np.mean(npt))


def _gate_history(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    cons = result.get("consolidation") or []
    last = cons[-1] if cons else {}
    return last.get("gate_history") or []


def _separation_series(gh: List[Dict[str, Any]]) -> List[float]:
    out = []
    for g in gh:
        goc, gow = g.get("gate_on_correct"), g.get("gate_on_wrong")
        if goc is None or gow is None or not (np.isfinite(goc) and np.isfinite(gow)):
            out.append(float("nan"))
        else:
            out.append(float(goc - gow))
    return out


def _buffer_diag(result: Dict[str, Any]) -> Dict[str, Any]:
    bd = result.get("buffer_diagnostics")
    if bd:
        return bd
    # legacy: only purity is available (from consolidation)
    cons = result.get("consolidation") or []
    last = cons[-1] if cons else {}
    return {"buffer_purity": last.get("buffer_purity", float("nan"))}


def process(result: Dict[str, Any], run_id: str):
    perf = performance_metrics(result)
    label = _canon(result.get("label", "?"))
    fam, sig, lf = METHOD_INFO.get(label, ("other", "?", None))
    ntype, nrate = _noise(result)
    multihead = result.get("multihead")
    if multihead is None:  # legacy inference
        multihead = result.get("dataset", "").startswith("split")
    gh = _gate_history(result)
    seps = _separation_series(gh)
    inv = run_inversion_summary(seps) if seps else {
        "final_gate_separation": float("nan"), "inversion_flag": False,
        "inversion_rate": float("nan"), "time_to_inversion": float("nan")}
    # overall gate<->correctness corr: mean of per-task pearson recorded by the method
    corrs = [g.get("corr_gate_correct") for g in gh
             if g.get("corr_gate_correct") is not None and np.isfinite(g.get("corr_gate_correct"))]
    gate_corr = float(np.mean(corrs)) if corrs else float("nan")
    last_gh = gh[-1] if gh else {}
    bd = _buffer_diag(result)
    cal = result.get("calibration") or {}
    clean_cal = result.get("cleanliness_calibration") or {}
    clean_group = clean_cal.get("clean") or {}
    noisy_group = clean_cal.get("mislabeled") or {}

    final_row = {
        "run_id": run_id,
        "benchmark": result.get("benchmark", result.get("dataset")),
        "dataset": result.get("dataset"),
        "method": label, "family": fam, "gate_signal": sig, "label_free": lf,
        "backbone": result.get("backbone", "mlp"),
        "noise_type": ntype, "noise_rate": round(nrate, 3),
        "condition": result.get("condition"), "seed": result.get("seed"),
        "multihead": bool(multihead),
        "average_accuracy": perf["average_accuracy"],
        "class_il_acc": perf["average_accuracy"] if not multihead else float("nan"),
        "task_il_acc": perf["average_accuracy"] if multihead else float("nan"),
        "mean_forgetting": perf["mean_forgetting"],
        "backward_transfer": perf["backward_transfer"],
        "forward_transfer": perf["forward_transfer"],
        "buffer_purity": bd.get("buffer_purity", last_gh.get("buffer_purity", float("nan"))),
        "buffer_diversity": bd.get("normalized_class_entropy", float("nan")),
        "buffer_class_balance": bd.get("buffer_class_balance", float("nan")),
        "class_count_variance": bd.get("class_count_variance", float("nan")),
        "minority_class_coverage": bd.get("minority_class_coverage", float("nan")),
        "n_classes_represented": bd.get("n_classes_represented", float("nan")),
        "gate_corr": gate_corr,
        "gate_on_correct": last_gh.get("gate_on_correct", float("nan")),
        "gate_on_wrong": last_gh.get("gate_on_wrong", float("nan")),
        "admission_rate": last_gh.get("admission_rate", float("nan")),
        "gate_separation": inv["final_gate_separation"],
        "inversion_flag": inv["inversion_flag"],
        "inversion_rate": inv["inversion_rate"],
        "time_to_inversion": inv["time_to_inversion"],
        "replay_loss_clean": bd.get("replay_loss_clean", float("nan")),
        "replay_loss_noisy": bd.get("replay_loss_noisy", float("nan")),
        "ece": cal.get("ece", float("nan")),
        "brier": cal.get("brier", float("nan")),
        "nll": cal.get("nll", float("nan")),
        "clean_mean_confidence": clean_group.get("mean_confidence", float("nan")),
        "mislabeled_mean_confidence": noisy_group.get("mean_confidence", float("nan")),
        "clean_ece": clean_group.get("ece", float("nan")),
        "mislabeled_ece": noisy_group.get("ece", float("nan")),
        "high_conf_mislabeled_fraction": clean_cal.get(
            "high_conf_mislabeled_fraction", float("nan")
        ),
        "total_time_s": result.get("total_time_s", float("nan")),
        "peak_memory_mb": result.get("peak_memory_mb", float("nan")),
        "n_params": result.get("n_params", float("nan")),
    }

    # per-eval (after each task) accuracy progression + gate separation
    R = np.array(result["acc_matrix"], dtype=float)
    per_eval = []
    for i in range(R.shape[0]):
        per_eval.append({
            "run_id": run_id, "benchmark": final_row["benchmark"], "dataset": final_row["dataset"],
            "method": label, "noise_type": ntype, "noise_rate": round(nrate, 3),
            "seed": result.get("seed"), "eval_after_task": i,
            "avg_acc_so_far": float(R[i, : i + 1].mean()),
            "learned_acc": float(R[i, i]),
            "gate_separation": seps[i] if i < len(seps) else float("nan"),
        })

    # per-task gate diagnostics
    gate_rows = []
    for g in gh:
        t = int(g.get("task", -1))
        goc, gow = g.get("gate_on_correct"), g.get("gate_on_wrong")
        sep = (goc - gow) if (goc is not None and gow is not None
                              and np.isfinite(goc) and np.isfinite(gow)) else float("nan")
        gate_rows.append({
            "run_id": run_id, "benchmark": final_row["benchmark"], "method": label,
            "noise_type": ntype, "noise_rate": round(nrate, 3), "seed": result.get("seed"),
            "task": t, "gate_mean": g.get("gate_mean", float("nan")),
            "gate_on_correct": goc, "gate_on_wrong": gow, "gate_separation": sep,
            "corr_gate_correct": g.get("corr_gate_correct", float("nan")),
            "buffer_purity": g.get("buffer_purity", float("nan")),
            "admission_rate": g.get("admission_rate", float("nan")),
        })

    buf_row = {
        "run_id": run_id, "benchmark": final_row["benchmark"], "method": label,
        "noise_type": ntype, "noise_rate": round(nrate, 3), "seed": result.get("seed"),
        "buffer_size": bd.get("buffer_size", float("nan")),
        "buffer_purity": final_row["buffer_purity"],
        "normalized_class_entropy": bd.get("normalized_class_entropy", float("nan")),
        "buffer_class_balance": bd.get("buffer_class_balance", float("nan")),
        "class_count_variance": bd.get("class_count_variance", float("nan")),
        "minority_class_coverage": bd.get("minority_class_coverage", float("nan")),
        "n_classes_represented": bd.get("n_classes_represented", float("nan")),
        "replay_loss_clean": bd.get("replay_loss_clean", float("nan")),
        "replay_loss_noisy": bd.get("replay_loss_noisy", float("nan")),
    }

    # reliability-diagram bins (pooled over tasks; only present in new-schema runs)
    cal_rows = []
    rb = cal.get("reliability_bins") or {}
    centers = rb.get("centers") or []
    for k, c in enumerate(centers):
        acc = rb["acc"][k]; conf = rb["conf"][k]
        cal_rows.append({
            "run_id": run_id, "benchmark": final_row["benchmark"], "method": label,
            "noise_type": ntype, "noise_rate": round(nrate, 3), "seed": result.get("seed"),
            "bin_center": c, "bin_acc": (acc if acc >= 0 else float("nan")),
            "bin_conf": (conf if conf >= 0 else float("nan")), "bin_count": rb["count"][k]})
    return final_row, per_eval, gate_rows, buf_row, cal_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results/neural_networks_submission/raw_logs"))
    ap.add_argument("--extra-input", nargs="*", default=[],
                    help="additional run-JSON dirs to fold in (e.g. legacy MNIST corpus)")
    ap.add_argument("--output", default=str(ROOT / "results/neural_networks_submission/csv"))
    args = ap.parse_args()

    # (file, is_primary). The primary --input dir is the submission's own runs and
    # is kept verbatim. --extra-input corpora (the legacy reliability dirs) are folded
    # in ONLY for MNIST-family datasets, so legacy SmallCNN-CIFAR runs cannot
    # masquerade as the new ResNet-18 CIFAR results (legacy JSONs carry no backbone).
    files: List = []
    pdir = Path(args.input)
    if pdir.exists():
        files += [(p, True) for p in sorted(pdir.glob("*.json")) if not p.name.startswith("_")]
    for d in args.extra_input:
        dd = Path(d)
        if dd.exists():
            files += [(p, False) for p in sorted(dd.glob("*.json")) if not p.name.startswith("_")]
    print(f"aggregating {len(files)} run JSONs (primary={pdir}, extra={args.extra_input})")

    def is_mnist(ds: str) -> bool:
        return "mnist" in str(ds).lower()

    final, per_eval, gate, buf, cal = [], [], [], [], []
    bad = dup = drop = 0
    seen: set = set()  # dedupe by run_id across input dirs (first dir wins)
    for fp, primary in files:
        if fp.stem in seen:
            dup += 1
            continue
        try:
            result = json.loads(fp.read_text())
            if "acc_matrix" not in result:
                continue
            if not primary and not is_mnist(result.get("dataset", "")):
                drop += 1  # legacy non-MNIST (e.g. SmallCNN CIFAR) -> excluded
                continue
            fr, pe, gr, br, cr = process(result, fp.stem)
            seen.add(fp.stem)
            final.append(fr); per_eval += pe; gate += gr; buf.append(br); cal += cr
        except Exception as e:
            bad += 1
            print(f"  skip {fp.name}: {e}")
    if drop:
        print(f"  ({drop} legacy non-MNIST runs excluded from extra-input)")
    if dup:
        print(f"  ({dup} duplicate run_ids skipped across input dirs)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final).to_csv(out / "final_metrics.csv", index=False)
    pd.DataFrame(per_eval).to_csv(out / "per_eval_metrics.csv", index=False)
    pd.DataFrame(gate).to_csv(out / "gate_diagnostics.csv", index=False)
    pd.DataFrame(buf).to_csv(out / "buffer_diagnostics.csv", index=False)
    pd.DataFrame(cal).to_csv(out / "calibration_bins.csv", index=False)
    print(f"wrote {len(final)} final rows ({bad} skipped) -> {out}/"
          f"{{final_metrics,per_eval_metrics,gate_diagnostics,buffer_diagnostics,calibration_bins}}.csv")


if __name__ == "__main__":
    main()
