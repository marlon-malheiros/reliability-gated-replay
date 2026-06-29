"""Continual-learning metrics computed from a single run result.

The accuracy matrix ``R[i][j]`` = test accuracy on task j after training task i
is the source of the performance/forgetting/transfer metrics (Lopez-Paz & Ranzato
conventions). Plasticity, representation and consolidation metrics are read from
the run's logged drift / probe-features / consolidation state.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .cka import cka_matrix, linear_cka, mean_cosine


def _R(result: Dict[str, Any]) -> np.ndarray:
    return np.array(result["acc_matrix"], dtype=float)


def performance_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    R = _R(result)
    T = R.shape[0]
    final = R[-1]
    diag = np.diag(R)
    max_acc = R.max(0)  # best ever per task
    forgetting = (max_acc - final)[: T - 1]  # last task can't be forgotten
    out = {
        "average_accuracy": float(final.mean()),
        "final_acc_per_task": final.tolist(),
        "learning_accuracy": float(diag.mean()),
        "mean_forgetting": float(forgetting.mean()) if T > 1 else 0.0,
        "forgetting_per_task": forgetting.tolist(),
        "max_acc_per_task": max_acc.tolist(),
    }
    if T > 1:
        bwt = np.mean([R[-1, j] - R[j, j] for j in range(T - 1)])
        init = np.array(result.get("init_acc", [0.0] * T))
        fwt = np.mean([R[j - 1, j] - init[j] for j in range(1, T)])
        out["backward_transfer"] = float(bwt)
        out["forward_transfer"] = float(fwt)
    else:
        out["backward_transfer"] = 0.0
        out["forward_transfer"] = 0.0
    return out


def plasticity_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    cons = result.get("consolidation", [{}])
    last = cons[-1] if cons else {}
    return {
        "plasticity_index": float(np.mean(np.diag(_R(result)))),  # learnability of new tasks
        "weight_drift_final": float(result.get("weight_drift", [0.0])[-1]),
        "weight_drift_per_task": result.get("weight_drift", []),
        "mean_grad_norm": float(np.mean(result.get("grad_norm", [0.0]))),
        "grad_norm_per_task": result.get("grad_norm", []),
        "protected_fraction": float(last.get("frac_consolidated", 0.0)),
        "effective_plasticity": float(last.get("mean_gating_factor", 1.0)),
    }


def representation_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    reps = result.get("representations", [])
    if len(reps) < 2:
        return {
            "representational_drift": 0.0,
            "feature_stability": 1.0,
            "mean_cosine": 1.0,
            "cka_matrix": [[1.0]],
        }
    M = cka_matrix(reps)
    drift = float(np.mean([1.0 - linear_cka(np.array(reps[0]), np.array(reps[t]))
                           for t in range(1, len(reps))]))
    stability = float(np.mean([M[t - 1, t] for t in range(1, len(reps))]))
    cos = float(np.mean([mean_cosine(np.array(reps[0]), np.array(reps[t]))
                         for t in range(1, len(reps))]))
    return {
        "representational_drift": drift,
        "feature_stability": stability,
        "mean_cosine": cos,
        "cka_matrix": M.tolist(),
    }


def consolidation_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    cons = result.get("consolidation", [{}])
    last = cons[-1] if cons else {}
    out = {
        "mean_P": float(last.get("mean_P", 0.0)),
        "var_P": float(last.get("var_P", 0.0)),
        "frac_consolidated": float(last.get("frac_consolidated", 0.0)),
        "corr_P_importance": float(last.get("corr_P_importance", 0.0)),
        "P_hist": last.get("P_hist"),
        "P_sample": last.get("P_sample"),
        "per_layer_mean_P": last.get("per_layer_mean_P", {}),
    }
    for k in ("closure_epoch", "closure_error", "closure_acc"):
        if k in last:
            out[k] = last[k]
    return out


def taskfree_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Task-free (stream) metrics from periodic all-tasks evaluation checkpoints.

    Forgetting here is the standard continual-eval form: per task, the best
    accuracy reached at any checkpoint minus the final accuracy. The "anytime"
    accuracy is the mean of the average-accuracy curve over the stream (an AUC),
    which rewards methods that stay accurate *throughout*, not just at the end.
    """
    cps = result.get("acc_checkpoints") or []
    if not cps:
        return {}
    per_task = np.array([cp["per_task_acc"] for cp in cps], dtype=float)  # (n_checkpoints, T)
    avg_curve = np.array([cp.get("avg_acc", np.mean(cp["per_task_acc"])) for cp in cps], dtype=float)
    final = per_task[-1]
    max_so_far = per_task.max(0)
    cons = (result.get("consolidation") or [{}])[-1] or {}
    return {
        "tf_final_avg_acc": float(final.mean()),
        "tf_anytime_avg_acc": float(avg_curve.mean()),
        "tf_forgetting": float(np.mean(max_so_far - final)),
        "tf_forgetting_per_task": (max_so_far - final).tolist(),
        "tf_final_acc_per_task": final.tolist(),
        "n_closure_events": int(cons.get("n_closure_events", 0)),
        "n_theta_star_updates": int(cons.get("n_theta_star_updates", 0)),
        "snapshot_schedule": cons.get("snapshot_schedule"),
    }


def efficiency_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_time_s": float(result.get("total_time_s", 0.0)),
        "peak_memory_mb": float(result.get("peak_memory_mb", 0.0)),
        "inference_cost_us_per_sample": float(result.get("inference_cost_us_per_sample", 0.0)),
        "n_params": int(result.get("n_params", 0)),
    }


def compute_run_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """All metrics for one run (scalars + arrays for plotting)."""
    m: Dict[str, Any] = {}
    m.update(performance_metrics(result))
    m.update(plasticity_metrics(result))
    m.update(representation_metrics(result))
    m.update(consolidation_metrics(result))
    if result.get("protocol") == "task_free":
        m.update(taskfree_metrics(result))
    m.update(efficiency_metrics(result))
    m["acc_matrix"] = result.get("acc_matrix")
    return m


# scalar keys aggregated across seeds (arrays excluded)
SCALAR_KEYS = [
    "average_accuracy", "learning_accuracy", "mean_forgetting", "backward_transfer",
    "forward_transfer", "plasticity_index", "weight_drift_final", "mean_grad_norm",
    "protected_fraction", "effective_plasticity", "representational_drift",
    "feature_stability", "mean_cosine", "mean_P", "var_P", "frac_consolidated",
    "corr_P_importance", "total_time_s", "peak_memory_mb",
    "inference_cost_us_per_sample", "n_params",
    # task-free protocol
    "tf_final_avg_acc", "tf_anytime_avg_acc", "tf_forgetting",
    "n_closure_events", "n_theta_star_updates",
]


def aggregate_over_seeds(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scalar metrics across seed runs -> per-key {mean,std,values}."""
    agg: Dict[str, Any] = {}
    for k in SCALAR_KEYS:
        vals = [float(r[k]) for r in run_metrics if k in r and r[k] is not None]
        if vals:
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "values": vals,
            }
    return agg
