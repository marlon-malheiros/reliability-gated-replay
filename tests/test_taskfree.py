"""Task-free online consolidation: stream builder, snapshot schedule, StreamTrainer.

Uses synthetic Permuted-MNIST (single-head, 10-way) so the whole task-free path is
exercised offline at unit scale -- no dataset downloads, no benchmark-scale training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.registry import build_benchmark
from datasets.streams import build_task_free_stream
from methods.pnn.schedule import SnapshotSchedule
from methods.registry import build_method
from methods.stream_trainer import StreamTrainer
from models.registry import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tf_bench():
    cfg = dict(
        name="permuted_mnist", source="synthetic", seed=0,
        n_tasks=3, multihead=False, val_fraction=0.2, max_train_per_task=60,
    )
    return build_benchmark(cfg, data_root=str(ROOT / "data"))


def _tf_method_cfg(schedule="error_gated", anchor=True, closure_mode="error_gated"):
    # high/loose thresholds + tiny detector period so closure events fire in a tiny run
    return dict(
        name="pnn", alpha=0.4, beta=4.0, gating="grad_scale", maturation=True,
        closure_mode=closure_mode, detector_period=2,
        importance={"method": "hybrid"},
        closure={"error_threshold": 5.0, "improvement_threshold": 5.0,
                 "stability_threshold": 5.0, "gamma": 3.0, "consecutive_epochs": 1,
                 "slope_window": 2, "ema_decay": 0.5,
                 "reopening_threshold": 50.0, "reopening_epochs": 2},
        reopening={"enabled": True, "factor": 0.3},
        consolidation={"schedule": schedule, "interval_steps": 3, "pulse": 0.0},
        anchor={"enabled": anchor, "lambda": 1.0, "normalize": True},
    )


def _run_cfg():
    return dict(train=dict(batch_size=32, eval_batch_size=64, probe_size=20),
                optimizer=dict(name="adam", lr=1e-3))


def _stream_cfg(blur_ratio=0.1, local_epochs=2, eval_period=4):
    return dict(blur_ratio=blur_ratio, local_epochs=local_epochs,
                eval_period=eval_period, seed=0)


# --- stream builder -------------------------------------------------------------
def test_stream_disjoint_when_blur_zero(tf_bench):
    s = build_task_free_stream(tf_bench, _stream_cfg(blur_ratio=0.0, local_epochs=1))
    # region_of is non-decreasing (each task owns a contiguous region)
    assert np.all(np.diff(s.region_of) >= 0)
    # disjoint: within each region only that task's data appears
    for t in range(s.n_tasks):
        pos = np.where(s.region_of == t)[0]
        assert np.all(s.source_task_of[pos] == t)
    # local_epochs=1, blur=0 -> stream length == total training samples
    total = sum(len(task.train.tensors[1]) for task in tf_bench.tasks)
    assert len(s) == total


def test_stream_blur_leaks_next_task(tf_bench):
    s = build_task_free_stream(tf_bench, _stream_cfg(blur_ratio=0.3, local_epochs=1))
    # at least one region contains a preview of a later task
    leaked = sum(
        np.any(s.source_task_of[np.where(s.region_of == t)[0]] != t)
        for t in range(s.n_tasks)
    )
    assert leaked >= 1
    assert len(s.boundaries) == s.n_tasks


# --- snapshot schedule ----------------------------------------------------------
def test_schedule_modes():
    none = SnapshotSchedule({"schedule": "none"})
    assert not any(none.fire(i, closure_event=True, at_boundary=True) for i in range(10))

    fixed = SnapshotSchedule({"schedule": "fixed_interval", "interval_steps": 3})
    fired = [i for i in range(1, 13) if fixed.fire(i, False, False)]
    assert fired == [3, 6, 9, 12]

    tb = SnapshotSchedule({"schedule": "task_boundary"})
    assert tb.fire(5, False, at_boundary=True)
    assert not tb.fire(6, False, at_boundary=False)

    eg = SnapshotSchedule({"schedule": "error_gated"})
    assert eg.fire(7, closure_event=True, at_boundary=False)
    assert not eg.fire(8, closure_event=False, at_boundary=False)


def test_random_schedule_count_matched():
    sched = SnapshotSchedule({"schedule": "random", "interval_steps": 5}, seed=1)
    n = 5000
    fires = sum(sched.fire(i, False, False) for i in range(n))
    assert 0.5 * (n / 5) < fires < 1.5 * (n / 5)  # ~ n/interval events


# --- StreamTrainer end-to-end ---------------------------------------------------
@pytest.mark.parametrize("schedule", ["none", "task_boundary", "fixed_interval",
                                       "random", "error_gated"])
def test_stream_trainer_runs_each_schedule(tf_bench, schedule):
    device = torch.device("cpu")
    model = build_model({"name": "mlp"}, tf_bench)
    method = build_method(_tf_method_cfg(schedule=schedule))
    res = StreamTrainer(_run_cfg(), tf_bench, model, method, device,
                        stream_cfg=_stream_cfg()).train()

    assert res["protocol"] == "task_free"
    assert len(res["acc_checkpoints"]) >= 1
    assert len(res["final_acc"]) == tf_bench.tasks.__len__()
    # stream log populated with the required diagnostic fields
    assert res["stream_log"], "stream log should be non-empty"
    rec = res["stream_log"][0]
    for key in ("step", "running_loss", "closure_signal", "mean_P",
                "anchor_loss", "anchor_grad_norm", "param_displacement",
                "p10", "p50", "p90"):
        assert key in rec
    cons = res["consolidation"][0]
    assert cons["snapshot_schedule"] == schedule
    # schedules that snapshot should record theta* updates (anchor enabled)
    if schedule in ("task_boundary", "fixed_interval", "error_gated"):
        assert cons["n_theta_star_updates"] >= 1, schedule


def test_taskboundary_snapshots_at_boundaries(tf_bench):
    device = torch.device("cpu")
    model = build_model({"name": "mlp"}, tf_bench)
    method = build_method(_tf_method_cfg(schedule="task_boundary"))
    trainer = StreamTrainer(_run_cfg(), tf_bench, model, method, device,
                            stream_cfg=_stream_cfg())
    res = trainer.train()
    updates = set(res["consolidation"][0]["theta_star_update_steps"])
    assert updates and updates.issubset(trainer._boundary_steps)


def test_no_anchor_makes_no_snapshots(tf_bench):
    device = torch.device("cpu")
    model = build_model({"name": "mlp"}, tf_bench)
    method = build_method(_tf_method_cfg(schedule="none", anchor=False))
    res = StreamTrainer(_run_cfg(), tf_bench, model, method, device,
                        stream_cfg=_stream_cfg()).train()
    assert res["consolidation"][0]["n_theta_star_updates"] == 0
    assert res["stream_log"][0]["anchor_loss"] == 0.0


@pytest.mark.parametrize("name,cfg", [
    ("mas", {"name": "mas", "lambda": 1.0, "interval_steps": 3}),
    ("online_ewc", {"name": "ewc", "online": True, "lambda": 10.0, "interval_steps": 3}),
    ("online_si", {"name": "si", "online": True, "lambda": 1.0, "interval_steps": 3}),
    ("adam", {"name": "adam"}),
])
def test_taskfree_baselines_run(tf_bench, name, cfg):
    device = torch.device("cpu")
    model = build_model({"name": "mlp"}, tf_bench)
    method = build_method(cfg)
    res = StreamTrainer(_run_cfg(), tf_bench, model, method, device,
                        stream_cfg=_stream_cfg()).train()
    assert len(res["final_acc"]) == len(tf_bench.tasks)
    assert np.isfinite(res["acc_checkpoints"][-1]["avg_acc"])


def test_taskfree_metrics_contract(tf_bench):
    """compute_run_metrics on a task-free result yields the tf_* keys the
    task-free analysis/report depend on, and they aggregate over seeds."""
    from analysis.metrics import aggregate_over_seeds, compute_run_metrics
    device = torch.device("cpu")
    model = build_model({"name": "mlp"}, tf_bench)
    method = build_method(_tf_method_cfg(schedule="error_gated"))
    res = StreamTrainer(_run_cfg(), tf_bench, model, method, device,
                        stream_cfg=_stream_cfg()).train()
    m = compute_run_metrics(res)
    for k in ("tf_final_avg_acc", "tf_anytime_avg_acc", "tf_forgetting",
              "n_closure_events", "n_theta_star_updates"):
        assert k in m, k
    assert 0.0 <= m["tf_final_avg_acc"] <= 1.0
    agg = aggregate_over_seeds([m, m])
    assert agg["tf_forgetting"]["n"] == 2 and "mean" in agg["tf_final_avg_acc"]
