"""Forgetting / BWT / FWT computed from a hand-built accuracy matrix."""
import numpy as np

from analysis.metrics import performance_metrics


def test_performance_metrics_known_values():
    R = [
        [0.90, 0.10, 0.10],
        [0.80, 0.95, 0.10],
        [0.70, 0.85, 0.92],
    ]
    result = {"acc_matrix": R, "init_acc": [0.1, 0.1, 0.1]}
    m = performance_metrics(result)
    assert np.isclose(m["average_accuracy"], np.mean([0.70, 0.85, 0.92]))
    assert np.isclose(m["learning_accuracy"], np.mean([0.90, 0.95, 0.92]))
    # forgetting on tasks 0,1: (0.90-0.70), (0.95-0.85)
    assert np.isclose(m["mean_forgetting"], np.mean([0.20, 0.10]))
    # BWT: ((0.70-0.90)+(0.85-0.95))/2
    assert np.isclose(m["backward_transfer"], (-0.20 - 0.10) / 2)
    # FWT: ((R[0][1]-0.1)+(R[1][2]-0.1))/2 = 0
    assert np.isclose(m["forward_transfer"], 0.0)


def test_single_task_no_forgetting():
    m = performance_metrics({"acc_matrix": [[0.9]], "init_acc": [0.1]})
    assert m["mean_forgetting"] == 0.0
    assert m["backward_transfer"] == 0.0
