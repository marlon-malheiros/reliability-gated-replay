"""Unit tests: gate-inversion detection (flag / rate / time-to-inversion).

Covers (Phase-1 deliverable e). Inversion = gate separation < 0 (the gate prefers
mislabeled samples). Time-to-inversion requires the separation to go negative AND
stay negative for >= 2 consecutive evaluations; a single-evaluation blip does not
count, and a never-inverting run records NaN.
"""
import numpy as np

from analysis.gate_metrics import (
    inversion_flag,
    inversion_rate,
    run_inversion_summary,
    time_to_inversion,
)


def test_inversion_flag():
    assert inversion_flag(-0.1) is True
    assert inversion_flag(0.1) is False
    assert inversion_flag(0.0) is False
    assert inversion_flag(float("nan")) is False


def test_inversion_rate():
    assert inversion_rate([0.3, 0.2, -0.1, -0.2]) == 0.5
    assert inversion_rate([0.3, 0.2]) == 0.0
    assert np.isnan(inversion_rate([float("nan")]))


def test_time_to_inversion_sustained():
    # benign then sustained inversion starting at index 2
    sep = [0.4, 0.2, -0.1, -0.3, -0.2]
    assert time_to_inversion(sep) == 2


def test_single_blip_does_not_count():
    # one negative at index 2 then recovers -> not a sustained inversion
    sep = [0.4, 0.2, -0.1, 0.3, 0.2]
    assert np.isnan(time_to_inversion(sep))


def test_never_inverts_is_nan():
    assert np.isnan(time_to_inversion([0.5, 0.4, 0.3, 0.2]))


def test_inversion_requires_two_consecutive_by_default():
    # negative only at the final step -> cannot satisfy "2 consecutive"
    assert np.isnan(time_to_inversion([0.4, 0.3, 0.2, -0.1]))
    # but a custom min_consecutive=1 catches it at the last index
    assert time_to_inversion([0.4, 0.3, 0.2, -0.1], min_consecutive=1) == 3


def test_run_inversion_summary_bundle():
    s = run_inversion_summary([0.4, 0.2, -0.1, -0.3])
    assert s["inversion_flag"] is True
    assert s["time_to_inversion"] == 2
    assert abs(s["inversion_rate"] - 0.5) < 1e-9
    assert s["final_gate_separation"] == -0.3

    safe = run_inversion_summary([0.4, 0.3, 0.2, 0.1])
    assert safe["inversion_flag"] is False
    assert np.isnan(safe["time_to_inversion"])
    assert safe["inversion_rate"] == 0.0
