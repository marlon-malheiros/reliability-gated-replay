"""Unit tests: ClosureDetector reopening configuration is actually used.

Regression test for the wiring bug found in the audit: the reopening *threshold*
and *patience* were configured under the ``reopening:`` YAML block, but the
detector read them from the ``closure:`` block it was handed -> they were silently
ignored (the defaults happened to match, masking it). These tests prove (1) the
detector honors its configured threshold/patience, and (2) ``PNNMethod`` now wires
the ``reopening:`` block into the detector.
"""
import yaml

from methods.pnn.closure import ClosureDetector
from methods.pnn.method import PNNMethod


def test_detector_uses_configured_threshold_and_patience():
    det = ClosureDetector({"reopening_threshold": 0.8, "reopening_epochs": 3})
    assert det.reopen_thr == 0.8
    assert det.reopen_M == 3


def test_reopening_fires_at_threshold_after_patience():
    # threshold 0.8, patience 2: reopening should fire on the 2nd consecutive epoch
    # whose EMA loss exceeds 0.8, and not before.
    det = ClosureDetector(
        {"reopening_threshold": 0.8, "reopening_epochs": 2, "ema_decay": 0.0}
    )  # ema_decay=0 -> E_t == val_loss (no smoothing), makes the test exact
    assert det.update(0.1)["just_reopened"] is False   # below threshold
    assert det.update(0.9)["just_reopened"] is False   # 1st above -> patience not met
    assert det.update(0.9)["just_reopened"] is True     # 2nd consecutive above -> fires
    # streak resets once back below threshold
    assert det.update(0.1)["just_reopened"] is False


def test_below_threshold_never_reopens():
    det = ClosureDetector({"reopening_threshold": 0.8, "reopening_epochs": 2, "ema_decay": 0.0})
    for _ in range(10):
        assert det.update(0.2)["just_reopened"] is False


def test_pnn_method_wires_reopening_block_into_detector():
    """The canonical ``reopening:`` block (threshold/epochs) reaches the detector."""
    cfg = {
        "name": "pnn",
        "closure": {"error_threshold": 0.15},
        "reopening": {"enabled": True, "threshold": 0.77, "epochs": 4, "factor": 0.3},
    }
    ccfg = PNNMethod(cfg)._closure_cfg()
    assert ccfg["reopening_threshold"] == 0.77
    assert ccfg["reopening_epochs"] == 4
    assert ccfg["reopen_enabled"] is True
    det = ClosureDetector(ccfg)
    assert det.reopen_thr == 0.77 and det.reopen_M == 4


def test_pnn_method_backcompat_closure_block_reopening():
    """Legacy task-free convention (reopening_* under ``closure:``) still works."""
    cfg = {
        "name": "pnn",
        "closure": {"error_threshold": 0.6, "reopening_threshold": 1.2, "reopening_epochs": 2},
        "reopening": {"enabled": True, "factor": 0.3},  # no threshold/epochs here
    }
    ccfg = PNNMethod(cfg)._closure_cfg()
    assert ccfg["reopening_threshold"] == 1.2   # taken from the closure block
    assert ccfg["reopening_epochs"] == 2


def test_shipped_pnn_yaml_reopening_is_wired():
    """The shipped configs/methods/pnn.yaml reopening values reach the detector."""
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[1] / "configs" / "methods" / "pnn.yaml"
    mcfg = yaml.safe_load(p.read_text())["method"]
    ccfg = PNNMethod(mcfg)._closure_cfg()
    reop = mcfg["reopening"]
    assert ccfg["reopening_threshold"] == reop["threshold"]
    assert ccfg["reopening_epochs"] == reop["epochs"]
