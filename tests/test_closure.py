"""Closure detector: error-gated triggering, gradual signal, reopening."""
from methods.pnn.closure import ClosureDetector


def _cfg():
    return dict(
        error_threshold=0.2, improvement_threshold=0.05, stability_threshold=0.02,
        gamma=5.0, consecutive_epochs=2, slope_window=2, ema_decay=0.5,
        reopen_enabled=True, reopening_threshold=0.5, reopening_epochs=2,
    )


def test_high_error_keeps_plastic():
    d = ClosureDetector(_cfg())
    sig = d.update(1.0)
    assert sig["closure_signal"] < 0.2
    assert not sig["closed"]


def test_low_stable_error_triggers_closure():
    d = ClosureDetector(_cfg())
    last = None
    # decay then a sustained low-stable tail so the EMA converges below threshold
    for loss in [1.0, 0.6, 0.3, 0.12, 0.06] + [0.05] * 8:
        last = d.update(loss)
    assert d.closed
    assert d.closure_epoch is not None
    assert last["closure_signal"] > 0.5


def test_signal_increases_as_error_falls():
    d = ClosureDetector(_cfg())
    first = d.update(1.0)["closure_signal"]
    for loss in [0.5, 0.2, 0.08, 0.05]:
        last = d.update(loss)["closure_signal"]
    assert last > first


def test_reopening_on_error_spike():
    d = ClosureDetector(_cfg())
    for loss in [0.05] * 5:           # close first
        d.update(loss)
    assert d.closed
    # sustained high error spike; the EMA needs a few epochs to cross the threshold,
    # then reopening fires once (when the streak reaches M).
    events = [d.update(0.9)["just_reopened"] for _ in range(6)]
    assert events[0] is False         # EMA lag: not immediate
    assert any(events)                # reopening does fire
