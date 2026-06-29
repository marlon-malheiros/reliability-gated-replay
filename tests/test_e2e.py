"""End-to-end smoke: train -> results JSON -> analysis (figures/tables/reports)."""
import json

from analysis.plots import make_all_figures
from analysis.reports import write_executive_summary, write_final_report
from analysis.tables import make_all_tables
from methods.registry import build_method
from methods.trainer import ContinualTrainer
from models.registry import build_model


def _run(method_cfg, bench, train_cfg, device):
    model = build_model({"name": "mlp"}, bench)
    method = build_method(method_cfg)
    res = ContinualTrainer(train_cfg, bench, model, method, device).train()
    res["label"] = method_cfg["name"]
    return res


def test_end_to_end_pipeline(tiny_bench, train_cfg, device, tmp_path):
    pnn_cfg = dict(
        name="pnn", alpha=0.3, beta=4.0, gating="grad_scale",
        importance={"method": "hybrid"},
        closure={"error_threshold": 0.6, "improvement_threshold": 0.2,
                 "stability_threshold": 0.2, "consecutive_epochs": 1},
        reopening={"enabled": True, "factor": 0.3},
    )
    runs = [_run(pnn_cfg, tiny_bench, train_cfg, device),
            _run({"name": "adam"}, tiny_bench, train_cfg, device)]

    # accuracy matrix is square and populated
    for r in runs:
        T = r["n_tasks"]
        assert len(r["acc_matrix"]) == T and all(len(row) == T for row in r["acc_matrix"])

    # build a minimal bundle as analyze_results would
    from analysis.metrics import aggregate_over_seeds, compute_run_metrics
    methods = {}
    for r in runs:
        m = compute_run_metrics(r)
        kind = "pnn" if r["label"] == "pnn" else "baseline"
        methods[r["label"]] = {"kind": kind, "runs": [m],
                               "agg": aggregate_over_seeds([m]), "sample_result": r}
    bundle = {"methods": methods, "pnn_label": "pnn", "dataset": "split_mnist",
              "n_seeds": 1, "note": "test", "manifest": {}}

    made = make_all_figures(bundle, tmp_path / "figures")
    assert len(made) >= 10
    assert (tmp_path / "figures" / "fig03_accuracy_over_tasks.png").exists()

    make_all_tables(bundle, tmp_path / "tables", {"train": {}, "optimizer": {}})
    assert (tmp_path / "tables" / "table4_baselines.csv").exists()

    es = write_executive_summary(bundle, {}, tmp_path / "report")
    fr = write_final_report(bundle, {}, tmp_path / "report")
    assert es.exists() and fr.exists()
    assert "PNN" in es.read_text()


def test_consolidation_state_present(tiny_bench, train_cfg, device):
    pnn_cfg = dict(name="pnn", alpha=0.3,
                   closure={"error_threshold": 0.6, "improvement_threshold": 0.2,
                            "stability_threshold": 0.2, "consecutive_epochs": 1})
    r = _run(pnn_cfg, tiny_bench, train_cfg, device)
    last = r["consolidation"][-1]
    assert "mean_P" in last and 0.0 <= last["mean_P"] <= 1.0
    assert "frac_consolidated" in last
