"""Table generation (Tables 1-10 of the spec) as CSV + Markdown.

Tables 1-3 are static / from config; 4-10 are computed from the analysis bundle.
We render Markdown without a hard ``tabulate`` dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import stats as st


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, outdir: Path, name: str, caption: str = "") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False)
    md = (f"**{caption}**\n\n" if caption else "") + _df_to_markdown(df) + "\n"
    (outdir / f"{name}.md").write_text(md)


def _fmt(bundle, label, key, pct=False):
    a = bundle["methods"][label]["agg"].get(key)
    if not a:
        return "-"
    m, s = a["mean"], a["std"]
    if pct:
        return f"{100*m:.1f}±{100*s:.1f}"
    return f"{m:.3f}±{s:.3f}"


def _methods(bundle, kinds):
    return sorted([k for k, v in bundle["methods"].items() if v["kind"] in kinds])


def _sort_by_metric(bundle, labels, metric="average_accuracy", descending=True):
    def key(label):
        return bundle["methods"][label]["agg"].get(metric, {}).get("mean", float("-inf"))

    return sorted(labels, key=key, reverse=descending)


def _comparison_methods(bundle):
    pnn = bundle.get("pnn_label")
    return sorted([k for k in bundle["methods"] if k != pnn])


# --- static tables ---------------------------------------------------------- #
def table1_datasets(outdir):
    df = pd.DataFrame([
        {"dataset": "Split-MNIST", "type": "continual (primary)", "tasks": 5,
         "classes/task": 2, "protocol": "task-incremental"},
        {"dataset": "MNIST", "type": "single-task", "tasks": 1,
         "classes/task": 10, "protocol": "standard"},
        {"dataset": "Fashion-MNIST", "type": "single-task", "tasks": 1,
         "classes/task": 10, "protocol": "standard"},
        {"dataset": "Permuted-MNIST", "type": "continual (optional)", "tasks": "N",
         "classes/task": 10, "protocol": "domain-incremental"},
    ])
    save_table(df, outdir, "table1_datasets", "Table 1 — Dataset configuration")


def table2_architectures(outdir):
    df = pd.DataFrame([
        {"model": "MLP (A)", "architecture": "784→512→256→head", "activation": "ReLU",
         "params": "~0.5M", "notes": "shared backbone, per-task heads"},
        {"model": "CNN (B)", "architecture": "[Conv-ReLU-Pool]×2→Dense(128)→head",
         "activation": "ReLU", "params": "~0.4M", "notes": "32/64 channels"},
    ])
    save_table(df, outdir, "table2_architectures", "Table 2 — Model architectures")


def table3_hyperparameters(outdir, config: Dict[str, Any]):
    opt = config.get("optimizer", {})
    tr = config.get("train", {})
    pnn = config.get("_pnn_defaults", {})
    clo = pnn.get("closure", {})
    rows = [
        ("optimizer", opt.get("name", "adam")),
        ("learning rate", opt.get("lr", 1e-3)),
        ("batch size", tr.get("batch_size", 128)),
        ("max epochs/task", tr.get("epochs", "-")),
        ("PNN α (maturation rate)", pnn.get("alpha", "-")),
        ("PNN β (gating)", pnn.get("beta", "-")),
        ("closure γ", clo.get("gamma", "-")),
        ("error threshold", clo.get("error_threshold", "-")),
        ("improvement threshold", clo.get("improvement_threshold", "-")),
        ("stability threshold", clo.get("stability_threshold", "-")),
        ("consecutive epochs N", clo.get("consecutive_epochs", "-")),
    ]
    df = pd.DataFrame(rows, columns=["hyperparameter", "value"])
    save_table(df, outdir, "table3_hyperparameters", "Table 3 — Hyperparameters")


# --- result tables ---------------------------------------------------------- #
def _result_table(bundle, labels, outdir, name, caption, extra_keys=()):
    rows = []
    for m in labels:
        row = {
            "method": m,
            "avg_acc": _fmt(bundle, m, "average_accuracy", pct=True),
            "forgetting": _fmt(bundle, m, "mean_forgetting", pct=True),
            "BWT": _fmt(bundle, m, "backward_transfer"),
            "FWT": _fmt(bundle, m, "forward_transfer"),
        }
        for k in extra_keys:
            row[k] = _fmt(bundle, m, k)
        rows.append(row)
    df = pd.DataFrame(rows)
    save_table(df, outdir, name, caption)
    return df


def table4_baselines(bundle, outdir):
    labels = _sort_by_metric(bundle, _methods(bundle, {"baseline"}))
    return _result_table(bundle, labels, outdir,
                         "table4_baselines", "Table 4 — Baseline results")


def table5_pnn(bundle, outdir):
    labels = _sort_by_metric(bundle, _methods(bundle, {"pnn"}))
    return _result_table(bundle, labels, outdir, "table5_pnn", "Table 5 — PNN results",
                         extra_keys=("mean_P", "frac_consolidated"))


def table6_ablations(bundle, outdir):
    labels = _sort_by_metric(bundle, _methods(bundle, {"ablation"}))
    if not labels:
        save_table(pd.DataFrame([{"note": "no ablation runs in this set"}]),
                   outdir, "table6_ablations", "Table 6 — Ablation results")
        return
    return _result_table(bundle, labels, outdir, "table6_ablations",
                         "Table 6 — Ablation results",
                         extra_keys=("mean_P", "frac_consolidated", "representational_drift"))


def table7_stats(bundle, outdir, metric="average_accuracy"):
    pnn = bundle.get("pnn_label")
    baselines = _comparison_methods(bundle)
    if not pnn:
        save_table(pd.DataFrame([{"note": "no PNN run"}]), outdir, "table7_stats",
                   "Table 7 — Statistical tests")
        return
    pnn_vals = bundle["methods"][pnn]["agg"].get(metric, {}).get("values", [])
    rows, pvals = [], []
    for b in baselines:
        bvals = bundle["methods"][b]["agg"].get(metric, {}).get("values", [])
        n = min(len(pnn_vals), len(bvals))
        comp = st.compare(pnn_vals[:n], bvals[:n]) if n >= 2 else {}
        rows.append({"comparison": b,
                     "mean_diff": f"{comp.get('mean_diff', float('nan')):.3f}",
                     "p_ttest": f"{comp.get('p_ttest', float('nan')):.4f}",
                     "p_wilcoxon": f"{comp.get('p_wilcoxon', float('nan')):.4f}"})
        pvals.append(comp.get("p_ttest", float("nan")))
    corr = st.holm_bonferroni(pvals)
    for r, cp, rej in zip(rows, corr["corrected"], corr["reject"]):
        r["p_holm"] = f"{cp:.4f}" if cp == cp else "nan"
        r["significant"] = "yes" if rej else "no"
    save_table(pd.DataFrame(rows), outdir, "table7_stats",
               f"Table 7 — Statistical tests (primary PNN vs comparisons, {metric}, paired)")


def table8_effect_sizes(bundle, outdir, metric="average_accuracy"):
    pnn = bundle.get("pnn_label")
    baselines = _comparison_methods(bundle)
    if not pnn:
        save_table(pd.DataFrame([{"note": "no PNN run"}]), outdir, "table8_effect_sizes",
                   "Table 8 — Effect sizes")
        return
    pnn_vals = bundle["methods"][pnn]["agg"].get(metric, {}).get("values", [])
    rows = []
    for b in baselines:
        bvals = bundle["methods"][b]["agg"].get(metric, {}).get("values", [])
        n = min(len(pnn_vals), len(bvals))
        comp = st.compare(pnn_vals[:n], bvals[:n]) if n >= 2 else {}
        rows.append({"comparison": b,
                     "cohens_d": f"{comp.get('cohens_d', float('nan')):.3f}",
                     "rank_biserial": f"{comp.get('rank_biserial', float('nan')):.3f}"})
    save_table(pd.DataFrame(rows), outdir, "table8_effect_sizes",
               "Table 8 — Effect sizes (primary PNN vs comparisons)")


def table9_cost(bundle, outdir):
    labels = _sort_by_metric(bundle, _methods(bundle, {"baseline", "pnn"}))
    rows = []
    for m in labels:
        rows.append({
            "method": m,
            "train_time_s": _fmt(bundle, m, "total_time_s"),
            "peak_mem_mb": _fmt(bundle, m, "peak_memory_mb"),
            "inference_us/sample": _fmt(bundle, m, "inference_cost_us_per_sample"),
        })
    save_table(pd.DataFrame(rows), outdir, "table9_cost",
               "Table 9 — Computational cost")


def _method_cfg(bundle, label: str) -> Dict[str, Any]:
    return bundle["methods"][label].get("sample_result", {}).get("method_cfg", {}) or {}


def _sample_result(bundle, label: str) -> Dict[str, Any]:
    return bundle["methods"][label].get("sample_result", {}) or {}


def _n_params(bundle, label: str) -> int:
    agg = bundle["methods"][label]["agg"].get("n_params", {})
    if agg.get("mean"):
        return int(round(float(agg["mean"])))
    return int(_sample_result(bundle, label).get("n_params", 0))


def _input_numel(bundle) -> int:
    data = bundle.get("manifest", {}).get("data", {})
    dataset = bundle.get("dataset", "")
    source = data.get("source", "")
    if "mnist" in dataset or "mnist" in source:
        return 28 * 28
    # Conservative image fallback for CIFAR-like future runs.
    if "cifar" in dataset or "cifar" in source:
        return 3 * 32 * 32
    return int(data.get("input_numel", 28 * 28))


def _state_estimate(bundle, label: str) -> Dict[str, Any]:
    """Estimate persistent method state.

    This intentionally reports simple storage estimates rather than pretending to
    know allocator/runtime overhead. Float tensors are counted as fp32. Replay
    storage is counted as raw input examples plus int64 label/task ids.
    """
    cfg = _method_cfg(bundle, label)
    name = cfg.get("name", label)
    n_params = max(_n_params(bundle, label), 0)
    n_tasks = int(_sample_result(bundle, label).get("n_tasks", 1))
    float_mb = n_params * 4 / (1024 ** 2)
    input_mb = _input_numel(bundle) * 4 / (1024 ** 2)
    raw_examples = 0
    raw_mb = 0.0
    state_tensors = 0
    state = "none"

    if name == "replay":
        raw_examples = int(cfg.get("buffer_size", 0))
        raw_mb = raw_examples * (input_mb + 16 / (1024 ** 2))
        state = f"raw replay buffer ({raw_examples} examples)"
    elif name == "ewc":
        state_tensors = 2 * n_tasks
        state = f"classic EWC anchors+Fisher ({n_tasks} tasks)"
    elif name == "si":
        state_tensors = 3
        state = "SI omega + path accumulator + anchor"
    elif name == "l2":
        state_tensors = 1
        state = "previous-task weight anchor"
    elif name == "pnn":
        anchor_enabled = bool(cfg.get("anchor", {}).get("enabled", False))
        # P, grad EMA, grad^2 EMA, plus optional anchor snapshot.
        state_tensors = 3 + (1 if anchor_enabled else 0)
        state = "P + gradient statistics"
        if anchor_enabled:
            state += " + anchor snapshot"

    state_mb = state_tensors * float_mb
    total_mb = state_mb + raw_mb
    return {
        "method": label,
        "n_params": n_params,
        "persistent_state_mb": f"{total_mb:.2f}",
        "param_state_mb": f"{state_mb:.2f}",
        "raw_buffer_mb": f"{raw_mb:.2f}",
        "raw_examples": raw_examples,
        "stores_raw_data": "yes" if raw_examples else "no",
        "state_summary": state,
    }


def table10_memory_efficiency(bundle, outdir):
    labels = _sort_by_metric(bundle, _methods(bundle, {"baseline", "pnn"}))
    rows = [_state_estimate(bundle, m) for m in labels]
    save_table(pd.DataFrame(rows), outdir, "table10_memory_efficiency",
               "Table 10 — Persistent memory and raw-data storage estimate")


def make_all_tables(bundle, outdir, config) -> None:
    outdir = Path(outdir)
    table1_datasets(outdir)
    table2_architectures(outdir)
    table3_hyperparameters(outdir, config)
    table4_baselines(bundle, outdir)
    table5_pnn(bundle, outdir)
    table6_ablations(bundle, outdir)
    table7_stats(bundle, outdir)
    table8_effect_sizes(bundle, outdir)
    table9_cost(bundle, outdir)
    table10_memory_efficiency(bundle, outdir)
