#!/usr/bin/env python
"""Generate the submission tables (1-6) from the aggregated CSVs.

Each table is written as a machine-readable ``.csv`` (the deliverable) and a
human-readable ``.md``. No number is hand-typed: everything is computed from
``final_metrics.csv`` / ``gate_diagnostics.csv``. Mean +/- std is over seeds.

    python scripts/make_nn_submission_tables.py \
        --input results/neural_networks_submission/csv \
        --output results/neural_networks_submission/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CIFAR = ["seq_cifar10", "split_cifar10", "seq_cifar100"]


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _ms(series, nd=3):
    s = series.dropna()
    if s.empty:
        return ""
    if len(s) == 1:
        return f"{s.iloc[0]:.{nd}f}"
    return f"{s.mean():.{nd}f}±{s.std(ddof=0):.{nd}f}"


def _to_md(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavored markdown table (avoids the tabulate dependency)."""
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([head, sep] + body)


def _write(df: pd.DataFrame, out: Path, name: str, note: str = ""):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    md = (f"<!-- {note} -->\n\n" if note else "") + _to_md(df)
    (out / f"{name}.md").write_text(md)
    print(f"  wrote {name}.csv ({len(df)} rows)")


# Table 1 -- benchmark x method coverage (run counts) + condition summary
def table1(final, out):
    if final.empty:
        _write(pd.DataFrame(), out, "table1_benchmark_method_coverage", "no runs yet"); return
    piv = (final.groupby(["benchmark", "method"]).run_id.nunique()
           .unstack(fill_value=0).reset_index())
    _write(piv, out, "table1_benchmark_method_coverage",
           "cell = number of completed runs (seeds x conditions)")


# Table 2 -- main final metrics on Seq-/Split-CIFAR-10
def table2(final, out):
    sub = final[final.benchmark.isin(["seq_cifar10", "split_cifar10"])]
    rows = []
    for (b, m, r), g in sub.groupby(["benchmark", "method", "noise_rate"]):
        rows.append({"benchmark": b, "method": m, "noise_rate": r,
                     "avg_accuracy": _ms(g.average_accuracy),
                     "forgetting": _ms(g.mean_forgetting),
                     "backward_transfer": _ms(g.backward_transfer),
                     "buffer_purity": _ms(g.buffer_purity),
                     "ece": _ms(g.ece), "n_seeds": g.seed.nunique()})
    df = pd.DataFrame(rows).sort_values(["benchmark", "noise_rate", "method"]) if rows else pd.DataFrame()
    _write(df, out, "table2_main_cifar10",
           "Seq-/Split-CIFAR-10 ResNet-18; mean+/-std over seeds. Empty => CIFAR grid still running.")


# Table 3 -- noisy-label benchmark (CIFAR-10N preferred; synthetic asym fallback)
def table3(final, out):
    c10n = final[final.benchmark.astype(str).str.startswith("cifar10n")]
    if not c10n.empty:
        src, note = c10n, "CIFAR-10N (real human-annotation noise)."
    else:
        src = final[(final.benchmark.isin(["seq_cifar10", "split_cifar10"]))
                    & (final.noise_type == "asymmetric")]
        note = ("FALLBACK: CIFAR-10N labels unavailable -> synthetic ASYMMETRIC noise on "
                "Seq-/Split-CIFAR-10 (clearly marked as fallback, per spec).")
    rows = []
    for (b, m, r), g in src.groupby(["benchmark", "method", "noise_rate"]):
        rows.append({"benchmark": b, "method": m, "noise_rate": r,
                     "avg_accuracy": _ms(g.average_accuracy), "forgetting": _ms(g.mean_forgetting),
                     "buffer_purity": _ms(g.buffer_purity), "gate_separation": _ms(g.gate_separation),
                     "inversion_rate": _ms(g.inversion_rate), "n_seeds": g.seed.nunique()})
    df = pd.DataFrame(rows).sort_values(["benchmark", "noise_rate", "method"]) if rows else pd.DataFrame()
    _write(df, out, "table3_noisy_label", note)


# Table 4 -- gate diagnostics across methods/noise (purity, corr, sep, inversion, TTI)
def table4(final, out):
    # prefer a benchmark that exhibits inversion; fall back across what exists
    bench = next((b for b in ["split_mnist", "split_cifar10", "seq_cifar10", "permuted_mnist"]
                  if b in set(final.benchmark)), None)
    if bench is None:
        _write(pd.DataFrame(), out, "table4_gate_diagnostics", "no runs"); return
    sub = final[final.benchmark == bench]
    rows = []
    for (m, r), g in sub.groupby(["method", "noise_rate"]):
        rows.append({"method": m, "noise_rate": r,
                     "buffer_purity": _ms(g.buffer_purity),
                     "gate_corr": _ms(g.gate_corr),
                     "gate_separation": _ms(g.gate_separation),
                     "inversion_rate": _ms(g.inversion_rate),
                     "time_to_inversion": _ms(g.time_to_inversion, nd=1),
                     "avg_accuracy": _ms(g.average_accuracy)})
    df = pd.DataFrame(rows).sort_values(["method", "noise_rate"])
    _write(df, out, "table4_gate_diagnostics", f"benchmark = {bench}; mean+/-std over seeds.")


# Table 6 -- ablation (Table 5 = baseline_implementation_status.csv, authored separately)
ABLATION = [
    ("er", "no gate / random replay admission"),
    ("gate_conf", "confidence gate only (label-free)"),
    ("gate_loss", "loss/error gate only (supervised)"),
    ("gate_predstab", "prediction-stability gate only (label-free)"),
    ("gate_coteach", "agreement / co-teaching gate only"),
    ("ewc", "consolidation only (EWC; PNN-anchor family, no replay gate)"),
    ("ewc_loss", "consolidation + reliability gate (gated Fisher)"),
    ("oracle", "oracle clean-sample selector (upper bound)"),
]


def table6(final, out):
    bench = next((b for b in ["permuted_mnist", "seq_cifar10", "split_mnist"]
                  if b in set(final.benchmark)), None)
    if bench is None:
        _write(pd.DataFrame(), out, "table6_ablation", "no runs"); return
    sub = final[final.benchmark == bench]
    rates = sorted(sub.noise_rate.unique())
    mod = next((r for r in rates if 0.3 <= r <= 0.45), (rates[len(rates) // 2] if rates else 0))
    high = max(rates) if rates else 0
    rows = []
    for m, desc in ABLATION:
        gm = sub[(sub.method == m) & (np.isclose(sub.noise_rate, mod))]
        gh = sub[(sub.method == m) & (np.isclose(sub.noise_rate, high))]
        rows.append({"ablation_arm": m, "description": desc,
                     f"acc@{int(mod*100)}%": _ms(gm.average_accuracy),
                     f"acc@{int(high*100)}%": _ms(gh.average_accuracy),
                     f"separation@{int(high*100)}%": _ms(gh.gate_separation),
                     f"inversion_rate@{int(high*100)}%": _ms(gh.inversion_rate)})
    _write(pd.DataFrame(rows), out, "table6_ablation",
           f"benchmark = {bench}; moderate={int(mod*100)}% high={int(high*100)}% symmetric noise.")


def table5_check(out):
    """Table 5 = baseline_implementation_status.csv (authored separately). Verify present."""
    src = out / "baseline_implementation_status.csv"
    if src.exists():
        print(f"  table5 baseline_implementation_status.csv present ({len(_read(src))} rows)")
    else:
        print("  WARNING: baseline_implementation_status.csv missing (author it manually)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results/neural_networks_submission/csv"))
    ap.add_argument("--output", default=str(ROOT / "results/neural_networks_submission/tables"))
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    final = _read(inp / "final_metrics.csv")
    print(f"tables from {len(final)} runs across {sorted(set(final.benchmark)) if not final.empty else '[]'}")
    table1(final, out)
    table2(final, out)
    table3(final, out)
    table4(final, out)
    table5_check(out)
    table6(final, out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
