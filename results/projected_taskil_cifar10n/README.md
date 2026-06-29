# Projected task-IL CIFAR-10N

This directory contains the compact Part-A artifacts. The 50 raw run JSONs are
kept in the linked raw-evidence deposit and are inventoried by
`evidence/raw_evidence_manifest.csv`.

- `configs/`: frozen configuration for each run.
- `csv/`: standard aggregator outputs for only the 50 bridge runs.
- `per_seed_metrics.csv`: selected release metrics for every run.
- `summary_mean_sd.csv`: five-seed mean and sample SD.
- `table_projected_taskil_cifar10n.tex`: paste-ready LaTeX table.
- `METHODOLOGICAL_NOTE.md`: required projection disclosure.
- `protocol.json`: shared training protocol, including the optimizer settings
  hard-coded by the runner.
- `experiment_manifest.csv`: completion status, timestamps, commands, and
  portable config paths for all 50 runs.

Regenerate the compact table artifacts with:

```bash
python scripts/build_projected_taskil_cifar10n_artifacts.py
```

The standard CSVs require the raw bridge JSONs. Point the standard aggregator
at a directory containing only those 50 files, then use this directory's
`csv/` folder as its output.
