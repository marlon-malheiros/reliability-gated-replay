# Seed-paired CIFAR uncertainty

These artifacts are computed from the existing five-seed CIFAR results. No
training is performed.

- `per_seed_differences.csv` records every matched seed pair and its frozen
  `max_train_per_task` value.
- `paired_summary.csv` contains mean differences, 10,000-resample percentile
  bootstrap intervals, paired Cohen's (d_z), and exact Wilcoxon p-values.
- `table_paired_cifar_claims.tex` is the generated LaTeX table.

Seeds 0--2 use 1,500 training examples per task and seeds 3--4 use 2,500. Every
comparison is paired within seed, so both methods in a pair use the same budget;
the protocol heterogeneity must nevertheless be disclosed.

Regenerate with:

```bash
python scripts/build_paired_cifar_uncertainty.py
```
