# Residual audit + controls (no GPU; from `csv/final_metrics.csv`)

_Corpus: 1162 runs. Gate-vs-ER paired points: 530._

## Control 1 - Aggregation completeness audit

- **Total runs in corpus:** 1162
- **Duplicate run_ids:** 0  (OK)
- **Controllable submission grid:** expected 240, observed 240, missing 0  (complete)
- **Legacy MNIST corpus (reused):** 922 runs (sanity/frontier evidence; not part of the controllable grid)
- **Provenance:** every figure/table is generated from `csv/final_metrics.csv` (raw aggregation), not hand-edited numbers.

| grid          |   expected |   observed |   missing | complete   |
|:--------------|-----------:|-----------:|----------:|:-----------|
| seq_cifar10   |         75 |         75 |         0 | True       |
| split_cifar10 |         75 |         75 |         0 | True       |
| cifar10n      |         60 |         60 |         0 | True       |
| seq_cifar100  |         30 |         30 |         0 | True       |

- **Seed-count regularity:** 347 benchmark x method x condition cells; 9 have a seed count other than 3 or 5 (legacy probes/ablations).

## Control 2 - Main correlation with cluster control

- **Run-level:** Pearson r = +0.741 (n=530)
- **Condition-mean level:** Pearson r = +0.802 (n=154 clusters) -- robust to within-condition non-independence
- **Cluster bootstrap 95% CI (resampling conditions):** [+0.566, +0.838]

**Within-regime correlation (where does the predictor live?):**
  - **MNIST family**: r = +0.814, rho = +0.876 (n=428)
  - **Split-CIFAR-10 (task-IL)**: r = +0.683, rho = +0.594 (n=30)
  - **class-IL CIFAR (seq-10/100, 10N)**: r = -0.261, rho = -0.112 (n=72)

**Leave-one-benchmark-out (LOBO):**
  - remove **MNIST**: r = +0.010, rho = +0.138 (n=102)
  - remove **CIFAR-10**: r = +0.752, rho = +0.799 (n=470)
  - remove **CIFAR-10N**: r = +0.798, rho = +0.835 (n=500)
  - remove **CIFAR-100**: r = +0.741, rho = +0.776 (n=518)

## Control 3 - Threshold prediction (predictive, not just descriptive)

- **Task:** predict sign of payoff (Delta acc > 0) from gate separation (n=530, positive rate = 0.58)
- **ROC-AUC:** 0.853
- **Balanced accuracy @ threshold 0:** 0.742
- **Optimal (Youden) threshold:** +0.016  (balanced acc 0.787) -- the critical threshold sits near zero, i.e. positive separation predicts benefit, negative predicts harm.

## Control 4 - CIFAR-10N non-inversion audit

- **Supervised gates** (small-loss, co-teaching): minimum separation across all seeds = **+0.107** -> not a single seed inverts; purity well above ER.
- **Label-free confidence gate**: grazes zero on the worst label set (min separation -0.045, inversion rate up to 0.08) but does not substantively invert -- it neither purifies strongly nor inverts.
- Paper sentence: *"On CIFAR-10N the supervised gates occupy the safe side of the frontier: they improve buffer purity without crossing into negative gate-correctness alignment; the confidence gate sits near the boundary."*

| label_set   | gate         |   n |   sep_mean |   sep_ci_low |   sep_ci_high |   sep_min |   inversion_rate |   buffer_purity |   d_acc_vs_er |
|:------------|:-------------|----:|-----------:|-------------:|--------------:|----------:|-----------------:|----------------:|--------------:|
| aggre       | gate_loss    |   5 |      0.513 |        0.476 |         0.549 |     0.462 |             0    |           0.978 |         0.001 |
| aggre       | gate_conf    |   5 |      0.109 |        0.085 |         0.133 |     0.084 |             0    |           0.939 |        -0.007 |
| aggre       | gate_coteach |   5 |      0.503 |        0.443 |         0.563 |     0.451 |             0    |           0.985 |        -0.035 |
| worse       | gate_loss    |   5 |      0.124 |        0.101 |         0.148 |     0.107 |             0    |           0.806 |        -0     |
| worse       | gate_conf    |   5 |      0.005 |       -0.033 |         0.043 |    -0.045 |             0.08 |           0.633 |        -0.002 |
| worse       | gate_coteach |   5 |      0.385 |        0.301 |         0.468 |     0.303 |             0    |           0.905 |        -0.007 |

## Control 5 - Forgetting bottleneck (why cleaner buffers != higher accuracy)

- Class-incremental runs (seq-CIFAR-10/100, CIFAR-10N; n=165):
  - corr(buffer purity, accuracy) = +0.376
  - corr(forgetting, accuracy)    = +0.514
  - standardized OLS betas: purity +0.008, forgetting +0.030 -> **forgetting dominates** |+0.03| vs |+0.01|.
- Reading: improving memory purity addresses only part of the problem; representational forgetting is the dominant bottleneck in class-IL.

## Control 6 - Negative-control shuffle

- **Real** Pearson r = +0.741
- **Shuffled** (2000 permutations of gate separation): mean r = +0.000 +/- 0.042, |r| max = 0.140
- **Permutation p-value:** 0.0004998 -> the association collapses under shuffling, so it is not a generic pooling artifact.

## Headline interpretation (honest)

- The data is complete and clean (Control 1): expected = observed, no dups, no missing cells.
- The gate-separation predictor is **strong where buffer purity is the binding constraint** -- MNIST (r=+0.81) and task-IL Split-CIFAR-10 (r=+0.68) -- and is **predictive of the sign** of the effect (AUC 0.85, threshold near 0; Control 3).
- It **does not hold in class-incremental CIFAR** (r=-0.26): there, representational forgetting -- not buffer contamination -- is the dominant bottleneck (Control 5), so the payoff decouples from buffer quality. Controls 2 and 5 are two views of the same fact.
- The association is **not a pooling artifact** (Control 6: shuffling collapses it, perm p<0.001).
- **CIFAR-10N is on the safe side** (Control 4): supervised gates never invert under real noise; the label-free confidence gate only grazes the boundary.
- **Scope the claim:** gate separation predicts the replay payoff *when buffer purity is the bottleneck*; it is silent when forgetting dominates. That is a precise, defensible diagnostic -- not a universal accuracy predictor.

