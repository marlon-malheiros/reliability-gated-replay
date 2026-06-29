# README_RESULTS — metric definitions & result organization

This file is the authoritative definition of every metric in the Neural Networks
submission package. All numbers in the figures/tables are computed from the CSVs in
`csv/` by `scripts/aggregate_nn_submission_results.py`
(metric code: [analysis/metrics.py](../../analysis/metrics.py) and
[analysis/gate_metrics.py](../../analysis/gate_metrics.py)); nothing is hand-typed.

## Directory layout

```
results/neural_networks_submission/
  repo_audit.md              # Phase-1 audit + repairs
  README_RESULTS.md          # this file
  experiment_manifest.csv    # one row per run (run_id, status, times, command, config)
  FINAL_AGENT_REPORT.md      # executive verdict + commands
  configs/<run_id>.json      # exact data+method+train config copied per run
  raw_logs/<run_id>.json     # full per-run result (acc matrix, gate history, calibration…)
  csv/
    final_metrics.csv        # one row per run — every final metric below
    per_eval_metrics.csv     # one row per (run, after-task-i): accuracy progression + gate sep
    gate_diagnostics.csv     # one row per (run, task): gate-on-correct/wrong/sep/corr/purity
    buffer_diagnostics.csv   # one row per run: purity/diversity/balance/replay-loss split
    calibration_bins.csv     # one row per (run, reliability-bin): for the reliability diagram
  figures/   fig1..fig7 (.png + .pdf, no titles)
  tables/    table1..table6 (.csv + .md) + baseline_implementation_status.csv
  scripts/   (symlinks/notes; the executable scripts live in repo-root scripts/)
```

## Experimental axes

* **Benchmarks.** Sanity: Permuted-MNIST, Rotated-MNIST (MLP). Main: Seq-CIFAR-10
  (class-IL, single head) and Split-CIFAR-10 (task-IL, per-task binary heads), both
  ResNet-18; Seq-CIFAR-100 (class-IL). Realistic noise: CIFAR-10N (or documented
  synthetic fallback).
* **Seeds.** 0,1,2 by default (code supports 0–4 unchanged: `--seeds 0,1,2,3,4`).
* **Noise.** Synthetic `clean / sym20 / sym40 / sym60 / asym20 / asym40` (train labels
  only; val/test stay clean, so reported accuracy IS clean-test accuracy). Symmetric =
  flip to a uniformly random other class; asymmetric = class-conditional `c→(c+1) mod K`.
  Realistic = CIFAR-10N human labels (`aggre`≈9%, `worse`≈40%, `random{1,2,3}`≈18%).
* **Correctness** is always defined against the **clean/reference label**, never the
  (possibly noisy) training label.

## Final metrics (`final_metrics.csv`)

Accuracy matrix `R[i][j]` = clean-test accuracy on task *j* after training task *i*
(Lopez-Paz & Ranzato). T = #tasks.

| metric | definition |
|---|---|
| `average_accuracy` | mean over tasks of the final row `R[T-1, :]` |
| `class_il_acc` | = `average_accuracy` when single-head (Seq-CIFAR-10/100), else NaN |
| `task_il_acc` | = `average_accuracy` when multi-head (Split-CIFAR-10), else NaN |
| `mean_forgetting` | mean over tasks j<T-1 of `max_i R[i,j] − R[T-1,j]` |
| `backward_transfer` | mean over j<T-1 of `R[T-1,j] − R[j,j]` |
| `forward_transfer` | mean over j>0 of `R[j-1,j] − init_acc[j]` (random-init reference) |
| `buffer_purity` | fraction of replay-buffer items whose **stored** label == **clean** label. NaN if no buffer / clean labels unavailable (never faked) |
| `buffer_diversity` | normalized class entropy of the buffer `H(class dist)/log K ∈ [0,1]` |
| `buffer_class_balance` | min/max class-count ratio in the buffer (1.0 = balanced) |
| `n_classes_represented` | number of distinct classes in the buffer |
| `gate_corr` | mean per-task Pearson corr(gate value, correctness). Pearson+Spearman are also computed per call in `gate_metrics.py` |
| `gate_on_correct` | mean gate value on truly-correct (clean) samples (final task) |
| `gate_on_wrong` | mean gate value on mislabeled samples (final task) |
| `gate_separation` | `gate_on_correct − gate_on_wrong` (final evaluation) |
| `inversion_flag` | True iff the gate-separation series sustains a negative run (≥2 evals) |
| `inversion_rate` | fraction of evaluations whose gate separation is negative |
| `time_to_inversion` | first eval index where separation goes negative and stays negative ≥2 consecutive evals; NaN if never |
| `replay_loss_clean` | final-model mean CE on buffer items whose stored==clean label |
| `replay_loss_noisy` | final-model mean CE on buffer items whose stored≠clean label |
| `ece` | Expected Calibration Error (15 equal-width top-label bins), mean over tasks' clean test sets |
| `brier` | multiclass Brier score, mean over tasks |
| `nll` | negative log-likelihood (cross-entropy of predicted probs), mean over tasks |
| `total_time_s` | wall-clock training time | 
| `peak_memory_mb` | CUDA peak memory (0 on CPU) |
| `n_params` | model parameter count |

### Inversion — the central definition

A gate is **inverted** when its reliability score is *higher* for incorrect/noisy
samples than for correct/clean ones:

```
gate_separation = mean(gate | correct) − mean(gate | wrong)
inversion_flag  = the per-evaluation separation series turns negative and STAYS
                  negative for ≥ 2 consecutive evaluations
time_to_inversion = index of the first evaluation in that sustained negative run
                    (NaN if it never inverts)
```

We also report Pearson and Spearman correlation between the gate value and true
correctness; a negative correlation is the inversion signature. Correctness uses the
clean/reference label.

### Buffer purity / diversity

`buffer_purity` = fraction of buffered samples whose **training (stored)** label
matches the **clean/reference** label. If clean labels are unavailable the metric is
reported as NaN (`unavailable`) — never fabricated. `buffer_diversity` reports the
number of represented classes and the normalized class entropy; class balance is the
min/max class-count ratio.

### Calibration

ECE (15 equal-width confidence bins — the *same* binning for every method), Brier
score, and NLL are computed on each task's clean test set with the final model and
averaged over tasks. Reliability-diagram bins (pooled top-label confidence vs accuracy)
are saved per run in `calibration_bins.csv` and drive Figure 7. (Legacy MNIST runs from
the earlier reliability suite predate this and have empty calibration fields.)

## Per-evaluation / gate / buffer CSVs

* `per_eval_metrics.csv` — for each run, one row per "after task *i*": `avg_acc_so_far`,
  `learned_acc` (`R[i,i]`), and `gate_separation` at task *i* (drives the inversion
  trajectory, Figure 5).
* `gate_diagnostics.csv` — for each run, one row per task: `gate_on_correct`,
  `gate_on_wrong`, `gate_separation`, `corr_gate_correct`, `buffer_purity`.
* `buffer_diagnostics.csv` — one row per run: purity, normalized class entropy, class
  balance, classes represented, and the clean/noisy replay-loss split.

## Power–safety frontier (Figure 6) definitions

```
benign_gain(method)       = acc(method, moderate_noise)        − acc(random_replay, moderate_noise)
inversion_penalty(method) = acc(random_replay, high/structured_noise) − acc(method, high/structured_noise)
```

`random_replay` = ER (reservoir admits everything). Defaults: benign = Permuted-MNIST @
40% symmetric; inversion = Split-MNIST @ 60% symmetric (the regimes where the phenomena
are established at large N). High benign_gain + low inversion_penalty (lower-right) is
ideal; high+high = powerful-but-unsafe; low+high = dominated.

## Method ↔ baseline mapping

| canonical | baseline | source |
|---|---|---|
| `er` | ER / random replay admission | baselines/replay.py |
| `derpp` | DER++ | baselines/derpp.py |
| `ewc`,`si` | EWC, SI | baselines/ewc.py, si.py |
| `er_ace` | ER-ACE | baselines/er_ace.py |
| `gate_loss` | loss/error-gated admission | gated_replay (signal=error) |
| `gate_conf` | confidence-gated admission | gated_replay (signal=confidence) |
| `gate_predstab`,`gate_reprstab` | stability-gated admission | gated_replay (pred/repr_stability) |
| `gate_spr` | SPR proxy (small-loss-over-time) | gated_replay (signal=loss_traj) |
| `gate_teacher` | teacher/agreement gate | gated_replay (agreement + EMA teacher) |
| `gate_coteach` | co-teaching/agreement | gated_replay (coteach) |
| `oracle` | oracle clean selector (upper bound) | gated_replay (oracle) |

See `tables/baseline_implementation_status.csv` for the full 16-baseline status,
including the approximated noisy-CL purifiers (SPR / PuriDivER / AER-AGR).
