#!/usr/bin/env bash
# Full Neural Networks submission benchmark: the decisive Priority-A CIFAR grid
# (Seq-/Split-CIFAR-10, ResNet-18, noise sweep) + realistic-noise CIFAR-10N, then
# aggregate (folding in the large MNIST reliability corpus as the sanity/frontier
# evidence) and regenerate all figures + tables.
#
#   bash scripts/run_nn_submission_full.sh                 # 3 seeds (default)
#   SEEDS=0,1,2,3,4 bash scripts/run_nn_submission_full.sh  # 5 seeds
#
# The exhaustive grid (all 6 benchmarks x 13 methods x 9 conditions x 5 seeds, ~2145
# runs) is `python scripts/run_nn_submission.py --preset full` -- compute-heavy; run
# selectively. This script runs the decisive, completable subset.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -d .venv ] && source .venv/bin/activate 2>/dev/null || true
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0,1,2}"
PKG="$REPO_ROOT/results/neural_networks_submission"

# Legacy MNIST reliability corpora reused as the sanity/frontier evidence (MNIST-only
# rows are folded in; legacy non-MNIST runs are excluded by the aggregator).
MNIST_CORPUS=(results/reliability/runs results/reliability_teacher/runs results/reliability_oracle/runs)

echo "==> [1/4] Priority-A CIFAR grid (Seq-/Split-CIFAR-10, ResNet-18), seeds=$SEEDS"
$PY scripts/run_nn_submission.py \
    --datasets seq_cifar10,split_cifar10 \
    --conditions sym20,sym40,sym60,asym40 \
    --methods er,derpp,ewc,gate_loss,gate_conf,gate_spr,oracle \
    --seeds "$SEEDS" --resume

echo "==> [2/4] Realistic-noise benchmark (CIFAR-10N if present, else synthetic fallback)"
$PY scripts/run_nn_submission.py --preset cifar10n --seeds "$SEEDS" --resume || \
  echo "    (CIFAR-10N skipped/failed -> synthetic asym40 fallback already covers Table 3)"

echo "==> [3/4] aggregate (CIFAR runs + MNIST corpus) -> csv/"
$PY scripts/aggregate_nn_submission_results.py \
    --input "$PKG/raw_logs" --extra-input "${MNIST_CORPUS[@]}" --output "$PKG/csv"

echo "==> [4/4] figures + tables"
$PY scripts/make_nn_submission_figures.py --input "$PKG/csv" --output "$PKG/figures"
$PY scripts/make_nn_submission_tables.py  --input "$PKG/csv" --output "$PKG/tables"
echo "==> FULL benchmark complete. See $PKG/ (csv, figures, tables)."
