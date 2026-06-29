#!/usr/bin/env bash
# Smoke test (CI gate): exercises the ENTIRE pipeline end-to-end in < ~10 min on
# CPU or a small GPU -- run a tiny grid, aggregate, build figures + tables. A fresh
# clone passing this proves the package is reproducible.
#   bash scripts/run_nn_submission_smoke.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -d .venv ] && source .venv/bin/activate 2>/dev/null || true
PY="${PYTHON:-python}"
SMOKE="$REPO_ROOT/results/neural_networks_submission/_smoke"

echo "==> [1/5] unit tests"
$PY -m pytest -q

echo "==> [2/5] tiny benchmark grid (Permuted-MNIST, 5 methods x 2 conditions)"
$PY scripts/run_nn_submission.py --preset smoke --output "$SMOKE"

echo "==> [3/5] aggregate -> CSVs"
$PY scripts/aggregate_nn_submission_results.py --input "$SMOKE/raw_logs" --output "$SMOKE/csv"

echo "==> [4/5] figures"
$PY scripts/make_nn_submission_figures.py --input "$SMOKE/csv" --output "$SMOKE/figures"

echo "==> [5/5] tables"
mkdir -p "$SMOKE/tables"
cp -n results/neural_networks_submission/tables/baseline_implementation_status.csv \
      "$SMOKE/tables/" 2>/dev/null || true
$PY scripts/make_nn_submission_tables.py --input "$SMOKE/csv" --output "$SMOKE/tables"

echo "==> SMOKE OK. Artifacts under $SMOKE (scratch; safe to delete)."
