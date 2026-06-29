#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Minimum-valid acceptance grid for the reliability-gated-replay paper.
# (See PROFESSOR_UPDATE.md for the rationale.)
#
#   Block A  CIFAR-10 (seq+split) -> 5 seeds         rigor on the main tables
#   Block B  real CIFAR-10N       -> 5 seeds         the real-human-noise result
#   Block C  CIFAR-100 frontier   -> 3 seeds, 2 reg. scale-to-100-classes check
#
# Same 5 methods everywhere: er, derpp, gate_loss, gate_conf, oracle.
# Fully resumable: re-running skips completed runs (existing CIFAR-10 seeds 0-2
# are reused, only 3-4 are computed).
#
# Usage:
#   bash scripts/run_minvalid_grid.sh           # serial   (~7 h, safest)
#   bash scripts/run_minvalid_grid.sh 2         # 2-way    (~4 h, OK on 4 GB GPU)
# Do NOT exceed 2 on a 4 GB card (each run ~1 GB + ~0.4 GB CUDA context).
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.."
PAR="${1:-1}"
M="er,derpp,gate_loss,gate_conf,oracle"
OUT=results/neural_networks_submission
LOG="$OUT/minvalid_run.log"
mkdir -p "$OUT"
echo "=== minvalid grid start $(date) | parallelism=$PAR ===" | tee -a "$LOG"

# Block B needs the real CIFAR-10N human labels; warn (not fail) if absent.
C10N=data/cifar-10n/CIFAR-10_human.pt
if [ ! -f "$C10N" ]; then
  echo "WARNING: $C10N missing -> Block B falls back to SYNTHETIC noise (not real labels)." | tee -a "$LOG"
fi

run_block () {                       # name  datasets  conditions  seed...
  local name="$1" ds="$2" conds="$3"; shift 3
  echo ">>> Block $name | ds=$ds | conds=$conds | seeds=$* | $(date)" | tee -a "$LOG"
  local i=0
  for s in "$@"; do
    python scripts/run_nn_submission.py --datasets "$ds" --methods "$M" \
      --conditions "$conds" --seeds "$s" --resume >>"$LOG" 2>&1 &
    i=$((i+1))
    if [ $((i % PAR)) -eq 0 ]; then wait; fi   # cap concurrency at PAR
  done
  wait
}

run_block A "seq_cifar10,split_cifar10" "sym20,sym60,asym40" 0 1 2 3 4
run_block B "cifar10n"                  "c10n_worse,c10n_aggre" 0 1 2 3 4
run_block C "seq_cifar100"              "sym20,sym60"           0 1 2

echo "=== minvalid grid DONE $(date) ===" | tee -a "$LOG"
echo "Next (finalize, ~30 min):" | tee -a "$LOG"
echo "  python scripts/aggregate_nn_submission_results.py --input $OUT/raw_logs \\" | tee -a "$LOG"
echo "      --extra-input results/reliability/runs results/reliability_teacher/runs results/reliability_oracle/runs \\" | tee -a "$LOG"
echo "      --output $OUT/csv" | tee -a "$LOG"
echo "  python scripts/stats_nn_submission.py" | tee -a "$LOG"
echo "  python scripts/make_nn_submission_figures.py ; python scripts/make_nn_submission_tables.py" | tee -a "$LOG"
