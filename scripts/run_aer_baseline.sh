#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Real AER baseline (Millunzi et al., BMVC 2024) + bridge-ER, matched to OUR protocol.
# Runs the official `er_ace_aer_abs` from the vendored Mammoth on Seq-CIFAR-10 under
# our conditions (Adam lr 1e-3, wd 0, 8 epochs, buffer 500, train bs 128, replay 32),
# so the comparison is apples-to-apples. One Seq-CIFAR-10 run reports BOTH Class-IL
# and Task-IL accuracy. Bridge-ER (sym60) validates cross-harness comparability.
#
# Grid:  AER  x {sym20, sym60, asym40} x seeds {0,1,2}   (9 runs)
#        ER   x {sym60}        x seeds {0,1}      (2 bridge runs)
# Each run -> its own stdout log (avoids Mammoth's shared logs.pyd write race).
#
# Usage:  bash scripts/run_aer_baseline.sh [PARALLEL]   (default 2; do not exceed 2 on 4GB)
# ---------------------------------------------------------------------------
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAM="$ROOT/projects/continual_learning/external/mammoth"
OUT="$ROOT/results/neural_networks_submission/aer"
mkdir -p "$OUT"
PAR="${1:-2}"
COMMON="--dataset seq-cifar10 --buffer_size 500 --optimizer adam --lr 0.001 --optim_wd 0 --n_epochs 8 --batch_size 128 --minibatch_size 32 --num_workers 0 --disable_noisy_labels_cache 1"

run () {   # model noise_type noise_rate seed
  local model="$1" nt="$2" nr="$3" seed="$4"
  local pct; pct=$(python3 -c "print(int($nr*100))")
  local short="sym"; [ "$nt" = "asymmetric" ] && short="asym"
  local log="$OUT/${model}__${short}${pct}__seed${seed}.log"
  if grep -qE '^END [0-9]+ rc=0$' "$log" 2>/dev/null; then
    echo "skip completed: $model $short$pct seed$seed" | tee -a "$OUT/run.log"
    return
  fi
  echo "START $(date +%s)" > "$log"
  ( cd "$MAM" && python utils/main.py --model "$model" --noise_type "$nt" \
      --noise_rate "$nr" --seed "$seed" $COMMON >> "$log" 2>&1
    echo "END $(date +%s) rc=$?" >> "$log" )
}

echo "=== AER baseline grid start $(date) | parallelism=$PAR ===" | tee "$OUT/run.log"
JOBS=(
  "er_ace_aer_abs symmetric 0.2 0" "er_ace_aer_abs symmetric 0.2 1" "er_ace_aer_abs symmetric 0.2 2"
  "er_ace_aer_abs symmetric 0.6 0" "er_ace_aer_abs symmetric 0.6 1" "er_ace_aer_abs symmetric 0.6 2"
  "er_ace_aer_abs asymmetric 0.4 0" "er_ace_aer_abs asymmetric 0.4 1" "er_ace_aer_abs asymmetric 0.4 2"
  "er symmetric 0.6 0" "er symmetric 0.6 1"
)
i=0
for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run $job & echo "launched: $job" | tee -a "$OUT/run.log"
  i=$((i+1)); [ $((i % PAR)) -eq 0 ] && wait
done
wait
echo "=== grid DONE $(date) ===" | tee -a "$OUT/run.log"

# ---- summarize accuracies + per-run wall time from each log ----
echo "model,noise,seed,class_il,task_il,seconds,rc" > "$OUT/summary.csv"
for log in "$OUT"/*__sym*__seed*.log "$OUT"/*__asym*__seed*.log; do
  [ -e "$log" ] || continue
  base=$(basename "$log" .log)
  model=${base%%__*}; noise=$(echo "$base" | grep -oE "(sym|asym)[0-9]+"); seed=${base##*seed}
  line=$(grep -aE "Accuracy for 5 task" "$log" | tail -1)
  cil=$(echo "$line" | grep -oE "Class-IL\]: [0-9.]+" | grep -oE "[0-9.]+$")
  til=$(echo "$line" | grep -oE "Task-IL\]: [0-9.]+" | grep -oE "[0-9.]+$")
  st=$(grep -oE "^START [0-9]+" "$log" | grep -oE "[0-9]+"); en=$(grep -oE "^END [0-9]+ rc=[0-9]+" "$log" | grep -oE "[0-9]+" | head -1)
  rc=$(grep -oE "rc=[0-9]+" "$log" | grep -oE "[0-9]+" | tail -1)
  echo "$model,$noise,$seed,${cil:-NA},${til:-NA},$(( ${en:-0} - ${st:-0} )),${rc:-NA}" >> "$OUT/summary.csv"
done
echo "=== summary -> $OUT/summary.csv ==="; cat "$OUT/summary.csv"
