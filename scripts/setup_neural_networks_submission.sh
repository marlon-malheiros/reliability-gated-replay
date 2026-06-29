#!/usr/bin/env bash
# Setup for the Neural Networks submission package.
#   bash scripts/setup_neural_networks_submission.sh                 # deps + datasets
#   bash scripts/setup_neural_networks_submission.sh --with-cifar10n # + fetch CIFAR-10N labels
#   bash scripts/setup_neural_networks_submission.sh --with-mammoth  # + clone/patch Mammoth (optional)
#
# Idempotent. Safe to re-run. Requires: bash, python (with pip), git.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATA="$REPO_ROOT/data"
PY="${PYTHON:-python}"

WITH_CIFAR10N=0
WITH_MAMMOTH=0
for a in "$@"; do
  case "$a" in
    --with-cifar10n) WITH_CIFAR10N=1 ;;
    --with-mammoth)  WITH_MAMMOTH=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

echo "==> [1/4] Python dependencies (requirements.txt)"
echo "    NOTE: install the matching CUDA torch wheel first if you want GPU; see requirements.txt."
$PY -m pip install -r requirements.txt

echo "==> [2/4] MNIST / Fashion-MNIST (IDX files)"
if [ -d "$DATA/mnist" ]; then echo "    mnist present."; else
  echo "    mnist absent — it will be downloaded on first run (allow_download=true)."; fi

echo "==> [3/4] CIFAR-10 / CIFAR-100 (torchvision)"
# The audited repo had data/cifar-10-batches-py as a SYMLINK into data/mammoth/.
# If that link is broken (Mammoth removed), drop it and download a real copy.
if [ -L "$DATA/cifar-10-batches-py" ] && [ ! -e "$DATA/cifar-10-batches-py" ]; then
  echo "    removing broken cifar-10 symlink"; rm -f "$DATA/cifar-10-batches-py"
fi
mkdir -p "$DATA"
$PY - "$DATA" <<'PY'
import sys
from torchvision.datasets import CIFAR10, CIFAR100
root = sys.argv[1]
CIFAR10(root, train=True, download=True); CIFAR10(root, train=False, download=True)
CIFAR100(root, train=True, download=True); CIFAR100(root, train=False, download=True)
print("    CIFAR-10/100 ready under", root)
PY

if [ "$WITH_CIFAR10N" = "1" ]; then
  echo "==> [extra] CIFAR-10N human labels"
  mkdir -p "$DATA/cifar-10n"
  URL="https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"
  if [ -f "$DATA/cifar-10n/CIFAR-10_human.pt" ]; then
    echo "    CIFAR-10N present."
  elif command -v curl >/dev/null; then
    curl -fL "$URL" -o "$DATA/cifar-10n/CIFAR-10_human.pt" \
      && echo "    CIFAR-10N downloaded." \
      || echo "    WARN: CIFAR-10N download failed; runner will use the synthetic-noise fallback."
  else
    echo "    WARN: curl not found; CIFAR-10N not fetched (synthetic-noise fallback will be used)."
  fi
fi

if [ "$WITH_MAMMOTH" = "1" ]; then
  echo "==> [extra] Mammoth (optional, pinned commit + local patch)"
  MROOT="projects/continual_learning/external/mammoth"
  COMMIT="e75a491c69fd729edeb01431afb753d9157d9a81"
  if [ ! -d "$MROOT/.git" ]; then
    git clone https://github.com/aimagelab/mammoth.git "$MROOT"
  fi
  (
    cd "$MROOT"
    git fetch --all -q
    git checkout "$COMMIT" -q
    PATCH="$REPO_ROOT/patches/mammoth_local_changes.patch"
    if git apply --check "$PATCH" 2>/dev/null; then
      git apply "$PATCH"
    elif ! git apply --reverse --check "$PATCH" 2>/dev/null; then
      echo "    ERROR: Mammoth patch neither applies cleanly nor is already applied." >&2
      exit 1
    fi
  )
  cp "$REPO_ROOT/patches/mammoth_models_pnn_derpp.py" "$MROOT/models/pnn_derpp.py"
  $PY -m pip install -r "$REPO_ROOT/patches/requirements-mammoth-stage0.txt"
  echo "    Mammoth @ $COMMIT with the archived patch applied."
fi

echo "==> [4/4] Import + GPU sanity"
$PY - <<'PY'
import torch, methods
print("    torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
print("    methods available:", len(methods.available_methods()))
PY
echo "==> setup complete. Next: bash scripts/run_nn_submission_smoke.sh"
