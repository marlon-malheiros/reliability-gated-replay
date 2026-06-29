# External Mammoth code — pinned commit + local patches

The native pipeline **does not require Mammoth** — every baseline (ER, DER++, EWC, SI)
is implemented natively under `baselines/` and runs through the shared
`methods/trainer.py:ContinualTrainer`, which also produces the gate / buffer-purity
/ inversion diagnostics Mammoth does not. Mammoth supplies the external AER
comparison and is reproduced here through a pinned upstream commit and patch.

## Pinned source

- Upstream: `https://github.com/aimagelab/mammoth.git`
- Commit: **`e75a491c69fd729edeb01431afb753d9157d9a81`**

## Local modifications

The working tree at `projects/continual_learning/external/mammoth/` had uncommitted
edits (it was a bare nested clone, **not** a git submodule, hence untracked by this
repo). They are captured here:

- `mammoth_local_changes.patch` — `git diff` of the tracked files
  (`backbone/__init__.py`, `datasets/__init__.py`, `models/__init__.py`,
  `utils/training.py`).
- `mammoth_models_pnn_derpp.py` — the untracked new model `models/pnn_derpp.py`.

## Reproduce the Mammoth clone (optional)

`scripts/setup_neural_networks_submission.sh --with-mammoth` automates the steps
below:

```bash
git clone https://github.com/aimagelab/mammoth.git projects/continual_learning/external/mammoth
cd projects/continual_learning/external/mammoth
git checkout e75a491c69fd729edeb01431afb753d9157d9a81
git apply ../../../../patches/mammoth_local_changes.patch
cp ../../../../patches/mammoth_models_pnn_derpp.py models/pnn_derpp.py
pip install -r ../../../../patches/requirements-mammoth-stage0.txt
```

The setup script performs the clone, patch, model-file copy, and dependency
installation. Launch the external AER grid from repository root with
`bash scripts/run_aer_baseline.sh`.
