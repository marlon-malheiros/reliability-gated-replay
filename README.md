# Reliability-Gated Replay

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21017758.svg)](https://doi.org/10.5281/zenodo.21017758)

Code, configurations, aggregated results, and figure-generation scripts for:

**Reliability-Gated Replay Reveals a Power--Safety Frontier for Unreliable Data
in Continual Learning**

This repository is the reproducibility package prepared for the manuscript
submitted to *Neural Networks*.

## Archive

Version-specific Zenodo archive for the submission:

- DOI: `10.5281/zenodo.21017758`
- URL: `https://doi.org/10.5281/zenodo.21017758`

## Contents

- `analysis/`: performance, gate-separation, calibration, and statistical tools.
- `baselines/`: replay, regularization, and continual-learning baselines.
- `configs/`: native experiment configuration files.
- `datasets/`: benchmark construction and label-noise implementations.
- `methods/`: replay gates, buffers, and training procedures.
- `models/`: MLP, CNN, and ResNet-18 implementations.
- `scripts/`: training, aggregation, statistics, audit, table, and figure scripts.
- `results/neural_networks_submission/`: compact aggregated manuscript results.
- `results/projected_taskil_cifar10n/`: projected task-IL CIFAR-10N configs and
  compact Part-A artifacts.
- `results/paired_uncertainty/`: seed-paired CIFAR uncertainty artifacts.
- `paper/`: manuscript source, bibliography, and publication figures.
- `patches/`: pinned patch bundle for the optional external Mammoth/AER baseline.
- `tests/`: unit and end-to-end tests.
- `evidence/`: portable SHA-256 manifest for the separately archived raw results.

Raw training logs and public datasets are not redistributed in the software
repository. The raw experimental evidence is planned as a linked archival data
record and is inventoried in `evidence/raw_evidence_manifest.csv`.

## Environment

The recorded environment uses Python 3.13 and is specified in both
`environment.yml` and `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate pnn-cl
pytest -q
```

For pip-based installation, install the appropriate PyTorch build for the
machine first, then run:

```bash
python -m pip install -r requirements.txt
```

## Public datasets

No dataset is stored in this repository.

- MNIST and Fashion-MNIST are downloaded by the dataset loader on first use.
- CIFAR-10 and CIFAR-100 are downloaded through `torchvision`.
- CIFAR-10N human annotations come from the public UCSC-REAL
  `cifar-10-100n` repository and are fetched only when `--with-cifar10n` is
  supplied.

Prepare dependencies and public datasets with:

```bash
bash scripts/setup_neural_networks_submission.sh --with-cifar10n
```

The experiments use clean reference labels only for diagnostics and oracle
controls. Realizable replay gates do not access clean reference labels.

## Reproducing compact analyses

The archived aggregate CSVs are sufficient to regenerate the principal tables,
statistics, and figures without training:

```bash
python scripts/make_nn_submission_tables.py
python scripts/stats_nn_submission.py
python scripts/build_paired_cifar_uncertainty.py
python scripts/build_projected_taskil_cifar10n_artifacts.py
python scripts/make_nn_submission_figures.py \
  --manuscript-output paper/figs
python scripts/make_cifar10n_figures.py --output paper/figs
```

The projected task-IL CIFAR-10N construction is validated without training by:

```bash
python scripts/sanity_check_cifar10n_taskil.py
```

Its required methodological disclosure is recorded in
`results/projected_taskil_cifar10n/METHODOLOGICAL_NOTE.md`.

## Training

Run the small end-to-end smoke test before launching full grids:

```bash
bash scripts/run_nn_submission_smoke.sh
```

The projected task-IL CIFAR-10N grid is selected with `--preset bridge`. Existing
run files can be skipped safely with `--resume`:

```bash
python scripts/run_nn_submission.py --preset bridge --resume
```

The external AER comparison uses a pinned Mammoth checkout. It is optional for
the native pipeline:

```bash
bash scripts/setup_neural_networks_submission.sh --with-mammoth
bash scripts/run_aer_baseline.sh
```

See `patches/README.md` for the exact upstream commit and archived modifications.

## Citation

A version-specific Zenodo DOI will be added after the GitHub release is archived.

## Authors and funding

- Marlon Malheiros
- Antonio Pereira

Signal Processing Laboratory, Universidade Federal do Pará (UFPA), Brazil.

Funding: Conselho Nacional de Desenvolvimento Científico e Tecnológico
(CNPq), Grant 309589/2023-1, and Coordenação de Aperfeiçoamento de Pessoal de
Nível Superior (CAPES).

## License

Software in this repository is released under the [MIT License](LICENSE).
