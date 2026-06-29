# Raw evidence archive

This software repository contains compact aggregated outputs under
`results/neural_networks_submission/`. The raw per-run result files are about
3.91 GB and are intentionally reserved for a separate Zenodo data record.
Keeping the deposits separate makes the source-code release small while
preserving the complete experimental evidence.

`raw_evidence_manifest.csv` inventories every raw evidence file using a stable
collection name, a collection-relative path, its byte size, and its SHA-256
digest. No machine-specific absolute paths appear in the manifest.

The data record should preserve these five collection directories:

| Manifest collection | Directory in the data record |
| --- | --- |
| `nn_submission_raw_logs` | `results/neural_networks_submission/raw_logs/` |
| `reviewer_controls_raw_logs` | `results/neural_networks_submission/reviewer_controls/raw_logs/` |
| `reliability_runs` | `results/reliability/runs/` |
| `reliability_teacher_runs` | `results/reliability_teacher/runs/` |
| `reliability_oracle_runs` | `results/reliability_oracle/runs/` |

After publication, the software and data records must link to one another using
Zenodo related identifiers. The data-record DOI will be added after it has been
reserved.

## Rebuilding the manifest

From the software repository root, run:

```bash
python scripts/build_raw_evidence_manifest.py \
  <SOURCE_REPOSITORY> evidence/raw_evidence_manifest.csv
```

## Verifying an extracted data record

For each manifest row, resolve `relative_path` below the directory associated
with `collection`, then compare its file size and SHA-256 digest. A verifier is
not yet included because the final packaging layout of the data record remains
to be approved.
