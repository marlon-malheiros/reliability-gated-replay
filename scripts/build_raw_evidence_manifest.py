#!/usr/bin/env python3
"""Build a portable SHA-256 manifest for the raw evidence collections.

The output contains collection names and paths relative to each collection. It
therefore does not expose machine-specific source paths and can be verified
after the collections are uploaded as a separate archival data deposit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


COLLECTIONS = {
    "nn_submission_raw_logs": Path("results/neural_networks_submission/raw_logs"),
    "reviewer_controls_raw_logs": Path(
        "results/neural_networks_submission/reviewer_controls/raw_logs"
    ),
    "reliability_runs": Path("results/reliability/runs"),
    "reliability_teacher_runs": Path("results/reliability_teacher/runs"),
    "reliability_oracle_runs": Path("results/reliability_oracle/runs"),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        type=Path,
        help="Root of the original repository containing the raw collections.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Destination CSV for the portable checksum manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    missing = [
        str(relative_path)
        for relative_path in COLLECTIONS.values()
        if not (source_root / relative_path).is_dir()
    ]
    if missing:
        raise SystemExit("Missing raw evidence collections: " + ", ".join(missing))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, int, str]] = []
    for collection, relative_root in COLLECTIONS.items():
        collection_root = source_root / relative_root
        for path in sorted(p for p in collection_root.rglob("*") if p.is_file()):
            rows.append(
                (
                    collection,
                    path.relative_to(collection_root).as_posix(),
                    path.stat().st_size,
                    sha256_file(path),
                )
            )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("collection", "relative_path", "size_bytes", "sha256"))
        writer.writerows(rows)

    total_bytes = sum(row[2] for row in rows)
    print(f"Wrote {len(rows)} entries ({total_bytes} bytes) to {output_csv}")


if __name__ == "__main__":
    main()
