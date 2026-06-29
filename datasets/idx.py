"""Raw IDX (MNIST / Fashion-MNIST) loader -- no torchvision dependency.

Downloads the four ``*-ubyte.gz`` files from a list of mirrors (first that
succeeds wins), caches them under ``data/<name>/``, and parses the IDX binary
format with numpy. We avoid torchvision deliberately: it is not installed and
pinning it to this torch build (2.12+cu130) is fragile.
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np

_MIRRORS = {
    "mnist": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ],
    "fashion_mnist": [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "https://storage.googleapis.com/fashion-mnist/",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    ],
}

_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _download(fname: str, mirrors: List[str], dest: Path, timeout: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    errors = []
    for base in mirrors:
        url = base + fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pnn-consolidation/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            return
        except Exception as e:  # try next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError(
        f"Could not download {fname} from any mirror.\n  " + "\n  ".join(errors)
    )


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        buf = f.read(int(np.prod(dims)))
        return np.frombuffer(buf, dtype=np.uint8).reshape(dims)


def load_idx_dataset(
    name: str, root: str | Path, allow_download: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_images, train_labels, test_images, test_labels).

    Images come back as uint8 with shape ``(N, 1, 28, 28)``; labels as int64.
    """
    if name not in _MIRRORS:
        raise KeyError(f"Unknown IDX dataset '{name}'. Known: {list(_MIRRORS)}")
    root = Path(root) / name
    paths = {f: root / f for f in _FILES}
    for f, p in paths.items():
        if not (p.exists() and p.stat().st_size > 0):
            if not allow_download:
                raise FileNotFoundError(f"Missing {p} and downloads are disabled.")
            _download(f, _MIRRORS[name], p)

    train_x = _read_idx(paths["train-images-idx3-ubyte.gz"])
    train_y = _read_idx(paths["train-labels-idx1-ubyte.gz"]).astype(np.int64)
    test_x = _read_idx(paths["t10k-images-idx3-ubyte.gz"])
    test_y = _read_idx(paths["t10k-labels-idx1-ubyte.gz"]).astype(np.int64)
    train_x = train_x[:, None, :, :]   # add channel dim
    test_x = test_x[:, None, :, :]
    return train_x, train_y, test_x, test_y
