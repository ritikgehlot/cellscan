"""Download the ELPV dataset (CC BY-NC-SA 4.0) and generate seeded splits.

The images are NOT committed to this repo (license + size); this script is
the reproducible path to the exact data + splits behind every reported
metric.

Usage:  python scripts/prepare_data.py [--source /path/to/elpv-dataset]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cellscan.config import DEFAULT  # noqa: E402
from cellscan.data import make_splits  # noqa: E402
from cellscan.utils import setup_logging  # noqa: E402

ELPV_REPO = "https://github.com/zae-bayern/elpv-dataset.git"
logger = logging.getLogger("prepare_data")


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=None,
                   help="existing clone of zae-bayern/elpv-dataset (skips download)")
    a = p.parse_args()

    src = a.source
    if src is None:
        src = DEFAULT.data_dir.parent / "_elpv_clone"
        if not src.exists():
            logger.info("Cloning ELPV dataset from %s ...", ELPV_REPO)
            subprocess.run(
                ["git", "clone", "--depth", "1", ELPV_REPO, str(src)],
                check=True,
            )
    data_root = src / "src" / "elpv_dataset" / "data"
    if not (data_root / "labels.csv").exists():
        raise FileNotFoundError(f"labels.csv not found under {data_root}")

    DEFAULT.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_root / "labels.csv", DEFAULT.data_dir / "labels.csv")
    dst_images = DEFAULT.data_dir / "images"
    if not dst_images.exists():
        n = len(list((data_root / "images").glob("*.png")))
        logger.info("Copying %d images ...", n)
        shutil.copytree(data_root / "images", dst_images)

    df = make_splits(DEFAULT)
    logger.info("Done. %d images | defect rate %.3f | splits -> %s",
                len(df), df["label"].mean(), DEFAULT.splits_csv)


if __name__ == "__main__":
    main()
