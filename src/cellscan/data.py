"""Dataset handling for the ELPV electroluminescence images.

Key decisions
-------------
* Splits are generated ONCE (seeded), stratified jointly on
  (binary label x cell type), and persisted to ``data/splits.csv``.
  Every downstream script reads that file, so train/val/test membership
  can never drift between runs -- this is what makes reported numbers
  reproducible.
* Grayscale images are replicated to 3 channels so the ImageNet-pretrained
  first conv layer stays useful.
* Augmentation is deliberately mild (h/v flips only). EL cell images are
  orientation-symmetric under flips, but rotations/crops would create
  border artifacts that look like the very cracks we are detecting.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .config import Config, DEFAULT

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_splits(cfg: Config = DEFAULT, labels_csv: Path | None = None) -> pd.DataFrame:
    """Create a seeded, jointly-stratified train/val/test split and save it.

    Stratification uses ``label x cell_type`` (4 strata) so that mono/poly
    proportions and defect rates match across splits.
    """
    labels_csv = labels_csv or cfg.data_dir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"{labels_csv} not found. Run scripts/prepare_data.py first."
        )
    df = pd.read_csv(labels_csv, sep=r"\s+", header=None,
                     names=["path", "proba", "cell_type"])
    df["label"] = (df["proba"] >= cfg.defect_threshold).astype(int)
    strata = df["label"].astype(str) + "_" + df["cell_type"]

    trainval_idx, test_idx = train_test_split(
        df.index, test_size=cfg.test_frac, stratify=strata, random_state=cfg.seed
    )
    val_rel = cfg.val_frac / (1.0 - cfg.test_frac)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_rel,
        stratify=strata.loc[trainval_idx], random_state=cfg.seed,
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    cfg.splits_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.splits_csv, index=False)
    logger.info("Wrote %s: %s", cfg.splits_csv,
                df["split"].value_counts().to_dict())
    return df


def build_transforms(cfg: Config = DEFAULT, train: bool = False) -> transforms.Compose:
    ops: list = [
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    ops += [transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class ELPVDataset(Dataset):
    """ELPV cells for one split, driven entirely by ``splits.csv``."""

    def __init__(self, split: str, cfg: Config = DEFAULT,
                 train_transforms: bool | None = None) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split: {split!r}")
        if not cfg.splits_csv.exists():
            raise FileNotFoundError(
                f"{cfg.splits_csv} missing. Run scripts/prepare_data.py first."
            )
        self.cfg = cfg
        df = pd.read_csv(cfg.splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"split {split!r} is empty in {cfg.splits_csv}")
        use_train_tf = train_transforms if train_transforms is not None else split == "train"
        self.transform = build_transforms(cfg, train=use_train_tf)
        logger.info("split=%s size=%d defect_rate=%.3f", split, len(self.df),
                    self.df["label"].mean())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.data_dir / row["path"])
        x = self.transform(img)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        meta = {"path": row["path"], "cell_type": row["cell_type"],
                "proba": float(row["proba"])}
        return x, y, meta

    def pos_weight(self) -> torch.Tensor:
        """Negative/positive ratio for BCEWithLogitsLoss (imbalance handling)."""
        pos = int(self.df["label"].sum())
        neg = len(self.df) - pos
        return torch.tensor(neg / max(pos, 1), dtype=torch.float32)
