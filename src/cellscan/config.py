"""Central configuration for CellScan.

A frozen dataclass instead of a YAML file: for a single-model project this
keeps every knob type-checked, greppable, and overridable from the CLI
without a config-parsing layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # --- paths ---
    data_dir: Path = PROJECT_ROOT / "data" / "elpv"
    splits_csv: Path = PROJECT_ROOT / "data" / "splits.csv"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- data ---
    img_size: int = 300          # native ELPV resolution; cracks are thin, avoid downsampling
    defect_threshold: float = 0.5  # binarize defect probability at >= 0.5
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # --- training ---
    backbone: str = "resnet18"
    epochs: int = 15
    batch_size: int = 32          # safe for 8 GB VRAM at 300x300 with AMP
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    amp: bool = True              # ignored automatically on CPU

    # --- evaluation ---
    ece_bins: int = 15
    decision_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}


DEFAULT = Config()
