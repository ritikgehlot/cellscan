"""Tests for the pieces most likely to silently corrupt results:
split integrity (leakage), tensor shapes, Grad-CAM output, and calibration.

Data-dependent tests skip cleanly if scripts/prepare_data.py has not run,
so `pytest` passes on a fresh clone and exercises everything after setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellscan.calibrate import expected_calibration_error, fit_temperature
from cellscan.config import DEFAULT
from cellscan.gradcam import GradCAM
from cellscan.model import build_model

HAS_DATA = DEFAULT.splits_csv.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="run scripts/prepare_data.py first")


# ---------- splits: the tests that protect every reported number ----------

@needs_data
def test_splits_no_overlap_and_full_coverage() -> None:
    df = pd.read_csv(DEFAULT.splits_csv)
    assert df["path"].is_unique, "duplicate image paths across splits"
    assert set(df["split"]) == {"train", "val", "test"}
    assert len(df) == 2624, "unexpected total image count for ELPV"


@needs_data
def test_splits_are_stratified() -> None:
    df = pd.read_csv(DEFAULT.splits_csv)
    overall = df["label"].mean()
    for split, g in df.groupby("split"):
        assert abs(g["label"].mean() - overall) < 0.03, (
            f"{split} defect rate drifted from overall")
        # both cell types present in every split
        assert set(g["cell_type"]) == {"mono", "poly"}


@needs_data
def test_split_sizes_match_config() -> None:
    df = pd.read_csv(DEFAULT.splits_csv)
    n = len(df)
    assert abs(len(df[df.split == "test"]) / n - DEFAULT.test_frac) < 0.01
    assert abs(len(df[df.split == "val"]) / n - DEFAULT.val_frac) < 0.01


# ---------- model ----------

def test_model_forward_shape() -> None:
    model = build_model(pretrained=False).eval()
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_gradcam_heatmap_valid() -> None:
    model = build_model(pretrained=False)
    cam, logit = GradCAM(model)(torch.randn(1, 3, 96, 96))
    assert cam.shape == (96, 96)
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert np.isfinite(logit)


# ---------- calibration ----------

def test_temperature_scaling_reduces_nll_on_overconfident_logits() -> None:
    torch.manual_seed(0)
    labels = torch.randint(0, 2, (500,)).float()
    # simulate an overconfident model: correct direction, inflated magnitude
    logits = (labels * 2 - 1) * 6.0 + torch.randn(500) * 4.0
    t = fit_temperature(logits, labels)
    assert t > 1.0, "overconfident logits should yield T > 1"
    bce = torch.nn.BCEWithLogitsLoss()
    assert bce(logits / t, labels) < bce(logits, labels)


def test_ece_perfect_and_worst_case() -> None:
    labels = np.array([0, 0, 1, 1], dtype=float)
    assert expected_calibration_error(labels, labels) == pytest.approx(0.0)
    assert expected_calibration_error(1 - labels, labels) == pytest.approx(1.0)
