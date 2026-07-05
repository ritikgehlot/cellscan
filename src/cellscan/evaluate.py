"""Evaluate a CellScan checkpoint on the held-out test split.

This script is the ONLY source of numbers for the README:
it writes ``reports/metrics.json`` plus a reliability diagram, a confusion
matrix, and the top misclassified images (failure cases). If a number is
not in metrics.json, it does not go in the README.

Run:  python -m cellscan.evaluate [--checkpoint models/cellscan_resnet18.pt]
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

from .calibrate import expected_calibration_error, reliability_curve
from .config import DEFAULT, Config
from .data import ELPVDataset
from .model import load_checkpoint
from .utils import collect_logits, setup_logging

logger = logging.getLogger(__name__)


def _plot_reliability(probs_raw: np.ndarray, probs_cal: np.ndarray,
                      labels: np.ndarray, n_bins: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for probs, name, marker in [(probs_raw, "uncalibrated", "o"),
                                (probs_cal, "temperature-scaled", "s")]:
        c, acc, _ = reliability_curve(probs, labels, n_bins)
        ax.plot(c, acc, marker=marker, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("predicted probability of defect")
    ax.set_ylabel("observed defect frequency")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=14)
    ax.set_xticks([0, 1], ["functional", "defective"])
    ax.set_yticks([0, 1], ["functional", "defective"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _dump_failures(probs: np.ndarray, labels: np.ndarray, metas: list[dict],
                   cfg: Config, k: int = 8) -> list[dict]:
    """Save the k most-confident mistakes -- honesty section of the README."""
    err = np.abs(probs - labels)
    wrong = np.where((probs >= 0.5).astype(int) != labels.astype(int))[0]
    worst = wrong[np.argsort(-err[wrong])][:k]
    out_dir = cfg.reports_dir / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    from PIL import Image
    for rank, i in enumerate(worst):
        m = metas[i]
        img = Image.open(cfg.data_dir / m["path"])
        dst = out_dir / f"fail_{rank:02d}_{Path(m['path']).name}"
        img.save(dst)
        records.append({"image": str(dst.relative_to(cfg.reports_dir.parent)),
                        "true_label": int(labels[i]),
                        "pred_prob_defect": round(float(probs[i]), 4),
                        "cell_type": m["cell_type"],
                        "raw_defect_proba_label": m["proba"]})
    return records


def evaluate(checkpoint: Path, cfg: Config = DEFAULT) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, temperature, ckpt_cfg = load_checkpoint(checkpoint, device)
    model.to(device)
    cfg = replace(cfg, img_size=int(ckpt_cfg.get("img_size", cfg.img_size)))

    ds = ELPVDataset("test", cfg, train_transforms=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    logits, labels_t, metas = collect_logits(model, dl, device)
    labels = labels_t.numpy()
    probs_raw = torch.sigmoid(logits).numpy()
    probs_cal = torch.sigmoid(logits / temperature).numpy()
    preds = (probs_cal >= cfg.decision_threshold).astype(int)

    cm = confusion_matrix(labels, preds)
    metrics: dict = {
        "checkpoint": str(checkpoint),
        "n_test": int(len(labels)),
        "temperature": round(temperature, 4),
        "auroc": round(float(roc_auc_score(labels, probs_cal)), 4),
        "average_precision": round(float(average_precision_score(labels, probs_cal)), 4),
        "f1_defective": round(float(f1_score(labels, preds)), 4),
        "precision_defective": round(float(precision_score(labels, preds)), 4),
        "recall_defective": round(float(recall_score(labels, preds)), 4),
        "ece_uncalibrated": round(expected_calibration_error(
            probs_raw, labels, cfg.ece_bins), 4),
        "ece_calibrated": round(expected_calibration_error(
            probs_cal, labels, cfg.ece_bins), 4),
        "confusion_matrix": cm.tolist(),
    }
    # per-cell-type breakdown: mono vs poly look very different
    types = np.array([m["cell_type"] for m in metas])
    for t in ("mono", "poly"):
        mask = types == t
        if mask.sum() and len(set(labels[mask])) == 2:
            metrics[f"auroc_{t}"] = round(
                float(roc_auc_score(labels[mask], probs_cal[mask])), 4)
            metrics[f"f1_{t}"] = round(
                float(f1_score(labels[mask], preds[mask])), 4)

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    _plot_reliability(probs_raw, probs_cal, labels, cfg.ece_bins,
                      cfg.reports_dir / "reliability_diagram.png")
    _plot_confusion(cm, cfg.reports_dir / "confusion_matrix.png")
    metrics["failure_cases"] = _dump_failures(probs_cal, labels, metas, cfg)

    out = cfg.reports_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", out)
    logger.info("AUROC=%.4f  F1=%.4f  ECE %.4f -> %.4f", metrics["auroc"],
                metrics["f1_defective"], metrics["ece_uncalibrated"],
                metrics["ece_calibrated"])
    return metrics


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Evaluate CellScan on test split")
    p.add_argument("--checkpoint", type=Path,
                   default=DEFAULT.models_dir / "cellscan_resnet18.pt")
    a = p.parse_args()
    evaluate(a.checkpoint)


if __name__ == "__main__":
    main()
