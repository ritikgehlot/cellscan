"""Post-hoc confidence calibration via temperature scaling.

Why this matters: a fine-tuned CNN's raw sigmoid outputs are usually
overconfident. Temperature scaling (Guo et al., 2017) fits ONE scalar T on
the validation set (never test) by minimizing NLL of ``sigmoid(logit / T)``.
It cannot change the ranking of predictions (AUROC is untouched); it only
makes the probability *mean something*, which is what the demo shows users.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    max_iter: int = 200) -> float:
    """Fit scalar temperature on validation logits/labels (both shape [N])."""
    logits = logits.detach().float().reshape(-1)
    labels = labels.detach().float().reshape(-1)
    log_t = torch.zeros(1, requires_grad=True)  # optimize log T so T > 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = bce(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(log_t.exp().item())
    logger.info("Fitted temperature T=%.4f", t)
    return t


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """Standard equal-width-bin ECE for binary probabilities."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_accuracy, bin_fraction) for plotting."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, fracs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        centers.append((lo + hi) / 2)
        accs.append(labels[mask].mean() if mask.any() else np.nan)
        fracs.append(mask.mean())
    return np.array(centers), np.array(accs), np.array(fracs)
