"""Small shared utilities: determinism, logging, batched inference."""
from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: str
                   ) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Run the model over a loader; return (logits[N], labels[N], metas)."""
    model.eval()
    logits, labels, metas = [], [], []
    for x, y, meta in loader:
        out = model(x.to(device)).squeeze(1).cpu()
        logits.append(out)
        labels.append(y)
        # meta arrives as dict-of-lists from the default collate
        n = len(y)
        metas.extend({k: (v[i].item() if torch.is_tensor(v[i]) else v[i])
                      for k, v in meta.items()} for i in range(n))
    return torch.cat(logits), torch.cat(labels), metas
