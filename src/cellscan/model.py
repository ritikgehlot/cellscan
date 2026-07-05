"""Model factory and checkpoint I/O.

ResNet-18 with a single-logit head:
* Small dataset (~1.8k training images) -> small backbone. ResNet-18 has
  enough capacity for crack/texture patterns without memorizing the set.
* One logit + BCEWithLogits (not 2-class softmax) keeps the imbalance
  handling (pos_weight), temperature scaling, and Grad-CAM all operating
  on a single scalar "defect evidence" value -- simpler and less error-prone.

Checkpoints bundle weights + config + calibration temperature, so the demo
app and evaluator can never accidentally mix a model with the wrong
preprocessing or temperature.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from .config import Config, DEFAULT

logger = logging.getLogger(__name__)


def build_model(cfg: Config = DEFAULT, pretrained: bool = True) -> nn.Module:
    if cfg.backbone != "resnet18":
        raise ValueError(f"unsupported backbone: {cfg.backbone}")
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def save_checkpoint(path: Path, model: nn.Module, cfg: Config,
                    temperature: float = 1.0,
                    extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "temperature": float(temperature),
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint -> %s (T=%.3f)", path, temperature)


def load_checkpoint(path: Path, map_location: str = "cpu"
                    ) -> tuple[nn.Module, float, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    img_size = int(payload["config"].get("img_size", DEFAULT.img_size))
    cfg = Config(img_size=img_size)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, float(payload.get("temperature", 1.0)), payload["config"]
