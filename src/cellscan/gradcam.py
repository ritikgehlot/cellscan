"""Minimal, dependency-free Grad-CAM for the single-logit ResNet.

Implemented from the original paper (Selvaraju et al., 2017) rather than
pulling a library: it is ~50 lines, and owning it means the demo has zero
risk of version drift in a third-party CAM package.

We hook ``layer4`` (last conv block): coarse 10x10 maps at 300px input,
which is the right granularity for "which region drove the defect score".
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        target = model.layer4  # type: ignore[union-attr]
        target.register_forward_hook(self._save_acts)
        target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m: nn.Module, _i, out: torch.Tensor) -> None:
        self._acts = out.detach()

    def _save_grads(self, _m: nn.Module, _gi, grad_out) -> None:
        self._grads = grad_out[0].detach()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """Return (heatmap HxW in [0,1], raw logit) for one image [1,3,H,W]."""
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("GradCAM expects a single image batch [1,3,H,W]")
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).squeeze()
        logit.backward()
        assert self._acts is not None and self._grads is not None
        weights = self._grads.mean(dim=(2, 3), keepdim=True)   # [1,C,1,1]
        cam = torch.relu((weights * self._acts).sum(dim=1)).squeeze(0)  # [h,w]
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=x.shape[-2:], mode="bilinear",
            align_corners=False,
        ).squeeze()
        cam -= cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), float(logit.detach().item())


def overlay(gray_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap onto a grayscale image, returning uint8 RGB."""
    import matplotlib.cm as mpl_cm

    g = gray_img.astype(float)
    g = (g - g.min()) / max(g.max() - g.min(), 1e-8)
    base = np.stack([g, g, g], axis=-1)
    heat = mpl_cm.jet(cam)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
