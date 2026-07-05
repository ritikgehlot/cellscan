"""Fine-tune ResNet-18 on ELPV.

Design notes
------------
* Model selection = best validation AUROC (threshold-free, robust to the
  31/69 class imbalance), not raw accuracy.
* AMP is used automatically on CUDA -- at 300x300/bs32 this fits well
  inside 8 GB VRAM and roughly halves step time on an RTX 4060.
* After training, temperature scaling is fitted on the validation set and
  stored inside the checkpoint, so calibration always ships with the model.

Run:  python -m cellscan.train [--epochs 15] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from .calibrate import fit_temperature
from .config import DEFAULT, Config
from .data import ELPVDataset
from .model import build_model, save_checkpoint
from .utils import collect_logits, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def train(cfg: Config, smoke: bool = False, pretrained: bool = True) -> dict:
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s amp=%s", device, cfg.amp and device == "cuda")

    train_ds = ELPVDataset("train", cfg)
    val_ds = ELPVDataset("val", cfg)
    if smoke:  # tiny slice to verify the full pipeline end-to-end
        train_ds = Subset(train_ds, range(0, len(train_ds), 12))  # ~150 imgs
        val_ds_small = Subset(val_ds, range(0, len(val_ds), 6))
        pos_weight = torch.tensor(2.0)
    else:
        val_ds_small = val_ds
        pos_weight = train_ds.pos_weight()

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=device == "cuda")
    val_dl = DataLoader(val_ds_small, batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers)

    model = build_model(cfg, pretrained=pretrained).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    best_auroc, best_state, history = -1.0, None, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,
                                enabled=cfg.amp and device == "cuda"):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        val_logits, val_labels, _ = collect_logits(model, val_dl, device)
        try:
            val_auroc = roc_auc_score(val_labels.numpy(), val_logits.numpy())
        except ValueError:  # single-class smoke batches
            val_auroc = float("nan")
        history.append({"epoch": epoch,
                        "train_loss": running / len(train_dl.dataset),
                        "val_auroc": val_auroc,
                        "secs": round(time.time() - t0, 1)})
        logger.info("epoch %02d  loss=%.4f  val_auroc=%.4f  (%.1fs)",
                    epoch, history[-1]["train_loss"], val_auroc,
                    history[-1]["secs"])
        if val_auroc == val_auroc and val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- calibration on the FULL validation split (never test) ---
    full_val_dl = DataLoader(val_ds, batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers)
    v_logits, v_labels, _ = collect_logits(model, full_val_dl, device)
    temperature = fit_temperature(v_logits, v_labels)

    ckpt = cfg.models_dir / ("cellscan_smoke.pt" if smoke else "cellscan_resnet18.pt")
    save_checkpoint(ckpt, model, cfg, temperature,
                    extra={"best_val_auroc": best_auroc, "history": history})
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    hist_path = cfg.reports_dir / ("train_history_smoke.json" if smoke
                                   else "train_history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    return {"best_val_auroc": best_auroc, "temperature": temperature,
            "checkpoint": str(ckpt)}


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Train CellScan")
    p.add_argument("--epochs", type=int, default=DEFAULT.epochs)
    p.add_argument("--batch-size", type=int, default=DEFAULT.batch_size)
    p.add_argument("--img-size", type=int, default=DEFAULT.img_size)
    p.add_argument("--smoke", action="store_true",
                   help="1-epoch tiny run to verify the pipeline")
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip ImageNet weights (offline CI environments)")
    a = p.parse_args()
    cfg = replace(DEFAULT, epochs=1 if a.smoke else a.epochs,
                  batch_size=a.batch_size,
                  img_size=96 if a.smoke else a.img_size)
    out = train(cfg, smoke=a.smoke, pretrained=not a.no_pretrained)
    logger.info("done: %s", out)


if __name__ == "__main__":
    main()
