"""Measure single-image and batch inference latency on the trained model.

Run:  python scripts/benchmark_latency.py
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cellscan.config import DEFAULT
from cellscan.data import build_transforms
from cellscan.model import load_checkpoint

CKPT = DEFAULT.models_dir / "cellscan_resnet18.pt"


def benchmark_inference():
    """Measure single-image and batch inference latency."""
    if not CKPT.exists():
        print(f"Checkpoint not found at {CKPT}")
        return

    print("Loading checkpoint...")
    model, temp, cfg = load_checkpoint(CKPT)
    model.eval()
    device = next(model.parameters()).device
    
    print(f"Device: {device}")
    
    dummy_img = torch.randn(1, 3, 300, 300).to(device)
    dummy_batch = torch.randn(32, 3, 300, 300).to(device)
    
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_img)
            _ = model(dummy_batch)
    
    print("Benchmarking single-image latency (100 iterations)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_img)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    single_latency_ms = (time.perf_counter() - t0) / 100 * 1000
    
    print("Benchmarking batch latency (10 iterations)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_batch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    batch_latency_ms = (time.perf_counter() - t0) / 10 * 1000
    
    throughput_per_sec = 1000 / single_latency_ms
    
    results = {
        "device": str(device),
        "single_image_latency_ms": round(single_latency_ms, 2),
        "batch_32_latency_ms": round(batch_latency_ms, 2),
        "throughput_images_per_second": round(throughput_per_sec, 1),
        "model_size_mb": round(CKPT.stat().st_size / (1024**2), 2),
    }
    
    DEFAULT.reports_dir.mkdir(exist_ok=True)
    out_file = DEFAULT.reports_dir / "performance.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Performance metrics saved to {out_file}")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    benchmark_inference()
