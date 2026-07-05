"""
Full CellScan pipeline: data prep -> train -> evaluate -> benchmark.

Run once: python scripts/run_full_pipeline.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_cmd(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"✗ Failed: {description}")
        sys.exit(1)
    print(f"✓ Done: {description}")


def main():
    print("🔆 CellScan Full Pipeline — Metrics Generation")
    print(f"Root: {ROOT}")
    
    if not (ROOT / "data" / "splits.csv").exists():
        run_cmd(
            "python scripts/prepare_data.py",
            "Download ELPV dataset & generate train/val/test splits"
        )
    else:
        print("✓ Data splits already exist, skipping prep")
    
    run_cmd(
        "python -m pytest -v",
        "Run test suite (split integrity, shape checks, calibration)"
    )
    
    ckpt_file = ROOT / "models" / "cellscan_resnet18.pt"
    if not ckpt_file.exists():
        run_cmd(
            "python -m cellscan.train",
            "Train ResNet-18 with AMP on ELPV (~15 epochs, ~5 min on RTX 4060)"
        )
    else:
        print(f"✓ Checkpoint already exists at {ckpt_file}, skipping training")
    
    run_cmd(
        "python -m cellscan.evaluate",
        "Evaluate on test set → reports/metrics.json + plots"
    )
    
    run_cmd(
        "python scripts/benchmark_latency.py",
        "Measure inference latency & throughput → reports/performance.json"
    )
    
    print(f"\n{'='*60}")
    print("✓ PIPELINE COMPLETE")
    print(f"{'='*60}")
    
    metrics_file = ROOT / "reports" / "metrics.json"
    perf_file = ROOT / "reports" / "performance.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        print(f"\n📊 Test Set Metrics (n={metrics['n_test']}):")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  F1 (defect): {metrics['f1_defective']:.4f}")
        print(f"  Precision / Recall: {metrics['precision_defective']:.4f} / {metrics['recall_defective']:.4f}")
        print(f"  ECE: {metrics['ece_uncalibrated']:.4f} (uncal) → {metrics['ece_calibrated']:.4f} (cal)")
    
    if perf_file.exists():
        with open(perf_file) as f:
            perf = json.load(f)
        print(f"\n⚡ Performance (RTX 4060):")
        print(f"  Single-image latency: {perf['single_image_latency_ms']:.1f} ms")
        print(f"  Batch-32 latency: {perf['batch_32_latency_ms']:.1f} ms")
        print(f"  Throughput: {perf['throughput_images_per_second']:.1f} img/sec")
        print(f"  Model size: {perf['model_size_mb']:.1f} MB")
    
    print(f"\n📝 Next: Update README.md with metrics from:")
    print(f"  - {metrics_file}")
    print(f"  - {perf_file}")


if __name__ == "__main__":
    main()
