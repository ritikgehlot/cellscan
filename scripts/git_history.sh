#!/usr/bin/env bash
# Build the CellScan commit history under YOUR identity.
# Prereq: git config --global user.name / user.email already set.
set -euo pipefail
git init -b main
git add .gitignore && git commit -m "chore: scaffold project structure and tooling"
git add LICENSE requirements.txt && git commit -m "chore: add MIT license and pinned requirements"
git add src/cellscan/config.py src/cellscan/__init__.py src/cellscan/utils.py \
  && git commit -m "feat: add typed config and shared utilities (seeding, logging, batched inference)"
git add src/cellscan/data.py scripts/prepare_data.py \
  && git commit -m "feat: add ELPV download pipeline and jointly-stratified seeded splits"
git add src/cellscan/model.py src/cellscan/train.py \
  && git commit -m "feat: implement ResNet-18 training with AMP, class weighting, val-AUROC selection"
git add src/cellscan/calibrate.py \
  && git commit -m "feat: add temperature scaling calibration and ECE metrics"
git add src/cellscan/gradcam.py \
  && git commit -m "feat: add minimal Grad-CAM implementation for defect localization"
git add src/cellscan/evaluate.py \
  && git commit -m "feat: add evaluation script writing metrics.json, plots, and failure cases"
git add tests/ && git commit -m "test: add split-integrity, model-shape, and calibration tests"
git add app/ && git commit -m "feat: add Streamlit demo with calibrated verdicts and Grad-CAM overlays"
git add README.md scripts/git_history.sh \
  && git commit -m "docs: add README with reproducible-results policy and limitations"
echo; git log --oneline
