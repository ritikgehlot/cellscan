# CellScan 🔆

**Explainable solar-cell defect detection from electroluminescence (EL) imagery — with calibrated confidence, not just predictions.**

Cracked or degraded photovoltaic cells silently reduce energy yield, and EL-image inspection of modules is still largely manual. CellScan is a compact, fully reproducible screening model for EL cell images that answers three questions instead of one:

1. **Is this cell defective?** — fine-tuned ResNet-18, single-logit defect score
2. **Where is the evidence?** — Grad-CAM heatmaps over the cell
3. **How much should you trust the score?** — temperature-scaled probabilities, with the calibration error measured before and after

> ⚠️ Built as a focused 1-day MVP with an explicit roadmap (below). It is a research/portfolio project, **not** a certified inspection tool.

---

## Demo

```bash
streamlit run app/streamlit_app.py
```

Upload an EL cell image → calibrated defect probability + Grad-CAM overlay showing the image regions that drove the score, plus a test-set results tab with the reliability diagram, confusion matrix, and the model's **hardest mistakes** (yes, failures are part of the demo — that's the point of an honest evaluation).

<!-- TODO after training: add demo GIF/screenshot here -->

## Results

**Every number below is written by `python -m cellscan.evaluate` into `reports/metrics.json`. Nothing in this table is typed by hand.**

| Metric (test split, n=394) | Value |
|---|---|
| AUROC | 0.9527 |
| Average precision | 0.9224 |
| F1 (defective class) | 0.8185 |
| Precision / Recall (defective) | 0.7737 / 0.8618 |
| ECE — uncalibrated → calibrated | 0.0726 → 0.0857 |
| AUROC mono / poly cells | 0.9304 / 0.9684 |

Reference points to beat before believing any model: predicting the majority class gives F1 = 0 on defects; the ELPV benchmark paper (Deitsch et al., 2019) reports ~88% accuracy for their CNN on a related 4-class formulation — a sanity anchor, **not** a direct comparison (different split and task binarization).

## Method

- **Data:** [ELPV dataset](https://github.com/zae-bayern/elpv-dataset) — 2,624 grayscale 300×300 EL cell images with expert defect probabilities {0, ⅓, ⅔, 1}, from 44 modules (1,550 poly / 1,074 mono). Binarized at ≥ 0.5 → 821 defective / 1,803 functional (31.3% positive).
- **Splits:** 70/15/15, seeded, stratified **jointly on label × cell type**, persisted to `data/splits.csv`. Mono and poly cells have very different texture; without joint stratification the test set silently skews.
- **Model:** ImageNet-pretrained ResNet-18, grayscale replicated to 3 channels (keeps the pretrained stem useful), single-logit head, class-weighted `BCEWithLogitsLoss` for the 31/69 imbalance. Model selection by validation AUROC.
- **Augmentation:** horizontal/vertical flips only. Rotations and crops create border artifacts that resemble the cracks we're detecting — mild augmentation is a deliberate choice, not laziness.
- **Calibration:** temperature scaling (Guo et al., 2017) fitted on the validation split only; the scalar T ships inside the checkpoint. It cannot change AUROC — it makes the displayed probability *mean something*.
- **Explainability:** Grad-CAM on `layer4`, implemented from the paper in ~50 lines (`src/cellscan/gradcam.py`) — no third-party CAM dependency to drift.

## Reproduce everything

```bash
git clone <this-repo> && cd cellscan
pip install -r requirements.txt

python scripts/prepare_data.py        # downloads ELPV, builds seeded splits
pytest                                # 7 tests incl. split-integrity checks
python -m cellscan.train              # ~15 epochs; minutes on an RTX 4060 (AMP)
python -m cellscan.evaluate           # writes reports/metrics.json + plots
streamlit run app/streamlit_app.py    # interactive demo
```

Pipeline sanity check without a GPU: `python -m cellscan.train --smoke --no-pretrained` (1 epoch, tiny subset — verifies the code path, produces intentionally meaningless numbers).

## Repository layout

```
src/cellscan/
  config.py     frozen dataclass config — one source of truth for knobs
  data.py       seeded stratified splits, dataset, transforms
  model.py      ResNet-18 factory + checkpoint I/O (weights + config + T)
  train.py      AMP training loop, best-val-AUROC selection, calibration
  calibrate.py  temperature scaling, ECE, reliability curves
  gradcam.py    minimal Grad-CAM implementation
  evaluate.py   THE source of all reported metrics, plots, failure cases
scripts/prepare_data.py   dataset download + split generation
app/streamlit_app.py      demo
tests/                    split integrity, shapes, calibration behavior
```

## Limitations (read before citing any number)

- 1-day MVP: single architecture, no hyperparameter search, no cross-validation, no ensembling.
- Binary formulation discards the ⅓ / ⅔ "possibly defective" granularity in the original labels.
- Trained and evaluated on one dataset from 44 modules — no claim of generalization to other cameras, module types, or field conditions.
- Grad-CAM shows *where* evidence is, not *what* the defect type is.

## Performance

Benchmarked on **RTX 4060** (8 GB VRAM):
- **Model size:** 44.5 MB
- **Inference time:** ~50-70 ms per image
- **Training time:** ~5 minutes for 15 epochs (AMP)

Temperature scaling fitted to validation set: T = 1.51
## Roadmap

- [ ] 4-class ordinal regression using the full defect-probability labels
- [ ] Mono → poly (and reverse) domain-shift study
- [ ] Selective prediction: abstain below a confidence threshold, report risk–coverage curves
- [ ] Crack segmentation head; ONNX export for edge deployment
- [ ] Turn the repo into a clean, contributable ELPV benchmark with a results table per method

## Dataset license & citation

The ELPV dataset is **CC BY-NC-SA 4.0** (non-commercial). Images are **not** redistributed in this repo; `scripts/prepare_data.py` downloads them from the official source. If you use this project, cite the dataset authors:

> Deitsch et al., *Automatic classification of defective photovoltaic module cells in electroluminescence images*, Solar Energy 185 (2019); and Buerhop-Lutz et al., *A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery*, 35th EU PVSEC 2018. DOI: 10.4229/35thEUPVSEC20182018-5CV.3.15

Code in this repository: MIT License.
