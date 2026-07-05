"""CellScan demo: upload an EL cell image, get a calibrated defect
probability and a Grad-CAM heatmap showing which region drove the score.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cellscan.config import DEFAULT  # noqa: E402
from cellscan.data import build_transforms  # noqa: E402
from cellscan.gradcam import GradCAM, overlay  # noqa: E402
from cellscan.model import load_checkpoint  # noqa: E402

CKPT = DEFAULT.models_dir / "cellscan_resnet18.pt"

st.set_page_config(page_title="CellScan", page_icon="🔆", layout="wide")
st.title("🔆 CellScan — Solar Cell Defect Detection")
st.caption(
    "Electroluminescence-image defect screening · ResNet-18 · Grad-CAM "
    "explainability · temperature-scaled confidence. Research/education demo "
    "on the ELPV dataset (CC BY-NC-SA 4.0) — not a certified inspection tool."
)


@st.cache_resource
def get_model():
    if not CKPT.exists():
        return None
    model, temperature, cfg = load_checkpoint(CKPT)
    return model, temperature, cfg


loaded = get_model()
if loaded is None:
    st.warning(
        f"No trained checkpoint at `{CKPT}`.\n\n"
        "Train one first:\n```\npython scripts/prepare_data.py\n"
        "python -m cellscan.train\n```"
    )
    st.stop()
model, temperature, ckpt_cfg = loaded
tf = build_transforms(DEFAULT, train=False)

tab_predict, tab_metrics = st.tabs(["🔍 Inspect a cell", "📊 Test-set results"])

with tab_predict:
    up = st.file_uploader("Upload an EL cell image (grayscale PNG/JPG)",
                          type=["png", "jpg", "jpeg"])
    if up is not None:
        img = Image.open(up).convert("L")
        x = tf(img).unsqueeze(0)
        cam, logit = GradCAM(model)(x)
        prob = float(torch.sigmoid(torch.tensor(logit / temperature)))

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.subheader("Input")
            st.image(img, use_container_width=True)
        with c2:
            st.subheader("Grad-CAM")
            heat = overlay(np.array(img.resize((300, 300))),
                           np.array(Image.fromarray(
                               (cam * 255).astype(np.uint8)).resize((300, 300))) / 255.0)
            st.image(heat, use_container_width=True,
                     caption="regions driving the defect score")
        with c3:
            st.subheader("Verdict")
            verdict = "DEFECTIVE" if prob >= 0.5 else "FUNCTIONAL"
            (st.error if verdict == "DEFECTIVE" else st.success)(
                f"**{verdict}**")
            st.metric("Calibrated P(defect)", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(
                f"Temperature T = {temperature:.3f} fitted on the validation "
                "split; probabilities are calibrated, not raw sigmoid output."
            )
    else:
        st.info("Upload a cell image, or grab one from `data/elpv/images/` "
                "after running `scripts/prepare_data.py`.")

with tab_metrics:
    metrics_path = DEFAULT.reports_dir / "metrics.json"
    if not metrics_path.exists():
        st.info("Run `python -m cellscan.evaluate` to generate test metrics.")
    else:
        m = json.loads(metrics_path.read_text())
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [
            ("AUROC", "auroc"), ("F1 (defective)", "f1_defective"),
            ("Precision", "precision_defective"),
            ("Recall", "recall_defective"), ("ECE (calibrated)", "ece_calibrated"),
        ]):
            col.metric(label, f"{m[key]:.3f}")
        c1, c2 = st.columns(2)
        rel = DEFAULT.reports_dir / "reliability_diagram.png"
        cmx = DEFAULT.reports_dir / "confusion_matrix.png"
        if rel.exists():
            c1.image(str(rel))
        if cmx.exists():
            c2.image(str(cmx))
        st.subheader("Hardest mistakes (honesty section)")
        fails = m.get("failure_cases", [])
        if fails:
            fc = st.columns(min(4, len(fails)))
            for i, rec in enumerate(fails[:8]):
                img_path = ROOT / rec["image"]
                if img_path.exists():
                    fc[i % 4].image(
                        str(img_path),
                        caption=(f"true={'defect' if rec['true_label'] else 'ok'} "
                                 f"pred={rec['pred_prob_defect']:.2f} "
                                 f"({rec['cell_type']})"),
                    )
