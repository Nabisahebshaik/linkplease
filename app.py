"""Streamlit dashboard for rural diabetic-retinopathy screening."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import predict
from src.reporting import build_html_report
from src.simulation import estimate_capacity

MODEL_PATH = Path("vgg16_model.h5")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


st.set_page_config(page_title="DR Screening", page_icon=":eye:", layout="wide")
st.title("Explainable Diabetic Retinopathy Screening")
st.caption("Decision support only: confirm every result with an ophthalmologist.")
uploaded = st.file_uploader("Upload a fundus photograph", type=("jpg", "jpeg", "png"))
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    try:
        model = load_model()
        result = predict(model, image)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
    else:
        quality = result["quality"]
        st.metric("Image quality", quality.grade.title())
        if result["label"] is None:
            st.error(quality.feedback)
            st.stop()
        label, confidence, explanation = result["label"], result["confidence"], result["explanation"]
        st.subheader(f"Result: {label.replace('_', ' ')} ({confidence:.1%} confidence)")
        if label in {"Severe", "Proliferate_DR"}:
            st.error("Immediate ophthalmologist referral recommended.")
        else:
            st.info("Routine clinical review recommended; this is not a diagnosis.")
        original, heatmap = st.columns(2)
        original.image(image, caption="Original fundus photo", use_container_width=True)
        heatmap.image(explanation, caption="Grad-CAM explanation", use_container_width=True)
        st.subheader("Structure and lesion candidates")
        st.json(result["lesions"])
        report = build_html_report(label, confidence, quality.grade, quality.feedback, result["lesions"])
        st.download_button("Download clinical report", report, "dr_screening_report.html", "text/html")

st.sidebar.subheader("Telemedicine capacity estimate")
capacity = estimate_capacity()
st.sidebar.write(f"Bottleneck: **{capacity.bottleneck}**")
st.sidebar.caption("Planning estimate for 100,000 patients; validate production rates in SimEvents.")
