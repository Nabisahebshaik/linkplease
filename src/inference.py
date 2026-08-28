"""Shared inference helpers for the command line and Streamlit app."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image

from .preprocessing import CLASS_NAMES, IMAGE_SIZE
from .quality import assess_quality, enhance
from .segmentation import detect_lesions
from .xai import make_gradcam_heatmap, overlay_heatmap


def predict(model: tf.keras.Model, image: Image.Image) -> dict:
    quality = assess_quality(image)
    if quality.grade == "ungradeable":
        return {"quality": quality, "label": None, "confidence": 0.0, "explanation": None, "lesions": {}}
    processed = enhance(image) if quality.grade == "borderline" else image
    resized = processed.convert("RGB").resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(resized)
    batch = tf.keras.applications.vgg16.preprocess_input(array)[None, ...]
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    heatmap = make_gradcam_heatmap(batch, model, index)
    lesions = detect_lesions(processed)
    return {
        "quality": quality,
        "label": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "explanation": overlay_heatmap(image, heatmap),
        "lesions": {name: int(mask.astype(bool).sum()) for name, mask in lesions.items()},
    }
