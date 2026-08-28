"""Grad-CAM explanations for VGG16 screening predictions."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image


def make_gradcam_heatmap(
    image_batch: tf.Tensor | np.ndarray,
    model: tf.keras.Model,
    class_index: int | None = None,
    last_conv_layer_name: str = "block5_conv3",
) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap for one preprocessed image."""
    image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    nested_base = None
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        try:
            nested_base = model.get_layer("vgg16")
            conv_layer = nested_base.get_layer(last_conv_layer_name)
        except ValueError as exc:
            raise ValueError(f"Model has no convolutional layer {last_conv_layer_name!r}") from exc
    if nested_base is None:
        grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
        model_input = image_batch
    else:
        # Keras 3 may deserialize nested layer outputs as a separate graph.
        grad_model = tf.keras.Model(nested_base.input, conv_layer.output)
        model_input = image_batch
    with tf.GradientTape() as tape:
        if nested_base is None:
            conv_outputs, predictions = grad_model(model_input, training=False)
        else:
            conv_outputs = grad_model(model_input, training=False)
            pooled = nested_base.get_layer("block5_pool")(conv_outputs)
            features = model.get_layer("flatten")(pooled)
            features = model.get_layer("dense")(features)
            features = model.get_layer("dropout")(features, training=False)
            predictions = model.get_layer("dense_1")(features)
        index = tf.argmax(predictions[0]) if class_index is None else class_index
        score = predictions[:, index]
    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients were not produced for the requested class")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)[0]
    cam = tf.maximum(cam, 0)
    maximum = tf.reduce_max(cam)
    return (cam / (maximum + tf.keras.backend.epsilon())).numpy()


def overlay_heatmap(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Colorize and blend a heatmap with the original image."""
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(original.size)
    # PIL's hot colormap is not available; use a compact blue-to-red gradient.
    values = np.asarray(heat).astype(np.float32) / 255
    rgb = np.stack([values, np.sqrt(values), 1 - values], axis=-1)
    colored = Image.fromarray(np.uint8(rgb * 255))
    return Image.blend(original.convert("RGB"), colored, alpha)
