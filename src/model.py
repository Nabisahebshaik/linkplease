"""VGG16 transfer-learning model."""

from __future__ import annotations

import tensorflow as tf

from .preprocessing import CLASS_NAMES, IMAGE_SIZE


def build_model(learning_rate: float = 1e-4, dropout: float = 0.5) -> tf.keras.Model:
    """Build a frozen ImageNet VGG16 backbone with a five-class head."""
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = base(inputs, training=False)
    x = tf.keras.layers.Flatten()(features)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="vgg16_diabetic_retinopathy")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
