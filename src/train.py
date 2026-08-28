"""Train and evaluate the diabetic-retinopathy classifier."""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from .model import build_model
from .preprocessing import CLASS_NAMES, SEED, load_dataset


def train(
    dataset_dir: str | None = None,
    labels_csv: str | None = None,
    output_path: str = "vgg16_model.h5",
    epochs: int = 15,
    batch_size: int = 32,
) -> dict:
    dataset_dir = dataset_dir or os.environ.get("DR_DATASET_DIR", "dataset/colored_images")
    labels_csv = labels_csv if labels_csv is not None else os.environ.get("DR_LABELS_CSV", "train.csv")
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    x_train, x_test, y_train, y_test = load_dataset(dataset_dir, labels_csv)
    model = build_model()
    Path("metrics").mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.2, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger("metrics/training_history.csv"),
    ]
    history = model.fit(
        x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, shuffle=True,
    )
    model = tf.keras.models.load_model(output_path)
    probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_true, y_pred = y_test.argmax(1), probabilities.argmax(1)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    with (Path("metrics") / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with Path("execution_log.md").open("a", encoding="utf-8") as log:
        log.write(
            f"\n- Training completed: epochs={len(history.epoch)}, "
            f"test_loss={model.evaluate(x_test, y_test, verbose=0)[0]:.4f}, "
            f"test_accuracy={report['accuracy']:.4f}. Metrics saved under `metrics/`.\n"
        )
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES, xticks_rotation="vertical")
    plt.tight_layout()
    plt.savefig(Path("metrics") / "confusion_matrix.png", dpi=150)
    plt.close()
    return {"test_loss_accuracy": model.evaluate(x_test, y_test, verbose=0), "report": report, "history": history.history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DR screening model.")
    parser.add_argument("--dataset-dir", default=None, help="Path to colored_images directory.")
    parser.add_argument("--labels-csv", default=None, help="Path to train.csv.")
    parser.add_argument("--output-path", default="vgg16_model.h5")
    args = parser.parse_args()
    train(args.dataset_dir, args.labels_csv, args.output_path)
