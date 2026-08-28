"""Build evaluation matrices for the trained DR model from the VS Code terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.preprocessing import CLASS_NAMES, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DR confusion matrix and classification report.")
    parser.add_argument("--model", default="vgg16_model.h5")
    parser.add_argument("--dataset-dir", default="dataset/colored_images")
    parser.add_argument("--labels-csv", default="train.csv")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    x_train, x_test, y_train, y_test = load_dataset(args.dataset_dir, args.labels_csv)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(x_test, batch_size=32, verbose=0).argmax(axis=1)
    actual = y_test.argmax(axis=1)
    output_dir = Path("metrics")
    output_dir.mkdir(exist_ok=True)

    matrix = confusion_matrix(actual, predictions, labels=range(len(CLASS_NAMES)))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    report = classification_report(
        actual, predictions, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        xticks_rotation="vertical", cmap="Blues", values_format="d"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    print(f"Evaluated {len(actual)} images")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Confusion matrix: {output_dir / 'confusion_matrix.png'}")
    print(f"CSV matrix: {output_dir / 'confusion_matrix.csv'}")
    print(f"Report: {output_dir / 'classification_report.json'}")


if __name__ == "__main__":
    main()
