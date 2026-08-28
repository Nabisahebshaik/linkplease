"""Dataset discovery, validation, balancing, and VGG16 preprocessing."""

from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

SEED = 42
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def collect_images(
    dataset_dir: str | Path = "dataset/colored_images",
    labels_csv: str | Path | None = "train.csv",
) -> pd.DataFrame:
    """Return image paths and folder labels, validating optional train.csv IDs."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    folders = [path for path in root.iterdir() if path.is_dir()]
    unknown = sorted(path.name for path in folders if path.name not in CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unexpected class folders: {unknown}")
    rows = [
        {"path": str(path), "image_id": path.stem, "label": path.parent.name}
        for label in CLASS_NAMES
        for path in sorted((root / label).glob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No supported images found below {root}")
    csv_path = Path(labels_csv) if labels_csv else None
    if csv_path and csv_path.is_file():
        csv = pd.read_csv(csv_path)
        id_column = next((c for c in ("id_code", "image_id", "id") if c in csv), None)
        diagnosis_column = "diagnosis" if "diagnosis" in csv else None
        if id_column and diagnosis_column:
            expected = frame[["image_id", "label"]].copy()
            expected["image_id"] = expected["image_id"].str.lower()
            actual = csv[[id_column, diagnosis_column]].copy()
            actual[id_column] = actual[id_column].astype(str).str.lower().str.replace(
                r"\.(jpg|jpeg|png)$", "", regex=True
            )
            numeric_to_class = dict(enumerate(CLASS_NAMES))
            actual["csv_label"] = actual[diagnosis_column].map(numeric_to_class)
            joined = expected.merge(actual[[id_column, "csv_label"]], left_on="image_id", right_on=id_column)
            mismatches = joined[joined["label"] != joined["csv_label"]]
            missing = len(expected) - len(joined)
            if len(mismatches) or missing:
                raise ValueError(
                    f"train.csv validation failed: {len(mismatches)} label mismatches and {missing} missing IDs"
                )
    return frame


def oversample(frame: pd.DataFrame, target_count: int = 1805) -> pd.DataFrame:
    """Balance every class by sampling with replacement, deterministically."""
    target = max(target_count, int(frame["label"].value_counts().max()))
    parts = [
        group.sample(target, replace=len(group) < target, random_state=SEED)
        for _, group in frame.groupby("label", sort=False)
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def load_image(path: str | Path) -> np.ndarray:
    image = tf.keras.utils.load_img(path, target_size=IMAGE_SIZE)
    return tf.keras.applications.vgg16.preprocess_input(
        tf.keras.utils.img_to_array(image)
    )


def load_dataset(
    dataset_dir: str | Path = "dataset/colored_images",
    labels_csv: str | Path | None = "train.csv",
    target_count: int = 1805,
    test_size: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, balance, preprocess, and stratify the image dataset."""
    balanced = oversample(collect_images(dataset_dir, labels_csv), target_count)
    images = np.stack([load_image(path) for path in balanced["path"]]).astype("float32")
    labels = tf.keras.utils.to_categorical(
        balanced["label"].map(CLASS_TO_INDEX).to_numpy(), num_classes=len(CLASS_NAMES)
    )
    return train_test_split(images, labels, test_size=test_size, stratify=labels.argmax(1), random_state=SEED)
