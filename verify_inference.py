"""Load the serialized model and run one inference for final verification."""

from pathlib import Path
import sys

from PIL import Image
from src.inference import predict
import tensorflow as tf

model_path = Path("vgg16_model.h5")
if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)
sample = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if sample:
    result = predict(model, Image.open(sample))
    if result["label"] is None:
        print(f"Image rejected: {result['quality'].feedback}")
    else:
        print(f"{result['label']}: {result['confidence']:.4f}")
else:
    print("Model loaded successfully; pass an image path to run inference.")
