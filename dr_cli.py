"""VS Code terminal workflow for local DR screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from PIL import Image

from src.inference import predict
from src.reporting import build_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local DR screening without Streamlit.")
    parser.add_argument("image", type=Path, help="Fundus image path.")
    parser.add_argument("--model", type=Path, default=Path("vgg16_model.h5"))
    parser.add_argument("--report", type=Path, help="Optional HTML report output path.")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    result = predict(tf.keras.models.load_model(args.model), Image.open(args.image))
    output = {
        "image": str(args.image),
        "quality": result["quality"].__dict__,
        "label": result["label"],
        "confidence": result["confidence"],
        "lesions": result["lesions"],
    }
    print(json.dumps(output, indent=2))
    if args.report and result["label"] is not None:
        html = build_html_report(
            result["label"], result["confidence"], result["quality"].grade,
            result["quality"].feedback, result["lesions"],
        )
        args.report.write_text(html, encoding="utf-8")
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
