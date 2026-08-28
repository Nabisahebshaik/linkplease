"""Fundus image quality assessment and conservative enhancement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityAssessment:
    focus: float
    illumination: float
    fov_fraction: float
    grade: str
    feedback: str


def assess_quality(image: Image.Image) -> QualityAssessment:
    """Measure focus, illumination, and field-of-view coverage."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    illumination = float(gray.mean() / 255.0)
    mask = gray > max(8, int(np.percentile(gray, 10)))
    fov_fraction = float(mask.mean())
    if focus < 20 or fov_fraction < 0.20:
        return QualityAssessment(focus, illumination, fov_fraction, "ungradeable",
                                 "Recapture with the lens centered, clean, and in focus.")
    if focus < 60 or illumination < 0.15 or illumination > 0.85:
        return QualityAssessment(focus, illumination, fov_fraction, "borderline",
                                 "Improve focus and even illumination before clinical review.")
    return QualityAssessment(focus, illumination, fov_fraction, "gradeable", "")


def enhance(image: Image.Image) -> Image.Image:
    """Apply illumination normalization, CLAHE, and light denoising."""
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    background = cv2.GaussianBlur(l_channel, (0, 0), 21)
    normalized = cv2.normalize(
        cv2.divide(l_channel, np.maximum(background, 1), scale=128),
        None, 0, 255, cv2.NORM_MINMAX,
    ).astype(np.uint8)
    enhanced_l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21))
