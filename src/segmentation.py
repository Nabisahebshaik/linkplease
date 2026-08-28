"""Classical, explainable retinal structure and lesion candidate extraction."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _green_channel(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def segment_vessels(image: Image.Image) -> np.ndarray:
    """Estimate vessels using blackhat morphology and adaptive thresholding."""
    gray = _green_channel(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.adaptiveThreshold(response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, -2)


def detect_lesions(image: Image.Image) -> dict[str, np.ndarray]:
    """Return transparent candidate masks for dark hemorrhages and bright exudates."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = cv2.inRange(rgb, (150, 150, 120), (255, 255, 255))
    dark = cv2.inRange(gray, 0, max(35, int(np.percentile(gray, 12))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return {
        "exudate_candidates": cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel),
        "hemorrhage_candidates": cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel),
    }


def localize_optic_disc(image: Image.Image) -> tuple[int, int, int] | None:
    """Estimate the brightest circular region as an optic-disc candidate."""
    gray = _green_channel(image)
    blurred = cv2.GaussianBlur(gray, (0, 0), 9)
    _, threshold = cv2.threshold(blurred, int(np.percentile(blurred, 98)), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return int(x), int(y), int(radius)
