"""
Unit tests for src/preprocessing.py
Run with: pytest tests/test_preprocessing.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import resize_image, remove_noise, normalize_lighting


@pytest.fixture
def dummy_image():
    """A small synthetic BGR image for fast testing (no dataset dependency)."""
    return np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)


def test_resize_image_width(dummy_image):
    resized = resize_image(dummy_image, target_width=200)
    assert resized.shape[1] == 200


def test_resize_image_aspect_ratio_preserved(dummy_image):
    original_ratio = dummy_image.shape[0] / dummy_image.shape[1]
    resized = resize_image(dummy_image, target_width=300)
    resized_ratio = resized.shape[0] / resized.shape[1]
    assert abs(original_ratio - resized_ratio) < 0.01


def test_remove_noise_median_shape_unchanged(dummy_image):
    denoised = remove_noise(dummy_image, method="median")
    assert denoised.shape == dummy_image.shape


def test_remove_noise_invalid_method_raises(dummy_image):
    with pytest.raises(ValueError):
        remove_noise(dummy_image, method="invalid")


def test_normalize_lighting_shape_unchanged(dummy_image):
    normalized = normalize_lighting(dummy_image)
    assert normalized.shape == dummy_image.shapesssss