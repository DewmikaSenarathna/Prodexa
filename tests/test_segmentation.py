import sys
import os
import numpy as np
import cv2
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.segmentation import (
    compute_intensity_mask, compute_saturation_mask, combine_masks,
    clean_mask, remove_small_components, segment_products,
)


@pytest.fixture
def synthetic_tray_image():
    
    image = np.full((300, 400, 3), 200, dtype=np.uint8)  
    cv2.rectangle(image, (40, 40), (140, 140), (0, 0, 255), thickness=-1)     
    cv2.rectangle(image, (220, 150), (340, 260), (255, 120, 0), thickness=-1) 
    return image


def test_compute_intensity_mask_shape(synthetic_tray_image):
    mask = compute_intensity_mask(synthetic_tray_image)
    assert mask.shape == synthetic_tray_image.shape[:2]
    assert mask.dtype == np.uint8


def test_compute_saturation_mask_flags_colourful_regions(synthetic_tray_image):
    mask = compute_saturation_mask(synthetic_tray_image)
    assert mask[90, 90] == 255


def test_combine_masks_is_union():
    a = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    b = np.array([[0, 0], [255, 0]], dtype=np.uint8)
    combined = combine_masks(a, b)
    assert combined[0, 1] == 255
    assert combined[1, 0] == 255
    assert combined[0, 0] == 0


def test_clean_mask_shape_unchanged(synthetic_tray_image):
    mask = compute_intensity_mask(synthetic_tray_image)
    cleaned = clean_mask(mask)
    assert cleaned.shape == mask.shape


def test_remove_small_components_drops_noise():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:12, 10:12] = 255      
    mask[50:80, 50:80] = 255     
    cleaned = remove_small_components(mask, min_area=50)
    assert cleaned[11, 11] == 0
    assert cleaned[65, 65] == 255


def test_segment_products_detects_two_blobs(synthetic_tray_image):
    result = segment_products(synthetic_tray_image, min_component_area=200)
    assert "mask" in result and "labels" in result
    assert result["mask"][90, 90] == 255
    assert result["mask"][200, 280] == 255
    assert result["mask"][10, 10] == 0
