import sys
import os
import numpy as np
import cv2
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.segmentation import segment_products
from src.detection import (
    detect_products, filter_contours, crop_products, draw_detections,
)


@pytest.fixture
def synthetic_tray_image():
    image = np.full((300, 400, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (140, 140), (0, 0, 255), thickness=-1)
    cv2.rectangle(image, (220, 150), (340, 260), (255, 120, 0), thickness=-1)
    return image


def test_detect_products_finds_both_blocks(synthetic_tray_image):
    seg = segment_products(synthetic_tray_image, min_component_area=200)
    detections = detect_products(synthetic_tray_image, seg)
    assert len(detections) == 2
    for det in detections:
        assert "bbox" in det and "id" in det


def test_detections_are_left_to_right(synthetic_tray_image):
    seg = segment_products(synthetic_tray_image, min_component_area=200)
    detections = detect_products(synthetic_tray_image, seg)
    x_positions = [d["bbox"][0] for d in detections]
    assert x_positions == sorted(x_positions)


def test_filter_contours_rejects_tiny_noise():
    contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]])  
    kept = filter_contours([contour], image_shape=(300, 400), min_area_ratio=0.01)
    assert kept == []


def test_crop_products_returns_expected_count(synthetic_tray_image):
    seg = segment_products(synthetic_tray_image, min_component_area=200)
    detections = detect_products(synthetic_tray_image, seg)
    crops = crop_products(synthetic_tray_image, detections)
    assert len(crops) == len(detections)
    for det_id, crop in crops:
        assert crop.size > 0


def test_draw_detections_shape_unchanged(synthetic_tray_image):
    seg = segment_products(synthetic_tray_image, min_component_area=200)
    detections = detect_products(synthetic_tray_image, seg)
    annotated = draw_detections(synthetic_tray_image, detections)
    assert annotated.shape == synthetic_tray_image.shape
