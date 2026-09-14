"""
segmentation.py

Product / background segmentation for the PRODEXA pipeline.

Input to this module is expected to be an already-preprocessed image
(see src/preprocessing.py: resized, denoised, lighting-normalized).

The RPC "checkout" images used for this project show products placed on a
fairly plain, low-saturation tray/table surface, so the segmentation
strategy combines two independent cues so it isn't fooled by either one
alone:

    1. Intensity (grayscale + Otsu threshold)  -> catches products whose
       brightness clearly differs from the tray.
    2. Saturation (HSV S channel + Otsu)       -> catches products that
       are colourful even when their brightness is close to the tray's.

The two masks are OR-ed together, cleaned up with morphological
operations, and (optionally) split with a distance-transform + watershed
step so that products placed close together are not merged into a
single blob.
"""

import cv2
import numpy as np


DEFAULT_MORPH_KERNEL = 5
MIN_HOLE_FILL_AREA = 30


def to_grayscale(image):
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _otsu_mask(channel):
    """
    Otsu-threshold a single-channel image and return a mask where the
    *minority* class (fewer pixels) is treated as foreground.

    Assumes the background (tray) occupies most of the frame, which holds
    for basket/checkout-tray layout photos.
    """
    _, mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_fraction = np.count_nonzero(mask) / mask.size
    if foreground_fraction > 0.5:
        mask = cv2.bitwise_not(mask)
    return mask


def compute_intensity_mask(image):
    """Foreground mask from grayscale intensity (Otsu)."""
    gray = to_grayscale(image)
    return _otsu_mask(gray)


def compute_saturation_mask(image):
    """Foreground mask from HSV saturation (Otsu)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return _otsu_mask(saturation)


def combine_masks(mask_a, mask_b):
    """Combine two binary masks with a logical OR."""
    return cv2.bitwise_or(mask_a, mask_b)


def fill_holes(mask):
    """Fill small enclosed holes inside foreground blobs (e.g. labels, glare)."""
    filled = mask.copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > MIN_HOLE_FILL_AREA:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def clean_mask(mask, kernel_size=DEFAULT_MORPH_KERNEL, open_iter=1, close_iter=2):
    """
    Remove speckle noise and close small gaps in a binary mask using
    morphological opening followed by closing, then fill interior holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return fill_holes(closed)


def remove_small_components(mask, min_area):
    """Drop connected components in a binary mask smaller than min_area."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, n_labels):  # label 0 is background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def separate_touching_objects(image, mask):
    """
    Split blobs of products that are touching/overlapping using a
    distance-transform + marker-based watershed.

    Returns a label image (int32, same H x W as `mask`) where:
        0/-1  -> background / watershed boundary
        1..N  -> individual product instances
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = dist / dist.max()
    else:
        dist_norm = dist
    _, sure_fg = cv2.threshold(dist_norm, 0.45, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # so background (from connectedComponents) is 1, not 0
    markers[unknown == 255] = 0    # region to be decided by watershed

    watershed_input = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_input.copy(), markers)
    return markers


def segment_products(image, min_component_area=150, use_watershed=True):
    """
    Full segmentation pipeline: preprocessed image -> cleaned binary
    foreground mask (+ optional per-instance label map).

    Args:
        image: preprocessed BGR image (np.ndarray, HxWx3).
        min_component_area: components smaller than this (in pixels) are
            discarded as noise before any watershed split.
        use_watershed: if True, also return an instance label map that
            attempts to separate touching products.

    Returns:
        dict with:
            "mask":   cleaned binary foreground mask (uint8, 0/255)
            "labels": instance label image (int32) if use_watershed else None
    """
    intensity_mask = compute_intensity_mask(image)
    saturation_mask = compute_saturation_mask(image)
    combined = combine_masks(intensity_mask, saturation_mask)

    cleaned = clean_mask(combined)
    cleaned = remove_small_components(cleaned, min_component_area)

    labels = separate_touching_objects(image, cleaned) if use_watershed else None

    return {"mask": cleaned, "labels": labels}
