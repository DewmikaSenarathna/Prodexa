import cv2
import numpy as np


DEFAULT_MORPH_KERNEL = 5
MIN_HOLE_FILL_AREA = 30


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _otsu_mask(channel):
   
    _, mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_fraction = np.count_nonzero(mask) / mask.size
    if foreground_fraction > 0.5:
        mask = cv2.bitwise_not(mask)
    return mask


def compute_intensity_mask(image):
    gray = to_grayscale(image)
    return _otsu_mask(gray)


def compute_saturation_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return _otsu_mask(saturation)


def combine_masks(mask_a, mask_b):
    return cv2.bitwise_or(mask_a, mask_b)


def fill_holes(mask):
    filled = mask.copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > MIN_HOLE_FILL_AREA:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def clean_mask(mask, kernel_size=DEFAULT_MORPH_KERNEL, open_iter=1, close_iter=2):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return fill_holes(closed)


def remove_small_components(mask, min_area):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, n_labels):  
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def separate_touching_objects(image, mask):
    
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
    markers = markers + 1         
    markers[unknown == 255] = 0    

    watershed_input = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_input.copy(), markers)
    return markers


def segment_products(image, min_component_area=150, use_watershed=True):
   
    intensity_mask = compute_intensity_mask(image)
    saturation_mask = compute_saturation_mask(image)
    combined = combine_masks(intensity_mask, saturation_mask)

    cleaned = clean_mask(combined)
    cleaned = remove_small_components(cleaned, min_component_area)

    labels = separate_touching_objects(image, cleaned) if use_watershed else None

    return {"mask": cleaned, "labels": labels}
