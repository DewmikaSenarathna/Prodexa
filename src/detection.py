import os
import cv2
import numpy as np


DEFAULT_MIN_AREA_RATIO = 0.0015   
DEFAULT_MAX_AREA_RATIO = 0.60     
DEFAULT_MIN_ASPECT = 0.15         
DEFAULT_PADDING = 6               


def _contours_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _contours_from_labels(labels):
    contours = []
    for label in np.unique(labels):
        if label <= 1:  
            continue
        instance_mask = np.uint8(labels == label) * 255
        instance_contours = _contours_from_mask(instance_mask)
        if instance_contours:
            contours.append(max(instance_contours, key=cv2.contourArea))
    return contours


def _is_valid_contour(contour, image_area, min_area_ratio, max_area_ratio, min_aspect):
    area = cv2.contourArea(contour)
    area_ratio = area / image_area
    if not (min_area_ratio <= area_ratio <= max_area_ratio):
        return False

    _, _, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return False
    aspect = min(w, h) / max(w, h)
    return aspect >= min_aspect


def filter_contours(contours, image_shape,
                     min_area_ratio=DEFAULT_MIN_AREA_RATIO,
                     max_area_ratio=DEFAULT_MAX_AREA_RATIO,
                     min_aspect=DEFAULT_MIN_ASPECT):
    image_area = image_shape[0] * image_shape[1]
    return [
        c for c in contours
        if _is_valid_contour(c, image_area, min_area_ratio, max_area_ratio, min_aspect)
    ]


def detect_products(image, segmentation_result,
                     min_area_ratio=DEFAULT_MIN_AREA_RATIO,
                     max_area_ratio=DEFAULT_MAX_AREA_RATIO,
                     min_aspect=DEFAULT_MIN_ASPECT):
    
    labels = segmentation_result.get("labels")
    if labels is not None:
        contours = _contours_from_labels(labels)
    else:
        contours = _contours_from_mask(segmentation_result["mask"])

    contours = filter_contours(contours, image.shape[:2],
                                min_area_ratio, max_area_ratio, min_aspect)

    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        detections.append({
            "bbox": (x, y, w, h),
            "area": cv2.contourArea(contour),
            "contour": contour,
        })

    detections.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    for idx, det in enumerate(detections, start=1):
        det["id"] = idx

    return detections


def crop_products(image, detections, padding=DEFAULT_PADDING):
    
    h_img, w_img = image.shape[:2]
    crops = []
    for det in detections:
        x, y, w, h = det["bbox"]
        x0 = max(x - padding, 0)
        y0 = max(y - padding, 0)
        x1 = min(x + w + padding, w_img)
        y1 = min(y + h + padding, h_img)
        crops.append((det["id"], image[y0:y1, x0:x1].copy()))
    return crops


def draw_detections(image, detections, labels=None, color=(0, 220, 0), thickness=2):

    annotated = image.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        if labels is not None:
            text = str(labels.get(det["id"], det["id"]))
        else:
            text = f"#{det['id']}"

        text_origin = (x, max(y - 8, 12))
        cv2.putText(annotated, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)   
        cv2.putText(annotated, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def save_detection_outputs(image, detections, output_dir, base_name):
    
    annotated_dir = os.path.join(output_dir, "annotated_images")
    crops_dir = os.path.join(output_dir, "crops", base_name)
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    annotated = draw_detections(image, detections)
    annotated_path = os.path.join(annotated_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    crop_paths = []
    for det_id, crop in crop_products(image, detections):
        crop_path = os.path.join(crops_dir, f"{base_name}_product_{det_id:02d}.jpg")
        cv2.imwrite(crop_path, crop)
        crop_paths.append(crop_path)

    return {"annotated_path": annotated_path, "crop_paths": crop_paths}


def print_detection_summary(detections, image_name=""):
    title = f"Detected products in {image_name}" if image_name else "Detected products"
    print(title)
    print("-" * len(title))
    print(f"Total products detected: {len(detections)}")
    for det in detections:
        x, y, w, h = det["bbox"]
        print(f"  #{det['id']:02d}  bbox=(x={x}, y={y}, w={w}, h={h})  area={det['area']:.0f}px")


def detect_and_report(image, segmentation_result, output_dir=None, base_name="image",
                       **filter_kwargs):
    
    detections = detect_products(image, segmentation_result, **filter_kwargs)
    print_detection_summary(detections, image_name=base_name)

    if output_dir is not None:
        save_detection_outputs(image, detections, output_dir, base_name)

    return detections
