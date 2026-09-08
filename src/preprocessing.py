import cv2
import numpy as np
import os
import json
import random
from collections import defaultdict


DEFAULT_TARGET_WIDTH = 800


def resize_image(image, target_width=DEFAULT_TARGET_WIDTH):
    
    h, w = image.shape[:2]
    scale = target_width / w
    new_dim = (target_width, int(h * scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


def remove_noise(image, method="median", ksize=5):
   
    if method == "median":
        return cv2.medianBlur(image, ksize)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    else:
        raise ValueError("method must be 'median' or 'gaussian'")


def normalize_lighting(image):
   
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_image(image_path, target_width=DEFAULT_TARGET_WIDTH, denoise_method="median"):
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = resize_image(image, target_width)
    image = remove_noise(image, method=denoise_method)
    image = normalize_lighting(image)
    return image


def get_sample_filenames(directory, limit=20):
    
    filenames = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                filenames.append(entry.name)
                if len(filenames) >= limit:
                    break
    return filenames


def batch_preprocess(filenames, source_dir, output_dir, target_width=DEFAULT_TARGET_WIDTH,
                      denoise_method="median"):
    
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0

    for fname in filenames:
        try:
            in_path = os.path.join(source_dir, fname)
            out_path = os.path.join(output_dir, fname)
            result = preprocess_image(in_path, target_width, denoise_method)
            cv2.imwrite(out_path, result)
            success_count += 1
        except Exception as e:
            print(f"Failed on {fname}: {e}")

    return success_count, len(filenames)




def load_image_category_map(annotations_path):
    """
    Build a mapping of image_id -> category info, and category_id -> category name/supercategory,
    from a COCO-format annotations JSON file.

    Args:
        annotations_path: path to instances_*.json

    Returns:
        Tuple of:
            image_to_category: dict mapping image_id -> category_id
            image_id_to_filename: dict mapping image_id -> filename
            category_info: dict mapping category_id -> {'name': ..., 'supercategory': ...}
    """
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    category_info = {
        cat["id"]: {"name": cat["name"], "supercategory": cat["supercategory"]}
        for cat in coco_data["categories"]
    }

    image_id_to_filename = {
        img["id"]: img["file_name"] for img in coco_data["images"]
    }

    # train2019 has 1 annotation per image, so this is a simple 1:1 map
    image_to_category = {
        ann["image_id"]: ann["category_id"] for ann in coco_data["annotations"]
    }

    return image_to_category, image_id_to_filename, category_info


def stratified_sample(annotations_path, images_per_category=80, seed=42):
    """
    Sample a balanced number of images per meta-category (supercategory) from
    a COCO-format annotations file.

    Args:
        annotations_path: path to instances_*.json
        images_per_category: number of images to sample per supercategory
        seed: random seed for reproducibility

    Returns:
        List of dicts: [{'filename': ..., 'supercategory': ..., 'category_name': ...}, ...]
    """
    random.seed(seed)
    image_to_category, image_id_to_filename, category_info = load_image_category_map(annotations_path)

    
    supercategory_to_image_ids = defaultdict(list)
    for image_id, category_id in image_to_category.items():
        supercategory = category_info[category_id]["supercategory"]
        supercategory_to_image_ids[supercategory].append(image_id)

    sampled = []
    for supercategory, image_ids in supercategory_to_image_ids.items():
        chosen_ids = random.sample(image_ids, min(images_per_category, len(image_ids)))
        for image_id in chosen_ids:
            category_id = image_to_category[image_id]
            sampled.append({
                "filename": image_id_to_filename[image_id],
                "supercategory": supercategory,
                "category_name": category_info[category_id]["name"]
            })

    return sampled