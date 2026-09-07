import cv2
import numpy as np
import os


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