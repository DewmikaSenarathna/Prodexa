"""
PRODEXA - Classification Module
Author: Member B

Extracts features from product images using a pretrained MobileNetV2 CNN
(as a frozen feature extractor) and trains/evaluates a classifier over the
RPC dataset's 17 meta-categories.

Design note: MobileNetV2 (ImageNet-pretrained) is used ONLY to produce a
fixed embedding per image — its classification head is removed, and no
weights are fine-tuned. A separate classifier (Random Forest) is trained
on top of these embeddings using our own labeled data. This is a legitimate
transfer-learning approach: the pretrained network is explained and
integrated into our own pipeline, not used as a black-box classifier.
"""

import os
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------------------------------------------
# Feature extraction (CNN-based)
# ---------------------------------------------------------------------------

CNN_INPUT_SIZE = (224, 224)  # MobileNetV2's expected input size

_feature_extractor = None  # lazy-loaded singleton, so the model loads only once


def get_feature_extractor():
    """
    Load MobileNetV2 (ImageNet weights, no top/classification head) once
    and reuse it across calls. Output is a 1280-dim embedding per image
    (from global average pooling).
    """
    global _feature_extractor
    if _feature_extractor is None:
        print("Loading MobileNetV2 (pretrained on ImageNet) as feature extractor...")
        base_model = MobileNetV2(
            input_shape=CNN_INPUT_SIZE + (3,),
            include_top=False,
            weights="imagenet",
            pooling="avg"  # global average pooling -> 1280-dim vector output
        )
        base_model.trainable = False  # freeze weights — used purely as a feature extractor
        _feature_extractor = base_model
        print("Feature extractor ready.")
    return _feature_extractor


def extract_features(image):
    """
    Extract a 1280-dim CNN embedding from a single BGR image using
    MobileNetV2's frozen convolutional base.

    Args:
        image: BGR image (NumPy array), any size

    Returns:
        1D NumPy feature vector (length 1280)
    """
    model = get_feature_extractor()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, CNN_INPUT_SIZE)

    batch = np.expand_dims(resized, axis=0).astype(np.float32)
    batch = preprocess_input(batch)  # MobileNetV2-specific normalization

    features = model.predict(batch, verbose=0)
    return features.flatten()


def extract_features_batch(images, batch_size=32):
    """
    Extract CNN embeddings for a list of images in batches — much faster
    than calling extract_features() one image at a time.

    Args:
        images: list of BGR images (NumPy arrays)
        batch_size: number of images to process per forward pass

    Returns:
        2D NumPy array, shape (len(images), 1280)
    """
    model = get_feature_extractor()

    all_features = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        processed_chunk = []
        for img in chunk:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, CNN_INPUT_SIZE)
            processed_chunk.append(resized)

        batch = np.array(processed_chunk, dtype=np.float32)
        batch = preprocess_input(batch)

        features = model.predict(batch, verbose=0)
        all_features.append(features)

    return np.vstack(all_features)


# ---------------------------------------------------------------------------
# Training data loading
# ---------------------------------------------------------------------------

def load_training_data(manifest_path, processed_dir, batch_size=32):
    """
    Load CNN feature embeddings and labels for all images listed in the
    dataset manifest. Uses the whole preprocessed exemplar image directly
    (no segmentation) — exemplar images already contain a single product
    per frame, and the segmentation module is tuned/validated for checkout
    tray conditions rather than the exemplar backdrop, so applying it here
    produced unreliable crops (verified visually).
    """
    images, y, filenames = [], [], []

    with open(manifest_path, "r") as f:
        for line in f:
            filename, supercategory, _ = line.strip().split("\t")
            image_path = os.path.join(processed_dir, filename)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            images.append(image)
            y.append(supercategory)
            filenames.append(filename)

    print(f"Extracting CNN features for {len(images)} images...")
    X = extract_features_batch(images, batch_size=batch_size)

    return X, np.array(y), filenames


def load_training_data_segmented(manifest_path, processed_dir, batch_size=32):
    """
    Same as load_training_data, but runs segmentation + detection first and
    extracts CNN features from the segmented product crop instead of the
    whole image — keeps training consistent with inference-time crops.
    """
    from src.segmentation import segment_products
    from src.detection import detect_products, crop_products

    images, y, filenames = [], [], []

    with open(manifest_path, "r") as f:
        for line in f:
            filename, supercategory, _ = line.strip().split("\t")
            image_path = os.path.join(processed_dir, filename)

            image = cv2.imread(image_path)
            if image is None:
                continue

            seg_result = segment_products(image)
            detections = detect_products(image, seg_result)

            if not detections:
                continue  # segmentation found nothing on this exemplar image — skip

            best_det = max(detections, key=lambda d: d["area"])
            crop = crop_products(image, [best_det])[0][1]

            images.append(crop)
            y.append(supercategory)
            filenames.append(filename)

    print(f"Extracting CNN features for {len(images)} segmented crops...")
    X = extract_features_batch(images, batch_size=batch_size)

    return X, np.array(y), filenames


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_classifier(X, y, test_size=0.2, seed=42):
    """
    Train a Random Forest classifier on CNN embeddings, with a stratified
    train/test split.

    Returns:
        model, label_encoder, X_test, y_test, y_pred
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, label_encoder, X_test, y_test, y_pred


def evaluate_classifier(y_test, y_pred, label_encoder):
    """Print accuracy and a full classification report."""
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    ))
    return acc


def save_model(model, label_encoder, scaler, model_dir="../models"):
    """Save trained classifier, label encoder, and scaler to disk."""

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    print(f"Saved model, label encoder, and scaler to {model_dir}")


def load_model(model_dir="../models"):
    """Load a previously trained classifier and label encoder from disk."""
    model = joblib.load(os.path.join(model_dir, "classifier.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return model, label_encoder


# ---------------------------------------------------------------------------
# Inference on new crops (used with segmentation module's output)
# ---------------------------------------------------------------------------

def classify_crop(crop, model, label_encoder):
    """
    Predict the category of a single product crop.

    Args:
        crop: BGR image (NumPy array) of a single detected product
        model: trained classifier
        label_encoder: fitted LabelEncoder

    Returns:
        (predicted_label, confidence)
    """
    features = extract_features(crop).reshape(1, -1)
    probs = model.predict_proba(features)[0]

    pred_idx = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]

    return predicted_label, confidence


def classify_detections(crops, model, label_encoder):
    """
    Classify a list of (det_id, crop) pairs, as returned by segmentation's crop_products().

    Args:
        crops: list of (det_id, crop_image) tuples
        model: trained classifier
        label_encoder: fitted LabelEncoder

    Returns:
        List of dicts: [{'det_id': ..., 'label': ..., 'confidence': ...}, ...]
    """
    results = []
    for det_id, crop in crops:
        label, confidence = classify_crop(crop, model, label_encoder)
        results.append({"det_id": det_id, "label": label, "confidence": float(confidence)})
    return results