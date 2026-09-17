

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




CNN_INPUT_SIZE = (224, 224)  

_feature_extractor = None  


def get_feature_extractor():
    
    global _feature_extractor
    if _feature_extractor is None:
        print("Loading MobileNetV2 (pretrained on ImageNet) as feature extractor...")
        base_model = MobileNetV2(
            input_shape=CNN_INPUT_SIZE + (3,),
            include_top=False,
            weights="imagenet",
            pooling="avg"  
        )
        base_model.trainable = False  
        _feature_extractor = base_model
        print("Feature extractor ready.")
    return _feature_extractor


def extract_features(image):
    
    model = get_feature_extractor()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, CNN_INPUT_SIZE)

    batch = np.expand_dims(resized, axis=0).astype(np.float32)
    batch = preprocess_input(batch)  

    features = model.predict(batch, verbose=0)
    return features.flatten()


def extract_features_batch(images, batch_size=32):
    
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




def load_training_data(manifest_path, processed_dir, batch_size=32):
    
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
                continue  

            best_det = max(detections, key=lambda d: d["area"])
            crop = crop_products(image, [best_det])[0][1]

            images.append(crop)
            y.append(supercategory)
            filenames.append(filename)

    print(f"Extracting CNN features for {len(images)} segmented crops...")
    X = extract_features_batch(images, batch_size=batch_size)

    return X, np.array(y), filenames




def train_classifier(X, y, test_size=0.2, seed=42):
    
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
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    ))
    return acc


def save_model(model, label_encoder, scaler, model_dir="../models"):
    

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    print(f"Saved model, label encoder, and scaler to {model_dir}")


def load_model(model_dir="../models"):

    model = joblib.load(os.path.join(model_dir, "classifier.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return model, label_encoder




def classify_crop(crop, model, label_encoder):
    
    features = extract_features(crop).reshape(1, -1)
    probs = model.predict_proba(features)[0]

    pred_idx = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]

    return predicted_label, confidence


def classify_detections(crops, model, label_encoder):
    
    results = []
    for det_id, crop in crops:
        label, confidence = classify_crop(crop, model, label_encoder)
        results.append({"det_id": det_id, "label": label, "confidence": float(confidence)})
    return results